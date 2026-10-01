"""Cross-platform Mojo racing and PPO; no Python on the execution path."""
from std.sys import argv
from std.time import perf_counter
from std.math import isfinite
from racer.core import Device, Params, Ptr, read_floats, GPU_AVAILABLE
from racer.env import spawn, step, observe, env_size, action_offset, BEAMS
from racer.network import memory, initialize, WEIGHTS
from racer.ppo import rollout, update
from racer.checkpoint import save, load
from racer.evaluate import evaluate
from racer.interactive import serve
from racer.terrain import validate


def benchmark_actions(i: Int, data: Ptr, map: Ptr, p: Params):
    var offset = action_offset(Int(p.envs)) + i * 2
    data[unsafe_offset=offset] = 0.1
    data[unsafe_offset=offset + 1] = 0.7


def main() raises:
    var args = argv()
    if len(args) < 3:
        print(
            "Usage: mojo main.mojo benchmark|train|eval|serve MAP.wrmap [auto|gpu|cpu]"
            " [environments] [steps|iterations] [checkpoint] [resume]"
        )
        return
    var mode = String(args[1])
    if mode != "benchmark" and mode != "train" and mode != "eval" and mode != "serve":
        raise Error("mode must be benchmark, train, eval, or serve")
    var map_data = read_floats(String(args[2]))
    var gpu = (
        GPU_AVAILABLE if len(args) < 4 or String(args[3]) == "auto" else String(args[3]) != "cpu"
    )
    if (
        len(args) > 3
        and String(args[3]) != "cpu"
        and String(args[3]) != "gpu"
        and String(args[3]) != "auto"
    ):
        raise Error(
            "device must be auto, cpu, or gpu (Metal, CUDA, or HIP is selected automatically)"
        )
    var n = Int(String(args[4])) if len(args) > 4 else 1024
    var iterations = Int(String(args[5])) if len(args) > 5 else 1000
    var path = String(args[6]) if len(args) > 6 else String("agent.wrppo")
    if n < 1 or n > 32768 or iterations < 1 or iterations > 1000000:
        raise Error("environments must be 1..32768 and iterations 1..1000000")
    validate(map_data)
    var device = Device(gpu, env_size(n) if mode == "benchmark" else memory(n).size, map_data)
    print("Device:", device.ctx.name())
    var seed: UInt32 = 42
    var first = 0
    var optimizer_step = 0
    if mode == "train":
        device.run[initialize](WEIGHTS, Params(Int32(n), seed))
        if len(args) > 7:
            first, optimizer_step, seed = load(device, String(args[7]), n)
        if first + iterations > 1000000:
            raise Error("total training iterations must be at most 1000000")
    elif mode == "eval" or (mode == "serve" and path != "-"):
        first, optimizer_step, seed = load(device, path, n)
    var p = Params(Int32(n), seed)
    device.run[spawn](n, p)
    device.run[observe](n * BEAMS, p)
    if mode == "benchmark":
        device.run[benchmark_actions](n, p)
        for _ in range(10):
            device.run[step](n, p)
            device.run[observe](n * BEAMS, p)
        device.ctx.synchronize()
        var start = perf_counter()
        for _ in range(iterations):
            device.run[step](n, p)
            device.run[observe](n * BEAMS, p)
        device.ctx.synchronize()
        var elapsed = perf_counter() - start
        print(
            n,
            "cars;",
            elapsed * 1000 / Float64(iterations),
            "ms/step;",
            Float64(n * iterations) / elapsed,
            "transitions/s",
        )
    elif mode == "serve":
        serve(device, n, seed)
    elif mode == "eval":
        evaluate(device, n, iterations, seed)
    else:
        device.ctx.synchronize()
        var start = perf_counter()
        for iteration in range(first, first + iterations):
            rollout(device, n, iteration, seed)
            update(device, n, iteration, optimizer_step, seed)
            var stats = device.data.create_sub_buffer[DType.float32](memory(n).stats, 8)
            with stats.map_to_host() as host:
                if not isfinite(host[2]) or not isfinite(host[3]) or not isfinite(host[4]):
                    raise Error("nonfinite PPO statistics")
                print("iteration", iteration + 1, "reward/step", host[2], "KL", host[4])
            if (iteration + 1) % 100 == 0:
                save(device, path, n, iteration + 1, optimizer_step, seed)
        save(device, path, n, first + iterations, optimizer_step, seed)
        var elapsed = perf_counter() - start
        print("Saved", path, ";", Float64(n * 32 * iterations) / elapsed, "training transitions/s")
