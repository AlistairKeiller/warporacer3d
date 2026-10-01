"""Cross-platform Mojo racing and PPO; no Python on the execution path."""
from std.sys import argv, stdin
from std.io.io import _fdopen
from std.time import perf_counter
from std.math import isfinite
from racer.device import Device, Params, Ptr, read_floats, GPU_AVAILABLE
from racer.simulation import (
    spawn,
    step,
    observe,
    env_size,
    action_offset,
    reward_offset,
    STATE,
    OBS,
)
from racer.network import memory, initialize, forward, WEIGHTS
from racer.ppo import rollout, update, input as policy_input, sample
from racer.checkpoint import save, load
from racer.terrain import validate
from racer.lidar import (
    RAYS,
    ROWS,
    BEAMS,
    RANGE,
    ELEVATION,
    MOUNT_FORWARD,
    MOUNT_HEIGHT,
)


def accumulate(i: Int, data: Ptr, map: Ptr, p: Params):
    var n = Int(p.envs)
    var mem = memory(n)
    var result = reward_offset(n) + i * 2
    var stats = mem.obs + i * 3
    data[unsafe_offset=stats] += data[unsafe_offset=result]
    if data[unsafe_offset=result + 1] == 1:
        data[unsafe_offset=stats + 1] += 1
    elif data[unsafe_offset=result + 1] == 3:
        data[unsafe_offset=stats + 2] += 1


def evaluate(mut device: Device, n: Int, steps: Int, seed: UInt32) raises:
    for t in range(steps):
        device.run[policy_input](n * OBS, Params(Int32(n), seed, -1))
        forward(device, n, n, False)
        device.run[sample](n, Params(Int32(n), seed, 0, Int32(t), 1))
        device.run[step](n, Params(Int32(n), seed))
        device.run[accumulate](n, Params(Int32(n), seed))
        device.run[observe](n * RAYS, Params(Int32(n), seed))
    var stats = device.data.create_sub_buffer[DType.float32](
        memory(n).obs, n * 3
    )
    var reward: Float64 = 0
    var collisions: Float64 = 0
    var finishes: Float64 = 0
    with stats.map_to_host() as host:
        for i in range(n):
            reward += Float64(host[i * 3])
            collisions += Float64(host[i * 3 + 1])
            finishes += Float64(host[i * 3 + 2])
    print(
        "reward/step",
        reward / Float64(n * steps),
        "failures/car",
        collisions / Float64(n),
        "finishes/car",
        finishes / Float64(n),
    )


def controls(i: Int, data: Ptr, map: Ptr, p: Params):
    var n = Int(p.envs)
    var a = action_offset(n) + i * 2
    data[unsafe_offset=a] = Float32(p.offset) / 100
    data[unsafe_offset=a + 1] = Float32(p.index) / 100
    if p.flag > 0:
        spawn(i, data, map, p)
    var r = reward_offset(n) + i * 2
    data[unsafe_offset=r] = 0
    data[unsafe_offset=r + 1] = 0


def serve(mut device: Device, n: Int, seed: UInt32) raises:
    print(
        '{"ready":true,"state":',
        STATE,
        ',"obs":',
        OBS,
        ',"beams":',
        BEAMS,
        ',"rows":',
        ROWS,
        ',"range":',
        RANGE,
        ',"elevation":',
        ELEVATION,
        ',"mount":[',
        MOUNT_FORWARD,
        ",0,",
        MOUNT_HEIGHT,
        "]}",
        flush=True,
    )
    var snapshot = device.data.create_sub_buffer[DType.float32](0, env_size(n))
    # Mojo 1.1 input() leaks a duplicated descriptor per call. Reuse one
    # buffered stream, whose context closes the descriptor on exit.
    with _fdopen["r"](stdin) as stream:
        while True:
            var command = stream.readline()
            if not command or command == "quit":
                return
            var words = command.split()
            if len(words) != 4:
                raise Error(
                    "viewer command requires steering, throttle, reset, policy"
                )
            device.run[controls](
                n,
                Params(
                    Int32(n),
                    seed,
                    Int32(Int(words[0])),
                    Int32(Int(words[1])),
                    Int32(Int(words[2])),
                ),
            )
            if Int(words[2]) > 0:
                device.run[observe](n * RAYS, Params(Int32(n), seed))
            if Int(words[3]) > 0:
                device.run[policy_input](n * OBS, Params(Int32(n), seed, -1))
                forward(device, n, n, False)
                device.run[sample](n, Params(Int32(n), seed, 0, 0, 1))
            device.run[step](n, Params(Int32(n), seed))
            device.run[observe](n * RAYS, Params(Int32(n), seed))
            var json = String("[")
            with snapshot.map_to_host() as host:
                for i in range(env_size(n)):
                    if i > 0:
                        json += ","
                    json += String(host[i])
            json += "]"
            print(json, flush=True)


def benchmark_actions(i: Int, data: Ptr, map: Ptr, p: Params):
    var offset = action_offset(Int(p.envs)) + i * 2
    data[unsafe_offset=offset] = 0.1
    data[unsafe_offset=offset + 1] = 0.7


def main() raises:
    var args = argv()
    if len(args) < 3:
        print(
            "Usage: mojo main.mojo benchmark|train|eval|serve MAP.wrmap"
            " [auto|gpu|cpu] [environments] [steps|iterations] [checkpoint]"
            " [resume]"
        )
        return
    var mode = String(args[1])
    if (
        mode != "benchmark"
        and mode != "train"
        and mode != "eval"
        and mode != "serve"
    ):
        raise Error("mode must be benchmark, train, eval, or serve")
    var map_data = read_floats(String(args[2]))
    var gpu = (
        GPU_AVAILABLE if len(args) < 4
        or String(args[3]) == "auto" else String(args[3]) != "cpu"
    )
    if (
        len(args) > 3
        and String(args[3]) != "cpu"
        and String(args[3]) != "gpu"
        and String(args[3]) != "auto"
    ):
        raise Error(
            "device must be auto, cpu, or gpu (Metal, CUDA, or HIP is selected"
            " automatically)"
        )
    var n = Int(String(args[4])) if len(args) > 4 else 1024
    var iterations = Int(String(args[5])) if len(args) > 5 else 1000
    var path = String(args[6]) if len(args) > 6 else String("agent.wrppo")
    if n < 1 or n > 32768 or iterations < 1 or iterations > 1000000:
        raise Error("environments must be 1..32768 and iterations 1..1000000")
    validate(map_data)
    var device = Device(
        gpu, env_size(n) if mode == "benchmark" else memory(n).size, map_data
    )
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
    device.run[observe](n * RAYS, p)
    if mode == "benchmark":
        device.run[benchmark_actions](n, p)
        for _ in range(10):
            device.run[step](n, p)
            device.run[observe](n * RAYS, p)
        device.ctx.synchronize()
        var start = perf_counter()
        for _ in range(iterations):
            device.run[step](n, p)
            device.run[observe](n * RAYS, p)
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
            var stats = device.data.create_sub_buffer[DType.float32](
                memory(n).stats, 8
            )
            with stats.map_to_host() as host:
                if (
                    not isfinite(host[2])
                    or not isfinite(host[3])
                    or not isfinite(host[4])
                ):
                    raise Error("nonfinite PPO statistics")
                print(
                    "iteration",
                    iteration + 1,
                    "reward/step",
                    host[2],
                    "KL",
                    host[4],
                )
            if (iteration + 1) % 100 == 0:
                save(device, path, n, iteration + 1, optimizer_step, seed)
        save(device, path, n, first + iterations, optimizer_step, seed)
        var elapsed = perf_counter() - start
        print(
            "Saved",
            path,
            ";",
            Float64(n * 32 * iterations) / elapsed,
            "training transitions/s",
        )
