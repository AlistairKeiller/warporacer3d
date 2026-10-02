"""racer: compile maps, train and evaluate PPO drivers, and serve the viewer."""
from std.sys import argv
from std.time import perf_counter
from std.math import isfinite
from std.python import Python, PythonObject
from layout import TileTensor, Coord, row_major
from layout.numpy import to_numpy
from racer.device import Device, Buffer, GPU_AVAILABLE, mat, read_floats, write_floats
from racer.compile import load_track, compile
from racer.simulation import Sim, STATE, IN, OUT, CRASH, FINISH
from racer.network import Policy, Activations
from racer.ppo import Trainer, ROLLOUT, checkpoint
from racer.lidar import BEAMS, RANGE, MOUNT_FORWARD, MOUNT_HEIGHT

comptime USAGE = """racer prepare   flat|ramp|bank|MAP.yaml OUT.wrmap [resolution=0.025]
racer train     MAP.wrmap [auto|gpu|cpu] [cars] [iterations] [checkpoint] [resume]
racer eval      MAP.wrmap [auto|gpu|cpu] [cars] [steps] [checkpoint]
racer benchmark MAP.wrmap [auto|gpu|cpu] [cars] [steps]
racer serve     MAP.wrmap [auto|gpu|cpu] [cars] [port] [checkpoint|-]"""


def act(
    device: Device,
    mut sim: Sim,
    mut policy: Policy,
    mut acts: Activations,
    mut scratch: Buffer,
    deterministic: Bool,
    t: Int,
) raises:
    """One environment step for every car, acting with the policy."""
    policy.forward(device, mat[IN](sim.obs, sim.n), acts)
    sim.sample(
        device,
        mat[OUT](acts.o, sim.n),
        policy.log_std(),
        mat[2](sim.actions, sim.n),
        mat[1](scratch, sim.n),
        mat[1](scratch, sim.n, sim.n),
        deterministic,
        t,
    )
    sim.physics(
        device,
        mat[2](sim.actions, sim.n),
        mat[1](sim.reward, sim.n),
        mat[1](sim.done, sim.n),
    )
    sim.sense(device, mat[IN](sim.obs, sim.n))


def as_bytes(values: List[Float32]) raises -> PythonObject:
    return to_numpy(
        TileTensor(Span(values), row_major(Coord(len(values))))
    ).tobytes()


def serve(
    device: Device,
    mut sim: Sim,
    mut policy: Policy,
    mut acts: Activations,
    mut scratch: Buffer,
    map_data: List[Float32],
    name: String,
    port: Int,
    trained: Bool,
) raises:
    """Drive the viewer: Python serves HTTP, each POST /step runs one step."""
    var info = String(
        '{"name":"', name, '","cars":', sim.n, ',"policy":', "true" if trained else "false",
        ',"state":', STATE, ',"obs":', IN, ',"beams":', BEAMS, ',"range":', RANGE,
        ',"mount":[', MOUNT_FORWARD, ",0,", MOUNT_HEIGHT, "]}",
    )
    Python.add_to_path(".")
    var bridge = Python.import_module("racer.serve").Bridge(
        port, open("viewer.html", "r").read(), info, as_bytes(map_data)
    )
    print("http://127.0.0.1:" + String(port) + " (Ctrl+C to stop)", flush=True)
    var tick = 0
    while True:
        var words = String(bridge.wait()).split()
        var steering = Float32(Int(String(words[0]))) / 100
        var throttle = Float32(Int(String(words[1]))) / 100
        if Int(String(words[2])) > 0:
            sim.spawn_all(device)
            sim.sense(device, mat[IN](sim.obs, sim.n))
        tick += 1
        if Int(String(words[3])) > 0 and trained:
            act(device, sim, policy, acts, scratch, True, tick)
        else:
            var actions = List[Float32](length=sim.n, fill=steering)
            actions.extend(List[Float32](length=sim.n, fill=throttle))
            device.write(sim.actions, actions)
            sim.physics(
                device,
                mat[2](sim.actions, sim.n),
                mat[1](sim.reward, sim.n),
                mat[1](sim.done, sim.n),
            )
            sim.sense(device, mat[IN](sim.obs, sim.n))
        var snapshot = device.read(sim.state)
        for buffer in [sim.obs, sim.actions, sim.reward, sim.done]:
            snapshot.extend(device.read(buffer))
        bridge.reply(as_bytes(snapshot))


def main() raises:
    var args = argv()
    if len(args) < 3:
        print(USAGE)
        return
    var mode = String(args[1])
    if mode == "prepare":
        var resolution = Float64(String(args[4])) if len(args) > 4 else 0.025
        var map = compile(load_track(String(args[2])), resolution)
        write_floats(String(args[3]), map)
        print(
            String(args[3]), ":", Int(map[2]), "x", Int(map[3]), "cells,",
            Int(map[8]), "segments,", Int(map[14]), "triangles,",
            Float64(len(map) * 4) / 1e6, "MB",
        )
        return
    if mode != "train" and mode != "eval" and mode != "benchmark" and mode != "serve":
        raise Error(USAGE)
    var map_data = read_floats(String(args[2]))
    var choice = String(args[3]) if len(args) > 3 else String("auto")
    var gpu = GPU_AVAILABLE if choice == "auto" else choice == "gpu"
    var n = Int(String(args[4])) if len(args) > 4 else 1024
    var count = Int(String(args[5])) if len(args) > 5 else (8765 if mode == "serve" else 1000)
    var path = String(args[6]) if len(args) > 6 else String("agent.wrppo")
    var device = Device(gpu)
    print("Device:", device.ctx.name())
    var sim = Sim(device, map_data, n, 42)
    var policy = Policy(device, 42)
    var acts = Activations(device, n)
    var scratch = device.alloc(2 * n)
    sim.spawn_all(device)
    sim.sense(device, mat[IN](sim.obs, n))
    if mode == "train":
        var trainer = Trainer(device, n)
        var first = 0
        if len(args) > 7:
            var saved = checkpoint(device, policy, String(args[7]))
            trainer.load(device, saved)
            first = Int(saved[3])
        var start = perf_counter()
        for iteration in range(first, first + count):
            var reward = trainer.rollout(device, sim, policy, iteration)
            var kl = trainer.update(device, policy, iteration)
            if not isfinite(kl):
                raise Error("training diverged (nonfinite KL)")
            print("iteration", iteration + 1, "reward/step", reward, "KL", kl)
            if (iteration + 1) % 100 == 0:
                trainer.save(device, policy, path, iteration + 1)
        trainer.save(device, policy, path, first + count)
        print(
            "Saved", path, ";",
            Float64(n * ROLLOUT * count) / (perf_counter() - start),
            "training transitions/s",
        )
        return
    var trained = mode == "eval" or (mode == "serve" and path != "-")
    if trained:
        _ = checkpoint(device, policy, path)
    if mode == "serve":
        serve(device, sim, policy, acts, scratch, map_data, String(args[2]), count, trained)
    elif mode == "benchmark":
        for t in range(10):
            act(device, sim, policy, acts, scratch, False, t)
        device.ctx.synchronize()
        var start = perf_counter()
        for t in range(count):
            act(device, sim, policy, acts, scratch, False, t)
        device.ctx.synchronize()
        var elapsed = perf_counter() - start
        print(n, "cars;", elapsed / Float64(count) * 1000, "ms/step;", Float64(n * count) / elapsed, "transitions/s")
    else:
        var reward: Float64 = 0
        var crashes = 0
        var finishes = 0
        for t in range(count):
            act(device, sim, policy, acts, scratch, True, t)
            for r in device.read(sim.reward):
                reward += Float64(r)
            for d in device.read(sim.done):
                crashes += Int(d == CRASH)
                finishes += Int(d == FINISH)
        print(
            "reward/step", reward / Float64(n * count),
            "crashes/car", Float64(crashes) / Float64(n),
            "finishes/car", Float64(finishes) / Float64(n),
        )
