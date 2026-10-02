"""Cross-platform Mojo racing and PPO; no Python on the execution path."""
from std.sys import argv
import std.os.path as os_path
from std.time import perf_counter
from std.math import isfinite
from racer.device import (
    Device,
    Params,
    Ptr,
    read_floats,
    write_floats,
    clamp,
    GPU_AVAILABLE,
)
from racer.http import Server
from racer.layout import (
    memory,
    env_size,
    action_offset,
    reward_offset,
    stats_offset,
    WEIGHTS,
    ROLLOUT,
    H,
    HA,
    MINIBATCHES,
    STATE,
    IN,
    BEAMS,
    ROWS,
)
from racer.simulation import spawn, POLICY, DETERMINISTIC, EVAL, SENSE_ONLY
from racer.network import initialize, pad_activations, transpose_w1
from racer.ppo import step, rollout, update
from racer.checkpoint import save, load
from racer.terrain import validate
from racer.compile import load_track, compile_track
from racer.lidar import RANGE, ELEVATION, MOUNT_FORWARD, MOUNT_HEIGHT


def evaluate(mut device: Device, n: Int, steps: Int, seed: UInt32) raises:
    for t in range(steps):
        step(
            device,
            n,
            Params(Int32(n), seed, 0, Int32(t), POLICY | DETERMINISTIC | EVAL),
        )
    var stats = device.read(stats_offset(n), n * 3)
    var reward: Float64 = 0
    var collisions: Float64 = 0
    var finishes: Float64 = 0
    for i in range(n):
        reward += Float64(stats[i * 3])
        collisions += Float64(stats[i * 3 + 1])
        finishes += Float64(stats[i * 3 + 2])
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


def serve(
    mut device: Device,
    n: Int,
    seed: UInt32,
    port: Int,
    map_path: String,
    map_data: List[Float32],
    policy: Bool,
) raises:
    """Serve viewer.html, the raw map, and binary state snapshots over HTTP."""
    var page = read_page()
    var pieces = map_path.split("/")
    var info = String(
        '{"name":"',
        String(pieces[len(pieces) - 1]),
        '","cars":',
        n,
        ',"policy":',
        "true" if policy else "false",
        ',"state":',
        STATE,
        ',"obs":',
        IN,
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
    )
    var server = Server(port)
    print("http://127.0.0.1:" + String(port) + " (Ctrl+C to stop)", flush=True)
    var tick = 0
    while True:
        var client = server.accept()
        try:
            var request = client.request()
            if request.method == "GET" and request.path == "/":
                client.respond(
                    "200 OK", "text/html; charset=utf-8", page.as_bytes()
                )
            elif request.method == "GET" and request.path == "/info":
                client.respond("200 OK", "application/json", info.as_bytes())
            elif request.method == "GET" and request.path == "/map":
                client.respond(
                    "200 OK",
                    "application/octet-stream",
                    Span(
                        unsafe_ptr=map_data.unsafe_ptr().unsafe_bitcast[
                            UInt8
                        ](),
                        length=len(map_data) * 4,
                    ),
                )
            elif request.method == "POST" and request.path == "/step":
                var words = request.body.split()
                if len(words) != 4:
                    raise Error(
                        "step body must be: steering throttle reset policy"
                    )
                var steering = clamp(Float32(Int(words[0])), -100, 100)
                var throttle = clamp(Float32(Int(words[1])), -100, 100)
                var reset = Int(words[2]) > 0
                var drive = Int(words[3]) > 0 and policy
                device.run[controls](
                    n,
                    Params(
                        Int32(n),
                        seed,
                        Int32(steering),
                        Int32(throttle),
                        Int32(1) if reset else Int32(0),
                    ),
                )
                if reset:
                    step(device, n, Params(Int32(n), seed, 0, 0, SENSE_ONLY))
                tick += 1
                step(
                    device,
                    n,
                    Params(
                        Int32(n),
                        seed,
                        0,
                        Int32(tick),
                        Int32(POLICY) if drive else Int32(0),
                    ),
                )
                var snapshot = device.read(0, env_size(n))
                client.respond(
                    "200 OK",
                    "application/octet-stream",
                    Span(
                        unsafe_ptr=snapshot.unsafe_ptr().unsafe_bitcast[
                            UInt8
                        ](),
                        length=len(snapshot) * 4,
                    ),
                )
            else:
                client.respond(
                    "404 Not Found", "text/plain", "not found".as_bytes()
                )
        except error:
            try:
                client.respond(
                    "400 Bad Request", "text/plain", String(error).as_bytes()
                )
            except:
                pass


def read_page() raises -> String:
    """viewer.html from the working directory or next to the build folder."""
    var executable = String(argv()[0])
    var candidates = List[String]()
    candidates.append("viewer.html")
    var slash = executable.rfind("/")
    if slash >= 0:
        candidates.append(executable[byte=0:slash] + "/../viewer.html")
        candidates.append(executable[byte=0:slash] + "/viewer.html")
    for path in candidates:
        if os_path.isfile(path):
            return open(path, "r").read()
    raise Error("viewer.html not found; run from the project directory")


def benchmark_actions(i: Int, data: Ptr, map: Ptr, p: Params):
    var offset = action_offset(Int(p.envs)) + i * 2
    data[unsafe_offset=offset] = 0.1
    data[unsafe_offset=offset + 1] = 0.7


def measure(
    mut device: Device, n: Int, steps: Int, flags: Int
) raises -> Float64:
    for t in range(10):
        step(device, n, Params(Int32(n), 7, 0, Int32(t), Int32(flags)))
    device.ctx.synchronize()
    var start = perf_counter()
    for t in range(steps):
        step(device, n, Params(Int32(n), 7, 0, Int32(t), Int32(flags)))
    device.ctx.synchronize()
    return (perf_counter() - start) / Float64(steps)


def main() raises:
    var args = argv()
    if len(args) < 3:
        print(
            "Usage: racer prepare flat|ramp|bank|MAP.yaml OUT.wrmap"
            " [resolution]\n       racer benchmark|train|eval MAP.wrmap"
            " [auto|gpu|cpu] [cars] [steps|iterations] [checkpoint] [resume]\n "
            "      racer serve MAP.wrmap [auto|gpu|cpu] [cars] [port]"
            " [checkpoint]"
        )
        return
    var mode = String(args[1])
    if mode == "prepare":
        if len(args) < 4:
            raise Error(
                "usage: racer prepare flat|ramp|bank|MAP.yaml OUT.wrmap"
                " [resolution]"
            )
        var resolution = Float64(String(args[4])) if len(args) > 4 else 0.025
        var compiled = compile_track(load_track(String(args[2])), resolution)
        validate(compiled)
        write_floats(String(args[3]), compiled)
        print(
            String(args[3]),
            ":",
            Int(compiled[2]),
            "x",
            Int(compiled[3]),
            "cells,",
            Int(compiled[4]),
            "segments,",
            Int(compiled[12]),
            "triangles,",
            Float64(len(compiled) * 4) / 1e6,
            "MB",
        )
        return
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
    var iterations = 1
    if mode != "serve":
        iterations = Int(String(args[5])) if len(args) > 5 else 1000
    var path = String(args[6]) if len(args) > 6 else String("agent.wrppo")
    if n < 1 or n > 32768 or iterations < 1 or iterations > 1000000:
        raise Error("environments must be 1..32768 and iterations 1..1000000")
    validate(map_data)
    var training = mode == "train"
    var device = Device(
        gpu, memory(n).size if training else memory(n).obs, map_data
    )
    print("Device:", device.ctx.name())
    var seed: UInt32 = 42
    var first = 0
    var optimizer_step = 0
    var p = Params(Int32(n), seed)
    if training:
        device.run[initialize](WEIGHTS, p)
        device.run[pad_activations](n * ROLLOUT // MINIBATCHES * HA, p)
        if len(args) > 7:
            first, optimizer_step, seed = load(device, String(args[7]), n)
        device.run[transpose_w1](HA * H, p)
        if first + iterations > 1000000:
            raise Error("total training iterations must be at most 1000000")
    elif mode == "eval" or (mode == "serve" and path != "-"):
        first, optimizer_step, seed = load(device, path, n)
    elif mode == "benchmark":
        device.run[initialize](WEIGHTS, p)
    p = Params(Int32(n), seed)
    device.run[spawn](n, p)
    step(device, n, Params(Int32(n), seed, 0, 0, SENSE_ONLY))
    if mode == "benchmark":
        device.run[benchmark_actions](n, p)
        var physics = measure(device, n, iterations, 0)
        var policy = measure(device, n, iterations, POLICY)
        print(
            n,
            "cars;",
            physics * 1000,
            "ms/step dynamics+lidar;",
            policy * 1000,
            "ms/step with policy;",
            Float64(n) / policy,
            "transitions/s",
        )
    elif mode == "serve":
        serve(
            device,
            n,
            seed,
            Int(String(args[5])) if len(args) > 5 else 8765,
            String(args[2]),
            map_data,
            path != "-",
        )
    elif mode == "eval":
        evaluate(device, n, iterations, seed)
    else:
        device.ctx.synchronize()
        var start = perf_counter()
        for iteration in range(first, first + iterations):
            rollout(device, n, iteration, seed)
            update(device, n, iteration, optimizer_step, seed)
            var stats = device.read(memory(n).stats, 8)
            if (
                not isfinite(stats[2])
                or not isfinite(stats[3])
                or not isfinite(stats[4])
            ):
                raise Error("nonfinite PPO statistics")
            print(
                "iteration",
                iteration + 1,
                "reward/step",
                stats[2],
                "KL",
                stats[4],
            )
            if (iteration + 1) % 100 == 0:
                save(device, path, n, iteration + 1, optimizer_step, seed)
        save(device, path, n, first + iterations, optimizer_step, seed)
        var elapsed = perf_counter() - start
        print(
            "Saved",
            path,
            ";",
            Float64(n * ROLLOUT * iterations) / elapsed,
            "training transitions/s",
        )
