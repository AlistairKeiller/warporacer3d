"""Dynamics, device parity, GAE, checkpoints, and PPO gradients vs finite differences.

    mojo -I . tests/test_mojo.mojo build/flat.wrmap build/ramp.wrmap [cpu]
"""
from std.sys import argv
from std.math import isfinite, exp, log, tanh, sqrt, sin, cos, floor
from racer.device import (
    Device,
    Params,
    Ptr,
    read_floats,
    uniform,
    GPU_AVAILABLE,
)
from racer.vehicle import Car, frame, integrate
from racer.layout import (
    memory,
    permutation,
    env_size,
    action_offset,
    reward_offset,
    STATE,
    OBS,
    IN,
    H,
    HA,
    OUT,
    W0,
    W1,
    W2,
    LOGSTD,
    WEIGHTS,
    TOTAL,
    ROLLOUT,
    MINIBATCHES,
    CHUNK,
)
from racer.simulation import spawn, SENSE_ONLY, POLICY
from racer.network import initialize, pad_activations, transpose_w1
from racer.ppo import step, gae, backprop
from racer.checkpoint import save, load
from racer.track import Mesh, Route, Track, Vec3, vec3, ribbon, demo_track
from racer.compile import compile_track
from racer.terrain import validate
from racer.lidar import RANGE, ELEVATION, MOUNT_FORWARD, MOUNT_HEIGHT
from racer.layout import BEAMS


def check(ok: Bool, message: StaticString) raises:
    if not ok:
        raise Error(String(message))


def vehicle_tests() raises:
    check(
        uniform(0) > 0 and uniform(0xD5F9EFC4) < 1,
        "random samples must exclude both endpoints",
    )
    var rest = Car(0, 0, 0, 0, 0, 0, 0)
    for _ in range(2400):
        integrate(rest, frame(0, 0, 0), 0, 0, 1.1, 1)
    check(
        rest.x == 0 and rest.u == 0 and rest.yaw == 0,
        "rest must remain stationary",
    )
    var full = rest
    var saturated = rest
    for _ in range(240):
        integrate(full, frame(full.heading, 0, 0), 1, 1, 1.1, 1)
        integrate(saturated, frame(saturated.heading, 0, 0), 3, 3, 1.1, 1)
    check(
        abs(full.u - saturated.u) < 1e-6
        and abs(full.heading - saturated.heading) < 1e-6,
        "actuators must saturate consistently",
    )
    var car = rest
    for _ in range(2400):
        integrate(car, frame(car.heading, 0, 0), 0, 1, 1.1, 1)
    check(
        car.x > 25 and car.u > 3 and car.u < 5 and abs(car.y) < 1e-6,
        "straight driving",
    )
    var speed = car.u
    for _ in range(240):
        integrate(car, frame(car.heading, 0, 0), 0, 0, 1.1, 1)
    check(car.u < speed and car.u > 0, "passive coasting must lose speed")
    car = rest
    for _ in range(480):
        integrate(car, frame(car.heading, 0, 0), 0.5, -1, 1.1, 1)
    check(car.u < -1 and car.yaw < 0 and isfinite(car.v), "reverse steering")
    car = Car(0, 0, 0, 2, 0, 0, 0)
    var uphill = car
    integrate(car, frame(0, 0, 0), 0, 0, 1.1, 1)
    integrate(uphill, frame(0, 0.3, 0), 0, 0, 1.1, 1)
    check(uphill.u < car.u, "uphill gravity must slow the car")
    for slope in range(-3, 4):
        var f = frame(0.6, Float32(slope) * 0.1, -0.2)
        check(
            abs(f.fx * f.fx + f.fy * f.fy + f.fz * f.fz - 1) < 1e-5,
            "forward frame length",
        )
        check(
            abs(f.lx * f.lx + f.ly * f.ly + f.lz * f.lz - 1) < 1e-5,
            "left frame length",
        )
        check(
            abs(f.fx * f.lx + f.fy * f.ly + f.fz * f.lz) < 1e-5,
            "orthogonal frame",
        )
    for n in range(1, 20):
        var seen = List[Bool](length=n * ROLLOUT, fill=False)
        for i in range(n * ROLLOUT):
            var j = permutation(i, n, 123)
            check(not seen[j], "minibatch permutation must be a bijection")
            seen[j] = True
    print("vehicle dynamics, terrain frames, and shuffle: passed")


def actions(i: Int, data: Ptr, map: Ptr, p: Params):
    var offset = action_offset(Int(p.envs)) + i * 2
    data[unsafe_offset=offset] = 0.03
    data[unsafe_offset=offset + 1] = 0.6


def drive(mut device: Device, n: Int, steps: Int) raises:
    var p = Params(Int32(n), 42)
    device.run[spawn](n, p)
    step(device, n, Params(Int32(n), 42, 0, 0, SENSE_ONLY))
    device.run[actions](n, p)
    for t in range(steps):
        step(device, n, Params(Int32(n), 42, 0, Int32(t), 0))


def boundary_fixture(i: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(1)
    data[unsafe_offset=mem.values + ROLLOUT] = 123
    for t in range(ROLLOUT):
        data[unsafe_offset=mem.values + t] = 0
        data[unsafe_offset=mem.rewards + t] = 1
        data[unsafe_offset=mem.done + t] = Float32(1) if t == 30 else Float32(0)


def environment_tests(map_path: String, gpu: Bool) raises:
    var n = 16
    var map = read_floats(map_path)
    var cpu = Device(False, memory(n).size, map)
    drive(cpu, n, 64)
    var snapshot = cpu.read(0, n * (STATE + IN))
    for i in range(n * (STATE + IN)):
        check(
            isfinite(snapshot[i]),
            "environment state and observation must remain finite",
        )
    step(cpu, n, Params(Int32(n), 42, 0, 0, SENSE_ONLY))
    var again = cpu.read(0, n * (STATE + IN))
    for i in range(n * (STATE + IN)):
        check(again[i] == snapshot[i], "sensing must not change state or RNG")
    if gpu:
        var metal = Device(True, memory(n).size, map)
        drive(metal, n, 64)
        var other = metal.read(0, n * (STATE + IN))
        for i in range(n * STATE):
            check(
                abs(snapshot[i] - other[i]) < 0.002, "CPU/GPU dynamics mismatch"
            )
        for i in range(n * STATE, n * (STATE + IN)):
            check(
                abs(snapshot[i] - other[i]) < 0.01, "CPU/GPU sensing mismatch"
            )
        print(map_path, "CPU/GPU dynamics and sensing: passed")
    var state = cpu.read(0, STATE)
    state[0] = 1000
    state[1] = 1000
    cpu.write(0, state)
    step(cpu, n, Params(Int32(n), 42, 0, 0, 0))
    var result = cpu.read(reward_offset(n), 2)
    check(result[1] == 1, "unsupported car must fail")
    state = cpu.read(0, STATE)
    check(
        state[9] == snapshot[9] + 1 and abs(state[0]) < 20,
        "terminal car must reset",
    )
    state[8] = 2999
    cpu.write(0, state)
    step(cpu, n, Params(Int32(n), 42, 0, 0, 0))
    result = cpu.read(reward_offset(n), 2)
    check(result[1] == 2, "finite-horizon episode must time out")
    check(cpu.read(8, 1)[0] == 0, "timeout must reset the episode clock")
    var p = Params(Int32(n), 42)
    cpu.run[initialize](WEIGHTS, p)
    save(cpu, "/tmp/warporacer3d-roundtrip.wrppo", n, 123, 456, 0xFEDCBA98)
    var other = Device(gpu, memory(1).size, map)
    var loaded = load(other, "/tmp/warporacer3d-roundtrip.wrppo", 1)
    check(
        loaded[0] == 123 and loaded[1] == 456 and loaded[2] == 0xFEDCBA98,
        "checkpoint counters and full uint32 seed",
    )
    var a = cpu.read(memory(n).weights, WEIGHTS * 4)
    var b = other.read(memory(1).weights, WEIGHTS * 4)
    for i in range(WEIGHTS * 4):
        check(
            a[i] == b[i], "checkpoint must preserve all parameters and moments"
        )
    print("terminal reset, pure observation, cross-device checkpoint: passed")
    var one = Device(False, memory(1).size, map)
    one.run[boundary_fixture](1, Params(1, 0))
    one.run[gae](1, Params(1, 0))
    var target = one.read(memory(1).target, ROLLOUT)
    check(abs(target[30] - 1) < 1e-5, "GAE must stop at terminal states")
    check(abs(target[29] - 1.9405) < 1e-5, "GAE recurrence")
    check(abs(target[31] - 122.77) < 1e-4, "rollout tail must bootstrap")
    print("GAE episode boundaries: passed")


# ===-------------------------------------------------------------------=== #
# PPO gradient: analytic backprop vs central finite differences (float64).
# ===-------------------------------------------------------------------=== #

comptime GRAD_N = 2
comptime GRAD_SAMPLES = GRAD_N * ROLLOUT
comptime GRAD_BATCH = GRAD_SAMPLES // MINIBATCHES


def gradient_fixture(i: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(GRAD_N)
    for j in range(IN):
        var value: Float32 = 0
        if j < OBS:
            value = 0.2 * Float32(((i * OBS + j) * 7919) % 1000) / 1000 - 0.1
        elif j == OBS:
            value = 1
        data[unsafe_offset=mem.obs + i * IN + j] = value
    data[unsafe_offset=mem.actions + i * 2] = -0.4 + 0.1 * Float32(i % 9)
    data[unsafe_offset=mem.actions + i * 2 + 1] = -0.2 + 0.05 * Float32(i % 11)
    data[unsafe_offset=mem.values + i] = 0.1 * Float32(i % 7) - 0.3
    data[unsafe_offset=mem.logp + i] = -0.5 - 0.3 * Float32(i % 8)
    data[unsafe_offset=mem.advantage + i] = Float32((i * 37) % 11) / 5 - 1
    data[unsafe_offset=mem.target + i] = Float32((i * 53) % 13) / 6 - 1
    if i == 0:
        data[unsafe_offset=mem.stats] = 0
        data[unsafe_offset=mem.stats + 1] = 1


def host_loss(w: List[Float64], arena: List[Float32]) -> Float64:
    """The clipped PPO objective of minibatch 0 in float64 on the host."""
    var mem = memory(GRAD_N)
    var total: Float64 = 0
    for b in range(GRAD_BATCH):
        var row = permutation(b, GRAD_N, 42)
        var h1 = List[Float64](length=HA, fill=0)
        var h2 = List[Float64](length=HA, fill=0)
        for j in range(H):
            var acc: Float64 = 0
            for k in range(IN):
                acc += (
                    Float64(arena[mem.obs + row * IN + k]) * w[W0 + k * H + j]
                )
            h1[j] = tanh(acc)
        h1[H] = 1
        for j in range(H):
            var acc: Float64 = 0
            for k in range(HA):
                acc += h1[k] * w[W1 + k * H + j]
            h2[j] = tanh(acc)
        h2[H] = 1
        var out = List[Float64](length=OUT, fill=0)
        for o in range(OUT):
            for k in range(HA):
                out[o] += h2[k] * w[W2 + o * HA + k]
        var logp: Float64 = 0
        for j in range(2):
            var error = Float64(arena[mem.actions + row * 2 + j]) - out[j]
            var ls = w[LOGSTD + j]
            logp -= 0.5 * error * error * exp(-2 * ls) + ls + 0.918938533204673
        var ratio = exp(logp - Float64(arena[mem.logp + row]))
        var adv = Float64(arena[mem.advantage + row])
        var clipped = min(max(ratio, 0.8), 1.2)
        total -= min(ratio * adv, clipped * adv) / GRAD_BATCH
        var old = Float64(arena[mem.values + row])
        var target = Float64(arena[mem.target + row])
        var value = out[2]
        var clipped_value = old + min(max(value - old, -0.2), 0.2)
        var d = value - target
        var dc = clipped_value - target
        total += 0.25 * max(d * d, dc * dc) / GRAD_BATCH
    return total


def gradient_tests(map_path: String, gpu: Bool) raises:
    var map = read_floats(map_path)
    var n = GRAD_N
    var p = Params(Int32(n), 42)
    var cpu = Device(False, memory(n).size, map)
    cpu.run[initialize](WEIGHTS, p)
    cpu.run[transpose_w1](HA * H, p)
    cpu.run[pad_activations](GRAD_BATCH * HA, p)
    cpu.run[gradient_fixture](GRAD_SAMPLES, p)
    var chunks = (GRAD_BATCH + CHUNK - 1) // CHUNK
    var q = Params(Int32(n), 42, 0)
    backprop(cpu, n, GRAD_BATCH, chunks, q)
    var arena = cpu.read(0, memory(n).size)
    var analytic = cpu.read(memory(n).gradient, WEIGHTS)
    var weights = List[Float64](capacity=WEIGHTS)
    for i in range(WEIGHTS):
        weights.append(Float64(arena[memory(n).weights + i]))
    var largest: Float32 = 0
    for i in range(WEIGHTS):
        largest = max(largest, abs(analytic[i]))
    check(largest > 1e-4, "gradient fixture must produce a nontrivial gradient")
    var worst: Float64 = 0
    var checked = 0
    for i in range(0, WEIGHTS, 97):
        var eps: Float64 = 1e-4
        var saved = weights[i]
        weights[i] = saved + eps
        var plus = host_loss(weights, arena)
        weights[i] = saved - eps
        var minus = host_loss(weights, arena)
        weights[i] = saved
        var numeric = (plus - minus) / (2 * eps)
        var error = abs(numeric - Float64(analytic[i]))
        var tolerance = 1e-5 + 5e-3 * abs(numeric)
        if error > tolerance:
            raise Error(
                "gradient "
                + String(i)
                + " analytic "
                + String(analytic[i])
                + " numeric "
                + String(numeric)
            )
        worst = max(worst, error)
        checked += 1
    for i in range(LOGSTD, WEIGHTS):
        var eps: Float64 = 1e-4
        var saved = weights[i]
        weights[i] = saved + eps
        var plus = host_loss(weights, arena)
        weights[i] = saved - eps
        var minus = host_loss(weights, arena)
        weights[i] = saved
        var numeric = (plus - minus) / (2 * eps)
        check(
            abs(numeric - Float64(analytic[i])) < 1e-5 + 5e-3 * abs(numeric),
            "log std gradient",
        )
    print(
        "analytic PPO gradient matches finite differences:",
        checked + 2,
        "entries, worst error",
        worst,
    )
    if gpu:
        var metal = Device(True, memory(n).size, map)
        metal.run[initialize](WEIGHTS, p)
        metal.run[transpose_w1](HA * H, p)
        metal.run[pad_activations](GRAD_BATCH * HA, p)
        metal.run[gradient_fixture](GRAD_SAMPLES, p)
        backprop(metal, n, GRAD_BATCH, chunks, q)
        var other = metal.read(memory(n).gradient, WEIGHTS)
        var scale: Float32 = 0
        for i in range(WEIGHTS):
            scale = max(scale, abs(analytic[i]))
        for i in range(WEIGHTS):
            check(
                abs(other[i] - analytic[i]) <= 1e-4 * scale + 1e-6,
                "CPU/GPU gradient mismatch",
            )
        print("CPU/GPU gradients agree")


def main() raises:
    var args = argv()
    vehicle_tests()
    var gpu = GPU_AVAILABLE if len(args) < 4 else String(args[3]) != "cpu"
    environment_tests(String(args[1]), gpu)
    environment_tests(String(args[2]), gpu)
    gradient_tests(String(args[1]), gpu)
    compiler_tests(gpu)


# ===-------------------------------------------------------------------=== #
# Map compiler: geometry checks, and a thin wall seen by the lidar.
# ===-------------------------------------------------------------------=== #


def quad(x0: Float64, y0: Float64, x1: Float64, y1: Float64) raises -> Mesh:
    var vertices = List[Vec3]()
    vertices.append(vec3(x0, y0, 0))
    vertices.append(vec3(x1, y0, 0))
    vertices.append(vec3(x1, y1, 0))
    vertices.append(vec3(x0, y1, 0))
    var faces = List[Int]()
    for index in [0, 1, 2, 0, 2, 3]:
        faces.append(index)
    return Mesh(vertices^, faces^)


def wall(x: Float64, y0: Float64, y1: Float64, height: Float64) raises -> Mesh:
    var vertices = List[Vec3]()
    vertices.append(vec3(x, y0, 0))
    vertices.append(vec3(x, y1, 0))
    vertices.append(vec3(x, y1, height))
    vertices.append(vec3(x, y0, height))
    var faces = List[Int]()
    for index in [0, 1, 2, 0, 2, 3]:
        faces.append(index)
    return Mesh(vertices^, faces^, True)


def straight(x0: Float64, x1: Float64) raises -> Route:
    var points = List[Vec3]()
    points.append(vec3(x0, 0, 0))
    points.append(vec3(x1, 0, 0))
    var widths = List[Float64]()
    widths.append(1.5)
    widths.append(1.5)
    return Route(points^, widths^, List[Vec3](), False)


def compiler_tests(gpu: Bool) raises:
    for kind in ["flat", "ramp", "bank"]:
        var data = compile_track(demo_track(String(kind)), 0.05)
        validate(data)
        var nx = Int(data[2])
        var ny = Int(data[3])
        check(Int(data[10]) > 0, "demo maps must have spawns")
        var slope: Float32 = 0
        for i in range(16 + 2 * nx * ny, 16 + 4 * nx * ny):
            slope = max(slope, abs(data[i]))
        if kind == "flat":
            check(slope == 0, "flat map must have no slope")
        else:
            check(slope > 0.05, "ramp and bank maps must have slopes")
    var failed = False
    try:
        _ = compile_track(demo_track("flat"), 1.0)
    except:
        failed = True
    check(failed, "invalid resolution must be rejected")
    # A figure-eight whose crossing stacks road at two heights.
    var points = List[Vec3]()
    var widths = List[Float64]()
    for i in range(256):
        var t = 2 * 3.141592653589793 * Float64(i) / 256
        points.append(vec3(8 * sin(t), 4 * sin(2 * t), 1 + cos(t)))
        widths.append(0.85)
    var eight = Route(points^, widths^, List[Vec3](), True)
    var parts = List[Mesh]()
    parts.append(ribbon(eight))
    failed = False
    try:
        _ = compile_track(Track(parts^, eight^, "eight"), 0.05)
    except error:
        failed = "overlapping" in String(error)
    check(failed, "bridges must be rejected")
    # A route that never touches the road has no spawn.
    var off = List[Mesh]()
    off.append(ribbon(demo_track("flat").route))
    failed = False
    try:
        _ = compile_track(Track(off^, straight(100, 101), "off"), 0.05)
    except error:
        failed = "spawn" in String(error)
    check(failed, "routes off the road must be rejected")
    # A 1.1 cm wall must still block its cell.
    var thin = List[Mesh]()
    thin.append(quad(-2, -2, 2, 2))
    thin.append(wall(0.011, -1, 1, 1))
    var narrow = List[Vec3]()
    narrow.append(vec3(-1, -1, 0))
    narrow.append(vec3(-1, 1, 0))
    var narrow_widths = List[Float64]()
    narrow_widths.append(1.5)
    narrow_widths.append(1.5)
    var data = compile_track(
        Track(
            thin^, Route(narrow^, narrow_widths^, List[Vec3](), False), "thin"
        ),
        0.05,
    )
    var nx = Int(data[2])
    var x = Int(floor((0.011 - data[6]) / data[8] + 0.5))
    var y = Int(floor((0 - data[7]) / data[8] + 0.5))
    check(data[16 + y * nx + x] < 0, "sub-cell wall must survive")
    print("map compiler geometry: passed")
    # Drive 32 cars at a thin wall across an open road and watch the lidar.
    var road = List[Mesh]()
    road.append(quad(-4, -3, 4, 3))
    road.append(wall(0.011, -3, 3, 0.6))
    var map = compile_track(Track(road^, straight(-3, 3), "wall"))
    validate(map)
    var n = 32
    var device = Device(gpu, memory(n).size, map)
    var p = Params(Int32(n), 42)
    device.run[spawn](n, p)
    step(device, n, Params(Int32(n), 42, 0, 0, SENSE_ONLY))
    var live = device.read(0, n * (STATE + IN))
    var middle = BEAMS // 2
    var angle = -3 * 3.141592653589793 / 4 + Float64(middle) * (
        3 * 3.141592653589793 / 2
    ) / Float64(BEAMS - 1)
    var seen_left = False
    var seen_right = False
    for i in range(n):
        var s = n * STATE + i * IN + 8
        check(live[i * STATE + 9] == 1, "every baked spawn is safe")
        var car_x = Float64(live[i * STATE])
        var forward = Float64(live[s + BEAMS + middle]) * Float64(RANGE)
        var down = Float64(live[s + middle]) * Float64(RANGE)
        var up = Float64(live[s + 2 * BEAMS + middle]) * Float64(RANGE)
        if car_x < -0.5:
            seen_left = True
            var expected = (0.011 - car_x - Float64(MOUNT_FORWARD)) / cos(angle)
            check(
                abs(forward - expected) < 1e-3, "forward beam must hit the wall"
            )
        elif car_x > 0.5 and car_x < 2.5:
            seen_right = True
            check(
                abs(down - Float64(MOUNT_HEIGHT) / sin(Float64(ELEVATION)))
                < 1e-4,
                "downward beam must hit the floor",
            )
            check(abs(forward - Float64(RANGE)) < 1e-6, "open road ahead")
        if car_x < -2:
            check(
                abs(up - Float64(RANGE)) < 1e-6, "upward beam clears the wall"
            )
    check(seen_left and seen_right, "spawns on both sides of the wall")
    device.run[full_throttle](n, p)
    var crashed = False
    var finished = False
    for t in range(600):
        step(device, n, Params(Int32(n), 42, 0, Int32(t), 0))
        var result = device.read(reward_offset(n), n * 2)
        var state = device.read(0, n * STATE)
        for i in range(n):
            if result[i * 2 + 1] > 0:
                check(
                    state[i * STATE + 8] == 0,
                    "terminal cars must respawn immediately",
                )
            if result[i * 2 + 1] == 1:
                crashed = True
            if result[i * 2 + 1] == 3:
                finished = True
    check(crashed, "wall impact must terminate")
    check(finished, "open route must finish")
    print("thin wall lidar, finish, and reset: passed")


def full_throttle(i: Int, data: Ptr, map: Ptr, p: Params):
    var offset = action_offset(Int(p.envs)) + i * 2
    data[unsafe_offset=offset] = 0
    data[unsafe_offset=offset + 1] = 1
