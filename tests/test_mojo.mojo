"""Dynamics, map geometry, lidar, device parity, GAE, and PPO gradients.

    uv run mojo -I . tests/test_mojo.mojo [cpu]
"""
from std.sys import argv
from std.math import isfinite, exp, log, tanh, cos, sin, sqrt, atan2
from std.testing import assert_true, assert_almost_equal
from racer.device import Device, mat, GPU_AVAILABLE
from racer.vehicle import Car, frame, integrate
from racer.map import Map
from racer.compile import (
    Mesh,
    Route,
    Track,
    Vec3,
    quad,
    demo,
    compile,
)
from racer.simulation import Sim, OBS, IN, OUT, X, Y, HEADING, EPISODE, STEPS, CRASH, FINISH, attitude
from racer.network import Policy, P, H, HA, W0, W1, W2, LOGSTD
from racer.ppo import Trainer, ROLLOUT, MINIBATCHES, permutation
from racer.lidar import BEAMS, RANGE, FOV, MOUNT_FORWARD, Vec, ray, walk


def close(a: List[Float32], b: List[Float32], tolerance: Float64, message: String) raises:
    for i in range(len(a)):
        assert_almost_equal(a[i], b[i], message, atol=tolerance, rtol=0)


def vehicle_tests() raises:
    var rest = Car(0, 0, 0, 0, 0, 0, 0)
    for _ in range(2400):
        integrate(rest, frame(0, 0, 0), 0, 0, 1.1, 1)
    assert_true(rest.x == 0 and rest.u == 0 and rest.yaw == 0, "rest stays at rest")
    var car = rest
    for _ in range(2400):
        integrate(car, frame(car.heading, 0, 0), 0, 1, 1.1, 1)
    assert_true(car.x > 25 and car.u > 3 and car.u < 5 and abs(car.y) < 1e-6, "straight driving")
    var speed = car.u
    for _ in range(240):
        integrate(car, frame(car.heading, 0, 0), 0, 0, 1.1, 1)
    assert_true(car.u < speed and car.u > 0, "coasting loses speed")
    car = rest
    for _ in range(480):
        integrate(car, frame(car.heading, 0, 0), 0.5, -1, 1.1, 1)
    assert_true(car.u < -1 and car.yaw < 0 and isfinite(car.v), "reverse steering")
    var level = Car(0, 0, 0, 2, 0, 0, 0)
    var uphill = level
    integrate(level, frame(0, 0, 0), 0, 0, 1.1, 1)
    integrate(uphill, frame(0, 0.3, 0), 0, 0, 1.1, 1)
    assert_true(uphill.u < level.u, "gravity slows the car uphill")
    for slope in range(-3, 4):
        var f = frame(0.6, Float32(slope) * 0.1, -0.2)
        assert_true(abs(f.fx * f.fx + f.fy * f.fy + f.fz * f.fz - 1) < 1e-5, "unit forward")
        assert_true(abs(f.fx * f.lx + f.fy * f.ly + f.fz * f.lz) < 1e-5, "orthogonal frame")
    for n in range(1, 20):
        var seen = List[Bool](length=n * ROLLOUT, fill=False)
        for i in range(n * ROLLOUT):
            var j = permutation(i, n, 123)
            assert_true(not seen[j], "the minibatch permutation is a bijection")
            seen[j] = True
    print("vehicle dynamics, terrain frames, and shuffle: passed")


def map_tests() raises:
    for kind in ["flat", "ramp", "bank"]:
        var data = compile(demo(String(kind)), 0.05)
        var map = Map(data)
        assert_true(map.spawns > 0 and map.triangles > 0, "demo maps have spawns and walls")
        var heights = Span(data)[map.height : map.height + map.nx * map.ny]
        var top: Float32 = 0
        for h in heights:
            top = max(top, h)
        assert_true((top > 1) == (kind == "ramp"), "only the ramp climbs")
        # Height and clearance must be continuous along the road, which the
        # old nearest-cell lookup was not (cars bobbed once per cell).
        var copy = Device(False).upload(data)
        var d = mat[1](copy, len(data))
        var previous = map.surface(d, 6, 0)
        for k in range(1, 2000):
            var angle = Float32(k) * 0.0005
            var here = map.surface(d, 6 * cos(angle), 6 * sin(angle))
            assert_true(here.clearance > 0.5, "the route centre is clear of the walls")
            assert_true(abs(here.height - previous.height) < 0.004, "height is continuous")
            assert_true(abs(here.clearance - previous.clearance) < 0.01, "clearance is continuous")
            previous = here
    print("map compiler geometry and continuity: passed")


def attitude_tests() raises:
    """The chassis attitude comes from the ground under the wheels, so the
    facets of a curved ribbon (whose two triangles per quad have different
    grades) do not pitch the car at every cell: no more stair-step bobbing."""
    var data = compile(demo("ramp"), 0.025)
    var map = Map(data)
    var cpu = Device(False)
    var copy = cpu.upload(data)
    var d = mat[1](copy, len(data))
    var previous: Float32 = 0
    for k in range(7500):
        var angle = Float32(k) * 0.005 / 6
        var a = attitude(map, d, 6 * cos(angle), 6 * sin(angle), angle + 1.5707964)
        var slope = -a.sx * sin(angle) + a.sy * cos(angle)
        assert_true(k == 0 or abs(slope - previous) < 0.005, "the grade under the wheels is smooth")
        previous = slope
    var n = 1
    var sim = Sim(cpu, data, n, 7)
    sim.spawn_all(cpu)
    cpu.write(sim.actions, [Float32(0), 1])
    var last: Float32 = 0
    var climbed = False
    for t in range(600):
        sim.physics(cpu, mat[2](sim.actions, n), mat[1](sim.reward, n), mat[1](sim.done, n))
        var s = cpu.read(sim.state)
        var a = attitude(map, d, s[X], s[Y], s[HEADING])
        var f = frame(s[HEADING], a.sx, a.sy)
        var pitch = atan2(f.fz, sqrt(f.fx * f.fx + f.fy * f.fy)) * 57.29578
        if t > 0 and cpu.read(sim.done)[0] == 0:
            assert_true(abs(pitch - last) < 1, "pitch changes smoothly between steps")
        climbed = climbed or abs(pitch) > 3
        last = pitch
    assert_true(climbed, "the car drove onto the ramp")
    print("wheel-based attitude is smooth along and while driving the ramp: passed")


def arena(width: Float64, wall_x: Float64) raises -> List[Float32]:
    """An open floor with a thin wall across the middle and a straight route."""
    var floor = Mesh(List[Vec3](), List[Int]())
    quad(floor, Vec3(-4, -3, 0, 0), Vec3(4, -3, 0, 0), Vec3(4, 3, 0, 0), Vec3(-4, 3, 0, 0))
    var wall = Mesh(List[Vec3](), List[Int](), True)
    quad(wall, Vec3(wall_x, -3, 0, 0), Vec3(wall_x, 3, 0, 0), Vec3(wall_x, 3, 0.6, 0), Vec3(wall_x, -3, 0.6, 0))
    var route = Route([Vec3(-3, 0, 0, 0), Vec3(3, 0, 0, 0)], [width, width], List[Vec3](), False)
    return compile(Track([floor^, wall^], route^))


def environment_tests(gpu: Bool) raises:
    var data = arena(1.5, 0.011)
    var n = 32
    var cpu = Device(False)
    var sim = Sim(cpu, data, n, 42)
    sim.spawn_all(cpu)
    sim.sense(cpu, mat[IN](sim.obs, n))
    var state = cpu.read(sim.state)
    var obs = cpu.read(sim.obs)
    var middle = BEAMS // 2
    var left = False
    var right = False
    for i in range(n):
        assert_true(state[EPISODE * n + i] == 1, "every spawn is valid")
        var x = state[X * n + i]
        var forward = obs[(8 + middle) * n + i] * RANGE
        if x < -0.5:
            left = True
            assert_almost_equal(forward, 0.011 - x - MOUNT_FORWARD, "the centre beam hits the wall", atol=2e-3, rtol=0)
        elif x > 0.5:
            right = True
            assert_almost_equal(forward, 4 - x - MOUNT_FORWARD, "the centre beam hits the far wall", atol=2e-3, rtol=0)
        assert_true(obs[OBS * n + i] == 1, "the bias input is one")
    assert_true(left and right, "spawns on both sides of the wall")
    var actions = List[Float32](length=n, fill=0)
    actions.extend(List[Float32](length=n, fill=1))
    cpu.write(sim.actions, actions)
    var crashed = False
    var finished = False
    for _ in range(600):
        sim.physics(cpu, mat[2](sim.actions, n), mat[1](sim.reward, n), mat[1](sim.done, n))
        var done = cpu.read(sim.done)
        var steps = cpu.read(sim.state)
        for i in range(n):
            if done[i] > 0:
                assert_true(steps[STEPS * n + i] == 0, "terminal cars respawn at once")
            crashed = crashed or done[i] == CRASH
            finished = finished or done[i] == FINISH
    assert_true(crashed and finished, "hitting the wall crashes; the open side finishes")
    sim.sense(cpu, mat[IN](sim.obs, n))
    for value in cpu.read(sim.obs):
        assert_true(isfinite(value), "observations stay finite")
    print("spawns, thin-wall lidar, crash, finish, and reset: passed")
    if not gpu:
        return
    var metal = Device(True)
    var other = Sim(metal, data, n, 42)
    other.spawn_all(metal)
    metal.write(other.actions, actions)
    sim = Sim(cpu, data, n, 42)
    sim.spawn_all(cpu)
    cpu.write(sim.actions, actions)
    for _ in range(60):
        sim.physics(cpu, mat[2](sim.actions, n), mat[1](sim.reward, n), mat[1](sim.done, n))
        other.physics(metal, mat[2](other.actions, n), mat[1](other.reward, n), mat[1](other.done, n))
    sim.sense(cpu, mat[IN](sim.obs, n))
    other.sense(metal, mat[IN](other.obs, n))
    close(cpu.read(sim.state), metal.read(other.state), 2e-3, "CPU/GPU dynamics agree")
    close(cpu.read(sim.obs), metal.read(other.obs), 1e-2, "CPU/GPU sensing agrees")
    print("CPU/GPU dynamics and sensing: passed")


def lidar_tests() raises:
    """The clearance-field march must hand over before anything the exact
    triangle walk would hit, so both agree on every beam of driven cars."""
    var n = 64
    var device = Device(False)
    var data = compile(demo("ramp"), 0.05)
    var sim = Sim(device, data, n, 7)
    sim.spawn_all(device)
    var actions = List[Float32](length=n, fill=0.3)
    actions.extend(List[Float32](length=n, fill=1))
    device.write(sim.actions, actions)
    for _ in range(90):
        sim.physics(device, mat[2](sim.actions, n), mat[1](sim.reward, n), mat[1](sim.done, n))
    sim.sense(device, mat[IN](sim.obs, n))
    var poses = device.read(sim.pose)
    var copy = device.upload(data)
    var d = mat[1](copy, len(data))
    assert_true(sim.map.slope > 0.3 and sim.map.slope < 0.4, "the ramp's slope bound")
    for i in range(n):
        var origin = Vec(poses[i], poses[n + i], poses[2 * n + i], 0)
        var forward = Vec(poses[3 * n + i], poses[4 * n + i], poses[5 * n + i], 0)
        var left = Vec(poses[6 * n + i], poses[7 * n + i], poses[8 * n + i], 0)
        for beam in range(0, BEAMS, 5):
            var azimuth = -FOV / 2 + Float32(beam) * (FOV / Float32(BEAMS - 1))
            var direction = cos(azimuth) * forward + sin(azimuth) * left
            var exact = walk(sim.map, d, origin, direction, RANGE)
            assert_almost_equal(ray(sim.map, d, origin, direction), exact, "the lidar march agrees with the exact walk", atol=1e-3, rtol=0)
    print("lidar march against the exact triangle walk: passed")


def gae_tests() raises:
    var device = Device(False)
    var trainer = Trainer(device, 1)
    var values = List[Float32](length=ROLLOUT + 1, fill=0)
    values[ROLLOUT] = 123
    device.write(trainer.value, values)
    device.write(trainer.reward, List[Float32](length=ROLLOUT, fill=1))
    var done = List[Float32](length=ROLLOUT, fill=0)
    done[30] = 1
    device.write(trainer.done, done)
    trainer.gae(device)
    var targets = device.read(trainer.target)
    assert_almost_equal(targets[30], 1, "GAE stops at terminal states", atol=1e-5, rtol=0)
    assert_almost_equal(targets[29], 1.9405, "GAE recurrence", atol=1e-5, rtol=0)
    assert_almost_equal(targets[31], 122.77, "the rollout tail bootstraps", atol=1e-4, rtol=0)
    print("GAE episode boundaries: passed")


# ===-------------------------------------------------------------------=== #
# PPO gradient: analytic backprop vs central finite differences (float64).
# ===-------------------------------------------------------------------=== #

comptime GRAD_N = 4
comptime BATCH = GRAD_N * ROLLOUT // MINIBATCHES


def fixture(device: Device, trainer: Trainer) raises:
    """Deterministic rollout storage: observations, actions, old log probs,
    values, advantages, and targets for GRAD_N cars."""
    var n = GRAD_N
    var obs = List[Float32](length=(ROLLOUT + 1) * IN * n, fill=0)
    var actions = List[Float32](length=ROLLOUT * 2 * n, fill=0)
    var logp = List[Float32](length=ROLLOUT * n, fill=0)
    var values = List[Float32](length=(ROLLOUT + 1) * n, fill=0)
    var advantage = List[Float32](length=ROLLOUT * n, fill=0)
    var target = List[Float32](length=ROLLOUT * n, fill=0)
    for s in range(ROLLOUT * n):
        var t = s // n
        var i = s % n
        for r in range(OBS):
            obs[(t * IN + r) * n + i] = 0.2 * Float32(((s * OBS + r) * 7919) % 1000) / 1000 - 0.1
        obs[(t * IN + OBS) * n + i] = 1
        actions[(t * 2) * n + i] = -0.4 + 0.1 * Float32(s % 9)
        actions[(t * 2 + 1) * n + i] = -0.2 + 0.05 * Float32(s % 11)
        values[s] = 0.1 * Float32(s % 7) - 0.3
        logp[s] = -0.5 - 0.3 * Float32(s % 8)
        advantage[s] = Float32((s * 37) % 11) / 5 - 1
        target[s] = Float32((s * 53) % 13) / 6 - 1
    device.write(trainer.obs, obs)
    device.write(trainer.actions, actions)
    device.write(trainer.logp, logp)
    device.write(trainer.value, values)
    device.write(trainer.advantage, advantage)
    device.write(trainer.target, target)


def host_loss(w: List[Float64], trainer: Trainer, device: Device) raises -> Float64:
    """The clipped PPO objective of minibatch 0 in float64 on the host."""
    var n = GRAD_N
    var obs = device.read(trainer.obs)
    var actions = device.read(trainer.actions)
    var logp = device.read(trainer.logp)
    var values = device.read(trainer.value)
    var advantage = device.read(trainer.advantage)
    var target = device.read(trainer.target)
    var total: Float64 = 0
    for b in range(BATCH):
        var s = permutation(b, n, 0)
        var t = s // n
        var i = s % n
        var h1 = List[Float64](length=HA, fill=1)
        var h2 = List[Float64](length=HA, fill=1)
        for j in range(H):
            var acc: Float64 = 0
            for k in range(IN):
                acc += Float64(obs[(t * IN + k) * n + i]) * w[W0 + j * IN + k]
            h1[j] = tanh(acc)
        for j in range(H):
            var acc: Float64 = 0
            for k in range(HA):
                acc += h1[k] * w[W1 + j * HA + k]
            h2[j] = tanh(acc)
        var out = List[Float64](length=OUT, fill=0)
        for o in range(OUT):
            for k in range(HA):
                out[o] += h2[k] * w[W2 + o * HA + k]
        var new_logp: Float64 = 0
        for a in range(2):
            var error = Float64(actions[(t * 2 + a) * n + i]) - out[a]
            var ls = w[LOGSTD + a]
            new_logp -= 0.5 * error * error * exp(-2 * ls) + ls + 0.918938533204673
        var ratio = exp(new_logp - Float64(logp[s]))
        var adv = Float64(advantage[s])
        var clipped = min(max(ratio, 0.8), 1.2)
        total -= min(ratio * adv, clipped * adv) / BATCH
        var old = Float64(values[s])
        var value = out[2]
        var clipped_value = old + min(max(value - old, -0.2), 0.2)
        var d = value - Float64(target[s])
        var dc = clipped_value - Float64(target[s])
        total += 0.25 * max(d * d, dc * dc) / BATCH
    return total


def gradient(device: Device, policy: Policy, trainer: Trainer) raises -> List[Float32]:
    """One minibatch's analytic gradient via the trainer's own kernels."""
    fixture(device, trainer)
    trainer.minibatch(device, policy, 0, 0, 0, 1)
    return device.read(trainer.grad)


def gradient_tests(gpu: Bool) raises:
    var cpu = Device(False)
    var policy = Policy(cpu, 42)
    var trainer = Trainer(cpu, GRAD_N)
    var analytic = gradient(cpu, policy, trainer)
    var weights = List[Float64]()
    for value in cpu.read(policy.theta):
        weights.append(Float64(value))
    var largest: Float32 = 0
    for value in analytic:
        largest = max(largest, abs(value))
    assert_true(largest > 1e-4, "the fixture produces a nontrivial gradient")
    var worst: Float64 = 0
    var checked = 0
    var i = 0
    while i < P:
        var eps: Float64 = 1e-4
        var saved = weights[i]
        weights[i] = saved + eps
        var plus = host_loss(weights, trainer, cpu)
        weights[i] = saved - eps
        var minus = host_loss(weights, trainer, cpu)
        weights[i] = saved
        var numeric = (plus - minus) / (2 * eps)
        var error = abs(numeric - Float64(analytic[i]))
        if error > 1e-5 + 5e-3 * abs(numeric):
            raise Error(
                "gradient " + String(i) + " analytic " + String(analytic[i]) + " numeric " + String(numeric)
            )
        worst = max(worst, error)
        checked += 1
        i += 97 if i < LOGSTD - 97 else 1
    print("analytic PPO gradient matches finite differences:", checked, "entries, worst error", worst)
    if gpu:
        var metal = Device(True)
        var other_policy = Policy(metal, 42)
        var other = Trainer(metal, GRAD_N)
        var fast = gradient(metal, other_policy, other)
        close(fast, analytic, Float64(2e-2 * largest + 1e-5), "CPU/GPU gradients agree")
        print("CPU/GPU gradients agree (GPU fp32 matmul may be TF32-like)")


def main() raises:
    var gpu = GPU_AVAILABLE if len(argv()) < 2 else String(argv()[1]) != "cpu"
    vehicle_tests()
    map_tests()
    attitude_tests()
    environment_tests(gpu)
    lidar_tests()
    gae_tests()
    gradient_tests(gpu)
