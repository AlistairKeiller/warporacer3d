"""Dynamics, corridor lidar/crash/progress, bridge layers, CPU/GPU parity, GAE, and PPO gradients.

    uv run mojo run -I . tests/test_mojo.mojo [cpu]
"""
from std.sys import argv
from std.math import isfinite, exp, tanh, floor, clamp, pi
from std.python import Python
from std.python.numpy import from_numpy_array
from std.testing import assert_true, assert_almost_equal
from racer.device import Device, mat, GPU_AVAILABLE
from racer.sim import Sim, Map, derivative, Vec, GRIP, G, STEER_MAX, ACCEL_MAX, SPEED_MAX, HORIZON, HALF_LOG_2PI
from racer.sim import STATE, IN, OBS, OUT, BEAMS, RANGE, X, Y, HEADING, SPEED, STEER, STEPS, EPISODE, FRICTION, RETURN, PROGRESS, CRASH
from racer.ppo import Policy, Activations, Trainer, P, H, HA, W0, W1, W2, LOGSTD, ROLLOUT, MINIBATCHES, GAMMA, LAMBDA

comptime CELL = Float32(0.05)
comptime NX = 281
comptime NY = 161
comptime WP = 74
comptime N = 4  # cars in the PPO tests
comptime BATCH = N * ROLLOUT // MINIBATCHES


def ramp(x: Float32) -> Float32:
    return max(x - 8, 0) * 0.5


def corridor() -> List[Float32]:
    """A 12 x 6 m corridor along +x from (0, -3): flat, then a 1:2 ramp face from x = 8,
    with waypoints every 0.15 m along y = 0 from x = 0.5 (see racer/track.py for the layout)."""
    var d: List[Float32] = [Float32(NX), Float32(NY), -1, -4, CELL, Float32(WP), 0.15, 1, 0.5, 0, 0, 0]
    for which in range(3):  # clearance, height, nearest waypoint
        for iy in range(NY):
            for ix in range(NX):
                var x = -1 + CELL * Float32(ix)
                var y = -4 + CELL * Float32(iy)
                var nearest = Float32(clamp(Int(floor((x - 0.5) / 0.15 + 0.5)), 0, WP - 1))
                d.append(min(x, min(12 - x, 3 - abs(y))) if which == 0 else (ramp(x) if which == 1 else nearest))
    for w in range(WP):
        d.extend([0.5 + 0.15 * Float32(w), 0, ramp(0.5 + 0.15 * Float32(w)), 0])
    return d^


def step(device: Device, sim: Sim) raises:
    var n = sim.n
    sim.step(device, mat[2](sim.actions, n), mat[1](sim.reward, n), mat[1](sim.done, n), mat[IN](sim.obs, n))


def physics_tests() raises:
    assert_true(derivative(Vec(0, 0, 0, 0), 0, 0, GRIP, 0, 0) == Vec(0), "a car at rest stays at rest")
    assert_true(derivative(Vec(0, 0, 0, 2), 0, 0, GRIP, 0.3, 0)[3] < 0, "an uphill slope slows the car")
    assert_true(derivative(Vec(0, 0, 0, 2), 0, 0, GRIP, 0, 0)[3] == 0, "coasting on the flat holds speed")
    var limit = derivative(Vec(0, 0, 0, 5), STEER_MAX, ACCEL_MAX, GRIP, 0, 0)
    assert_true(abs(limit[2] * 5) <= GRIP * G + 1e-4 and limit[3] <= 1e-4, "the friction circle caps the tyres")
    assert_true(derivative(Vec(0, 0, 0, 2), 0.2, 0, GRIP, 0, 0)[2] > 0.5, "steering yaws the car")
    print("dynamics: passed")


def corridor_tests() raises:
    var track = corridor()
    var cpu = Device(False)
    var n = 4  # at rest facing the ramp; full throttle into the wall; straight; steering left
    var sim = Sim(cpu, Span(track), n, 1)
    var s = List[Float32](length=STATE * n, fill=0)
    for i in range(n):
        s[X * n + i], s[FRICTION * n + i], s[PROGRESS * n + i] = 1, 1, Float32(0.5 / 0.15)
    s[HEADING * n + 1] = Float32(pi)
    cpu.write(sim.state, s)
    cpu.write(sim.actions, [0, 0, 0, 1, 0, 1, 1, 1])
    step(cpu, sim)
    var o = cpu.read(sim.obs)
    assert_almost_equal(o[(4 + BEAMS // 2) * n] * RANGE, 7.1, "the centre beam stops on the ramp face", atol=0.1, rtol=0)
    assert_almost_equal(o[(4 + BEAMS // 2) * n + 1] * RANGE, 0.73, "the centre beam reads the wall", atol=0.1, rtol=0)
    var crashed = False
    for _ in range(60):
        step(cpu, sim)
        var done = cpu.read(sim.done)
        assert_true(done[0] == 0 and done[2] == 0 and done[3] == 0, "the open road does not crash")
        if done[1] == CRASH and not crashed:
            s = cpu.read(sim.state)
            assert_true(s[STEPS * n + 1] == 0 and s[EPISODE * n + 1] == 1, "a crash respawns at once")
            crashed = True
    assert_true(crashed, "driving into the wall crashes")
    s = cpu.read(sim.state)
    o = cpu.read(sim.obs)
    assert_true(s[X * n] == 1 and s[SPEED * n] == 0 and s[HEADING * n] == 0, "a car at rest stays at rest")
    assert_true(s[SPEED * n + 2] == SPEED_MAX and s[X * n + 2] > 3.5 and abs(s[Y * n + 2]) < 1e-4, "full throttle")
    assert_almost_equal(s[RETURN * n + 2], 10 * (s[X * n + 2] - 1), "continuous progress reward", atol=0.01, rtol=0)
    assert_true(s[Y * n + 3] > 0.1 and s[X * n + 3] < s[X * n + 2], "steering turns the car left")
    assert_true(o[n + 2] == 1 and o[OBS * n + 2] == 1, "speed and bias observations")
    for value in o:
        assert_true(isfinite(value), "observations stay finite")
    print("corridor lidar, crash, respawn, speed cap, progress, steering: passed")


def bridge_tests(track: Span[Float32, _]) raises:
    var cpu = Device(False)
    var map = Map(track)
    var copy = cpu.upload(track)
    var d = mat[1](copy, len(track))
    assert_true(map.layers == 2 and map.probe(d, 0, 0, 0).layer == 0 and map.probe(d, 0, 0, 0.5).layer == 1, "layer by height")
    assert_true(map.probe(d, 0, 0, 0).clearance > 1 and map.probe(d, 0, 0, 0.9).clearance > 1, "both roads cross at the origin")
    for w in range(0, map.waypoints, 2):
        var x, y, z = map.waypoint(d, w, 0), map.waypoint(d, w, 1), map.waypoint(d, w, 2)
        var spot = map.probe(d, x, y, z)
        assert_true(spot.clearance > 0.9, "the route centre is clear of walls")
        assert_almost_equal(map.surface(d, spot.layer, x, y).height, z, "the surface follows the route", atol=0.03, rtol=0)
        assert_almost_equal(map.progress(d, spot.layer, x, y), Float32(w), "progress counts waypoints", atol=1e-3, rtol=0)
    print("bridge layers, surface, clearance, progress: passed")


def drive(gpu: Bool, track: Span[Float32, _]) raises -> List[Float32]:
    """Poses and observations of 8 cars after a reset and 10 steps with fixed actions."""
    var device = Device(gpu)
    var n = 8
    var sim = Sim(device, track, n, 42)
    sim.reset(device)
    device.write(sim.actions, [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1, 1, 1, 1, 1, 1, 1, 1])
    for _ in range(10):
        step(device, sim)
    var result = device.read(sim.state)
    result.resize((STEER + 1) * n, 0)
    result.extend(device.read(sim.obs))
    return result^


def layer(x: List[Float64], w: List[Float64], at: Int, rows: Int, squash: Bool) -> List[Float64]:
    """`rows` x len(x) weights at `at` applied to x, with a trailing constant 1."""
    var y = List[Float64](length=rows + 1, fill=1)
    for j in range(rows):
        var acc: Float64 = 0
        for k in range(len(x)):
            acc += x[k] * w[at + j * len(x) + k]
        y[j] = tanh(acc) if squash else acc
    return y^


def host_loss(w: List[Float64], device: Device, trainer: Trainer) raises -> Float64:
    """Minibatch 0's clipped PPO objective in float64 with raw advantages."""
    var obs = device.read(trainer.obs)
    var actions = device.read(trainer.actions)
    var logp = device.read(trainer.logp)
    var value = device.read(trainer.value)
    var advantage = device.read(trainer.advantage)
    var total: Float64 = 0
    for b in range(BATCH):
        var s = b * MINIBATCHES
        var x = List[Float64]()
        for r in range(IN):
            x.append(Float64(obs[(s // N * IN + r) * N + s % N]))
        var o = layer(layer(layer(x, w, W0, H, True), w, W1, H, True), w, W2, OUT, False)
        var new_logp: Float64 = 0
        for a in range(2):
            var error = Float64(actions[(s // N * 2 + a) * N + s % N]) - o[a]
            new_logp -= 0.5 * error * error * exp(-2 * w[LOGSTD + a]) + w[LOGSTD + a] + Float64(HALF_LOG_2PI)
        var ratio, adv = exp(new_logp - Float64(logp[s])), Float64(advantage[s])
        total -= min(ratio * adv, clamp(ratio, 0.8, 1.2) * adv) / BATCH
        var e = abs(o[2] - Float64(value[s] + advantage[s]))  # Huber value loss, matching `minibatch`
        total += (0.25 * e * e if e <= 1 else 0.5 * e - 0.25) / BATCH
    return total


def ppo_tests(gpu: Bool, track: Span[Float32, _]) raises:
    """One rollout on `flat`: GAE recomputed on the host, the analytic gradient of
    minibatch 0 against finite differences, then an update with a finite KL."""
    var device = Device(gpu)
    var sim = Sim(device, track, N, 42)
    sim.reset(device)
    var s = device.read(sim.state)
    for i in range(N):
        s[STEPS * N + i] = Float32(HORIZON - 5 - 7 * i)  # episodes end mid-rollout
    device.write(sim.state, s)
    var policy = Policy(device, 42)
    var acts = Activations(device, N)
    var trainer = Trainer(device, N)
    assert_true(isfinite(trainer.rollout(device, sim, policy, acts, 0)), "rollout reward")
    var reward = device.read(trainer.reward)
    var done = device.read(trainer.done)
    var value = device.read(trainer.value)
    var advantage = device.read(trainer.advantage)
    var last = device.read(acts.o)  # the policy on the final observations: the bootstrap values
    var ended = 0
    for i in range(N):
        var running: Float32 = 0
        var next = last[2 * N + i]
        for t in range(ROLLOUT - 1, -1, -1):
            var k = t * N + i
            var alive = Float32(1 if done[k] == 0 else 0)
            ended += Int(done[k] != 0)
            running = reward[k] + GAMMA * alive * next - value[k] + GAMMA * LAMBDA * alive * running
            assert_almost_equal(advantage[k], running, "GAE advantage", atol=1e-4, rtol=0)
            next = value[k]
    assert_true(ended == N, "every car's episode ended once during the rollout")
    trainer.minibatch(device, policy, 0, 0, 1)
    var analytic = device.read(trainer.grad)
    var w = List[Float64]()
    for value in device.read(policy.theta):
        w.append(Float64(value))
    var worst: Float64 = 0
    var largest: Float64 = 0
    var i = 0
    while i < P:
        w[i] += 1e-4
        var plus = host_loss(w, device, trainer)
        w[i] -= 2e-4
        var numeric = (plus - host_loss(w, device, trainer)) / 2e-4
        w[i] += 1e-4
        var error = abs(numeric - Float64(analytic[i]))
        if error > (3e-2 * abs(numeric) + 1e-3 if gpu else 5e-3 * abs(numeric) + 1e-5):
            raise Error("gradient " + String(i) + " analytic " + String(analytic[i]) + " numeric " + String(numeric))
        worst, largest = max(worst, error), max(largest, abs(numeric))
        i += 293 if i < LOGSTD - 293 else 1
    assert_true(largest > 1e-3, "the rollout produces a nontrivial gradient")
    var kl = trainer.update(device, policy)
    assert_true(isfinite(kl) and kl >= 0 and kl < 0.1, "KL after one update")
    print("GAE, PPO gradient (worst error", worst, "of", largest, ") and update on", "GPU" if gpu else "CPU", ": passed")


def main() raises:
    var gpu = GPU_AVAILABLE if len(argv()) < 2 else String(argv()[1]) != "cpu"
    Python.add_to_path(".")
    var tracks = Python.import_module("racer.track")
    var flat = tracks.load("flat")
    var bridge = tracks.load("bridge")
    physics_tests()
    corridor_tests()
    bridge_tests(from_numpy_array[DType.float32](bridge))
    ppo_tests(False, from_numpy_array[DType.float32](flat))
    if gpu:
        var a = drive(False, from_numpy_array[DType.float32](flat))
        var b = drive(True, from_numpy_array[DType.float32](flat))
        for i in range(len(a)):
            assert_almost_equal(a[i], b[i], "CPU/GPU poses and observations agree", atol=3e-3, rtol=0)
        print("CPU/GPU parity: passed")
        ppo_tests(True, from_numpy_array[DType.float32](flat))
