"""Behavior, device parity, GAE boundaries, and checkpoint round trips."""
from std.sys import argv
from std.math import isfinite
from racer.device import (
    Device,
    Params,
    Ptr,
    read_floats,
    uniform,
    GPU_AVAILABLE,
)
from racer.vehicle import Car, frame, integrate
from racer.lidar import RAYS
from racer.simulation import (
    spawn,
    step,
    observe,
    STATE,
    OBS,
    action_offset,
    reward_offset,
)
from racer.network import memory, initialize, WEIGHTS, ROLLOUT
from racer.ppo import gae, permutation
from racer.checkpoint import save, load


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


def boundary_fixture(i: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(1)
    data[unsafe_offset=mem.output + 2] = 123
    for t in range(ROLLOUT):
        data[unsafe_offset=mem.values + t] = 0
        data[unsafe_offset=mem.rewards + t] = 1
        data[unsafe_offset=mem.done + t] = Float32(1) if t == 30 else Float32(0)


def environment_tests(map_path: String, gpu: Bool) raises:
    var n = 16
    var map = read_floats(map_path)
    var cpu = Device(False, memory(n).size, map)
    var p = Params(Int32(n), 42)
    cpu.run[spawn](n, p)
    cpu.run[actions](n, p)
    for _ in range(64):
        cpu.run[step](n, p)
        cpu.run[observe](n * RAYS, p)
    cpu.ctx.synchronize()
    var snapshot = List[Float32](length=n * STATE, fill=0)
    with cpu.data.map_to_host() as h:
        for i in range(n * STATE):
            snapshot[i] = h[i]
    cpu.run[observe](n * RAYS, p)
    cpu.ctx.synchronize()
    with cpu.data.map_to_host() as h:
        for i in range(n * STATE):
            check(h[i] == snapshot[i], "observe must not change state or RNG")
        for i in range(n * (STATE + OBS)):
            check(
                isfinite(h[i]),
                "environment state and observation must remain finite",
            )
    if gpu:
        var metal = Device(True, memory(n).size, map)
        metal.run[spawn](n, p)
        metal.run[actions](n, p)
        for _ in range(64):
            metal.run[step](n, p)
            metal.run[observe](n * RAYS, p)
        metal.ctx.synchronize()
        with cpu.data.map_to_host() as a:
            with metal.data.map_to_host() as b:
                for i in range(n * STATE):
                    check(abs(a[i] - b[i]) < 0.002, "CPU/GPU dynamics mismatch")
                for i in range(n * STATE, n * (STATE + OBS)):
                    check(abs(a[i] - b[i]) < 0.01, "CPU/GPU sensing mismatch")
        print(map_path, "CPU/GPU dynamics and sensing: passed")
    with cpu.data.map_to_host() as h:
        h[0] = 1000
        h[1] = 1000
    cpu.run[step](n, p)
    cpu.ctx.synchronize()
    with cpu.data.map_to_host() as h:
        check(h[reward_offset(n) + 1] == 1, "unsupported car must fail")
        check(
            h[9] == snapshot[9] + 1 and abs(h[0]) < 20,
            "terminal car must reset",
        )
        h[8] = 2999
    cpu.run[step](n, p)
    with cpu.data.map_to_host() as h:
        check(
            h[reward_offset(n) + 1] == 2, "finite-horizon episode must time out"
        )
        check(h[8] == 0, "timeout must reset the episode clock")
    cpu.run[initialize](WEIGHTS, p)
    save(cpu, "/tmp/warporacer3d-roundtrip.wrppo", n, 123, 456, 0xFEDCBA98)
    var other = Device(gpu, memory(1).size, map)
    var loaded = load(other, "/tmp/warporacer3d-roundtrip.wrppo", 1)
    check(
        loaded[0] == 123 and loaded[1] == 456 and loaded[2] == 0xFEDCBA98,
        "checkpoint counters and full uint32 seed",
    )
    with cpu.data.map_to_host() as a:
        with other.data.map_to_host() as b:
            for i in range(WEIGHTS * 4):
                check(
                    a[memory(n).weights + i] == b[memory(1).weights + i],
                    "checkpoint must preserve all parameters and moments",
                )
    print("terminal reset, pure observation, cross-device checkpoint: passed")
    var one = Device(False, memory(1).size, map)
    one.run[boundary_fixture](1, Params(1, 0))
    one.run[gae](1, Params(1, 0))
    with one.data.map_to_host() as h:
        check(
            abs(h[memory(1).target + 30] - 1) < 1e-5,
            "GAE must stop at terminal states",
        )
        check(abs(h[memory(1).target + 29] - 1.9405) < 1e-5, "GAE recurrence")
        check(
            abs(h[memory(1).target + 31] - 122.77) < 1e-4,
            "rollout tail must bootstrap",
        )
    print("GAE episode boundaries: passed")


def main() raises:
    var args = argv()
    vehicle_tests()
    var gpu = GPU_AVAILABLE if len(args) < 4 else String(args[3]) != "cpu"
    environment_tests(String(args[1]), gpu)
    environment_tests(String(args[2]), gpu)
