"""Fused integration, terminal/reset logic, and parallel distance-field lidar."""
from std.math import cos, sin, sqrt
from .core import Ptr, Params, clamp, uniform
from .terrain import surface, progress, ray, RANGE
from .vehicle import Car, frame, integrate

from std.sys import get_defined_int

comptime BEAMS = get_defined_int["BEAMS", 64]()
comptime OBS = 8 + BEAMS
comptime STATE = 14
# Arena begins with state[N,14], observation[N,72], actions[N,2], rewards[N,2].
# State: x,y,heading,u,v,yaw,steer,progress,steps,episode,grip,motor,return,route segment.


def obs_offset(n: Int) -> Int:
    return n * STATE


def action_offset(n: Int) -> Int:
    return n * (STATE + OBS)


def reward_offset(n: Int) -> Int:
    return n * (STATE + OBS + 2)


def env_size(n: Int) -> Int:
    return n * (STATE + OBS + 4)


def spawn(i: Int, data: Ptr, map: Ptr, p: Params):
    var k = i * STATE
    var episode = data[unsafe_offset=k + 9] + 1
    var seed = p.seed + UInt32(i) * 0x9E3779B9 + UInt32(episode) * 0x85EBCA6B
    var route = 16 + 4 * Int(map[unsafe_offset=2]) * Int(map[unsafe_offset=3])
    var spawns = route + 10 * Int(map[unsafe_offset=4])
    var index = Int(uniform(seed) * map[unsafe_offset=10])
    var segment = Int(map[unsafe_offset=spawns + index])
    var s = route + segment * 10
    for j in range(STATE):
        data[unsafe_offset=k + j] = 0
    var x = map[unsafe_offset=s] + 0.5 * map[unsafe_offset=s + 3]
    var y = map[unsafe_offset=s + 1] + 0.5 * map[unsafe_offset=s + 4]
    data[unsafe_offset=k] = x
    data[unsafe_offset=k + 1] = y
    data[unsafe_offset=k + 2] = map[unsafe_offset=s + 9]
    data[unsafe_offset=k + 7] = map[unsafe_offset=s + 7] + 0.5 * map[unsafe_offset=s + 6]
    data[unsafe_offset=k + 9] = episode
    data[unsafe_offset=k + 13] = Float32(segment)
    data[unsafe_offset=k + 10] = 1.1 * (0.85 + 0.3 * uniform(seed + 1))
    data[unsafe_offset=k + 11] = 0.85 + 0.3 * uniform(seed + 2)


def step(i: Int, data: Ptr, map: Ptr, p: Params):
    var n = Int(p.envs)
    var k = i * STATE
    var action = action_offset(n) + i * 2
    var car = Car(
        data[unsafe_offset=k],
        data[unsafe_offset=k + 1],
        data[unsafe_offset=k + 2],
        data[unsafe_offset=k + 3],
        data[unsafe_offset=k + 4],
        data[unsafe_offset=k + 5],
        data[unsafe_offset=k + 6],
    )
    var reason: Float32 = 0
    var travel_total: Float32 = 0
    for _ in range(4):
        var ground = surface(map, car.x, car.y)
        var old_frame = frame(car.heading, ground.sx, ground.sy)
        var old_x = car.x
        var old_y = car.y
        integrate(
            car,
            old_frame,
            data[unsafe_offset=action],
            data[unsafe_offset=action + 1],
            data[unsafe_offset=k + 10],
            data[unsafe_offset=k + 11],
        )
        var next_ground = surface(map, car.x, car.y)
        var next_frame = frame(car.heading, next_ground.sx, next_ground.sy)
        # Transport velocity when the road grade changes. Turning itself is
        # already accounted for by the bicycle's rotating-frame equations.
        var same_heading = frame(car.heading, ground.sx, ground.sy)
        var vx = same_heading.fx * car.u + same_heading.lx * car.v
        var vy = same_heading.fy * car.u + same_heading.ly * car.v
        var vz = same_heading.fz * car.u + same_heading.lz * car.v
        car.u = vx * next_frame.fx + vy * next_frame.fy + vz * next_frame.fz
        car.v = vx * next_frame.lx + vy * next_frame.ly + vz * next_frame.lz
        # Three overlapping circles enclose the 58 x 38 cm chassis. Inflate
        # their radius by half the swept distance, including turning.
        travel_total += (1.0 / 240.0) * sqrt(car.u * car.u + car.v * car.v)
        var travel = sqrt((car.x - old_x) * (car.x - old_x) + (car.y - old_y) * (car.y - old_y))
        var clearance = min(next_ground.clearance, ground.clearance)
        for end in range(-1, 2, 2):
            var old_end_x = old_x + Float32(end) * 0.15 * old_frame.fx
            var old_end_y = old_y + Float32(end) * 0.15 * old_frame.fy
            var next_end_x = car.x + Float32(end) * 0.15 * next_frame.fx
            var next_end_y = car.y + Float32(end) * 0.15 * next_frame.fy
            var dx = next_end_x - old_end_x
            var dy = next_end_y - old_end_y
            travel = max(travel, sqrt(dx * dx + dy * dy))
            clearance = min(
                clearance,
                surface(map, old_end_x, old_end_y).clearance,
            )
            clearance = min(
                clearance,
                surface(map, next_end_x, next_end_y).clearance,
            )
        if clearance < 0.24 + 0.5 * travel + 0.001:
            reason = 1
            break
    var ground = surface(map, car.x, car.y)
    var projection = progress(map, car.x, car.y, Int32(data[unsafe_offset=k + 13]))
    var next_progress = projection[0]
    data[unsafe_offset=k + 13] = projection[1]
    if projection[2] + 0.19 > projection[3]:
        reason = 1
    var delta = next_progress - data[unsafe_offset=k + 7]
    var total = map[unsafe_offset=9]
    if map[unsafe_offset=5] > 0:
        delta = delta - total if delta > total * 0.5 else delta
        delta = delta + total if delta < -total * 0.5 else delta
    # Projection jumps cannot manufacture progress reward.
    delta = clamp(delta, -travel_total * 1.25 - 0.001, travel_total * 1.25 + 0.001)
    var reward = 10 * delta - 0.001 * car.steer * car.steer
    var steps = data[unsafe_offset=k + 8] + 1
    if reason > 0:
        reward -= 2
    elif map[unsafe_offset=5] == 0 and next_progress > total - 0.3:
        reason = 3
        reward += 2
    elif steps >= 3000:
        reason = 2
    data[unsafe_offset=k] = car.x
    data[unsafe_offset=k + 1] = car.y
    data[unsafe_offset=k + 2] = car.heading
    data[unsafe_offset=k + 3] = car.u
    data[unsafe_offset=k + 4] = car.v
    data[unsafe_offset=k + 5] = car.yaw
    data[unsafe_offset=k + 6] = car.steer
    data[unsafe_offset=k + 7] = next_progress
    data[unsafe_offset=k + 8] = steps
    data[unsafe_offset=k + 12] += reward
    var output = reward_offset(n) + i * 2
    data[unsafe_offset=output] = reward
    data[unsafe_offset=output + 1] = reason
    if reason > 0:
        spawn(i, data, map, p)


def observe(i: Int, data: Ptr, map: Ptr, p: Params):
    var n = Int(p.envs)
    var env = i // BEAMS
    var beam = i % BEAMS
    var k = env * STATE
    var o = obs_offset(n) + env * OBS
    var x = data[unsafe_offset=k]
    var y = data[unsafe_offset=k + 1]
    var heading = data[unsafe_offset=k + 2]
    if beam == 0:
        var ground = surface(map, x, y)
        var f = frame(heading, ground.sx, ground.sy)
        data[unsafe_offset=o] = data[unsafe_offset=k + 6] / 0.4189
        data[unsafe_offset=o + 1] = data[unsafe_offset=k + 3] / 5
        data[unsafe_offset=o + 2] = data[unsafe_offset=k + 4] / 5
        data[unsafe_offset=o + 3] = data[unsafe_offset=k + 5] / 10
        data[unsafe_offset=o + 4] = -f.fz
        data[unsafe_offset=o + 5] = -f.lz
        data[unsafe_offset=o + 6] = f.nz
        data[unsafe_offset=o + 7] = min(ground.clearance, 2) / 2
    var angle = heading - 2.35619449 + Float32(beam) * (4.71238898 / Float32(BEAMS - 1))
    data[unsafe_offset=o + 8 + beam] = ray(map, x, y, cos(angle), sin(angle)) / RANGE
