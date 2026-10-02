"""One fused environment step: policy, sampling, dynamics, resets, and lidar.

On the GPU each environment is one thread block: the block evaluates the
policy cooperatively, thread 0 samples and integrates, then every thread
casts one lidar ray. The CPU runs the same phases as loops per environment.
"""
from std.math import sqrt, exp, log
from std.memory import unsafe_stack_allocation, AddressSpace
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from .device import Ptr, Params, clamp, uniform, normal, LANES
from .layout import (
    RAYS,
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
    ROLLOUT,
    obs_offset,
    action_offset,
    reward_offset,
    stats_offset,
    memory,
)
from .terrain import surface, progress
from .lidar import RANGE, ray, mount, direction
from .vehicle import Car, frame, integrate
from .network import hidden, dot, row_hidden

comptime THREADS = H
# Step flags (Params.flag).
comptime POLICY = 1  # evaluate the network and sample actions
comptime DETERMINISTIC = 2  # use the action mean
comptime RECORD = 4  # store the transition at rollout step Params.offset
comptime VALUE_ONLY = 8  # only bootstrap the critic (no dynamics or lidar)
comptime EVAL = 16  # accumulate per-car reward, failures, and finishes
comptime SENSE_ONLY = 32  # refresh observations without stepping


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
    data[unsafe_offset=k] = (
        map[unsafe_offset=s] + 0.5 * map[unsafe_offset=s + 3]
    )
    data[unsafe_offset=k + 1] = (
        map[unsafe_offset=s + 1] + 0.5 * map[unsafe_offset=s + 4]
    )
    data[unsafe_offset=k + 2] = map[unsafe_offset=s + 9]
    data[unsafe_offset=k + 7] = (
        map[unsafe_offset=s + 7] + 0.5 * map[unsafe_offset=s + 6]
    )
    data[unsafe_offset=k + 9] = episode
    data[unsafe_offset=k + 13] = Float32(segment)
    data[unsafe_offset=k + 10] = 1.1 * (0.85 + 0.3 * uniform(seed + 1))
    data[unsafe_offset=k + 11] = 0.85 + 0.3 * uniform(seed + 2)


def advance(i: Int, data: Ptr, map: Ptr, p: Params):
    """Four 240 Hz substeps, progress reward, terminal checks, and respawn."""
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
        var travel = sqrt(
            (car.x - old_x) * (car.x - old_x)
            + (car.y - old_y) * (car.y - old_y)
        )
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
    var projection = progress(
        map, car.x, car.y, Int32(data[unsafe_offset=k + 13])
    )
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
    delta = clamp(
        delta, -travel_total * 1.25 - 0.001, travel_total * 1.25 + 0.001
    )
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


def sense(i: Int, beam: Int, data: Ptr, map: Ptr, n: Int):
    """One lidar ray of car i; beam 0 also writes the proprioceptive inputs."""
    var k = i * STATE
    var o = obs_offset(n) + i * IN
    var x = data[unsafe_offset=k]
    var y = data[unsafe_offset=k + 1]
    var heading = data[unsafe_offset=k + 2]
    var ground = surface(map, x, y)
    var f = frame(heading, ground.sx, ground.sy)
    if beam == 0:
        data[unsafe_offset=o] = data[unsafe_offset=k + 6] / 0.4189
        data[unsafe_offset=o + 1] = data[unsafe_offset=k + 3] / 5
        data[unsafe_offset=o + 2] = data[unsafe_offset=k + 4] / 5
        data[unsafe_offset=o + 3] = data[unsafe_offset=k + 5] / 10
        data[unsafe_offset=o + 4] = -f.fz
        data[unsafe_offset=o + 5] = -f.lz
        data[unsafe_offset=o + 6] = f.nz
        data[unsafe_offset=o + 7] = min(ground.clearance, 2) / 2
    data[unsafe_offset=o + 8 + beam] = (
        ray(map, mount(x, y, ground.height, f), direction(beam, f)) / RANGE
    )


@inline(.always)
def feature(k: Int, i: Int, data: Ptr, n: Int) -> Float32:
    """Network input k of car i: clamped observation, then 1, then zeros."""
    if k < OBS:
        return clamp(data[unsafe_offset=obs_offset(n) + i * IN + k], -10, 10)
    return 1 if k == OBS else 0


def sample(
    i: Int, output: Pointer[Float32, _, address_space=_], data: Ptr, p: Params
):
    """Draw car i's action from the policy output (or bootstrap its value)."""
    var n = Int(p.envs)
    var mem = memory(n)
    var index = Int(p.offset) * n + i
    if p.flag & VALUE_ONLY:
        data[unsafe_offset=mem.values + ROLLOUT * n + i] = output[
            unsafe_offset=2
        ]
        return
    var probability: Float32 = 0
    for action in range(2):
        var std = exp(data[unsafe_offset=mem.weights + LOGSTD + action])
        var noise = normal(
            p.seed
            + UInt32(Int(p.index) * n + i) * 0x9E3779B9
            + UInt32(action) * 0x85EBCA6B
        )
        noise = 0 if p.flag & DETERMINISTIC else noise
        var value = output[unsafe_offset=action] + std * noise
        data[unsafe_offset=action_offset(n) + i * 2 + action] = value
        if p.flag & RECORD:
            data[unsafe_offset=mem.actions + index * 2 + action] = value
        probability -= 0.5 * noise * noise + log(std) + 0.918938533
    if p.flag & RECORD:
        data[unsafe_offset=mem.logp + index] = probability
        data[unsafe_offset=mem.values + index] = output[unsafe_offset=2]


def policy_gpu(data: Ptr, map: Ptr, p: Params):
    """One block of H threads per car: cooperative MLP, then thread 0 samples.
    """
    var i = block_idx.x
    var t = thread_idx.x
    var n = Int(p.envs)
    var mem = memory(n)
    var xs = unsafe_stack_allocation[
        IN, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var h1 = unsafe_stack_allocation[
        HA, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var h2 = unsafe_stack_allocation[
        HA, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var output = unsafe_stack_allocation[
        OUT, DType.float32, address_space=AddressSpace.SHARED
    ]()
    for k in range(t, IN, H):
        var value = feature(k, i, data, n)
        xs[unsafe_offset=k] = value
        if p.flag & RECORD:
            data[
                unsafe_offset=mem.obs + (Int(p.offset) * n + i) * IN + k
            ] = value
    if t >= H - 16:
        h1[unsafe_offset=t + 16] = 1 if t == H - 16 else 0
        h2[unsafe_offset=t + 16] = 1 if t == H - 16 else 0
    barrier()
    h1[unsafe_offset=t] = hidden[1](
        t, xs, data.unsafe_offset(mem.weights + W0), IN
    )
    barrier()
    h2[unsafe_offset=t] = hidden[1](
        t, h1, data.unsafe_offset(mem.weights + W1), HA
    )
    barrier()
    if t < OUT:
        output[unsafe_offset=t] = dot[1](
            h2, data.unsafe_offset(mem.weights + W2 + t * HA), HA
        )
    barrier()
    if t == 0:
        sample(i, output, data, p)


def physics(i: Int, data: Ptr, map: Ptr, p: Params):
    """Integrate car i from the action buffer and record the transition."""
    var n = Int(p.envs)
    var mem = memory(n)
    advance(i, data, map, p)
    var reward = data[unsafe_offset=reward_offset(n) + i * 2]
    var reason = data[unsafe_offset=reward_offset(n) + i * 2 + 1]
    if p.flag & RECORD:
        var index = Int(p.offset) * n + i
        data[unsafe_offset=mem.rewards + index] = reward
        data[unsafe_offset=mem.done + index] = 1 if reason > 0 else 0
    if p.flag & EVAL:
        var stats = stats_offset(n) + i * 3
        data[unsafe_offset=stats] += reward
        if reason == 1:
            data[unsafe_offset=stats + 1] += 1
        elif reason == 3:
            data[unsafe_offset=stats + 2] += 1


def lidar(r: Int, data: Ptr, map: Ptr, p: Params):
    sense(r // RAYS, r % RAYS, data, map, Int(p.envs))


def rollout_cpu(i: Int, data: Ptr, map: Ptr, p: Params):
    var n = Int(p.envs)
    var mem = memory(n)
    var xs = unsafe_stack_allocation[IN, DType.float32]()
    var h1 = unsafe_stack_allocation[HA, DType.float32]()
    var h2 = unsafe_stack_allocation[HA, DType.float32]()
    var output = unsafe_stack_allocation[OUT, DType.float32]()
    if p.flag & (POLICY | VALUE_ONLY):
        for k in range(IN):
            var value = feature(k, i, data, n)
            xs[unsafe_offset=k] = value
            if p.flag & RECORD:
                data[
                    unsafe_offset=mem.obs + (Int(p.offset) * n + i) * IN + k
                ] = value
        row_hidden(xs, data.unsafe_offset(mem.weights + W0), IN, h1)
        for j in range(H, HA):
            h1[unsafe_offset=j] = 1 if j == H else 0
        row_hidden(h1, data.unsafe_offset(mem.weights + W1), HA, h2)
        for j in range(H, HA):
            h2[unsafe_offset=j] = 1 if j == H else 0
        for j in range(OUT):
            output[unsafe_offset=j] = dot[LANES](
                h2, data.unsafe_offset(mem.weights + W2 + j * HA), HA
            )
    if p.flag & (POLICY | VALUE_ONLY):
        sample(i, output, data, p)
    if p.flag & VALUE_ONLY:
        return
    if not (p.flag & SENSE_ONLY):
        physics(i, data, map, p)
    for beam in range(RAYS):
        sense(i, beam, data, map, n)
