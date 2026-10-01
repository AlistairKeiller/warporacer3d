"""Device-resident rollout, GAE, clipped PPO, and Adam for the fixed network."""
from std.math import exp, log, sqrt
from .device import Device, Params, Ptr, clamp, normal, uniform
from .simulation import (
    OBS,
    obs_offset,
    action_offset,
    reward_offset,
    step,
    observe,
)
from .lidar import RAYS
from .network import (
    memory,
    forward,
    backward,
    ROLLOUT,
    PART,
    WEIGHTS,
    LOGSTD,
    HIDDEN,
)


def input(i: Int, data: Ptr, map: Ptr, p: Params):
    var n = Int(p.envs)
    var mem = memory(n)
    var value = clamp(data[unsafe_offset=obs_offset(n) + i], -10, 10)
    data[unsafe_offset=mem.x + i] = value
    if p.offset >= 0:
        data[unsafe_offset=mem.obs + Int(p.offset) * n * OBS + i] = value


def sample(i: Int, data: Ptr, map: Ptr, p: Params):
    var n = Int(p.envs)
    var mem = memory(n)
    var index = Int(p.offset) * n + i
    var probability: Float32 = 0
    for action in range(2):
        var mean = data[unsafe_offset=mem.output + i * 3 + action]
        var std = exp(data[unsafe_offset=mem.weights + LOGSTD + action])
        var noise = normal(
            p.seed
            + UInt32(Int(p.index) * n + i) * 0x9E3779B9
            + UInt32(action) * 0x85EBCA6B
        )
        noise = 0 if p.flag > 0 else noise
        var value = mean + std * noise
        data[unsafe_offset=action_offset(n) + i * 2 + action] = value
        data[unsafe_offset=mem.actions + index * 2 + action] = value
        probability -= 0.5 * noise * noise + log(std) + 0.918938533
    data[unsafe_offset=mem.logp + index] = probability
    data[unsafe_offset=mem.values + index] = data[
        unsafe_offset=mem.output + i * 3 + 2
    ]


def transition(i: Int, data: Ptr, map: Ptr, p: Params):
    var n = Int(p.envs)
    var mem = memory(n)
    var index = Int(p.offset) * n + i
    data[unsafe_offset=mem.rewards + index] = data[
        unsafe_offset=reward_offset(n) + i * 2
    ]
    data[unsafe_offset=mem.done + index] = Float32(1) if data[
        unsafe_offset=reward_offset(n) + i * 2 + 1
    ] > 0 else Float32(0)


def gae(i: Int, data: Ptr, map: Ptr, p: Params):
    var n = Int(p.envs)
    var mem = memory(n)
    var next_value = data[unsafe_offset=mem.output + i * 3 + 2]
    var advantage: Float32 = 0
    for t in range(ROLLOUT - 1, -1, -1):
        var index = t * n + i
        var alive = 1 - data[unsafe_offset=mem.done + index]
        var value = data[unsafe_offset=mem.values + index]
        var delta = (
            data[unsafe_offset=mem.rewards + index]
            + 0.99 * alive * next_value
            - value
        )
        advantage = delta + 0.99 * 0.95 * alive * advantage
        data[unsafe_offset=mem.advantage + index] = advantage
        data[unsafe_offset=mem.target + index] = value + advantage
        next_value = value


def advantage_moments(i: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(Int(p.envs))
    var sum: Float32 = 0
    var square: Float32 = 0
    var rewards: Float32 = 0
    var count = Int(p.envs) * ROLLOUT
    for j in range(i * PART, min((i + 1) * PART, count)):
        var v = data[unsafe_offset=mem.advantage + j]
        sum += v
        square += v * v
        rewards += data[unsafe_offset=mem.rewards + j]
    data[unsafe_offset=mem.partial + i * 3] = sum
    data[unsafe_offset=mem.partial + i * 3 + 1] = square
    data[unsafe_offset=mem.partial + i * 3 + 2] = rewards


def normalize(i: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(Int(p.envs))
    var count = Int(p.envs) * ROLLOUT
    var sum: Float32 = 0
    var square: Float32 = 0
    var rewards: Float32 = 0
    for j in range((count + PART - 1) // PART):
        sum += data[unsafe_offset=mem.partial + j * 3]
        square += data[unsafe_offset=mem.partial + j * 3 + 1]
        rewards += data[unsafe_offset=mem.partial + j * 3 + 2]
    var mean = sum / Float32(count)
    data[unsafe_offset=mem.stats] = mean
    data[unsafe_offset=mem.stats + 1] = 1 / sqrt(
        max(square / Float32(count) - mean * mean, Float32(1e-8))
    )
    data[unsafe_offset=mem.stats + 2] = rewards / Float32(count)


def permutation(index: Int, n: Int, seed: UInt32) -> Int:
    # Two modular shears form a bijection for every N, without a shuffle buffer.
    var row = index // n
    var col = index % n
    row = (row + Int(uniform(UInt32(col) + seed) * ROLLOUT)) % ROLLOUT
    col = (col + Int(uniform(UInt32(row) + seed + 1) * Float32(n))) % n
    return row * n + col


def gather(i: Int, data: Ptr, map: Ptr, p: Params):
    var n = Int(p.envs)
    var mem = memory(n)
    var row = i // OBS
    var col = i % OBS
    var index = permutation(Int(p.offset) + row, n, p.seed)
    var value = data[unsafe_offset=mem.obs + index * OBS + col]
    data[unsafe_offset=mem.x + i] = value
    data[unsafe_offset=mem.xt + col * (n * ROLLOUT // 4) + row] = value


def loss(i: Int, data: Ptr, map: Ptr, p: Params):
    var n = Int(p.envs)
    var mem = memory(n)
    var batch = n * ROLLOUT // 4
    var index = permutation(Int(p.offset) + i, n, p.seed)
    var logp: Float32 = 0
    var error = SIMD[DType.float32, 2](0)
    var invvar = SIMD[DType.float32, 2](0)
    for j in range(2):
        error[j] = (
            data[unsafe_offset=mem.actions + index * 2 + j]
            - data[unsafe_offset=mem.output + i * 3 + j]
        )
        var ls = data[unsafe_offset=mem.weights + LOGSTD + j]
        invvar[j] = exp(-2 * ls)
        logp -= 0.5 * error[j] * error[j] * invvar[j] + ls + 0.918938533
    var logratio = logp - data[unsafe_offset=mem.logp + index]
    var ratio = exp(logratio)
    var adv = (
        data[unsafe_offset=mem.advantage + index]
        - data[unsafe_offset=mem.stats]
    ) * data[unsafe_offset=mem.stats + 1]
    var dlogp = -adv * ratio / Float32(batch)
    if (adv >= 0 and ratio > 1.2) or (adv < 0 and ratio < 0.8):
        dlogp = 0
    for j in range(2):
        data[unsafe_offset=mem.d3 + i * 3 + j] = dlogp * error[j] * invvar[j]
        data[unsafe_offset=mem.output + i * 3 + j] = dlogp * (
            error[j] * error[j] * invvar[j] - 1
        )
    var value = data[unsafe_offset=mem.output + i * 3 + 2]
    var old = data[unsafe_offset=mem.values + index]
    var target = data[unsafe_offset=mem.target + index]
    var clipped = old + clamp(value - old, -0.2, 0.2)
    var delta = value - target
    var clipped_delta = clipped - target
    var derivative = delta
    if clipped_delta * clipped_delta > delta * delta:
        derivative = clipped_delta if abs(value - old) <= 0.2 else 0
    data[unsafe_offset=mem.d3 + i * 3 + 2] = 0.5 * derivative / Float32(batch)
    data[unsafe_offset=mem.output + i * 3 + 2] = ratio - 1 - logratio


def std_gradient(i: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(Int(p.envs))
    var sum: Float32 = 0
    for row in range(Int(p.envs) * ROLLOUT // 4):
        sum += data[unsafe_offset=mem.output + row * 3 + i]
    if i < 2:
        data[unsafe_offset=mem.gradient + LOGSTD + i] = sum
    else:
        data[unsafe_offset=mem.stats + 4] = sum / Float32(
            Int(p.envs) * ROLLOUT // 4
        )


def gradient_moments(i: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(Int(p.envs))
    var square: Float32 = 0
    for j in range(i * PART, min((i + 1) * PART, WEIGHTS)):
        var g = data[unsafe_offset=mem.gradient + j]
        square += g * g
    data[unsafe_offset=mem.partial + i] = square


def clip_gradient(i: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(Int(p.envs))
    var square: Float32 = 0
    for j in range((WEIGHTS + PART - 1) // PART):
        square += data[unsafe_offset=mem.partial + j]
    data[unsafe_offset=mem.stats + 3] = min(
        Float32(1), 0.5 / (sqrt(square) + 1e-8)
    )
    data[unsafe_offset=mem.stats + 5] = 1 / (
        1 - exp(Float32(p.index) * Float32(-0.105360515658))
    )
    data[unsafe_offset=mem.stats + 6] = 1 / (
        1 - exp(Float32(p.index) * Float32(-0.001000500334))
    )


def adam(i: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(Int(p.envs))
    var g = (
        data[unsafe_offset=mem.gradient + i] * data[unsafe_offset=mem.stats + 3]
    )
    var m = 0.9 * data[unsafe_offset=mem.first + i] + 0.1 * g
    var v = 0.999 * data[unsafe_offset=mem.second + i] + 0.001 * g * g
    data[unsafe_offset=mem.first + i] = m
    data[unsafe_offset=mem.second + i] = v
    var value = data[unsafe_offset=mem.weights + i] - p.scale * m * data[
        unsafe_offset=mem.stats + 5
    ] / (sqrt(v * data[unsafe_offset=mem.stats + 6]) + 1e-5)
    data[unsafe_offset=mem.weights + i] = (
        clamp(value, -2, 0) if i >= LOGSTD else value
    )


def rollout(mut device: Device, n: Int, iteration: Int, seed: UInt32) raises:
    for t in range(ROLLOUT):
        var p = Params(Int32(n), seed, Int32(t), Int32(iteration * ROLLOUT + t))
        device.run[input](n * OBS, p)
        forward(device, n, n, False)
        device.run[sample](n, p)
        device.run[step](n, p)
        device.run[transition](n, p)
        device.run[observe](n * RAYS, p)
    device.run[input](n * OBS, Params(Int32(n), seed, -1))
    forward(device, n, n, False)
    device.run[gae](n, Params(Int32(n), seed))
    device.run[advantage_moments](
        (n * ROLLOUT + PART - 1) // PART, Params(Int32(n), seed)
    )
    device.run[normalize](1, Params(Int32(n), seed))


def update(
    mut device: Device,
    n: Int,
    iteration: Int,
    mut optimizer_step: Int,
    seed: UInt32,
) raises:
    var batch = n * ROLLOUT // 4
    for epoch in range(4):
        var shuffle = seed + UInt32(iteration * 4 + epoch) * 0x9E3779B9
        for minibatch in range(4):
            var p = Params(Int32(n), shuffle, Int32(minibatch * batch))
            device.run[gather](batch * OBS, p)
            forward(device, n, batch, True)
            device.run[loss](batch, p)
            device.run[std_gradient](3, p)
            backward(device, n, batch)
            device.run[gradient_moments]((WEIGHTS + PART - 1) // PART, p)
            optimizer_step += 1
            device.run[clip_gradient](
                1, Params(Int32(n), seed, 0, Int32(optimizer_step))
            )
            device.run[adam](WEIGHTS, Params(Int32(n), seed, 0, 0, 0, 0.0003))
