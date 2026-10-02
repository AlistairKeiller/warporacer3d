"""Device-resident rollout, GAE, clipped PPO, and Adam for the fixed network."""
from std.math import exp, log, sqrt
from .device import Device, Params, Ptr, clamp, GPU_AVAILABLE
from .layout import (
    RAYS,
    IN,
    OUT,
    H,
    HA,
    W1,
    W2,
    LOGSTD,
    WEIGHTS,
    TOTAL,
    ROLLOUT,
    EPOCHS,
    MINIBATCHES,
    PART,
    CHUNK,
    MOMENT_PARTS,
    memory,
    permutation,
)
from .simulation import (
    policy_gpu,
    physics,
    lidar,
    rollout_cpu,
    THREADS,
    POLICY,
    RECORD,
    VALUE_ONLY,
    SENSE_ONLY,
)
from .network import (
    gather,
    gather_row,
    SLICE,
    layer1_cpu,
    layer2_cpu,
    layer3,
    back2,
    back2_cpu,
    back1_cpu,
    partials_cpu,
    partials_tail,
    reduce_cpu,
    layer1_gpu,
    layer2_gpu,
    back1_gpu,
    partials_w0_gpu,
    partials_w1_gpu,
    reduce,
)


def step(mut device: Device, n: Int, p: Params) raises:
    """One environment step for every car (see the simulation flags).

    The GPU uses three launches (block-per-car policy, per-car dynamics,
    per-ray lidar) so each phase runs at its natural parallelism; the CPU
    does all three in one parallel pass over cars.
    """
    comptime if GPU_AVAILABLE:
        if device.gpu:
            if p.flag & (POLICY | VALUE_ONLY):
                device.blocks[policy_gpu, THREADS](n, p)
            if p.flag & VALUE_ONLY:
                return
            if not (p.flag & SENSE_ONLY):
                device.run[physics](n, p)
            device.run[lidar](n * RAYS, p)
            return
    device.run[rollout_cpu](n, p)


def gae(i: Int, data: Ptr, map: Ptr, p: Params):
    var n = Int(p.envs)
    var mem = memory(n)
    var next_value = data[unsafe_offset=mem.values + ROLLOUT * n + i]
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


def loss(i: Int, data: Ptr, map: Ptr, p: Params):
    """Clipped surrogate and value loss derivatives for minibatch row i."""
    var n = Int(p.envs)
    var mem = memory(n)
    var batch = n * ROLLOUT // MINIBATCHES
    var index = permutation(Int(p.offset) + i, n, p.seed)
    var logp: Float32 = 0
    var error = SIMD[DType.float32, 2](0)
    var invvar = SIMD[DType.float32, 2](0)
    for j in range(2):
        error[j] = (
            data[unsafe_offset=mem.actions + index * 2 + j]
            - data[unsafe_offset=mem.output + i * OUT + j]
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
        data[unsafe_offset=mem.d3 + i * OUT + j] = dlogp * error[j] * invvar[j]
        data[unsafe_offset=mem.extra + i * OUT + j] = dlogp * (
            error[j] * error[j] * invvar[j] - 1
        )
    var value = data[unsafe_offset=mem.output + i * OUT + 2]
    var old = data[unsafe_offset=mem.values + index]
    var target = data[unsafe_offset=mem.target + index]
    var clipped = old + clamp(value - old, -0.2, 0.2)
    var delta = value - target
    var clipped_delta = clipped - target
    var derivative = delta
    if clipped_delta * clipped_delta > delta * delta:
        derivative = clipped_delta if abs(value - old) <= 0.2 else 0
    data[unsafe_offset=mem.d3 + i * OUT + 2] = 0.5 * derivative / Float32(batch)
    data[unsafe_offset=mem.extra + i * OUT + 2] = ratio - 1 - logratio


def gradient_moments(i: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(Int(p.envs))
    var square: Float32 = 0
    for j in range(i * PART, min((i + 1) * PART, WEIGHTS)):
        var g = data[unsafe_offset=mem.gradient + j]
        square += g * g
    data[unsafe_offset=mem.moments + i] = square


def clip(i: Int, data: Ptr, map: Ptr, p: Params):
    """Global-norm clipping factor (max norm 0.5) from the moment partials."""
    var mem = memory(Int(p.envs))
    var square: Float32 = 0
    for j in range(MOMENT_PARTS):
        square += data[unsafe_offset=mem.moments + j]
    data[unsafe_offset=mem.stats + 3] = min(
        Float32(1), 0.5 / (sqrt(square) + 1e-8)
    )


def adam(i: Int, data: Ptr, map: Ptr, p: Params):
    """Bias-corrected Adam on the clipped gradient; p.index is the step."""
    var mem = memory(Int(p.envs))
    var scale = data[unsafe_offset=mem.stats + 3]
    var first = 1 / (1 - exp(Float32(p.index) * Float32(-0.105360515658)))
    var second = 1 / (1 - exp(Float32(p.index) * Float32(-0.001000500334)))
    var g = data[unsafe_offset=mem.gradient + i] * scale
    var m = 0.9 * data[unsafe_offset=mem.first + i] + 0.1 * g
    var v = 0.999 * data[unsafe_offset=mem.second + i] + 0.001 * g * g
    data[unsafe_offset=mem.first + i] = m
    data[unsafe_offset=mem.second + i] = v
    var value = data[unsafe_offset=mem.weights + i] - p.scale * m * first / (
        sqrt(v * second) + 1e-5
    )
    value = clamp(value, -2, 0) if i >= LOGSTD else value
    data[unsafe_offset=mem.weights + i] = value
    if W1 <= i and i < W2:
        data[
            unsafe_offset=mem.w1t + ((i - W1) % H) * HA + (i - W1) // H
        ] = value


def rollout(mut device: Device, n: Int, iteration: Int, seed: UInt32) raises:
    for t in range(ROLLOUT):
        step(
            device,
            n,
            Params(
                Int32(n),
                seed,
                Int32(t),
                Int32(iteration * ROLLOUT + t),
                POLICY | RECORD,
            ),
        )
    step(device, n, Params(Int32(n), seed, 0, 0, VALUE_ONLY))
    device.run[gae](n, Params(Int32(n), seed))
    device.run[advantage_moments](
        (n * ROLLOUT + PART - 1) // PART, Params(Int32(n), seed)
    )
    device.run[normalize](1, Params(Int32(n), seed))


def backprop(
    mut device: Device, n: Int, batch: Int, chunks: Int, p: Params
) raises:
    """Forward, loss derivatives, backward, and split-K weight gradients."""
    comptime if GPU_AVAILABLE:
        if device.gpu:
            device.run[gather](batch * (IN // SLICE), p)
            device.tiles[layer1_gpu](batch, H, IN, 1, p)
            device.tiles[layer2_gpu](batch, H, HA, 1, p)
            device.run[layer3](batch * OUT, p)
            device.run[loss](batch, p)
            device.run[back2](batch * H, p)
            device.tiles[back1_gpu](batch, H, H, 1, p)
            device.tiles[partials_w0_gpu](IN, H, CHUNK, chunks, p)
            device.tiles[partials_w1_gpu](HA, H, CHUNK, chunks, p)
            device.run[partials_tail](chunks * (TOTAL - W2), p)
            device.run[reduce](TOTAL, p)
            return
    device.run[gather_row](batch, p)
    device.run[layer1_cpu](batch, p)
    device.run[layer2_cpu](batch, p)
    device.run[layer3](batch * OUT, p)
    device.run[loss](batch, p)
    device.run[back2_cpu](batch, p)
    device.run[back1_cpu](batch, p)
    device.run[partials_cpu](chunks, p)
    device.run[reduce_cpu]((TOTAL + PART - 1) // PART, p)


def update(
    mut device: Device,
    n: Int,
    iteration: Int,
    mut optimizer_step: Int,
    seed: UInt32,
) raises:
    var batch = n * ROLLOUT // MINIBATCHES
    var chunks = (batch + CHUNK - 1) // CHUNK
    for epoch in range(EPOCHS):
        var shuffle = seed + UInt32(iteration * EPOCHS + epoch) * 0x9E3779B9
        for minibatch in range(MINIBATCHES):
            var p = Params(Int32(n), shuffle, Int32(minibatch * batch))
            backprop(device, n, batch, chunks, p)
            device.run[gradient_moments](MOMENT_PARTS, p)
            device.run[clip](1, p)
            optimizer_step += 1
            device.run[adam](
                WEIGHTS,
                Params(Int32(n), seed, 0, Int32(optimizer_step), 0, 0.0003),
            )
