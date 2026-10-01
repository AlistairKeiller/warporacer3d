"""Shared tanh backbone, Gaussian actor, scalar critic, and explicit backprop."""
from std.math import tanh, sqrt
from layout import TileTensor, row_major
from linalg.matmul import matmul
from .device import Device, Params, Ptr, normal, clamp, GPU_AVAILABLE
from .simulation import OBS, env_size

comptime HIDDEN = 64
comptime W0 = 0
comptime B0 = OBS * HIDDEN
comptime W1 = B0 + HIDDEN
comptime B1 = W1 + HIDDEN * HIDDEN
comptime W2 = B1 + HIDDEN
comptime B2 = W2 + HIDDEN * 3
comptime LOGSTD = B2 + 3
comptime WEIGHTS = LOGSTD + 2
comptime ROLLOUT = 32
comptime PART = 256


@fieldwise_init
struct Memory(TrivialRegisterPassable):
    var weights: Int
    var gradient: Int
    var first: Int
    var second: Int
    var obs: Int
    var actions: Int
    var values: Int
    var logp: Int
    var rewards: Int
    var done: Int
    var advantage: Int
    var target: Int
    var x: Int
    var h1: Int
    var h2: Int
    var output: Int
    var d1: Int
    var d2: Int
    var d3: Int
    var xt: Int
    var h1t: Int
    var h2t: Int
    var partial: Int
    var stats: Int
    var size: Int


def memory(n: Int) -> Memory:
    var samples = n * ROLLOUT
    var batch = samples // 4
    var w = env_size(n)
    var g = w + WEIGHTS
    var m = g + WEIGHTS
    var v = m + WEIGHTS
    var obs = v + WEIGHTS
    var actions = obs + samples * OBS
    var values = actions + samples * 2
    var logp = values + samples
    var rewards = logp + samples
    var done = rewards + samples
    var advantage = done + samples
    var target = advantage + samples
    var x = target + samples
    var h1 = x + batch * OBS
    var h2 = h1 + batch * HIDDEN
    var output = h2 + batch * HIDDEN
    var d1 = output + batch * 3
    var d2 = d1 + batch * HIDDEN
    var d3 = d2 + batch * HIDDEN
    var xt = d3 + batch * 3
    var h1t = xt + batch * OBS
    var h2t = h1t + batch * HIDDEN
    var partial = h2t + batch * HIDDEN
    var stats = partial + 3 * ((max(samples, WEIGHTS) + PART - 1) // PART)
    return Memory(
        w,
        g,
        m,
        v,
        obs,
        actions,
        values,
        logp,
        rewards,
        done,
        advantage,
        target,
        x,
        h1,
        h2,
        output,
        d1,
        d2,
        d3,
        xt,
        h1t,
        h2t,
        partial,
        stats,
        stats + 8,
    )


def initialize(i: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(Int(p.envs))
    var value: Float32 = 0
    if i < B0:
        value = normal(p.seed + UInt32(i) * 0x9E3779B9) * sqrt(2 / Float32(OBS))
    elif W1 <= i and i < B1:
        value = normal(p.seed + UInt32(i) * 0x9E3779B9) * 0.176776695
    elif W2 <= i and i < B2:
        var scale: Float32 = 0.125 if (i - W2) % 3 == 2 else 0.00125
        value = normal(p.seed + UInt32(i) * 0x9E3779B9) * scale
    elif i >= LOGSTD:
        value = -0.5
    data[unsafe_offset=mem.weights + i] = value


def mm[
    transpose_b: Bool = False
](
    mut device: Device, dest: Int, lhs: Int, rhs: Int, m: Int, n: Int, k: Int
) raises:
    var a = device.data.create_sub_buffer[DType.float32](lhs, m * k)
    var b = device.data.create_sub_buffer[DType.float32](rhs, k * n)
    var c = device.data.create_sub_buffer[DType.float32](dest, m * n)
    var at = TileTensor(a, row_major(Int32(m), Int32(k)))
    var bt = TileTensor(
        b,
        row_major(
            Int32(n if transpose_b else k), Int32(k if transpose_b else n)
        ),
    )
    var ct = TileTensor(c, row_major(Int32(m), Int32(n)))
    comptime if GPU_AVAILABLE:
        if device.gpu:
            matmul[transpose_b=transpose_b, target="gpu"](
                ct, at, bt, device.ctx
            )
            return
    matmul[transpose_b=transpose_b, target="cpu"](ct, at, bt)


def activation(i: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(Int(p.envs))
    var layer = Int(p.index)
    var width = 3 if layer == 2 else HIDDEN
    var offset = mem.h1 if layer == 0 else (
        mem.h2 if layer == 1 else mem.output
    )
    var bias = B0 if layer == 0 else (B1 if layer == 1 else B2)
    var value = (
        data[unsafe_offset=offset + i]
        + data[unsafe_offset=mem.weights + bias + i % width]
    )
    value = value if layer == 2 else tanh(value)
    data[unsafe_offset=offset + i] = value
    if layer != 2 and p.flag > 0:
        var transposed = mem.h1t if layer == 0 else mem.h2t
        data[
            unsafe_offset=transposed + (i % width) * Int(p.offset) + i // width
        ] = value


def forward(mut device: Device, n: Int, batch: Int, training: Bool) raises:
    var mem = memory(n)
    mm(device, mem.h1, mem.x, mem.weights + W0, batch, HIDDEN, OBS)
    device.run[activation](
        batch * HIDDEN, Params(Int32(n), 0, Int32(batch), 0, Int32(training))
    )
    mm(device, mem.h2, mem.h1, mem.weights + W1, batch, HIDDEN, HIDDEN)
    device.run[activation](
        batch * HIDDEN, Params(Int32(n), 0, Int32(batch), 1, Int32(training))
    )
    mm(device, mem.output, mem.h2, mem.weights + W2, batch, 3, HIDDEN)
    device.run[activation](batch * 3, Params(Int32(n), 0, Int32(batch), 2, 0))


def derivative(i: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(Int(p.envs))
    var activation = mem.h1 if p.index == 0 else mem.h2
    var delta = mem.d1 if p.index == 0 else mem.d2
    var value = data[unsafe_offset=activation + i]
    data[unsafe_offset=delta + i] *= 1 - value * value


def bias_gradient(i: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(Int(p.envs))
    var width = 3 if p.index == 2 else HIDDEN
    var delta = mem.d1 if p.index == 0 else (mem.d2 if p.index == 1 else mem.d3)
    var bias = B0 if p.index == 0 else (B1 if p.index == 1 else B2)
    var sum: Float32 = 0
    for row in range(Int(p.offset)):
        sum += data[unsafe_offset=delta + row * width + i]
    data[unsafe_offset=mem.gradient + bias + i] = sum


def backward(mut device: Device, n: Int, batch: Int) raises:
    var mem = memory(n)
    mm(device, mem.gradient + W2, mem.h2t, mem.d3, HIDDEN, 3, batch)
    device.run[bias_gradient](3, Params(Int32(n), 0, Int32(batch), 2))
    mm[True](device, mem.d2, mem.d3, mem.weights + W2, batch, HIDDEN, 3)
    device.run[derivative](batch * HIDDEN, Params(Int32(n), 0, 0, 1))
    mm(device, mem.gradient + W1, mem.h1t, mem.d2, HIDDEN, HIDDEN, batch)
    device.run[bias_gradient](HIDDEN, Params(Int32(n), 0, Int32(batch), 1))
    mm[True](device, mem.d1, mem.d2, mem.weights + W1, batch, HIDDEN, HIDDEN)
    device.run[derivative](batch * HIDDEN, Params(Int32(n), 0, 0, 0))
    mm(device, mem.gradient + W0, mem.xt, mem.d1, OBS, HIDDEN, batch)
    device.run[bias_gradient](HIDDEN, Params(Int32(n), 0, Int32(batch), 0))
