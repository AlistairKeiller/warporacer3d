"""Shared tanh MLP with a Gaussian actor and a scalar critic.

Activations are feature-major (one column per sample) with a constant-1 row,
so biases are ordinary weight columns and every forward, backward, and
weight-gradient product is a plain MAX matmul (`transpose_b` for gradients).
"""
from std.math import tanh, sqrt
from std.random.philox import NormalRandom
from std.utils import IndexList
from layout import Coord
from .device import Device, Buffer, Mat, mat
from .simulation import IN, OBS, OUT

comptime H = 64
comptime HA = H + 1
comptime W0 = 0  # [H, IN]
comptime W1 = W0 + H * IN  # [H, HA]
comptime W2 = W1 + H * HA  # [OUT, HA]
comptime LOGSTD = W2 + OUT * HA  # [2]
comptime P = LOGSTD + 2


struct Activations:
    """Hidden layers and outputs for `cols` samples; row H of each hidden
    layer is the constant 1 that feeds the next layer's bias column."""

    var cols: Int
    var h1: Buffer
    var h2: Buffer
    var o: Buffer

    def __init__(out self, device: Device, cols: Int) raises:
        self.cols = cols
        self.h1 = device.alloc(HA * cols)
        self.h2 = device.alloc(HA * cols)
        self.o = device.alloc(OUT * cols)
        var h1 = mat[HA](self.h1, cols)
        var h2 = mat[HA](self.h2, cols)

        def ones(i: Int) {var}:
            h1[H, i] = 1
            h2[H, i] = 1

        device.run(ones, cols)


@__parameter
def activate[
    dtype: DType, width: SIMDLength, *, alignment: Int = 1
](idx: IndexList[2], v: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return tanh(v.cast[DType.float32]()).cast[dtype]()


def through[
    rows: Int
](device: Device, result: Mat[HA], wt: Mat[HA], delta: Mat[rows], h: Mat[HA]) raises:
    """result = tanh'(h) * (wt @ delta): the gradient back through one layer."""

    @__parameter
    @__copy_capture(h)
    def mask[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], v: SIMD[dtype, width]) -> SIMD[dtype, width]:
        var a = h.load[width=width](Coord(idx[0], idx[1]))
        return (v.cast[DType.float32]() * (1 - a * a)).cast[dtype]()

    device.gemm[epilogue=mask](result, wt, delta)


struct Policy:
    var theta: Buffer  # [P]: W0, W1, W2, log std
    # Transposed copies of W1 and W2 for the backward products: MAX's GPU
    # matmul offers transpose_b but not transpose_a, and these are tiny next
    # to the activations. Refreshed after every update.
    var w1t: Buffer  # [HA, H]
    var w2t: Buffer  # [HA, OUT]

    def __init__(out self, device: Device, seed: Int) raises:
        self.theta = device.alloc(P)
        self.w1t = device.alloc(HA * H)
        self.w2t = device.alloc(HA * OUT)
        var theta = mat[1](self.theta, P)

        def init(i: Int) {var}:
            """Gaussian fan-in init; bias columns start at zero, log std at -0.5."""
            var scale: Float32 = 0
            if i < W1:
                scale = sqrt(2 / Float32(OBS)) if i % IN < OBS else 0
            elif i < W2:
                scale = sqrt(1 / Float32(H)) if i % HA < H else 0
            elif i < LOGSTD:
                var actor = (i - W2) // HA < 2  # tiny actor weights: start neutral
                if i % HA < H:
                    scale = 0.00125 if actor else 0.125
            else:
                theta[0, i] = -0.5
                return
            if scale == 0:
                theta[0, i] = 0
                return
            var z = NormalRandom(seed=UInt64(seed), subsequence=UInt64(i))
            theta[0, i] = scale * z.step_normal_4()[0]

        device.run(init, P)
        self.transpose(device)

    def transpose(self, device: Device) raises:
        var w1 = mat[H](self.theta, HA, W1)
        var w2 = mat[OUT](self.theta, HA, W2)
        var w1t = mat[HA](self.w1t, H)
        var w2t = mat[HA](self.w2t, OUT)

        def kernel(k: Int, j: Int) {var}:
            if j < H:
                w1t[k, j] = w1[j, k]
            if j < OUT:
                w2t[k, j] = w2[j, k]

        device.run(kernel, HA, H)

    def log_std(self) -> Mat[1]:
        return mat[1](self.theta, 2, LOGSTD)

    def forward(self, device: Device, x: Mat[IN], a: Activations) raises:
        device.gemm[epilogue=activate](
            mat[H](a.h1, a.cols), mat[H](self.theta, IN, W0), x
        )
        device.gemm[epilogue=activate](
            mat[H](a.h2, a.cols), mat[H](self.theta, HA, W1), mat[HA](a.h1, a.cols)
        )
        device.gemm(
            mat[OUT](a.o, a.cols), mat[OUT](self.theta, HA, W2), mat[HA](a.h2, a.cols)
        )

    def backward(
        self,
        device: Device,
        x: Mat[IN],
        a: Activations,
        d3: Mat[OUT],
        d2: Buffer,
        d1: Buffer,
        grad: Buffer,
    ) raises:
        """Backpropagate output gradients d3 into `grad` (W0, W1, W2 entries)."""
        var h1 = mat[HA](a.h1, a.cols)
        var h2 = mat[HA](a.h2, a.cols)
        through(device, mat[HA](d2, a.cols), mat[HA](self.w2t, OUT), d3, h2)
        through(
            device, mat[HA](d1, a.cols), mat[HA](self.w1t, H), mat[H](d2, a.cols), h1
        )
        device.gemm[transpose_b=True](
            mat[H](grad, IN, W0), mat[H](d1, a.cols), x
        )
        device.gemm[transpose_b=True](
            mat[H](grad, HA, W1), mat[H](d2, a.cols), h1
        )
        device.gemm[transpose_b=True](mat[OUT](grad, HA, W2), d3, h2)
