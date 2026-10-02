"""Shared tanh backbone, Gaussian actor, scalar critic, and explicit backprop.

Inputs and hidden activations carry a constant-1 lane so biases are weight
rows. The last layer is stored transposed (OUT x HA), and a transposed copy
of the middle layer is kept next to the weights, so every dot product reads
contiguous memory.

GPU matmuls (forward layers, the hidden backprop, and the split-K weight
gradients) share one 64x64 shared-memory tiled kernel whose operands are
thin loader functions. The CPU runs register-blocked rows of the same
arithmetic, 64 lanes at a time.
"""
from std.math import tanh, sqrt
from std.sys import is_gpu
from std.memory import unsafe_stack_allocation, AddressSpace
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from .device import Ptr, Vec, Params, normal, LANES
from .layout import (
    Memory,
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
    CHUNK,
    PART,
    ROLLOUT,
    MINIBATCHES,
    memory,
    permutation,
)

# Inner dot products use one lane per GPU thread and SIMD vectors on the CPU.
comptime DOT = 1 if is_gpu() else LANES


@inline(.always)
def hidden[
    width: Int
](j: Int, x: Pointer[Float32, _, address_space=_], w: Ptr, rows: Int) -> Vec[
    width
]:
    """tanh of lanes j..j+width of `x @ w` for a `rows` x H weight block."""
    var acc = Vec[width](0)
    for k in range(rows):
        acc += (
            Vec[width](x[unsafe_offset=k])
            * w.unsafe_offset(k * H + j).unsafe_load[width=width]()
        )
    return tanh(acc)


@inline(.always)
def dot[
    width: Int
](a: Pointer[Float32, _, address_space=_], b: Ptr, count: Int) -> Float32:
    """Contiguous dot product; `width` lanes at a time (count % width == 0)."""
    var acc = Vec[width](0)
    for k in range(0, count, width):
        acc += (
            a.unsafe_offset(k).unsafe_load[width=width]()
            * b.unsafe_offset(k).unsafe_load[width=width]()
        )
    return acc.reduce_add()


@inline(.always)
def row_hidden(
    x: Pointer[Float32, _, address_space=_],
    w: Ptr,
    rows: Int,
    dest: MutPointer[Float32, _, address_space=_],
):
    """All H outputs of one row, `tanh(x @ w)`, with register accumulators."""
    comptime assert H == 4 * LANES, "a hidden row is exactly four vectors"
    var a0 = Vec[LANES](0)
    var a1 = Vec[LANES](0)
    var a2 = Vec[LANES](0)
    var a3 = Vec[LANES](0)
    for k in range(rows):
        var xv = Vec[LANES](x[unsafe_offset=k])
        var wk = w.unsafe_offset(k * H)
        a0 += xv * wk.unsafe_load[width=LANES]()
        a1 += xv * wk.unsafe_offset(LANES).unsafe_load[width=LANES]()
        a2 += xv * wk.unsafe_offset(2 * LANES).unsafe_load[width=LANES]()
        a3 += xv * wk.unsafe_offset(3 * LANES).unsafe_load[width=LANES]()
    dest.unsafe_store(0, tanh(a0))
    dest.unsafe_store(LANES, tanh(a1))
    dest.unsafe_store(2 * LANES, tanh(a2))
    dest.unsafe_store(3 * LANES, tanh(a3))


def initialize(i: Int, data: Ptr, map: Ptr, p: Params):
    """Gaussian init of the real rows; bias and padding rows start at zero."""
    var mem = memory(Int(p.envs))
    var value: Float32 = 0
    var noise = normal(p.seed + UInt32(i) * 0x9E3779B9)
    if i < W1:
        if i // H < OBS:
            value = noise * sqrt(2 / Float32(OBS))
    elif i < W2:
        if (i - W1) // H < H:
            value = noise * 0.176776695
    elif i < LOGSTD:
        var row = (i - W2) // HA
        if (i - W2) % HA < H:
            value = noise * (Float32(0.125) if row == 2 else Float32(0.00125))
    else:
        value = -0.5
    data[unsafe_offset=mem.weights + i] = value


def transpose_w1(e: Int, data: Ptr, map: Ptr, p: Params):
    """Refresh the transposed middle layer after weights change wholesale."""
    var mem = memory(Int(p.envs))
    data[unsafe_offset=mem.w1t + (e % H) * HA + e // H] = data[
        unsafe_offset=mem.weights + W1 + e
    ]


def pad_activations(i: Int, data: Ptr, map: Ptr, p: Params):
    """Constant lanes of the minibatch activations: one 1, then zeros."""
    var mem = memory(Int(p.envs))
    if i % HA >= H:
        var value: Float32 = 1 if i % HA == H else 0
        data[unsafe_offset=mem.h1 + i] = value
        data[unsafe_offset=mem.h2 + i] = value


comptime SLICE = 8


def gather(e: Int, data: Ptr, map: Ptr, p: Params):
    """x[b] = obs[shuffle(b)] for the minibatch at p.offset, SLICE floats
    per thread so the permutation is computed once per slice."""
    var n = Int(p.envs)
    var mem = memory(n)
    var b = e // (IN // SLICE)
    var k = e % (IN // SLICE) * SLICE
    var row = permutation(Int(p.offset) + b, n, p.seed)
    data.unsafe_offset(mem.x + b * IN + k).unsafe_store(
        0, data.unsafe_offset(mem.obs + row * IN + k).unsafe_load[width=SLICE]()
    )


def gather_row(b: Int, data: Ptr, map: Ptr, p: Params):
    var n = Int(p.envs)
    var mem = memory(n)
    var row = permutation(Int(p.offset) + b, n, p.seed)
    for k in range(0, IN, LANES):
        data.unsafe_offset(mem.x + b * IN + k).unsafe_store(
            0,
            data.unsafe_offset(mem.obs + row * IN + k).unsafe_load[
                width=LANES
            ](),
        )


# ===-------------------------------------------------------------------=== #
# Generic 64x64 tiled matmul for the GPU: C[m, n] = sum_k A[m, k] * B[k, n].
# ===-------------------------------------------------------------------=== #

comptime Loader = def(Int, Int, Int, Ptr, Memory, Params) thin -> Float32
comptime Storer = def(Int, Int, Int, Float32, Ptr, Memory, Params) thin -> None
comptime TILE = 64
comptime TK = 16
comptime TILE_THREADS = 256


def tiled[
    a_at: Loader,
    b_at: Loader,
    store: Storer,
    a_rowwise: Bool,
](
    data: Ptr,
    map: Ptr,
    p: Params,
    m_count: Int32,
    n_count: Int32,
    k_count: Int32,
):
    """One 64x64 output tile per 256-thread block; grid z selects a replica
    (e.g. a split-K chunk). `a_rowwise` says A is contiguous along k."""
    var mem = memory(Int(p.envs))
    var tx = thread_idx.x
    var m0 = block_idx.y * TILE
    var n0 = block_idx.x * TILE
    var z = block_idx.z
    var M = Int(m_count)
    var N = Int(n_count)
    var K = Int(k_count)
    var a_tile = unsafe_stack_allocation[
        TILE * TK, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var b_tile = unsafe_stack_allocation[
        TK * TILE, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var tm = (tx // 16) * 4
    var tn = (tx % 16) * 4
    var acc0 = SIMD[DType.float32, 4](0)
    var acc1 = SIMD[DType.float32, 4](0)
    var acc2 = SIMD[DType.float32, 4](0)
    var acc3 = SIMD[DType.float32, 4](0)
    for k0 in range(0, K, TK):
        comptime for r in range(4):
            var q = tx + r * TILE_THREADS
            var ml: Int
            var kl: Int
            comptime if a_rowwise:
                ml = q // TK
                kl = q % TK
            else:
                kl = q // TILE
                ml = q % TILE
            var value: Float32 = 0
            if m0 + ml < M and k0 + kl < K:
                value = a_at(m0 + ml, k0 + kl, z, data, mem, p)
            a_tile[unsafe_offset=ml * TK + kl] = value
            var kb = q // TILE
            var nl = q % TILE
            value = 0
            if k0 + kb < K and n0 + nl < N:
                value = b_at(k0 + kb, n0 + nl, z, data, mem, p)
            b_tile[unsafe_offset=kb * TILE + nl] = value
        barrier()
        for kk in range(TK):
            var b = b_tile.unsafe_offset(kk * TILE + tn).unsafe_load[width=4]()
            var a = a_tile.unsafe_offset(tm * TK + kk)
            acc0 += a[unsafe_offset=0] * b
            acc1 += a[unsafe_offset=TK] * b
            acc2 += a[unsafe_offset=2 * TK] * b
            acc3 += a[unsafe_offset=3 * TK] * b
        barrier()
    comptime for c in range(4):
        var n = n0 + tn + c
        if n < N:
            if m0 + tm < M:
                store(m0 + tm, n, z, acc0[c], data, mem, p)
            if m0 + tm + 1 < M:
                store(m0 + tm + 1, n, z, acc1[c], data, mem, p)
            if m0 + tm + 2 < M:
                store(m0 + tm + 2, n, z, acc2[c], data, mem, p)
            if m0 + tm + 3 < M:
                store(m0 + tm + 3, n, z, acc3[c], data, mem, p)


# Operand loaders and stores for the forward layers and the hidden backprop.


def x_at(b: Int, k: Int, z: Int, data: Ptr, mem: Memory, p: Params) -> Float32:
    return data[unsafe_offset=mem.x + b * IN + k]


def w0_at(k: Int, j: Int, z: Int, data: Ptr, mem: Memory, p: Params) -> Float32:
    return data[unsafe_offset=mem.weights + W0 + k * H + j]


def h1_at(b: Int, k: Int, z: Int, data: Ptr, mem: Memory, p: Params) -> Float32:
    return data[unsafe_offset=mem.h1 + b * HA + k]


def w1_at(k: Int, j: Int, z: Int, data: Ptr, mem: Memory, p: Params) -> Float32:
    return data[unsafe_offset=mem.weights + W1 + k * H + j]


def w1t_at(
    j: Int, k: Int, z: Int, data: Ptr, mem: Memory, p: Params
) -> Float32:
    return data[unsafe_offset=mem.w1t + j * HA + k]


def d2_at(b: Int, j: Int, z: Int, data: Ptr, mem: Memory, p: Params) -> Float32:
    return data[unsafe_offset=mem.d2 + b * H + j]


def store_h1(
    b: Int, j: Int, z: Int, v: Float32, data: Ptr, mem: Memory, p: Params
):
    data[unsafe_offset=mem.h1 + b * HA + j] = tanh(v)


def store_h2(
    b: Int, j: Int, z: Int, v: Float32, data: Ptr, mem: Memory, p: Params
):
    data[unsafe_offset=mem.h2 + b * HA + j] = tanh(v)


def store_d1(
    b: Int, k: Int, z: Int, v: Float32, data: Ptr, mem: Memory, p: Params
):
    var h = data[unsafe_offset=mem.h1 + b * HA + k]
    data[unsafe_offset=mem.d1 + b * H + k] = v * (1 - h * h)


# Split-K gradient operands: m indexes a weight row, k a row of chunk z.


@inline(.always)
def chunk_row(k: Int, z: Int, p: Params) -> Int:
    var b = z * CHUNK + k
    return b if b < Int(p.envs) * ROLLOUT // MINIBATCHES else -1


def x_t_at(
    m: Int, k: Int, z: Int, data: Ptr, mem: Memory, p: Params
) -> Float32:
    var b = chunk_row(k, z, p)
    return 0 if b < 0 else data[unsafe_offset=mem.x + b * IN + m]


def d1_at(k: Int, j: Int, z: Int, data: Ptr, mem: Memory, p: Params) -> Float32:
    var b = chunk_row(k, z, p)
    return 0 if b < 0 else data[unsafe_offset=mem.d1 + b * H + j]


def h1_t_at(
    m: Int, k: Int, z: Int, data: Ptr, mem: Memory, p: Params
) -> Float32:
    var b = chunk_row(k, z, p)
    return 0 if b < 0 else data[unsafe_offset=mem.h1 + b * HA + m]


def d2_rows_at(
    k: Int, j: Int, z: Int, data: Ptr, mem: Memory, p: Params
) -> Float32:
    var b = chunk_row(k, z, p)
    return 0 if b < 0 else data[unsafe_offset=mem.d2 + b * H + j]


def store_w0(
    k: Int, j: Int, z: Int, v: Float32, data: Ptr, mem: Memory, p: Params
):
    data[unsafe_offset=mem.partial + z * TOTAL + W0 + k * H + j] = v


def store_w1(
    k: Int, j: Int, z: Int, v: Float32, data: Ptr, mem: Memory, p: Params
):
    data[unsafe_offset=mem.partial + z * TOTAL + W1 + k * H + j] = v


comptime layer1_gpu = tiled[x_at, w0_at, store_h1, True]
comptime layer2_gpu = tiled[h1_at, w1_at, store_h2, True]
comptime back1_gpu = tiled[d2_at, w1t_at, store_d1, True]
comptime partials_w0_gpu = tiled[x_t_at, d1_at, store_w0, False]
comptime partials_w1_gpu = tiled[h1_t_at, d2_rows_at, store_w1, False]


# ===-------------------------------------------------------------------=== #
# Per-element GPU kernels and per-row CPU kernels.
# ===-------------------------------------------------------------------=== #


def layer1_cpu(b: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(Int(p.envs))
    row_hidden(
        data.unsafe_offset(mem.x + b * IN),
        data.unsafe_offset(mem.weights + W0),
        IN,
        data.unsafe_offset(mem.h1 + b * HA),
    )


def layer2_cpu(b: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(Int(p.envs))
    row_hidden(
        data.unsafe_offset(mem.h1 + b * HA),
        data.unsafe_offset(mem.weights + W1),
        HA,
        data.unsafe_offset(mem.h2 + b * HA),
    )


def layer3(e: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(Int(p.envs))
    var b = e // OUT
    var j = e % OUT
    data[unsafe_offset=mem.output + e] = dot[DOT](
        data.unsafe_offset(mem.h2 + b * HA),
        data.unsafe_offset(mem.weights + W2 + j * HA),
        HA,
    )


def back2(e: Int, data: Ptr, map: Ptr, p: Params):
    """d2 = (1 - h2^2) * d3 @ W2 for lanes starting at e (DOT per task)."""
    var mem = memory(Int(p.envs))
    var b = e // H
    var j = e % H
    var acc = Vec[DOT](0)
    for o in range(OUT):
        acc += (
            Vec[DOT](data[unsafe_offset=mem.d3 + b * OUT + o])
            * data.unsafe_offset(mem.weights + W2 + o * HA + j).unsafe_load[
                width=DOT
            ]()
        )
    var h = data.unsafe_offset(mem.h2 + b * HA + j).unsafe_load[width=DOT]()
    data.unsafe_offset(mem.d2 + e).unsafe_store(0, acc * (1 - h * h))


def back2_cpu(b: Int, data: Ptr, map: Ptr, p: Params):
    for j in range(0, H, LANES):
        back2(b * H + j, data, map, p)


def back1(e: Int, data: Ptr, map: Ptr, p: Params):
    """d1 = (1 - h1^2) * d2 @ W1^T for hidden unit e % H of row e // H."""
    var mem = memory(Int(p.envs))
    var b = e // H
    var k = e % H
    var h = data[unsafe_offset=mem.h1 + b * HA + k]
    data[unsafe_offset=mem.d1 + e] = (1 - h * h) * dot[DOT](
        data.unsafe_offset(mem.d2 + b * H),
        data.unsafe_offset(mem.weights + W1 + k * H),
        H,
    )


def back1_cpu(b: Int, data: Ptr, map: Ptr, p: Params):
    """One row of d1 = (1 - h1^2) * d2 @ W1^T with register accumulators."""
    var mem = memory(Int(p.envs))
    var a0 = Vec[LANES](0)
    var a1 = Vec[LANES](0)
    var a2 = Vec[LANES](0)
    var a3 = Vec[LANES](0)
    for j in range(H):
        var d = Vec[LANES](data[unsafe_offset=mem.d2 + b * H + j])
        var w = data.unsafe_offset(mem.w1t + j * HA)
        a0 += d * w.unsafe_load[width=LANES]()
        a1 += d * w.unsafe_offset(LANES).unsafe_load[width=LANES]()
        a2 += d * w.unsafe_offset(2 * LANES).unsafe_load[width=LANES]()
        a3 += d * w.unsafe_offset(3 * LANES).unsafe_load[width=LANES]()
    var h = data.unsafe_offset(mem.h1 + b * HA)
    var dest = data.unsafe_offset(mem.d1 + b * H)
    var h0 = h.unsafe_load[width=LANES]()
    var h1 = h.unsafe_offset(LANES).unsafe_load[width=LANES]()
    var h2 = h.unsafe_offset(2 * LANES).unsafe_load[width=LANES]()
    var h3 = h.unsafe_offset(3 * LANES).unsafe_load[width=LANES]()
    dest.unsafe_store(0, a0 * (1 - h0 * h0))
    dest.unsafe_store(LANES, a1 * (1 - h1 * h1))
    dest.unsafe_store(2 * LANES, a2 * (1 - h2 * h2))
    dest.unsafe_store(3 * LANES, a3 * (1 - h3 * h3))


def partials_tail(e: Int, data: Ptr, map: Ptr, p: Params):
    """Split-K partial sums for the small entries: W2 rows, log std, KL."""
    var n = Int(p.envs)
    var mem = memory(n)
    var batch = n * ROLLOUT // MINIBATCHES
    var z = e // (TOTAL - W2)
    var w = W2 + e % (TOTAL - W2)
    var sum: Float32 = 0
    for b in range(z * CHUNK, min((z + 1) * CHUNK, batch)):
        if w < LOGSTD:
            sum += (
                data[unsafe_offset=mem.h2 + b * HA + (w - W2) % HA]
                * data[unsafe_offset=mem.d3 + b * OUT + (w - W2) // HA]
            )
        else:
            sum += data[unsafe_offset=mem.extra + b * OUT + w - LOGSTD]
    data[unsafe_offset=mem.partial + z * TOTAL + w] = sum


@inline(.always)
def outer_rows(
    column: Ptr,
    stride: Int,
    rows_ptr: Ptr,
    row_stride: Int,
    count: Int,
    dest: Ptr,
):
    """out[0:H] = sum over `count` rows of column[r*stride] * rows[r*row_stride + 0:H].
    """
    var a0 = Vec[LANES](0)
    var a1 = Vec[LANES](0)
    var a2 = Vec[LANES](0)
    var a3 = Vec[LANES](0)
    for r in range(count):
        var a = Vec[LANES](column[unsafe_offset=r * stride])
        var d = rows_ptr.unsafe_offset(r * row_stride)
        a0 += a * d.unsafe_load[width=LANES]()
        a1 += a * d.unsafe_offset(LANES).unsafe_load[width=LANES]()
        a2 += a * d.unsafe_offset(2 * LANES).unsafe_load[width=LANES]()
        a3 += a * d.unsafe_offset(3 * LANES).unsafe_load[width=LANES]()
    dest.unsafe_store(0, a0)
    dest.unsafe_store(LANES, a1)
    dest.unsafe_store(2 * LANES, a2)
    dest.unsafe_store(3 * LANES, a3)


def partials_cpu(z: Int, data: Ptr, map: Ptr, p: Params):
    """All split-K partial sums of chunk z: one weight row per pass."""
    var n = Int(p.envs)
    var mem = memory(n)
    var batch = n * ROLLOUT // MINIBATCHES
    var first = z * CHUNK
    var count = min(CHUNK, batch - first)
    var acc = data.unsafe_offset(mem.partial + z * TOTAL)
    var x = data.unsafe_offset(mem.x + first * IN)
    var d1 = data.unsafe_offset(mem.d1 + first * H)
    for k in range(IN):
        outer_rows(
            x.unsafe_offset(k), IN, d1, H, count, acc.unsafe_offset(W0 + k * H)
        )
    var h1 = data.unsafe_offset(mem.h1 + first * HA)
    var d2 = data.unsafe_offset(mem.d2 + first * H)
    for k in range(HA):
        outer_rows(
            h1.unsafe_offset(k), HA, d2, H, count, acc.unsafe_offset(W1 + k * H)
        )
    var h2 = data.unsafe_offset(mem.h2 + first * HA)
    var d3 = data.unsafe_offset(mem.d3 + first * OUT)
    for o in range(OUT):
        for k in range(0, HA, LANES):
            var sum = Vec[LANES](0)
            for r in range(count):
                sum += (
                    Vec[LANES](d3[unsafe_offset=r * OUT + o])
                    * h2.unsafe_offset(r * HA + k).unsafe_load[width=LANES]()
                )
            acc.unsafe_offset(W2 + o * HA + k).unsafe_store(0, sum)
    for j in range(OUT):
        var sum: Float32 = 0
        for r in range(count):
            sum += data[unsafe_offset=mem.extra + (first + r) * OUT + j]
        acc[unsafe_offset=LOGSTD + j] = sum


def reduce(w: Int, data: Ptr, map: Ptr, p: Params):
    """Sum the split-K partials into the gradient (and the KL statistic)."""
    var n = Int(p.envs)
    var mem = memory(n)
    var batch = n * ROLLOUT // MINIBATCHES
    var sum: Float32 = 0
    for c in range((batch + CHUNK - 1) // CHUNK):
        sum += data[unsafe_offset=mem.partial + c * TOTAL + w]
    if w < WEIGHTS:
        data[unsafe_offset=mem.gradient + w] = sum
    else:
        data[unsafe_offset=mem.stats + 4] = sum / Float32(batch)


def reduce_cpu(block: Int, data: Ptr, map: Ptr, p: Params):
    """The same sums for PART consecutive entries, vectorized over entries."""
    var n = Int(p.envs)
    var mem = memory(n)
    var batch = n * ROLLOUT // MINIBATCHES
    var chunks = (batch + CHUNK - 1) // CHUNK
    var start = block * PART
    var stop = min(start + PART, TOTAL)
    var w = start
    while w + LANES <= stop:
        var sum = Vec[LANES](0)
        for c in range(chunks):
            sum += data.unsafe_offset(mem.partial + c * TOTAL + w).unsafe_load[
                width=LANES
            ]()
        data.unsafe_offset(mem.gradient + w).unsafe_store(0, sum)
        w += LANES
    while w < stop:
        reduce(w, data, map, p)
        w += 1
