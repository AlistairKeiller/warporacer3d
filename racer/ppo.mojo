"""A tanh MLP with a Gaussian actor and a scalar critic, trained by clipped PPO.

Activations are feature-major (one column per sample) with a constant-1 row, so
biases are weight columns and every forward, backward and weight-gradient
product is a MAX `matmul` (`transpose_b` for the gradients; `matmul` has no
`transpose_a`, so transposed weight copies are kept for backpropagation).
Everything per sample or per weight is a kernel; the statistics an iteration
needs are row sums into a small `stats` buffer, read back only to print.
"""
from std.math import tanh, sqrt, exp, clamp
from std.random.philox import NormalRandom
from std.utils import IndexList
from layout import Coord
from .device import Device, Buffer, Mat, mat, save_floats, load_floats
from .sim import Sim, IN, OBS, OUT, HALF_LOG_2PI

comptime H = 64
comptime HA = H + 1  # hidden units plus the constant 1
comptime W0 = 0  # [H, IN]
comptime W1 = W0 + H * IN  # [H, HA]
comptime W2 = W1 + H * HA  # [OUT, HA]
comptime LOGSTD = W2 + OUT * HA  # [2]
comptime P = LOGSTD + 2
comptime ROLLOUT = 32
comptime EPOCHS = 4
comptime MINIBATCHES = 4
comptime GAMMA = Float32(0.99)
comptime LAMBDA = Float32(0.95)
comptime CLIP = Float32(0.2)
comptime LR = Float32(3e-4)
comptime MAX_NORM = Float32(0.5)
# Slots of the `stats` buffer: reward sum, advantage sum and sum of squares, squared gradient norm, KL sum.
comptime REWARD = 0
comptime ADV = 1
comptime NORM = 3
comptime KL = 4
comptime STATS = 5


@__parameter
def activate[
    dtype: DType, width: SIMDLength, *, alignment: Int = 1
](idx: IndexList[2], v: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return tanh(v.cast[DType.float32]()).cast[dtype]()


struct Activations:
    """Hidden layers and outputs for `cols` samples; row H of each hidden layer
    stays the constant 1 that feeds the next layer's bias column."""

    var cols: Int
    var h1: Buffer
    var h2: Buffer
    var o: Buffer

    def __init__(out self, device: Device, cols: Int) raises:
        self.cols = cols
        self.h1 = device.alloc(HA * cols, fill=1)
        self.h2 = device.alloc(HA * cols, fill=1)
        self.o = device.alloc(OUT * cols)


struct Policy:
    var theta: Buffer  # [P]: W0, W1, W2, log std
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
                scale = (Float32(0.00125) if (i - W2) // HA < 2 else Float32(0.125)) if i % HA < H else 0
            else:
                theta[0, i] = -0.5
                return
            var rng = NormalRandom(seed=UInt64(seed), subsequence=UInt64(i))
            theta[0, i] = scale * rng.step_normal_4()[0]

        device.run(init, P)
        self.transpose(device)

    def transpose(self, device: Device) raises:
        var w1 = mat[H](self.theta, HA, W1)
        var w2 = mat[OUT](self.theta, HA, W2)
        var w1t = mat[HA](self.w1t, H)
        var w2t = mat[HA](self.w2t, OUT)

        def kernel(k: Int, j: Int) {var}:
            w1t[k, j] = w1[j, k]
            if j < OUT:
                w2t[k, j] = w2[j, k]

        device.run(kernel, HA, H)

    def log_std(self) -> Mat[1]:
        return mat[1](self.theta, 2, LOGSTD)

    def forward(self, device: Device, x: Mat[IN], a: Activations) raises:
        device.gemm[epilogue=activate](mat[H](a.h1, a.cols), mat[H](self.theta, IN, W0), x)
        device.gemm[epilogue=activate](mat[H](a.h2, a.cols), mat[H](self.theta, HA, W1), mat[HA](a.h1, a.cols))
        device.gemm(mat[OUT](a.o, a.cols), mat[OUT](self.theta, HA, W2), mat[HA](a.h2, a.cols))

    def backward(
        self, device: Device, x: Mat[IN], a: Activations, d3: Mat[OUT], d2: Buffer, d1: Buffer, grad: Buffer
    ) raises:
        """Backpropagate output gradients d3 into `grad` (the W0, W1, W2 entries)."""
        var h1 = mat[HA](a.h1, a.cols)
        var h2 = mat[HA](a.h2, a.cols)
        self.through(device, mat[HA](d2, a.cols), mat[HA](self.w2t, OUT), d3, h2)
        self.through(device, mat[HA](d1, a.cols), mat[HA](self.w1t, H), mat[H](d2, a.cols), h1)
        device.gemm[transpose_b=True](mat[H](grad, IN, W0), mat[H](d1, a.cols), x)
        device.gemm[transpose_b=True](mat[H](grad, HA, W1), mat[H](d2, a.cols), h1)
        device.gemm[transpose_b=True](mat[OUT](grad, HA, W2), d3, h2)

    def through[
        rows: Int
    ](self, device: Device, result: Mat[HA], wt: Mat[HA], delta: Mat[rows], h: Mat[HA]) raises:
        """result = tanh'(h) * (wt @ delta): the gradient back through one layer."""

        @__parameter
        @__copy_capture(h)
        def mask[
            dtype: DType, width: SIMDLength, *, alignment: Int = 1
        ](idx: IndexList[2], v: SIMD[dtype, width]) -> SIMD[dtype, width]:
            var a = h.load[width=width](Coord(idx[0], idx[1]))
            return (v.cast[DType.float32]() * (1 - a * a)).cast[dtype]()

        device.gemm[epilogue=mask](result, wt, delta)


struct Trainer:
    """Rollout storage for n cars, one minibatch of scratch, and Adam state."""

    var n: Int
    var batch: Int
    var step: Int
    var obs: Buffer  # [ROLLOUT + 1][IN, n]
    var actions: Buffer  # [ROLLOUT][2, n]
    var logp: Buffer  # [ROLLOUT * n]
    var value: Buffer  # [ROLLOUT * n]
    var reward: Buffer  # [ROLLOUT * n]
    var done: Buffer  # [ROLLOUT * n]
    var advantage: Buffer  # [ROLLOUT * n]
    var work: Buffer  # [2, ROLLOUT * n] scratch for the sums behind the statistics
    var x: Buffer  # [IN, batch]
    var acts: Activations
    var d3: Buffer  # [OUT + 3, batch]: output gradients, then d log std (2) and the per-sample KL estimate
    var d2: Buffer  # [HA, batch]
    var d1: Buffer  # [HA, batch]
    var grad: Buffer  # [P]
    var first: Buffer  # [P] Adam moments
    var second: Buffer
    var stats: Buffer  # [STATS] device-side scalars; Adam restarts from zero on resume

    def __init__(out self, device: Device, n: Int) raises:
        self.n = n
        self.batch = n * ROLLOUT // MINIBATCHES
        self.step = 0
        self.obs = device.alloc((ROLLOUT + 1) * IN * n)
        self.actions = device.alloc(ROLLOUT * 2 * n)
        self.logp = device.alloc(ROLLOUT * n)
        self.value = device.alloc(ROLLOUT * n)
        self.reward = device.alloc(ROLLOUT * n)
        self.done = device.alloc(ROLLOUT * n)
        self.advantage = device.alloc(ROLLOUT * n)
        self.work = device.alloc(max(2 * ROLLOUT * n, P))
        self.x = device.alloc(IN * self.batch)
        self.acts = Activations(device, self.batch)
        self.d3 = device.alloc((OUT + 3) * self.batch)
        self.d2 = device.alloc(HA * self.batch)
        self.d1 = device.alloc(HA * self.batch)
        self.grad = device.alloc(P)
        self.first = device.alloc(P)
        self.second = device.alloc(P)
        self.stats = device.alloc(STATS)

    def rollout(self, device: Device, sim: Sim, policy: Policy, acts: Activations, iteration: Int) raises -> Float32:
        """ROLLOUT steps of every car, then advantages; returns the mean reward per step."""
        var n = self.n
        device.ctx.enqueue_copy(self.obs.create_sub_buffer[DType.float32](0, IN * n), sim.obs)
        for t in range(ROLLOUT):
            policy.forward(device, mat[IN](self.obs, n, t * IN * n), acts)
            sim.sample(
                device,
                mat[OUT](acts.o, n),
                policy.log_std(),
                mat[2](self.actions, n, t * 2 * n),
                mat[1](self.logp, n, t * n),
                mat[1](self.value, n, t * n),
                False,
                iteration * ROLLOUT + t,
            )
            sim.step(
                device,
                mat[2](self.actions, n, t * 2 * n),
                mat[1](self.reward, n, t * n),
                mat[1](self.done, n, t * n),
                mat[IN](self.obs, n, (t + 1) * IN * n),
            )
        device.ctx.enqueue_copy(sim.obs, self.obs.create_sub_buffer[DType.float32](ROLLOUT * IN * n, IN * n))
        policy.forward(device, mat[IN](self.obs, n, ROLLOUT * IN * n), acts)
        var o = mat[OUT](acts.o, n)
        var reward = mat[1](self.reward, ROLLOUT * n)
        var done = mat[1](self.done, ROLLOUT * n)
        var value = mat[1](self.value, ROLLOUT * n)
        var advantage = mat[1](self.advantage, ROLLOUT * n)

        def gae(i: Int) {var}:
            """Generalized advantage estimates for car i, bootstrapped from the final value."""
            var next = o[2, i]
            var running: Float32 = 0
            for t in range(ROLLOUT - 1, -1, -1):
                var k = t * n + i
                var alive: Float32 = 1 if done[0, k] == 0 else 0
                var delta = reward[0, k] + GAMMA * alive * next - value[0, k]
                running = delta + GAMMA * LAMBDA * alive * running
                advantage[0, k] = running
                next = value[0, k]

        device.run(gae, n)
        device.sum(reward, mat[1](self.stats, STATS), REWARD)
        return device.read(self.stats)[REWARD] / Float32(ROLLOUT * n)

    def update(mut self, device: Device, policy: Policy) raises -> Float32:
        """EPOCHS passes of clipped PPO over the minibatches; returns the KL
        estimate of the last minibatch."""
        var samples = ROLLOUT * self.n
        var stats = mat[1](self.stats, STATS)
        var advantage = mat[1](self.advantage, samples)
        var work = mat[2](self.work, samples)

        def moments(i: Int) {var}:
            work[0, i] = advantage[0, i]
            work[1, i] = advantage[0, i] * advantage[0, i]

        device.run(moments, samples)
        device.sum(work, stats, ADV)
        var read = device.read(self.stats)
        var adv_mean = read[ADV] / Float32(samples)
        var adv_scale = 1 / sqrt(max(read[ADV + 1] / Float32(samples) - adv_mean * adv_mean, 1e-8))
        for epoch in range(EPOCHS):
            for m in range(MINIBATCHES):
                self.minibatch(device, policy, (m + epoch) % MINIBATCHES, adv_mean, adv_scale)
                self.step += 1
                self.adam(device, policy)
        device.sum(mat[1](self.d3, self.batch, (OUT + 2) * self.batch), stats, KL)
        return device.read(self.stats)[KL] / Float32(self.batch)

    def minibatch(self, device: Device, policy: Policy, m: Int, adv_mean: Float32, adv_scale: Float32) raises:
        """Gradient of the clipped PPO loss over every MINIBATCHES-th sample
        from `m`, left in `grad` with its squared norm in `stats`."""
        var n = self.n
        var batch = self.batch
        var obs = mat[1](self.obs, len(self.obs))
        var actions = mat[1](self.actions, len(self.actions))
        var logp = mat[1](self.logp, ROLLOUT * n)
        var value = mat[1](self.value, ROLLOUT * n)
        var advantage = mat[1](self.advantage, ROLLOUT * n)
        var x = mat[IN](self.x, batch)
        var o = mat[OUT](self.acts.o, batch)
        var d3 = mat[OUT + 3](self.d3, batch)
        var grad = mat[1](self.grad, P)
        var work = mat[1](self.work, P)
        var log_std = policy.log_std()

        def gather(b: Int) {var}:
            var s = b * MINIBATCHES + m
            for r in range(IN):
                x[r, b] = obs[0, (s // n * IN + r) * n + s % n]

        def loss(b: Int) {var}:
            """Clipped surrogate and value loss derivatives for column b."""
            var s = b * MINIBATCHES + m
            var t = s // n
            var i = s % n
            var new_logp: Float32 = 0
            var error = SIMD[DType.float32, 2](0)
            var invvar = SIMD[DType.float32, 2](0)
            for a in range(2):
                error[a] = actions[0, (t * 2 + a) * n + i] - o[a, b]
                invvar[a] = exp(-2 * log_std[0, a])
                new_logp -= 0.5 * error[a] * error[a] * invvar[a] + log_std[0, a] + HALF_LOG_2PI
            var logratio = new_logp - logp[0, s]
            var ratio = exp(logratio)
            var a_hat = (advantage[0, s] - adv_mean) * adv_scale
            var dlogp = -a_hat * ratio / Float32(batch)
            if (a_hat >= 0 and ratio > 1 + CLIP) or (a_hat < 0 and ratio < 1 - CLIP):
                dlogp = 0
            for a in range(2):
                d3[a, b] = dlogp * error[a] * invvar[a]
                d3[OUT + a, b] = dlogp * (error[a] * error[a] * invvar[a] - 1)
            # Huber value loss: returns are tens, so a bounded error keeps the shared trunk balanced.
            d3[2, b] = 0.5 * clamp(o[2, b] - value[0, s] - advantage[0, s], -1, 1) / Float32(batch)
            d3[OUT + 2, b] = ratio - 1 - logratio

        def square(i: Int) {var}:
            work[0, i] = grad[0, i] * grad[0, i]

        device.run(gather, batch)
        policy.forward(device, x, self.acts)
        device.run(loss, batch)
        policy.backward(device, x, self.acts, mat[OUT](self.d3, batch), self.d2, self.d1, self.grad)
        device.sum(mat[2](self.d3, batch, OUT * batch), grad, LOGSTD)  # the log std gradient
        device.run(square, P)
        device.sum(work, mat[1](self.stats, STATS), NORM)

    def adam(self, device: Device, policy: Policy) raises:
        var step = self.step
        var theta = mat[1](policy.theta, P)
        var grad = mat[1](self.grad, P)
        var first = mat[1](self.first, P)
        var second = mat[1](self.second, P)
        var stats = mat[1](self.stats, STATS)

        def kernel(i: Int) {var}:
            var g = grad[0, i] * min(Float32(1), MAX_NORM / (sqrt(stats[0, NORM]) + 1e-8))
            var m = 0.9 * first[0, i] + 0.1 * g
            var v = 0.999 * second[0, i] + 0.001 * g * g
            first[0, i] = m
            second[0, i] = v
            var m_hat = m / (1 - Float32(0.9) ** Float32(step))
            var v_hat = v / (1 - Float32(0.999) ** Float32(step))
            var value = theta[0, i] - LR * m_hat / (sqrt(v_hat) + 1e-5)
            theta[0, i] = min(max(value, -2), 0) if i >= LOGSTD else value

        device.run(kernel, P)
        policy.transpose(device)



def save(device: Device, policy: Policy, path: String, iteration: Int) raises:
    """The weights and the iteration count as one float32 .npy file."""
    var values: List[Float32] = [Float32(iteration)]
    values.extend(device.read(policy.theta))
    save_floats(path, values)


def load(device: Device, policy: Policy, path: String) raises -> Int:
    """Restore saved weights into the policy; returns the iteration they were saved at."""
    var values = load_floats(path)
    if len(values) != 1 + P:
        raise Error("checkpoint does not match the network size")
    device.write(policy.theta, Span(values)[1:])
    policy.transpose(device)
    return Int(values[0])
