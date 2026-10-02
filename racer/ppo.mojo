"""Rollouts, GAE, the clipped PPO objective, and Adam, all on the device.

Scalar statistics (advantage moments, gradient norm, KL) are reduced on the
host; everything per sample or per weight runs as a kernel.
"""
from std.math import exp, log, sqrt, isfinite
from std.random.philox import Random
from std.utils import IndexList
from max.algorithm.reduction import mean, variance, sum
from layout import Coord, Idx
from .device import Device, Buffer, Mat, mat, read_floats, write_floats
from .simulation import Sim, STATE, IN, OUT
from .network import Policy, Activations, P, LOGSTD, H, HA

comptime ROLLOUT = 32
comptime EPOCHS = 4
comptime MINIBATCHES = 4
comptime GAMMA = Float32(0.99)
comptime LAMBDA = Float32(0.95)
comptime CLIP = Float32(0.2)
comptime LR = Float32(3e-4)
comptime MAX_NORM = Float32(0.5)
comptime CHECKPOINT_MAGIC = 271828
comptime CHECKPOINT_VERSION = 4


def permutation(index: Int, n: Int, seed: Int) -> Int:
    """Minibatch shuffle: two modular shears form a bijection for every N."""
    var row = index // n
    var col = index % n
    var rows = Random(seed=UInt64(seed), subsequence=UInt64(col))
    row = (row + Int(rows.step_uniform()[0] * ROLLOUT)) % ROLLOUT
    var cols = Random(seed=UInt64(seed) + 1, subsequence=UInt64(row))
    col = (col + Int(cols.step_uniform()[0] * Float32(n))) % n
    return row * n + col


struct Trainer:
    """Rollout storage for n cars, one minibatch of scratch, and Adam state."""

    var n: Int
    var batch: Int
    var step: Int
    var acts: Activations  # for stepping all n cars
    var obs: Buffer  # [ROLLOUT + 1][IN, n]
    var actions: Buffer  # [ROLLOUT][2, n]
    var logp: Buffer  # [ROLLOUT * n]
    var value: Buffer  # [(ROLLOUT + 1) * n]
    var reward: Buffer  # [ROLLOUT * n]
    var done: Buffer  # [ROLLOUT * n]
    var advantage: Buffer  # [ROLLOUT * n]
    var target: Buffer  # [ROLLOUT * n]
    var x: Buffer  # [IN, batch]
    var batch_acts: Activations
    var d3: Buffer  # [OUT, batch]
    var extra: Buffer  # [3, batch]: d log std (2), per-sample KL estimate
    var d2: Buffer  # [HA, batch]
    var d1: Buffer  # [HA, batch]
    var grad: Buffer  # [P]
    var first: Buffer  # [P] Adam moments
    var second: Buffer

    def __init__(out self, device: Device, n: Int) raises:
        self.n = n
        self.batch = n * ROLLOUT // MINIBATCHES
        self.step = 0
        self.acts = Activations(device, n)
        self.obs = device.alloc((ROLLOUT + 1) * IN * n)
        self.actions = device.alloc(ROLLOUT * 2 * n)
        self.logp = device.alloc(ROLLOUT * n)
        self.value = device.alloc((ROLLOUT + 1) * n)
        self.reward = device.alloc(ROLLOUT * n)
        self.done = device.alloc(ROLLOUT * n)
        self.advantage = device.alloc(ROLLOUT * n)
        self.target = device.alloc(ROLLOUT * n)
        self.x = device.alloc(IN * self.batch)
        self.batch_acts = Activations(device, self.batch)
        self.d3 = device.alloc(OUT * self.batch)
        self.extra = device.alloc(3 * self.batch)
        self.d2 = device.alloc(HA * self.batch)
        self.d1 = device.alloc(HA * self.batch)
        self.grad = device.alloc(P)
        self.first = device.alloc(P)
        self.second = device.alloc(P)

    def rollout(
        mut self, device: Device, mut sim: Sim, mut policy: Policy, iteration: Int
    ) raises -> Float32:
        """ROLLOUT steps of every car; returns the mean reward per step."""
        var n = self.n
        sim.sense(device, mat[IN](self.obs, n))
        for t in range(ROLLOUT):
            policy.forward(device, mat[IN](self.obs, n, t * IN * n), self.acts)
            sim.sample(
                device,
                mat[OUT](self.acts.o, n),
                policy.log_std(),
                mat[2](self.actions, n, t * 2 * n),
                mat[1](self.logp, n, t * n),
                mat[1](self.value, n, t * n),
                False,
                iteration * ROLLOUT + t,
            )
            sim.physics(
                device,
                mat[2](self.actions, n, t * 2 * n),
                mat[1](self.reward, n, t * n),
                mat[1](self.done, n, t * n),
            )
            sim.sense(device, mat[IN](self.obs, n, (t + 1) * IN * n))
        policy.forward(device, mat[IN](self.obs, n, ROLLOUT * IN * n), self.acts)
        var o = mat[OUT](self.acts.o, n)
        var value = mat[1](self.value, n, ROLLOUT * n)

        def bootstrap[width: Int, alignment: Int = 1](c: Coord) {var}:
            value[0, Int(c[0].value())] = o[2, Int(c[0].value())]

        device.run(bootstrap, Coord(n))
        self.gae(device)
        return mean(Span(device.read(self.reward)))

    def gae(mut self, device: Device) raises:
        """Generalized advantage estimates and value targets per car."""
        var n = self.n
        var reward = mat[1](self.reward, ROLLOUT * n)
        var done = mat[1](self.done, ROLLOUT * n)
        var values = mat[1](self.value, (ROLLOUT + 1) * n)
        var advantage = mat[1](self.advantage, ROLLOUT * n)
        var target = mat[1](self.target, ROLLOUT * n)

        def kernel[width: Int, alignment: Int = 1](c: Coord) {var}:
            var i = Int(c[0].value())
            var next = values[0, ROLLOUT * n + i]
            var running: Float32 = 0
            for t in range(ROLLOUT - 1, -1, -1):
                var k = t * n + i
                var alive: Float32 = 1 if done[0, k] == 0 else 0
                var delta = reward[0, k] + GAMMA * alive * next - values[0, k]
                running = delta + GAMMA * LAMBDA * alive * running
                advantage[0, k] = running
                target[0, k] = values[0, k] + running
                next = values[0, k]

        device.run(kernel, Coord(n))

    def update(
        mut self, device: Device, mut policy: Policy, iteration: Int
    ) raises -> Float32:
        """EPOCHS passes of clipped PPO over shuffled minibatches; returns KL."""
        var adv = device.read(self.advantage)
        var adv_mean = mean(Span(adv))
        var adv_scale = 1 / sqrt(max(variance(Span(adv), adv_mean, 0), 1e-8))
        var kl: Float32 = 0
        for epoch in range(EPOCHS):
            for m in range(MINIBATCHES):
                var dls, divergence = self.minibatch(
                    device, policy, m * self.batch, 1000003 * iteration + 7 * epoch, adv_mean, adv_scale
                )
                kl = divergence
                var norm = sqrt(squares(device.read(self.grad)) + (dls * dls).reduce_add())
                self.step += 1
                self.adam(device, policy, dls, min(Float32(1), MAX_NORM / (norm + 1e-8)))
        return kl

    def minibatch(
        mut self,
        device: Device,
        mut policy: Policy,
        start: Int,
        shuffle: Int,
        adv_mean: Float32,
        adv_scale: Float32,
    ) raises -> Tuple[SIMD[DType.float32, 2], Float32]:
        """Gradient of the clipped PPO loss over samples `start` onward of the
        shuffle, left in `grad`; returns the log std gradient and the KL."""
        var n = self.n
        var batch = self.batch
        var obs = mat[1](self.obs, len(self.obs))
        var actions = mat[1](self.actions, len(self.actions))
        var logp = mat[1](self.logp, ROLLOUT * n)
        var values = mat[1](self.value, (ROLLOUT + 1) * n)
        var advantage = mat[1](self.advantage, ROLLOUT * n)
        var target = mat[1](self.target, ROLLOUT * n)
        var x = mat[IN](self.x, batch)
        var o = mat[OUT](self.batch_acts.o, batch)
        var d3 = mat[OUT](self.d3, batch)
        var extra = mat[3](self.extra, batch)
        var log_std = policy.log_std()

        def gather[width: Int, alignment: Int = 1](c: Coord) {var}:
            var b = Int(c[0].value())
            var s = permutation(start + b, n, shuffle)
            for r in range(IN):
                x[r, b] = obs[0, (s // n * IN + r) * n + s % n]

        def loss[width: Int, alignment: Int = 1](c: Coord) {var}:
            """Clipped surrogate and value loss derivatives for column b."""
            var b = Int(c[0].value())
            var s = permutation(start + b, n, shuffle)
            var t = s // n
            var i = s % n
            var new_logp: Float32 = 0
            var error = SIMD[DType.float32, 2](0)
            var invvar = SIMD[DType.float32, 2](0)
            for a in range(2):
                error[a] = actions[0, (t * 2 + a) * n + i] - o[a, b]
                invvar[a] = exp(-2 * log_std[0, a])
                new_logp -= (
                    0.5 * error[a] * error[a] * invvar[a] + log_std[0, a] + 0.918938533
                )
            var logratio = new_logp - logp[0, s]
            var ratio = exp(logratio)
            var a_hat = (advantage[0, s] - adv_mean) * adv_scale
            var dlogp = -a_hat * ratio / Float32(batch)
            if (a_hat >= 0 and ratio > 1 + CLIP) or (a_hat < 0 and ratio < 1 - CLIP):
                dlogp = 0
            for a in range(2):
                d3[a, b] = dlogp * error[a] * invvar[a]
                extra[a, b] = dlogp * (error[a] * error[a] * invvar[a] - 1)
            var value = o[2, b]
            var old = values[0, s]
            var clipped = old + min(max(value - old, -CLIP), CLIP)
            var delta = value - target[0, s]
            var clipped_delta = clipped - target[0, s]
            var derivative = delta
            if clipped_delta * clipped_delta > delta * delta:
                derivative = clipped_delta if abs(value - old) <= CLIP else 0
            d3[2, b] = 0.5 * derivative / Float32(batch)
            extra[2, b] = ratio - 1 - logratio

        device.run(gather, Coord(batch))
        policy.forward(device, x, self.batch_acts)
        device.run(loss, Coord(batch))
        policy.backward(device, x, self.batch_acts, d3, self.d2, self.d1, self.grad)
        var tails = device.read(self.extra)
        return (
            SIMD[DType.float32, 2](sum(Span(tails)[0:batch]), sum(Span(tails)[batch : 2 * batch])),
            mean(Span(tails)[2 * batch : 3 * batch]),
        )

    def adam(
        mut self, device: Device, mut policy: Policy, dls: SIMD[DType.float32, 2], scale: Float32
    ) raises:
        var step = self.step
        var theta = mat[1](policy.theta, P)
        var grad = mat[1](self.grad, P)
        var first = mat[1](self.first, P)
        var second = mat[1](self.second, P)

        def kernel[width: Int, alignment: Int = 1](c: Coord) {var}:
            var i = Int(c[0].value())
            var g = (grad[0, i] if i < LOGSTD else dls[i - LOGSTD]) * scale
            var m = 0.9 * first[0, i] + 0.1 * g
            var v = 0.999 * second[0, i] + 0.001 * g * g
            first[0, i] = m
            second[0, i] = v
            var m_hat = m / (1 - Float32(0.9) ** Float32(step))
            var v_hat = v / (1 - Float32(0.999) ** Float32(step))
            var value = theta[0, i] - LR * m_hat / (sqrt(v_hat) + 1e-5)
            theta[0, i] = min(max(value, -2), 0) if i >= LOGSTD else value

        device.run(kernel, Coord(P))
        policy.transpose(device)

    def save(
        self, device: Device, mut policy: Policy, path: String, iteration: Int
    ) raises:
        var values = List[Float32]()
        for value in [CHECKPOINT_MAGIC, CHECKPOINT_VERSION, P, iteration, self.step]:
            values.append(Float32(value))
        values.extend(device.read(policy.theta))
        values.extend(device.read(self.first))
        values.extend(device.read(self.second))
        write_floats(path, values)

    def load(mut self, device: Device, values: List[Float32]) raises:
        """Restore the Adam state saved next to the weights (see `checkpoint`)."""
        self.step = Int(values[4])
        device.write(self.first, slice(values, 5 + P, 5 + 2 * P))
        device.write(self.second, slice(values, 5 + 2 * P, 5 + 3 * P))


def checkpoint(device: Device, mut policy: Policy, path: String) raises -> List[Float32]:
    """Load a checkpoint into the policy; returns it so a Trainer can resume."""
    var values = read_floats(path)
    if (
        len(values) != 5 + 3 * P
        or values[0] != CHECKPOINT_MAGIC
        or values[1] != CHECKPOINT_VERSION
    ):
        raise Error("checkpoint format or network size does not match")
    device.write(policy.theta, slice(values, 5, 5 + P))
    policy.transpose(device)
    return values^


def squares(values: List[Float32]) -> Float32:
    """Sum of squares of the weight gradient (the log std part is separate)."""
    var total: Float32 = 0
    for i in range(LOGSTD):
        total += values[i] * values[i]
    return total


def slice(values: List[Float32], start: Int, stop: Int) -> List[Float32]:
    var result = List[Float32](capacity=stop - start)
    result.extend(Span(values)[start:stop])
    return result^
