"""Arena layout shared by every kernel: one float32 buffer, offsets by env count.

Live region (also what the viewer snapshots):
  state[N,14], obs[N,IN], actions[N,2], result[N,2], stats[N,3]
State: x,y,heading,u,v,yaw,steer,progress,steps,episode,grip,motor,return,route segment.
Observations are padded from OBS to IN with a constant 1 (the bias input)
and zeros, so biases are ordinary weight rows and every matmul width is a
multiple of 16.
"""
from std.sys import get_defined_int
from .device import uniform

comptime BEAMS = get_defined_int["BEAMS", 64]()
comptime ROWS = 3
comptime RAYS = ROWS * BEAMS
comptime STATE = 14
comptime OBS = 8 + RAYS
comptime IN = (OBS + 1 + 15) // 16 * 16
comptime H = 64
comptime HA = H + 16
comptime OUT = 3
comptime W0 = 0
comptime W1 = W0 + IN * H
comptime W2 = W1 + HA * H
comptime LOGSTD = W2 + HA * OUT
comptime WEIGHTS = LOGSTD + 2
# Gradient partial sums also carry the per-sample KL estimate.
comptime TOTAL = WEIGHTS + 1
comptime ROLLOUT = 32
comptime EPOCHS = 4
comptime MINIBATCHES = 4
comptime PART = 256
# Rows of one split-K chunk when accumulating weight gradients.
comptime CHUNK = 128
comptime MOMENT_PARTS = (WEIGHTS + PART - 1) // PART


def obs_offset(n: Int) -> Int:
    return n * STATE


def action_offset(n: Int) -> Int:
    return n * (STATE + IN)


def reward_offset(n: Int) -> Int:
    return n * (STATE + IN + 2)


def stats_offset(n: Int) -> Int:
    return n * (STATE + IN + 4)


def env_size(n: Int) -> Int:
    return n * (STATE + IN + 7)


@fieldwise_init
struct Memory(TrivialRegisterPassable):
    var weights: Int
    var gradient: Int
    var first: Int
    var second: Int
    var w1t: Int
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
    var extra: Int
    var d3: Int
    var d2: Int
    var d1: Int
    var partial: Int
    var moments: Int
    var stats: Int
    var size: Int


def memory(n: Int) -> Memory:
    var samples = n * ROLLOUT
    var batch = samples // MINIBATCHES
    var chunks = (batch + CHUNK - 1) // CHUNK
    var weights = env_size(n)
    var gradient = weights + WEIGHTS
    var first = gradient + WEIGHTS
    var second = first + WEIGHTS
    var w1t = second + WEIGHTS
    var obs = w1t + HA * H
    var actions = obs + samples * IN
    var values = actions + samples * 2
    var logp = values + samples + n
    var rewards = logp + samples
    var done = rewards + samples
    var advantage = done + samples
    var target = advantage + samples
    var x = target + samples
    var h1 = x + batch * IN
    var h2 = h1 + batch * HA
    var output = h2 + batch * HA
    var extra = output + batch * OUT
    var d3 = extra + batch * OUT
    var d2 = d3 + batch * OUT
    var d1 = d2 + batch * H
    var partial = d1 + batch * H
    var moments = partial + chunks * TOTAL
    var stats = moments + MOMENT_PARTS
    return Memory(
        weights,
        gradient,
        first,
        second,
        w1t,
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
        extra,
        d3,
        d2,
        d1,
        partial,
        moments,
        stats,
        stats + 8,
    )


def permutation(index: Int, n: Int, seed: UInt32) -> Int:
    """Minibatch shuffle: two modular shears form a bijection for every N."""
    var row = index // n
    var col = index % n
    row = (row + Int(uniform(UInt32(col) + seed) * ROLLOUT)) % ROLLOUT
    col = (col + Int(uniform(UInt32(row) + seed + 1) * Float32(n))) % n
    return row * n + col
