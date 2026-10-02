"""Vectorized environments: spawn, physics with rewards and resets, sensing,
and Gaussian action sampling from a policy output.

State is one row per quantity and one column per car (feature-major), the
same layout the network uses, so the viewer snapshots rows directly.
"""
from std.math import sqrt, exp
from std.random.philox import Random, NormalRandom
from .device import Device, Buffer, Mat, mat
from .map import Map
from .vehicle import Car, frame, integrate, DT, SUBSTEPS
from .lidar import BEAMS, RANGE, ray, mount, direction

# State rows.
comptime X = 0
comptime Y = 1
comptime HEADING = 2
comptime U = 3
comptime V = 4
comptime YAW = 5
comptime STEER = 6
comptime PROGRESS = 7
comptime STEPS = 8
comptime EPISODE = 9
comptime GRIP = 10
comptime MOTOR = 11
comptime RETURN = 12
comptime SEGMENT = 13
comptime STATE = 14
# Observation rows: proprioception, beams, then a constant 1 (the bias input).
comptime OBS = 8 + BEAMS
comptime IN = OBS + 1
comptime OUT = 3  # steering mean, throttle mean, value
# Episode end reasons.
comptime CRASH = 1
comptime TIMEOUT = 2
comptime FINISH = 3
comptime HORIZON = 3000


def spawn(s: Mat[STATE], map: Map, d: Mat[1], i: Int, seed: Int):
    var episode = s[EPISODE, i] + 1
    var rng = Random(
        seed=UInt64(seed), subsequence=UInt64(i), offset=UInt64(episode)
    )
    var r = rng.step_uniform()
    var segment = Int(d[0, map.spawn + Int(r[0] * Float32(map.spawns))])
    for row in range(STATE):
        s[row, i] = 0
    s[X, i] = map.segment(d, segment, 0) + 0.5 * map.segment(d, segment, 3)
    s[Y, i] = map.segment(d, segment, 1) + 0.5 * map.segment(d, segment, 4)
    s[HEADING, i] = map.segment(d, segment, 8)
    s[PROGRESS, i] = map.progress(d, s[X, i], s[Y, i], segment)[0]
    s[EPISODE, i] = episode
    s[SEGMENT, i] = Float32(segment)
    s[GRIP, i] = 1.1 * (0.85 + 0.3 * r[1])
    s[MOTOR, i] = 0.85 + 0.3 * r[2]


def advance(
    s: Mat[STATE],
    actions: Mat[2],
    reward: Mat[1],
    done: Mat[1],
    map: Map,
    d: Mat[1],
    i: Int,
    seed: Int,
):
    """Integrate car i for one 60 Hz step, score it, and respawn if it ended."""
    var car = Car(
        s[X, i], s[Y, i], s[HEADING, i], s[U, i], s[V, i], s[YAW, i], s[STEER, i]
    )
    var reason: Float32 = 0
    var travelled: Float32 = 0
    for _ in range(SUBSTEPS):
        var ground = map.surface(d, car.x, car.y)
        var before = frame(car.heading, ground.sx, ground.sy)
        var old_x = car.x
        var old_y = car.y
        integrate(
            car, before, actions[0, i], actions[1, i], s[GRIP, i], s[MOTOR, i]
        )
        var next_ground = map.surface(d, car.x, car.y)
        var after = frame(car.heading, next_ground.sx, next_ground.sy)
        # Carry the velocity across a change of road grade.
        var same = frame(car.heading, ground.sx, ground.sy)
        var vx = same.fx * car.u + same.lx * car.v
        var vy = same.fy * car.u + same.ly * car.v
        var vz = same.fz * car.u + same.lz * car.v
        car.u = vx * after.fx + vy * after.fy + vz * after.fz
        car.v = vx * after.lx + vy * after.ly + vz * after.lz
        # Three circles along the chassis, inflated by half the swept distance.
        travelled += DT * sqrt(car.u * car.u + car.v * car.v)
        var swept = sqrt((car.x - old_x) ** 2 + (car.y - old_y) ** 2)
        var clearance = min(next_ground.clearance, ground.clearance)
        for end in range(-1, 2, 2):
            var ox = old_x + Float32(end) * 0.15 * before.fx
            var oy = old_y + Float32(end) * 0.15 * before.fy
            var nx = car.x + Float32(end) * 0.15 * after.fx
            var ny = car.y + Float32(end) * 0.15 * after.fy
            swept = max(swept, sqrt((nx - ox) ** 2 + (ny - oy) ** 2))
            clearance = min(clearance, map.surface(d, ox, oy).clearance)
            clearance = min(clearance, map.surface(d, nx, ny).clearance)
        if clearance < 0.24 + 0.5 * swept + 0.001:
            reason = CRASH
            break
    var projection = map.progress(d, car.x, car.y, Int(s[SEGMENT, i]))
    s[SEGMENT, i] = projection[1]
    if projection[2] + 0.19 > projection[3]:
        reason = CRASH
    var delta = projection[0] - s[PROGRESS, i]
    if map.closed:
        delta = delta - map.total if delta > map.total / 2 else delta
        delta = delta + map.total if delta < -map.total / 2 else delta
    # A projection jump cannot manufacture progress.
    delta = min(max(delta, -1.25 * travelled - 0.001), 1.25 * travelled + 0.001)
    var score = 10 * delta - 0.001 * car.steer * car.steer
    var steps = s[STEPS, i] + 1
    if reason > 0:
        score -= 2
    elif not map.closed and projection[0] > map.total - 0.3:
        reason = FINISH
        score += 2
    elif steps >= HORIZON:
        reason = TIMEOUT
    s[X, i], s[Y, i], s[HEADING, i] = car.x, car.y, car.heading
    s[U, i], s[V, i], s[YAW, i], s[STEER, i] = car.u, car.v, car.yaw, car.steer
    s[PROGRESS, i] = projection[0]
    s[STEPS, i] = steps
    s[RETURN, i] += score
    reward[0, i] = score
    done[0, i] = reason
    if reason > 0:
        spawn(s, map, d, i, seed)


struct Sim:
    var n: Int
    var seed: Int
    var map: Map
    var terrain: Buffer
    var state: Buffer  # [STATE, n]
    var obs: Buffer  # [IN, n]
    var actions: Buffer  # [2, n]
    var reward: Buffer  # [n]
    var done: Buffer  # [n]

    def __init__(
        out self, device: Device, map_data: List[Float32], n: Int, seed: Int
    ) raises:
        self.n = n
        self.seed = seed
        self.map = Map(map_data)
        self.terrain = device.upload(map_data)
        self.state = device.alloc(STATE * n)
        self.obs = device.alloc(IN * n)
        self.actions = device.alloc(2 * n)
        self.reward = device.alloc(n)
        self.done = device.alloc(n)

    def spawn_all(mut self, device: Device) raises:
        var s = mat[STATE](self.state, self.n)
        var d = mat[1](self.terrain, len(self.terrain))
        var map = self.map
        var seed = self.seed

        def kernel(i: Int) {var}:
            spawn(s, map, d, i, seed)

        device.run(kernel, self.n)

    def physics(
        mut self, device: Device, actions: Mat[2], reward: Mat[1], done: Mat[1]
    ) raises:
        var s = mat[STATE](self.state, self.n)
        var d = mat[1](self.terrain, len(self.terrain))
        var map = self.map
        var seed = self.seed

        def kernel(i: Int) {var}:
            advance(s, actions, reward, done, map, d, i, seed)

        device.run(kernel, self.n)

    def sense(mut self, device: Device, obs: Mat[IN]) raises:
        """Proprioception and one lidar scan per car into `obs`."""
        var s = mat[STATE](self.state, self.n)
        var d = mat[1](self.terrain, len(self.terrain))
        var map = self.map

        def proprio(i: Int) {var}:
            var ground = map.surface(d, s[X, i], s[Y, i])
            var f = frame(s[HEADING, i], ground.sx, ground.sy)
            obs[0, i] = s[STEER, i] / 0.4189
            obs[1, i] = s[U, i] / 5
            obs[2, i] = s[V, i] / 5
            obs[3, i] = s[YAW, i] / 10
            obs[4, i] = -f.fz
            obs[5, i] = -f.lz
            obs[6, i] = f.nz
            obs[7, i] = min(ground.clearance, 2) / 2
            obs[OBS, i] = 1

        def lidar(i: Int, beam: Int) {var}:
            var ground = map.surface(d, s[X, i], s[Y, i])
            var f = frame(s[HEADING, i], ground.sx, ground.sy)
            var origin = mount(s[X, i], s[Y, i], ground.height, f)
            obs[8 + beam, i] = ray(map, d, origin, direction(beam, f)) / RANGE

        device.run(proprio, self.n)
        device.run(lidar, self.n, BEAMS)

    def sample(
        mut self,
        device: Device,
        output: Mat[OUT],
        log_std: Mat[1],
        actions: Mat[2],
        logp: Mat[1],
        value: Mat[1],
        deterministic: Bool,
        step: Int,
    ) raises:
        """Gaussian actions from the policy output; also records log
        probabilities and the critic's value estimates."""
        var seed = self.seed

        def kernel(i: Int) {var}:
            var rng = NormalRandom(
                seed=UInt64(seed) + 1, subsequence=UInt64(i), offset=UInt64(step)
            )
            var noise = rng.step_normal_4()
            var probability: Float32 = 0
            for a in range(2):
                var std = exp(log_std[0, a])
                var z = 0 if deterministic else noise[a]
                actions[a, i] = output[a, i] + std * z
                probability -= 0.5 * z * z + log_std[0, a] + 0.918938533
            logp[0, i] = probability
            value[0, i] = output[2, i]

        device.run(kernel, self.n)
