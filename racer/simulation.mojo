"""Vectorized environments: spawn, physics with rewards and resets, sensing,
and Gaussian action sampling from a policy output.

State is one row per quantity and one column per car (feature-major), the
same layout the network uses, so the viewer snapshots rows directly.
"""
from std.math import sqrt, exp, cos, sin, log, pi
from std.random.philox import Random, NormalRandom
from .device import Device, Buffer, Mat, mat
from .map import Map, Surface, POSITION, DELTA, COURSE
from .vehicle import Car, Frame, frame, integrate, DT, SUBSTEPS, AXLE, TRACK, STEER_MAX
from .lidar import BEAMS, RANGE, POSE, mount, pose, scan

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
# Proprioception is scaled to about -1..1 by these.
comptime SPEED_SCALE = Float32(5)  # m/s
comptime YAW_SCALE = Float32(10)  # rad/s
comptime CLEARANCE_SCALE = Float32(2)  # m
# Episode end reasons.
comptime CRASH = 1
comptime TIMEOUT = 2
comptime FINISH = 3
comptime HORIZON = 3000  # steps
# Collision: three circles along the chassis must keep this clearance.
comptime CHASSIS = Float32(0.15)  # m, circle centres ahead of and behind the middle
comptime RADIUS = Float32(0.24)  # m, clearance each circle needs from walls
comptime EDGE = Float32(0.19)  # m, lateral margin kept inside the road width
# Per-car randomisation: tire grip and motor strength vary by +-15%.
comptime GRIP_NOMINAL = Float32(1.1)
comptime VARIATION = Float32(0.3)
# Reward: route progress per step, a steering penalty, and terminal bonuses.
comptime PROGRESS_REWARD = Float32(10)  # per metre along the route
comptime STEER_PENALTY = Float32(0.001)  # per squared radian
comptime TERMINAL = Float32(2)  # subtracted for a crash, added for a finish
comptime FINISH_WITHIN = Float32(0.3)  # m of the route end
comptime SLACK = Float32(1.25)  # route progress credited per metre travelled, at most
comptime HALF_LOG_2PI = Float32(0.5 * log(2 * pi))


def attitude(map: Map, d: Mat[1], x: Float32, y: Float32, heading: Float32) -> Surface:
    """The ground as the chassis feels it: height and slope from the surface
    under the four wheels, so a faceted mesh is averaged over the wheelbase
    instead of pitching the car at every facet. Clearance is at the centre."""
    var c = cos(heading)
    var s = sin(heading)
    var front = map.height_at(d, x + AXLE * c, y + AXLE * s)
    var rear = map.height_at(d, x - AXLE * c, y - AXLE * s)
    var left = map.height_at(d, x - 0.5 * TRACK * s, y + 0.5 * TRACK * c)
    var right = map.height_at(d, x + 0.5 * TRACK * s, y - 0.5 * TRACK * c)
    var along = (front - rear) / (2 * AXLE)
    var across = (left - right) / TRACK
    return Surface(
        map.clearance_at(d, x, y),
        0.25 * (front + rear + left + right),
        along * c - across * s,
        along * s + across * c,
    )


def spawn(s: Mat[STATE], map: Map, d: Mat[1], i: Int, seed: Int):
    """Reset car i to the middle of a random spawn segment, facing along it."""
    var episode = s[EPISODE, i] + 1
    var rng = Random(
        seed=UInt64(seed), subsequence=UInt64(i), offset=UInt64(episode)
    )
    var r = rng.step_uniform()
    var segment = Int(d[0, map.spawn + Int(r[0] * Float32(map.spawns))])
    for row in range(STATE):
        s[row, i] = 0
    s[X, i] = map.segment(d, segment, POSITION) + 0.5 * map.segment(d, segment, DELTA)
    s[Y, i] = map.segment(d, segment, POSITION + 1) + 0.5 * map.segment(d, segment, DELTA + 1)
    s[HEADING, i] = map.segment(d, segment, COURSE)
    s[PROGRESS, i] = map.progress(d, s[X, i], s[Y, i], segment).distance
    s[EPISODE, i] = episode
    s[SEGMENT, i] = Float32(segment)
    s[GRIP, i] = GRIP_NOMINAL * (1 - VARIATION / 2 + VARIATION * r[1])
    s[MOTOR, i] = 1 - VARIATION / 2 + VARIATION * r[2]


def collides(map: Map, d: Mat[1], before: Car, after: Car, f0: Frame, f1: Frame, ground: Float32) -> Bool:
    """Whether the move from `before` to `after` brings any of the three
    chassis circles (inflated by half the swept distance) too close to a wall.
    `ground` is the clearance already known at the middle circle."""
    var clearance = ground
    var swept: Float32 = 0
    for end in range(-1, 2, 2):
        var ox = before.x + Float32(end) * CHASSIS * f0.fx
        var oy = before.y + Float32(end) * CHASSIS * f0.fy
        var nx = after.x + Float32(end) * CHASSIS * f1.fx
        var ny = after.y + Float32(end) * CHASSIS * f1.fy
        swept = max(swept, sqrt((nx - ox) ** 2 + (ny - oy) ** 2))
        clearance = min(clearance, map.clearance_at(d, ox, oy))
        clearance = min(clearance, map.clearance_at(d, nx, ny))
    swept = max(swept, sqrt((after.x - before.x) ** 2 + (after.y - before.y) ** 2))
    return clearance < RADIUS + 0.5 * swept + 0.001


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
        var ground = attitude(map, d, car.x, car.y, car.heading)
        var before = frame(car.heading, ground.sx, ground.sy)
        var old = car
        integrate(
            car, before, actions[0, i], actions[1, i], s[GRIP, i], s[MOTOR, i]
        )
        var next_ground = attitude(map, d, car.x, car.y, car.heading)
        var after = frame(car.heading, next_ground.sx, next_ground.sy)
        # Carry the velocity across a change of road grade.
        var same = frame(car.heading, ground.sx, ground.sy)
        var vx = same.fx * car.u + same.lx * car.v
        var vy = same.fy * car.u + same.ly * car.v
        var vz = same.fz * car.u + same.lz * car.v
        car.u = vx * after.fx + vy * after.fy + vz * after.fz
        car.v = vx * after.lx + vy * after.ly + vz * after.lz
        travelled += DT * sqrt(car.u * car.u + car.v * car.v)
        var clearance = min(next_ground.clearance, ground.clearance)
        if collides(map, d, old, car, before, after, clearance):
            reason = CRASH
            break
    var projection = map.progress(d, car.x, car.y, Int(s[SEGMENT, i]))
    s[SEGMENT, i] = projection.segment
    if projection.offset + EDGE > projection.half_width:
        reason = CRASH
    var delta = projection.distance - s[PROGRESS, i]
    if map.closed:
        delta = delta - map.total if delta > map.total / 2 else delta
        delta = delta + map.total if delta < -map.total / 2 else delta
    # A projection jump cannot manufacture progress.
    delta = min(max(delta, -SLACK * travelled - 0.001), SLACK * travelled + 0.001)
    var score = PROGRESS_REWARD * delta - STEER_PENALTY * car.steer * car.steer
    var steps = s[STEPS, i] + 1
    if reason > 0:
        score -= TERMINAL
    elif not map.closed and projection.distance > map.total - FINISH_WITHIN:
        reason = FINISH
        score += TERMINAL
    elif steps >= HORIZON:
        reason = TIMEOUT
    s[X, i], s[Y, i], s[HEADING, i] = car.x, car.y, car.heading
    s[U, i], s[V, i], s[YAW, i], s[STEER, i] = car.u, car.v, car.yaw, car.steer
    s[PROGRESS, i] = projection.distance
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
    var pose: Buffer  # [POSE, n]: lidar origin and beam basis per car

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
        self.pose = device.alloc(POSE * n)

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
        self.proprio(device, obs)
        self.lidar(device, obs)

    def proprio(mut self, device: Device, obs: Mat[IN]) raises:
        """Proprioception into `obs`, and the lidar pose of every car."""
        var s = mat[STATE](self.state, self.n)
        var d = mat[1](self.terrain, len(self.terrain))
        var p = mat[POSE](self.pose, self.n)
        var map = self.map

        def kernel(i: Int) {var}:
            var ground = attitude(map, d, s[X, i], s[Y, i], s[HEADING, i])
            var f = frame(s[HEADING, i], ground.sx, ground.sy)
            pose(p, i, mount(s[X, i], s[Y, i], ground.height, f), f)
            obs[0, i] = s[STEER, i] / STEER_MAX
            obs[1, i] = s[U, i] / SPEED_SCALE
            obs[2, i] = s[V, i] / SPEED_SCALE
            obs[3, i] = s[YAW, i] / YAW_SCALE
            obs[4, i] = -f.fz
            obs[5, i] = -f.lz
            obs[6, i] = f.nz
            obs[7, i] = min(ground.clearance, CLEARANCE_SCALE) / CLEARANCE_SCALE
            obs[OBS, i] = 1

        device.run(kernel, self.n)

    def lidar(mut self, device: Device, obs: Mat[IN]) raises:
        """One beam per thread from the poses `proprio` recorded."""
        var d = mat[1](self.terrain, len(self.terrain))
        var p = mat[POSE](self.pose, self.n)
        var map = self.map

        def kernel(i: Int, beam: Int) {var}:
            obs[8 + beam, i] = scan(map, d, p, i, beam) / RANGE

        device.run(kernel, self.n, BEAMS)

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
                probability -= 0.5 * z * z + log_std[0, a] + HALF_LOG_2PI
            logp[0, i] = probability
            value[0, i] = output[2, i]

        device.run(kernel, self.n)
