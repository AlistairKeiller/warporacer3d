"""Vectorized racing environments on a track from racer/track.py.

A kinematic bicycle with a friction circle follows the terrain: the slope under
the car tilts gravity along and across its heading and sets the tyre load. The
planar lidar sphere-marches the clearance grid and stops at walls or at the floor.
State is one row per quantity and one column per car (feature-major), the
layout the network uses, so the viewer snapshots rows directly.
"""
from std.math import cos, sin, sqrt, floor, clamp, exp, log, pi
from std.random.philox import Random, NormalRandom
from std.sys import get_defined_int
from .device import Device, Buffer, Mat, mat

comptime Vec = SIMD[DType.float32, 4]
# Car (F1TENTH).
comptime WHEELBASE = Float32(0.3302)
comptime HALF_DIAG = Float32(0.3)  # chassis radius for wall contact
comptime GRIP = Float32(1.05)  # tyre friction coefficient
comptime G = Float32(9.81)
comptime STEER_MAX = Float32(0.4189)  # rad
comptime STEER_RATE = Float32(3.2)  # rad/s
comptime ACCEL_MAX = Float32(9.51)  # m/s^2
comptime SPEED_MAX = Float32(5)  # m/s
comptime SUBSTEPS = 4  # per 60 Hz environment step
comptime DT = Float32(1.0 / 240)
comptime HORIZON = 3000  # steps per episode
# Lidar (Hokuyo UST-10LX: 270 degrees, 10 m, 1081 beams; every tenth by default).
comptime BEAMS = get_defined_int["BEAMS", 108]()
comptime RANGE = Float32(10)
comptime FOV = Float32(4.71238898)
comptime MOUNT = Vec(0.27, 0, 0.23, 0)  # forward and up from the chassis centre
# State rows.
comptime X = 0
comptime Y = 1
comptime Z = 2
comptime HEADING = 3
comptime SPEED = 4
comptime STEER = 5
comptime STEPS = 6
comptime PROGRESS = 7  # along the route, in waypoints
comptime EPISODE = 8
comptime FRICTION = 9  # per-episode grip scale
comptime RETURN = 10
comptime STATE = 11
# Observation rows: steer, speed, slope along, slope across, beams, then a constant 1 (the bias input).
comptime OBS = 4 + BEAMS
comptime IN = OBS + 1
comptime OUT = 3  # steering mean, throttle mean, value
comptime CRASH = 1
comptime TIMEOUT = 2
comptime HALF_LOG_2PI = Float32(0.5 * log(2 * pi))
# Map header and grids (see racer/track.py).
comptime HEADER = 12
comptime CLEARANCE = 0
comptime HEIGHT = 1
comptime NEAREST = 2


@fieldwise_init
struct Surface(TrivialRegisterPassable):
    var height: Float32
    var sx: Float32  # d height / dx
    var sy: Float32

    def slopes(self, heading: Float32) -> Tuple[Float32, Float32]:
        """Rise per metre ahead and to the left."""
        var c = cos(heading)
        var s = sin(heading)
        return (self.sx * c + self.sy * s, -self.sx * s + self.sy * c)


@fieldwise_init
struct Spot(TrivialRegisterPassable):
    """What is under a point: its layer, and that layer's clearance and height."""

    var layer: Int
    var clearance: Float32
    var height: Float32


struct Map(TrivialRegisterPassable):
    """Header fields; the map floats themselves (`d`) are passed alongside
    because kernel-visible structs cannot hold views."""

    var nx: Int
    var ny: Int
    var x0: Float32
    var y0: Float32
    var cell: Float32
    var waypoints: Int
    var spacing: Float32
    var layers: Int
    var slope: Float32  # bound on the terrain gradient, for the lidar march
    var route: Int  # offset of the waypoint table

    def __init__(out self, d: Span[Float32, _]) raises:
        if len(d) < HEADER:
            raise Error("not a track array")
        self.nx, self.ny = Int(d[0]), Int(d[1])
        self.x0, self.y0, self.cell = d[2], d[3], d[4]
        self.waypoints, self.spacing, self.layers, self.slope = Int(d[5]), d[6], Int(d[7]), d[8]
        self.route = HEADER + 3 * self.layers * self.nx * self.ny

    @inline(.always)
    def grid(self, layer: Int, which: Int) -> Int:
        return HEADER + (3 * layer + which) * self.nx * self.ny

    def at(self, d: Mat[1], layer: Int, which: Int, x: Float32, y: Float32) -> Float32:
        """Nearest cell of a grid; off the map every grid reads -1."""
        var ix = Int(floor((x - self.x0) / self.cell + 0.5))
        var iy = Int(floor((y - self.y0) / self.cell + 0.5))
        if ix < 0 or iy < 0 or ix >= self.nx or iy >= self.ny:
            return -1
        return d[0, self.grid(layer, which) + iy * self.nx + ix]

    def probe(self, d: Mat[1], x: Float32, y: Float32, z: Float32) -> Spot:
        """The drivable layer whose road is nearest to height z, or layer 0 off the road."""
        var best = Spot(0, -1, self.at(d, 0, HEIGHT, x, y))
        var gap = Float32(1e30)
        for l in range(self.layers):
            var spot = Spot(l, self.at(d, l, CLEARANCE, x, y), self.at(d, l, HEIGHT, x, y))
            if spot.clearance > 0 and abs(spot.height - z) < gap:
                best, gap = spot, abs(spot.height - z)
        return best

    def surface(self, d: Mat[1], layer: Int, x: Float32, y: Float32) -> Surface:
        """Bilinear height and its gradient."""
        var fx = (x - self.x0) / self.cell
        var fy = (y - self.y0) / self.cell
        var ix = clamp(Int(floor(fx)), 0, self.nx - 2)
        var iy = clamp(Int(floor(fy)), 0, self.ny - 2)
        var tx = clamp(fx - Float32(ix), 0, 1)
        var ty = clamp(fy - Float32(iy), 0, 1)
        var k = self.grid(layer, HEIGHT) + iy * self.nx + ix
        var h = Vec(d[0, k], d[0, k + 1], d[0, k + self.nx], d[0, k + self.nx + 1])
        return Surface(
            (h * Vec((1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty)).reduce_add(),
            ((h[1] - h[0]) * (1 - ty) + (h[3] - h[2]) * ty) / self.cell,
            ((h[2] - h[0]) * (1 - tx) + (h[3] - h[1]) * tx) / self.cell,
        )

    @inline(.always)
    def waypoint(self, d: Mat[1], w: Int, field: Int) -> Float32:
        return d[0, self.route + 4 * w + field]

    def progress(self, d: Mat[1], layer: Int, x: Float32, y: Float32) -> Float32:
        """Position along the route in waypoints: the nearest waypoint plus the
        projection onto its tangent, so progress is continuous."""
        var w = Int(self.at(d, layer, NEAREST, x, y))
        var heading = self.waypoint(d, w, 3)
        var along = (x - self.waypoint(d, w, 0)) * cos(heading) + (y - self.waypoint(d, w, 1)) * sin(heading)
        return Float32(w) + along / self.spacing


def derivative(p: Vec, steer: Float32, accel: Float32, grip: Float32, along: Float32, across: Float32) -> Vec:
    """Kinematic bicycle on a slope: p = (x, y, heading, speed); `along` and `across`
    are the road's rise per metre ahead and to the left. Tyre forces are capped by
    a friction circle, with gravity taking its share of the lateral budget."""
    var nz = 1 / sqrt(1 + along * along + across * across)
    var load = grip * G * nz
    var v = max(abs(p[3]), 0.5) * Float32(1 if p[3] >= 0 else -1)
    var slide = G * across * nz  # gravity pulling the car across the road
    var lateral = clamp(v * p[3] * sin(steer) / cos(steer) / WHEELBASE + slide, -load, load)
    var yaw = (lateral - slide) / v
    var longitudinal = sqrt(max(load * load - lateral * lateral, 0))
    var dv = clamp(accel, -longitudinal, longitudinal) - G * along * nz
    return Vec(p[3] * cos(p[2]), p[3] * sin(p[2]), yaw, dv)


def spawn(s: Mat[STATE], map: Map, d: Mat[1], i: Int, seed: Int):
    var episode = s[EPISODE, i] + 1
    var rng = Random(seed=UInt64(seed), subsequence=UInt64(i), offset=UInt64(episode))
    var r = rng.step_uniform()
    var w = min(Int(r[0] * Float32(map.waypoints)), map.waypoints - 1)
    for row in range(STATE):
        s[row, i] = 0
    s[X, i], s[Y, i], s[Z, i] = map.waypoint(d, w, 0), map.waypoint(d, w, 1), map.waypoint(d, w, 2)
    s[HEADING, i] = map.waypoint(d, w, 3)
    s[PROGRESS, i] = Float32(w)
    s[EPISODE, i] = episode
    s[FRICTION, i] = 0.85 + 0.3 * r[1]


def advance(
    s: Mat[STATE], act: Mat[2], reward: Mat[1], done: Mat[1], obs: Mat[IN], map: Map, d: Mat[1], i: Int, seed: Int
):
    """Integrate car i for one 60 Hz step, score it, respawn it if its episode
    ended, and write its proprioceptive observations."""
    var p = Vec(s[X, i], s[Y, i], s[HEADING, i], s[SPEED, i])
    var spot = map.probe(d, p[0], p[1], s[Z, i])
    var ground = map.surface(d, spot.layer, p[0], p[1])
    var steer = s[STEER, i]
    var rate = clamp(act[0, i], -1, 1) * STEER_RATE
    var accel = clamp(act[1, i], -1, 1) * ACCEL_MAX
    var grip = GRIP * s[FRICTION, i]
    var crashed = False
    for _ in range(SUBSTEPS):
        var along, across = ground.slopes(p[2])
        var k1 = derivative(p, steer, accel, grip, along, across)
        var k2 = derivative(p + k1 * (DT / 2), steer + rate * DT / 2, accel, grip, along, across)
        var k3 = derivative(p + k2 * (DT / 2), steer + rate * DT / 2, accel, grip, along, across)
        var k4 = derivative(p + k3 * DT, steer + rate * DT, accel, grip, along, across)
        p += (k1 + 2 * (k2 + k3) + k4) * (DT / 6)
        p[3] = clamp(p[3], -SPEED_MAX, SPEED_MAX)
        steer = clamp(steer + rate * DT, -STEER_MAX, STEER_MAX)
        spot = map.probe(d, p[0], p[1], spot.height)
        ground = map.surface(d, spot.layer, p[0], p[1])
        # Walls, and steps too steep to drive (ramp sides, deck edges), end the episode.
        crashed = crashed or spot.clearance < HALF_DIAG or ground.sx * ground.sx + ground.sy * ground.sy > 1
    if p[2] > Float32(pi):
        p[2] -= Float32(2 * pi)
    if p[2] < -Float32(pi):
        p[2] += Float32(2 * pi)
    # Progress along the route, wrapped the short way around the loop.
    var progress = map.progress(d, spot.layer, p[0], p[1])
    var dp = progress - s[PROGRESS, i]
    var half = Float32(map.waypoints) / 2
    dp = dp - 2 * half if dp > half else (dp + 2 * half if dp < -half else dp)
    var score = -2 if crashed else 10 * dp * map.spacing - 0.001 * steer * steer
    var steps = s[STEPS, i] + 1
    var reason = Float32(CRASH if crashed else (TIMEOUT if steps >= HORIZON else 0))
    s[X, i], s[Y, i], s[Z, i], s[HEADING, i] = p[0], p[1], spot.height, p[2]
    s[SPEED, i], s[STEER, i], s[STEPS, i], s[PROGRESS, i] = p[3], steer, steps, progress
    s[RETURN, i] += score
    reward[0, i] = score
    done[0, i] = reason
    if reason > 0:
        spawn(s, map, d, i, seed)
        ground = map.surface(d, map.probe(d, s[X, i], s[Y, i], s[Z, i]).layer, s[X, i], s[Y, i])
    var along, across = ground.slopes(s[HEADING, i])
    obs[0, i] = s[STEER, i] / STEER_MAX
    obs[1, i] = s[SPEED, i] / SPEED_MAX
    obs[2, i] = along
    obs[3, i] = across
    obs[OBS, i] = 1


def scan(map: Map, d: Mat[1], s: Mat[STATE], obs: Mat[IN], i: Int, beam: Int) -> Float32:
    """Range of one beam of car i, swept right to left through the body plane
    (tilted by the slopes `advance` observed). Sphere tracing: each step is bounded
    by the clearance under the ray and by the height left above the floor, so
    walls, ramps and overpasses all register."""
    var heading = s[HEADING, i]
    var forward = Vec(cos(heading), sin(heading), obs[2, i], 0)
    var left = Vec(-sin(heading), cos(heading), obs[3, i], 0)
    var azimuth = -FOV / 2 + Float32(beam) * (FOV / Float32(BEAMS - 1))
    var direction = cos(azimuth) * forward + sin(azimuth) * left
    direction /= sqrt((direction * direction).reduce_add())
    var origin = Vec(s[X, i], s[Y, i], s[Z, i], 0) + MOUNT[0] * forward + Vec(0, 0, MOUNT[2], 0)
    var horizontal = sqrt(direction[0] * direction[0] + direction[1] * direction[1])
    var descent = 1.5 * map.slope * horizontal + max(-direction[2], 0)  # how fast the floor can close in
    var t: Float32 = 0
    var last: Float32 = 0  # the previous sample, for steps the slope bound misses
    var last_floor: Float32 = 0
    while t < RANGE:
        var p = origin + t * direction
        var spot = map.probe(d, p[0], p[1], p[2])
        var floor = p[2] - spot.height - 0.03
        if floor < 0:  # under a step: back off to where the ray met it
            return min(last + (t - last) * last_floor / (last_floor - floor), RANGE)
        var step = min(spot.clearance - map.cell, floor / descent)
        if step < 0.5 * map.cell:
            break
        last, last_floor = t, floor
        t += step
    return min(t, RANGE)


struct Sim:
    var n: Int
    var seed: Int
    var map: Map
    var data: Buffer  # the track array
    var state: Buffer  # [STATE, n]
    var obs: Buffer  # [IN, n]
    var actions: Buffer  # [2, n]
    var reward: Buffer  # [n]
    var done: Buffer  # [n]

    def __init__(out self, device: Device, track: Span[Float32, _], n: Int, seed: Int) raises:
        self.n = n
        self.seed = seed
        self.map = Map(track)
        self.data = device.upload(track)
        self.state = device.alloc(STATE * n)
        self.obs = device.alloc(IN * n)
        self.actions = device.alloc(2 * n)
        self.reward = device.alloc(n)
        self.done = device.alloc(n)

    def reset(self, device: Device) raises:
        """Spawn every car, then a step at rest fills the observations."""
        var s = mat[STATE](self.state, self.n)
        var actions = mat[2](self.actions, self.n)
        var d = mat[1](self.data, len(self.data))
        var map = self.map
        var seed = self.seed

        def kernel(i: Int) {var}:
            spawn(s, map, d, i, seed)
            actions[0, i] = 0
            actions[1, i] = 0

        device.run(kernel, self.n)
        self.step(device, actions, mat[1](self.reward, self.n), mat[1](self.done, self.n), mat[IN](self.obs, self.n))

    def step(self, device: Device, actions: Mat[2], reward: Mat[1], done: Mat[1], obs: Mat[IN]) raises:
        """Physics, rewards and resets for every car, then one lidar beam per thread."""
        var s = mat[STATE](self.state, self.n)
        var d = mat[1](self.data, len(self.data))
        var map = self.map
        var seed = self.seed

        def physics(i: Int) {var}:
            advance(s, actions, reward, done, obs, map, d, i, seed)

        def lidar(i: Int, beam: Int) {var}:
            obs[4 + beam, i] = scan(map, d, s, obs, i, beam) / RANGE

        device.run(physics, self.n)
        device.run(lidar, self.n, BEAMS)

    def sample(
        self,
        device: Device,
        output: Mat[OUT],
        log_std: Mat[1],
        actions: Mat[2],
        logp: Mat[1],
        value: Mat[1],
        deterministic: Bool,
        step: Int,
    ) raises:
        """Gaussian actions from the policy output, their log probabilities, and the critic's values."""
        var seed = self.seed

        def kernel(i: Int) {var}:
            var rng = NormalRandom(seed=UInt64(seed) + 1, subsequence=UInt64(i), offset=UInt64(step))
            var noise = rng.step_normal_4()
            var probability: Float32 = 0
            for a in range(2):
                var z = 0 if deterministic else noise[a]
                actions[a, i] = output[a, i] + exp(log_std[0, a]) * z
                probability -= 0.5 * z * z + log_std[0, a] + HALF_LOG_2PI
            logp[0, i] = probability
            value[0, i] = output[2, i]

        device.run(kernel, self.n)
