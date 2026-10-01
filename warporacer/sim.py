"""Independent Newton worlds with Warp controls, race logic, and 3D lidar."""

from contextlib import contextmanager
from dataclasses import dataclass

import newton
import numpy as np
import torch
import warp as wp

from warporacer.vehicle import CarSpec, build_car, motor_controls, steer_targets

wp.set_module_options({"enable_backward": False})

ACT_DIM = 2
PROPRIO_DIM = 14  # steering, linear/angular velocity, gravity, four wheel contacts


@dataclass(frozen=True)
class SimConfig:
    dt: float = 1 / 60
    substeps: int = 16
    iterations: int = 12
    use_graph: bool = True
    max_steps: int = 10_000
    rollover_steps: int = 30
    lidar_beams: int = 108
    lidar_elevations: tuple = (-15.0, 0.0, 15.0)
    lidar_fov: float = 270.0
    lidar_range: float = 20.0
    lidar_mount: tuple = (0.20, 0.0, 0.06)

    def __post_init__(self):
        if (
            min(
                self.dt,
                self.substeps,
                self.iterations,
                self.max_steps,
                self.rollover_steps,
                self.lidar_beams,
                self.lidar_range,
            )
            <= 0
        ):
            raise ValueError(
                "Simulation steps, limits, and lidar dimensions must be positive"
            )
        if not self.lidar_elevations:
            raise ValueError("Lidar requires at least one elevation")


@wp.struct
class RouteData:
    points: wp.array[wp.vec3]
    delta: wp.array[wp.vec3]
    up: wp.array[wp.vec3]
    rotation: wp.array[wp.quat]
    widths: wp.array[float]
    distance: wp.array[float]
    lengths: wp.array[float]
    spawns: wp.array[int]
    total: float
    closed: int


@wp.struct
class Episodes:
    target: wp.array[float]
    motor: wp.array[float]
    progress: wp.array[float]
    position: wp.array[wp.vec3]
    steps: wp.array[int]
    tipped: wp.array[int]
    resets: wp.array[int]
    crashed: wp.array[int]
    grounded: wp.array2d[int]


@wp.func
def route_delta(a: float, b: float, route: RouteData):
    delta = a - b
    if route.closed != 0:
        delta -= wp.floor(delta / route.total + 0.5) * route.total
    return delta


@wp.func
def project_route(
    p: wp.vec3, previous: float, window: float, route: RouteData
) -> wp.vec4:
    """Nearest reachable segment: (arc length, lateral error, width, index)."""
    best = wp.vec4(previous, 0.0, 1.0, 0.0)
    best_distance = float(1.0e20)  # noqa: UP018 — mutable Warp loop variable
    for j in range(route.lengths.shape[0]):
        midpoint = route.distance[j] + 0.5 * route.lengths[j]
        if (
            wp.abs(route_delta(midpoint, previous, route))
            > window + 0.5 * route.lengths[j]
        ):
            continue
        d = route.delta[j]
        t = wp.clamp(wp.dot(p - route.points[j], d) / wp.dot(d, d), 0.0, 1.0)
        s = route.distance[j] + t * route.lengths[j]
        if wp.abs(route_delta(s, previous, route)) <= window:
            error = p - (route.points[j] + t * d)
            distance = wp.dot(error, error)
            if distance < best_distance:
                best_distance = distance
                left = wp.normalize(wp.cross(route.up[j], wp.normalize(d)))
                best = wp.vec4(s, wp.dot(error, left), route.widths[j], float(j))
    return best


@wp.kernel
def valid_spawns(
    route: RouteData,
    mesh: wp.uint64,
    wheelbase: float,
    width: float,
    valid: wp.array[int],
):
    j = wp.tid()
    center = route.points[j] + 0.5 * route.delta[j]
    pose = wp.transform(center, route.rotation[j])
    supported = int(1)  # noqa: UP018, RUF046 — mutable Warp loop variable
    for k in range(4):
        x = wp.where(k < 2, 0.5 * wheelbase, -0.5 * wheelbase)
        y = wp.where(k % 2 == 0, 0.5 * width, -0.5 * width)
        ray = wp.mesh_query_ray(
            mesh, wp.transform_point(pose, wp.vec3(x, y, 0.3)), -route.up[j], 0.6
        )
        if (
            not ray.result
            or wp.abs(ray.t - 0.3) > 0.01
            or wp.dot(ray.normal, route.up[j]) < 0.5
        ):
            supported = 0
    valid[j] = supported


@wp.kernel
def reset_cars(
    mask: wp.array[int],
    route: RouteData,
    episodes: Episodes,
    bodies: wp.array2d[int],
    rest: wp.array[wp.transform],
    wheels: wp.array2d[int],
    friction: wp.array[float],
    q0: wp.array[wp.transform],
    q1: wp.array[wp.transform],
    v0: wp.array[wp.spatial_vector],
    v1: wp.array[wp.spatial_vector],
    f0: wp.array[wp.spatial_vector],
    f1: wp.array[wp.spatial_vector],
    steering: wp.array2d[int],
    drive: wp.array2d[int],
    target_q: wp.array[float],
    joint_f: wp.array[float],
    seed: int,
    offset: int,
    height: float,
    mu: float,
):
    i = wp.tid()
    if mask[i] == 0:
        return
    rng = wp.rand_init(seed, offset + i + episodes.resets[i] * 1000003)
    j = route.spawns[wp.randi(rng, 0, route.spawns.shape[0])]
    pose = wp.transform(
        route.points[j] + 0.5 * route.delta[j] + route.up[j] * height, route.rotation[j]
    )
    for k in range(bodies.shape[1]):
        body = bodies[i, k]
        q0[body] = pose * rest[k]
        q1[body] = q0[body]
        v0[body] = wp.spatial_vector()
        v1[body] = wp.spatial_vector()
        f0[body] = wp.spatial_vector()
        f1[body] = wp.spatial_vector()
    for k in range(4):
        friction[wheels[i, k]] = mu * (0.85 + 0.30 * wp.randf(rng))
        episodes.grounded[i, k] = 0
        joint_f[drive[i, k]] = 0.0
    for k in range(2):
        target_q[steering[i, k]] = 0.0
    episodes.target[i] = 0.0
    episodes.motor[i] = 0.85 + 0.30 * wp.randf(rng)
    episodes.progress[i] = route.distance[j] + 0.5 * route.lengths[j]
    episodes.position[i] = wp.transform_get_translation(pose)
    episodes.steps[i] = 0
    episodes.tipped[i] = 0
    episodes.crashed[i] = 0
    episodes.resets[i] += 1


@wp.kernel
def contact_events(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape_body: wp.array[int],
    shape_world: wp.array[int],
    role: wp.array[int],
    wheel_number: wp.array[int],
    q: wp.array[wp.transform],
    root: wp.array[int],
    episodes: Episodes,
):
    j = wp.tid()
    if j >= count[0]:
        return
    a = int(shape0[j])
    b = int(shape1[j])
    if a < 0 or b < 0:
        return
    pa = wp.vec3(point0[j])
    pb = wp.vec3(point1[j])
    if shape_body[a] >= 0:
        pa = wp.transform_point(q[shape_body[a]], pa)
    if shape_body[b] >= 0:
        pb = wp.transform_point(q[shape_body[b]], pb)
    separation = wp.dot(pb - pa, normal[j]) - margin0[j] - margin1[j]
    if separation > 0.001:
        return
    # Roles: chassis=1, wheel=2, road=3, obstacle=4.
    car = int(a)
    map_shape = int(b)
    n = -normal[j]
    if role[b] <= 2:
        car = int(b)
        map_shape = int(a)
        n = normal[j]
    if role[car] > 2 or role[map_shape] < 3:
        return
    i = shape_world[car]
    if role[car] == 2:
        wp.atomic_max(episodes.grounded, i, wheel_number[car], 1)
    up = wp.quat_rotate(wp.transform_get_rotation(q[root[i]]), wp.vec3(0.0, 0.0, 1.0))
    # Newton also reports nearby, separated pairs; only penetration is a crash.
    if separation < 0.0 and (
        role[car] == 1 or role[map_shape] == 4 or wp.dot(n, up) < 0.3
    ):
        wp.atomic_max(episodes.crashed, i, 1)


@wp.kernel
def race_step(
    q: wp.array[wp.transform],
    root: wp.array[int],
    route: RouteData,
    episodes: Episodes,
    max_steps: int,
    rollover_steps: int,
    low: wp.vec3,
    high: wp.vec3,
    reward: wp.array[float],
    done: wp.array[int],
    reason: wp.array[int],
):
    i = wp.tid()
    pose = q[root[i]]
    p = wp.transform_get_translation(pose)
    travel = wp.length(p - episodes.position[i])
    hit = project_route(p, episodes.progress[i], wp.max(0.7, 2.0 * travel + 0.2), route)
    ds = wp.clamp(route_delta(hit[0], episodes.progress[i], route), -travel, travel)
    up = wp.quat_rotate(wp.transform_get_rotation(pose), wp.vec3(0.0, 0.0, 1.0))
    tipped = wp.dot(up, route.up[int(hit[3])]) < 0.1
    episodes.tipped[i] = wp.where(tipped, episodes.tipped[i] + 1, 0)
    episodes.steps[i] += 1
    lost = p[2] < low[2] - 3.0 or p[0] < low[0] - 5.0 or p[0] > high[0] + 5.0
    lost = lost or p[1] < low[1] - 5.0 or p[1] > high[1] + 5.0
    lost = (
        lost or not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2])
    )
    lost = lost or wp.abs(hit[1]) > hit[2] + 0.5
    why = int(0)  # noqa: UP018, RUF046 — mutable Warp variable
    if episodes.crashed[i] != 0:
        why = 1
    elif lost:
        why = 2
    elif episodes.tipped[i] >= rollover_steps:
        why = 3
    elif route.closed == 0 and hit[0] >= route.total - 0.15:
        why = 5
    elif episodes.steps[i] >= max_steps:
        why = 4
    reward[i] = 100.0 * ds / route.total - 0.05 * hit[1] * hit[1]
    if why > 0 and why < 4:
        reward[i] = -25.0
    elif why == 5:
        reward[i] += 10.0
    reason[i] = why
    done[i] = wp.int32(why != 0)
    episodes.progress[i] = hit[0]
    episodes.position[i] = p


@wp.kernel
def proprioception(
    q: wp.array[wp.transform],
    v: wp.array[wp.spatial_vector],
    root: wp.array[int],
    steering: wp.array3d[int],
    episodes: Episodes,
    obs: wp.array2d[float],
    poses: wp.array[wp.transform],
):
    i = wp.tid()
    pose = q[root[i]]
    rot = wp.transform_get_rotation(pose)
    linear = wp.quat_rotate_inv(rot, wp.spatial_top(v[root[i]]))
    angular = wp.quat_rotate_inv(rot, wp.spatial_bottom(v[root[i]]))
    gravity = wp.quat_rotate_inv(rot, wp.vec3(0.0, 0.0, -1.0))
    angle = float(0.0)  # noqa: UP018 — mutable Warp loop variable
    for k in range(2):
        parent = wp.transform_get_rotation(q[steering[i, k, 0]])
        child = wp.transform_get_rotation(q[steering[i, k, 1]])
        twist = wp.quat_twist_angle_signed(
            wp.vec3(0.0, 0.0, 1.0), wp.quat_inverse(parent) * child
        )
        angle += wp.atan2(wp.sin(twist), wp.cos(twist))
    obs[i, 0] = 0.5 * angle
    for k in range(3):
        obs[i, 1 + k] = linear[k]
        obs[i, 4 + k] = angular[k]
        obs[i, 7 + k] = gravity[k]
    for k in range(4):
        obs[i, 10 + k] = float(episodes.grounded[i, k])
    poses[i] = pose


@wp.kernel
def lidar(
    q: wp.array[wp.transform],
    root: wp.array[int],
    mesh: wp.uint64,
    directions: wp.array[wp.vec3],
    mount: wp.vec3,
    max_range: float,
    obs: wp.array2d[float],
):
    i, j = wp.tid()
    pose = q[root[i]]
    ray = wp.mesh_query_ray(
        mesh,
        wp.transform_point(pose, mount),
        wp.transform_vector(pose, directions[j]),
        max_range,
    )
    obs[i, PROPRIO_DIM + j] = wp.where(ray.result, ray.t, max_range)


class PhysicsBatch:
    """One map shared by homogeneous, independent car worlds."""

    def __init__(self, track, n, env, start):
        self.track, self.env, self.start, self.n = track, env, start, n
        device, car = env.device, env.car

        def array(a, dtype):
            return wp.array(a, dtype=dtype, device=device)

        def world_ids(starts):
            starts = starts.numpy()
            return np.stack([np.arange(starts[i], starts[i + 1]) for i in range(n)])

        self.route = RouteData()
        for name, values, dtype in [
            ("points", track.route.points, wp.vec3),
            ("delta", track.route.delta, wp.vec3),
            ("up", track.route.up, wp.vec3),
            ("rotation", track.route.rotations, wp.quat),
            ("widths", track.route.half_width, float),
            ("distance", track.route.distance, float),
            ("lengths", track.route.lengths, float),
        ]:
            setattr(self.route, name, array(values, dtype))
        self.route.total, self.route.closed = track.route.total, int(track.route.closed)
        builder = newton.ModelBuilder()
        self.meshes = []
        vertices, triangles = [], []
        vertex_offset = 0
        for part in track.parts:
            triangles.append(part.triangles + vertex_offset)
            vertex_offset += len(part.vertices)
            vertices.append(part.vertices)
            mesh = newton.Mesh(
                part.vertices, part.triangles.ravel(), compute_inertia=False
            )
            builder.add_shape_mesh(
                -1,
                mesh=mesh,
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0, mu=car.friction, gap=0.003
                ),
                color=(0.65, 0.2, 0.15) if part.obstacle else (0.3, 0.35, 0.4),
            )
            self.meshes.append(mesh)
        self.mesh = wp.Mesh(
            points=array(np.concatenate(vertices), wp.vec3),
            indices=array(np.concatenate(triangles).ravel(), int),
        )
        self.route.spawns = array(np.arange(len(track.route.lengths)), int)
        valid = wp.zeros(len(track.route.lengths), dtype=int, device=device)
        wp.launch(
            valid_spawns,
            dim=len(valid),
            inputs=[self.route, self.mesh.id, car.wheelbase, car.track_width, valid],
            device=device,
        )
        candidates = np.flatnonzero(valid.numpy())
        if not len(candidates):
            raise ValueError(
                f"{track.name}: no route segments support all four wheels; check route placement and triangle winding"
            )
        self.route.spawns = array(candidates, int)
        template, ids = build_car(car)
        self.rest = array(template.body_q, wp.transform)
        builder.replicate(template, n)
        self.model = builder.finalize(device=device)
        self.solver = newton.solvers.SolverXPBD(
            self.model, iterations=env.config.iterations
        )
        # Cars collide with map meshes; all internal car contacts are filtered.
        parts, shapes = len(track.parts), self.model.shape_count
        pairs = np.column_stack(
            [
                np.repeat(np.arange(parts), shapes - parts),
                np.tile(np.arange(parts, shapes), parts),
            ]
        )
        triangle_pairs = min(
            1_000_000,
            (shapes - parts) * sum(len(part.triangles) for part in track.parts),
        )
        self.pipeline = newton.CollisionPipeline(
            self.model,
            rigid_contact_max=256 * n,
            max_triangle_pairs=max(256 * n, triangle_pairs),
            broad_phase="explicit",
            shape_pairs_filtered=array(pairs, wp.vec2i),
        )
        self.contacts = self.pipeline.contacts()
        self.state, self.other = self.model.state(), self.model.state()
        self.control = self.model.control()
        # Resolve IDs by world and parentage, rather than relying on buffer strides.
        bodies = world_ids(self.model.body_world_start)
        self.bodies = array(bodies, int)
        self.root = array(bodies[:, ids["root"]], int)
        joints = world_ids(self.model.joint_world_start)
        steering = joints[:, ids["steering"]]
        self.steering_bodies = array(
            np.stack(
                [
                    self.model.joint_parent.numpy()[steering],
                    self.model.joint_child.numpy()[steering],
                ],
                axis=-1,
            ),
            int,
        )
        d_start = self.model.joint_qd_start.numpy()
        self.steering_target = array(
            self.model.joint_target_q_start.numpy()[joints[:, ids["steering"]]], int
        )
        self.drive_dof = array(d_start[joints[:, ids["drive"]]], int)
        wheel_bodies = bodies[:, [b for b, _ in ids["wheels"]]]
        self.wheel_bodies = array(wheel_bodies, int)
        shape_body = self.model.shape_body.numpy()
        # The chassis and each wheel have one collision shape.
        dynamic_shapes = np.flatnonzero(shape_body >= 0)
        body_shape = np.full(self.model.body_count, -1)
        body_shape[shape_body[dynamic_shapes]] = dynamic_shapes
        wheels = body_shape[wheel_bodies]
        self.wheels = array(wheels, int)
        roles = np.array(
            [4 if p.obstacle else 3 for p in track.parts]
            + [0] * (self.model.shape_count - len(track.parts))
        )
        wheel_number = np.full(self.model.shape_count, -1)
        roles[body_shape[bodies[:, ids["root"]]]] = 1
        roles[wheels] = 2
        wheel_number[wheels] = np.arange(4)
        self.roles, self.wheel_number = array(roles, int), array(wheel_number, int)
        self.episodes = Episodes()
        for name in ("target", "motor", "progress"):
            setattr(self.episodes, name, wp.zeros(n, dtype=float, device=device))
        for name in ("steps", "tipped", "resets", "crashed"):
            setattr(self.episodes, name, wp.zeros(n, dtype=int, device=device))
        self.episodes.position = wp.zeros(n, dtype=wp.vec3, device=device)
        self.episodes.grounded = wp.zeros((n, 4), dtype=int, device=device)
        sl = slice(start, start + n)
        self.actions, self.obs, self.reward, self.done, self.reason, self.poses = (
            a[sl]
            for a in (
                env.actions,
                env.obs_w,
                env.reward_w,
                env.done_w,
                env.reason_w,
                env.poses,
            )
        )

    def launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.env.device)

    def reset(self):
        self.launch(
            reset_cars,
            self.n,
            [
                self.done,
                self.route,
                self.episodes,
                self.bodies,
                self.rest,
                self.wheels,
                self.model.shape_material_mu,
                self.state.body_q,
                self.other.body_q,
                self.state.body_qd,
                self.other.body_qd,
                self.state.body_f,
                self.other.body_f,
                self.steering_target,
                self.drive_dof,
                self.control.joint_target_q,
                self.control.joint_f,
                self.env.seed + self.env.loads,
                self.start,
                self.env.car.ride_height,
                self.env.car.friction,
            ],
        )

    def detect_contacts(self):
        self.episodes.grounded.zero_()
        self.pipeline.collide(self.state, self.contacts)
        c, m = self.contacts, self.model
        self.launch(
            contact_events,
            c.rigid_contact_max,
            [
                c.rigid_contact_count,
                c.rigid_contact_shape0,
                c.rigid_contact_shape1,
                c.rigid_contact_point0,
                c.rigid_contact_point1,
                c.rigid_contact_normal,
                c.rigid_contact_margin0,
                c.rigid_contact_margin1,
                m.shape_body,
                m.shape_world,
                self.roles,
                self.wheel_number,
                self.state.body_q,
                self.root,
                self.episodes,
            ],
        )

    def step(self):
        cfg, car, e = self.env.config, self.env.car, self.episodes
        self.launch(
            steer_targets,
            self.n,
            [self.actions, e.target, cfg.dt, car.steer_rate, car.steer_max],
        )
        e.crashed.zero_()
        for _ in range(cfg.substeps):
            self.state.clear_forces()
            self.launch(
                motor_controls,
                self.n,
                [
                    self.actions,
                    e.target,
                    e.motor,
                    self.steering_target,
                    self.drive_dof,
                    self.wheel_bodies,
                    self.root,
                    self.state.body_q,
                    self.state.body_qd,
                    self.control.joint_target_q,
                    self.control.joint_f,
                    self.state.body_f,
                    car.drive_torque,
                    car.speed_limit,
                    car.wheel_radius,
                    car.drag,
                ],
            )
            self.detect_contacts()
            self.solver.step(
                self.state,
                self.other,
                self.control,
                self.contacts,
                cfg.dt / cfg.substeps,
            )
            self.state, self.other = self.other, self.state
        self.detect_contacts()
        low, high = self.track.bounds
        self.launch(
            race_step,
            self.n,
            [
                self.state.body_q,
                self.root,
                self.route,
                e,
                cfg.max_steps,
                cfg.rollover_steps,
                wp.vec3(low),
                wp.vec3(high),
                self.reward,
                self.done,
                self.reason,
            ],
        )
        self.reset()
        self.observe()

    def observe(self):
        cfg = self.env.config
        self.detect_contacts()
        self.launch(
            proprioception,
            self.n,
            [
                self.state.body_q,
                self.state.body_qd,
                self.root,
                self.steering_bodies,
                self.episodes,
                self.obs,
                self.poses,
            ],
        )
        self.launch(
            lidar,
            (self.n, len(self.env.directions)),
            [
                self.state.body_q,
                self.root,
                self.mesh.id,
                self.env.directions,
                wp.vec3(cfg.lidar_mount),
                cfg.lidar_range,
                self.obs,
            ],
        )


class Env:
    """step(actions) returns persistent Torch buffers; finished cars auto-reset."""

    def __init__(
        self, tracks, num_envs=256, seed=0, device=None, car=None, config=None
    ):
        wp.init()
        self.device = wp.get_device(device)
        self.torch_device = wp.device_to_torch(self.device)
        # A Warp-owned blocking stream keeps Newton's temporary buffers alive.
        self.stream = wp.Stream(self.device) if self.device.is_cuda else None
        self.torch_stream = wp.stream_to_torch(self.stream) if self.stream else None
        self.num_envs, self.seed, self.loads = num_envs, seed, -1
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.car, self.config = car or CarSpec(), config or SimConfig()
        cfg, d = self.config, self.device
        theta = np.radians(
            np.linspace(-cfg.lidar_fov / 2, cfg.lidar_fov / 2, cfg.lidar_beams)
        )
        phi, theta = np.meshgrid(np.radians(cfg.lidar_elevations), theta, indexing="ij")
        directions = np.stack(
            [np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)],
            axis=-1,
        )
        directions = directions.reshape(-1, 3)
        self.obs_dim, self.act_dim = PROPRIO_DIM + len(directions), ACT_DIM
        with self.scope():
            self.directions = wp.array(directions, dtype=wp.vec3, device=d)
            self.actions = wp.zeros((num_envs, ACT_DIM), dtype=float, device=d)
            self.obs_w = wp.zeros((num_envs, self.obs_dim), dtype=float, device=d)
            self.reward_w = wp.zeros(num_envs, dtype=float, device=d)
            self.done_w = wp.zeros(num_envs, dtype=int, device=d)
            self.reason_w = wp.zeros(num_envs, dtype=int, device=d)
            self.poses = wp.zeros(num_envs, dtype=wp.transform, device=d)
            self.act_t, self.obs, self.rew, self.done, self.reason, self.poses_t = (
                wp.to_torch(a)
                for a in (
                    self.actions,
                    self.obs_w,
                    self.reward_w,
                    self.done_w,
                    self.reason_w,
                    self.poses,
                )
            )
        self.viewer = None
        self.rotate(tracks)

    @contextmanager
    def scope(self):
        """Order environment work between operations on the caller's Torch stream."""
        if self.stream is None:
            yield
            return
        caller = torch.cuda.current_stream(self.torch_device)
        if caller == self.torch_stream and wp.get_stream(self.device) == self.stream:
            yield
            return
        self.torch_stream.wait_stream(caller)
        try:
            with (
                wp.ScopedStream(self.stream, sync_exit=True),
                torch.cuda.stream(self.torch_stream),
            ):
                yield
        finally:
            caller.wait_stream(self.torch_stream)

    def rotate(self, tracks):
        self.tracks = list(tracks) if isinstance(tracks, (list, tuple)) else [tracks]
        if not self.tracks or len(self.tracks) > self.num_envs:
            raise ValueError("Need between one and num_envs active maps")
        self.loads += 1
        with self.scope():
            if hasattr(self, "batches") and self.device.is_cuda:
                wp.synchronize_device(self.device)
            self.graphs = {}
            self.batches = []
            bounds = np.linspace(0, self.num_envs, len(self.tracks) + 1, dtype=int)
            for track, a, b in zip(self.tracks, bounds[:-1], bounds[1:]):
                self.batches.append(PhysicsBatch(track, int(b - a), self, int(a)))
            self.done_w.fill_(1)
            for batch in self.batches:
                batch.reset()
                batch.observe()
            self.done_w.zero_()
            self.reward_w.zero_()
            self.reason_w.zero_()

    def step(self, actions):
        if tuple(actions.shape) != (self.num_envs, ACT_DIM):
            raise ValueError(f"Expected actions ({self.num_envs}, {ACT_DIM})")
        with self.scope():
            if self.stream and actions.is_cuda:
                actions.record_stream(self.torch_stream)
            self.act_t.copy_(actions.detach())
            if not self.config.use_graph:
                for batch in self.batches:
                    batch.step()
            else:
                # Odd substep counts alternate state buffers and need two graphs.
                key = id(self.batches[0].state)
                graph = self.graphs.get(key)
                if graph is None:
                    with wp.ScopedCapture(device=self.device) as capture:
                        for batch in self.batches:
                            batch.step()
                    graph = self.graphs[key] = capture.graph
                elif self.config.substeps % 2:
                    for batch in self.batches:
                        batch.state, batch.other = batch.other, batch.state
                wp.capture_launch(graph)
        return self.obs, self.rew, self.done

    def observe(self):
        with self.scope():
            for batch in self.batches:
                batch.observe()
        return self.obs

    def reset(self, mask=None):
        """Reset selected cars without changing other worlds or their RNG state."""
        with self.scope():
            if mask is None:
                self.done.fill_(1)
            else:
                mask = torch.as_tensor(
                    mask, device=self.torch_device, dtype=self.done.dtype
                )
                if tuple(mask.shape) != (self.num_envs,):
                    raise ValueError(f"Expected reset mask ({self.num_envs},)")
                if self.stream:
                    mask.record_stream(self.torch_stream)
                self.done.copy_(mask)
            for batch in self.batches:
                batch.reset()
                batch.observe()
            self.done_w.zero_()
            self.reward_w.zero_()
            self.reason_w.zero_()
        return self.obs
