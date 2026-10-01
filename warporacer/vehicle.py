"""A jointed four-wheel car. Newton owns suspension, rolling contacts, and integration."""

from dataclasses import dataclass
from itertools import combinations

import newton
import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})


@dataclass(frozen=True)
class CarSpec:
    length: float = 0.58
    width: float = 0.38
    height: float = 0.10
    wheelbase: float = 0.3302
    track_width: float = 0.36
    mass: float = 3.0
    wheel_radius: float = 0.065
    wheel_width: float = 0.045
    wheel_mass: float = 0.12
    axle_drop: float = 0.08
    travel: float = 0.04
    spring: float = 500.0
    damping: float = 20.0
    friction: float = 1.1
    steer_max: float = 0.4189
    steer_rate: float = 3.2
    steer_stiffness: float = 10.0
    steer_damping: float = 0.1
    drive_torque: float = 0.55
    speed_limit: float = 5.0
    drag: float = 0.75

    @property
    def ride_height(self):
        return self.wheel_radius + self.axle_drop - self.mass * 9.81 / (4 * self.spring)


def build_car(spec):
    """Return a builder and named IDs; separate front steering and axle hinges avoid angle wrapping."""
    builder = newton.ModelBuilder()
    inertia = np.diag(
        [
            spec.mass * (spec.width**2 + spec.height**2) / 12,
            spec.mass * (spec.length**2 + spec.height**2) / 12,
            spec.mass * (spec.length**2 + spec.width**2) / 12,
        ]
    )
    root = builder.add_link(
        mass=spec.mass, inertia=wp.mat33(inertia), lock_inertia=True, label="chassis"
    )
    joints = [builder.add_joint_free(root)]
    builder.add_shape_box(
        root,
        hx=spec.length / 2,
        hy=spec.width / 2,
        hz=spec.height / 2,
        cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=0.5, gap=0.003),
    )
    steering, drive, suspension, wheels = [], [], [], []
    for x, y in [
        (spec.wheelbase / 2, spec.track_width / 2),
        (spec.wheelbase / 2, -spec.track_width / 2),
        (-spec.wheelbase / 2, spec.track_width / 2),
        (-spec.wheelbase / 2, -spec.track_width / 2),
    ]:
        mount = wp.transform(wp.vec3(x, y, -spec.axle_drop), wp.quat_identity())
        compressed = wp.transform(
            wp.vec3(x, y, -spec.axle_drop + spec.mass * 9.81 / (4 * spec.spring)),
            wp.quat_identity(),
        )
        carrier = builder.add_link(
            xform=compressed, mass=0.05, inertia=wp.mat33(np.eye(3) * 5e-4)
        )
        suspension.append(
            builder.add_joint_prismatic(
                root,
                carrier,
                parent_xform=mount,
                axis=newton.Axis.Z,
                limit_lower=-spec.travel,
                limit_upper=spec.travel,
                target_ke=spec.spring,
                target_kd=spec.damping,
            )
        )
        joints.append(suspension[-1])
        wheel = builder.add_link(xform=compressed, label=f"wheel_{len(wheels)}")
        density = spec.wheel_mass / (np.pi * spec.wheel_radius**2 * spec.wheel_width)
        shape = builder.add_shape_cylinder(
            wheel,
            radius=spec.wheel_radius,
            half_height=spec.wheel_width / 2,
            xform=wp.transform(q=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2)),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=density, mu=spec.friction, gap=0.003
            ),
            color=(0.12, 0.12, 0.12),
        )
        parent = carrier
        if x > 0:
            parent = builder.add_link(
                xform=compressed, mass=0.05, inertia=wp.mat33(np.eye(3) * 5e-4)
            )
            steer_frame = wp.transform(
                q=wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -np.pi / 2)
            )
            steering.append(
                builder.add_joint_revolute(
                    carrier,
                    parent,
                    axis=newton.Axis.X,
                    parent_xform=steer_frame,
                    child_xform=steer_frame,
                    limit_lower=-spec.steer_max,
                    limit_upper=spec.steer_max,
                    target_ke=spec.steer_stiffness,
                    target_kd=spec.steer_damping,
                )
            )
            joints.append(steering[-1])
        roll_frame = wp.transform(
            q=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2)
        )
        joint = builder.add_joint_revolute(
            parent,
            wheel,
            axis=newton.Axis.X,
            parent_xform=roll_frame,
            child_xform=roll_frame,
        )
        drive.append(joint)
        wheels.append((wheel, shape))
        joints.append(joint)
    builder.add_articulation(joints)
    for a, b in combinations(range(builder.shape_count), 2):
        builder.add_shape_collision_filter_pair(a, b)
    return builder, {
        "root": root,
        "steering": steering,
        "drive": drive,
        "suspension": suspension,
        "wheels": wheels,
    }


@wp.kernel
def steer_targets(
    actions: wp.array2d[float],
    target: wp.array[float],
    dt: float,
    rate: float,
    limit: float,
):
    i = wp.tid()
    target[i] = wp.clamp(
        target[i] + wp.clamp(actions[i, 0], -1.0, 1.0) * rate * dt, -limit, limit
    )


@wp.kernel
def motor_controls(
    actions: wp.array2d[float],
    target: wp.array[float],
    motor_scale: wp.array[float],
    steering_target: wp.array2d[int],
    drive_dof: wp.array2d[int],
    wheel_bodies: wp.array2d[int],
    root: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    target_q: wp.array[float],
    force: wp.array[float],
    body_force: wp.array[wp.spatial_vector],
    torque: float,
    speed_limit: float,
    radius: float,
    drag: float,
):
    i = wp.tid()
    for k in range(2):
        target_q[steering_target[i, k]] = target[i]
    speed = wp.quat_rotate_inv(
        wp.transform_get_rotation(body_q[root[i]]), wp.spatial_top(body_qd[root[i]])
    )[0]
    body_force[root[i]] = wp.spatial_vector(
        -drag * wp.spatial_top(body_qd[root[i]]), wp.vec3()
    )
    drive = wp.clamp(actions[i, 1], -1.0, 1.0) * torque * motor_scale[i]
    if (speed > speed_limit and drive > 0.0) or (speed < -speed_limit and drive < 0.0):
        drive = 0.0
    for k in range(4):
        wheel = wheel_bodies[i, k]
        axis = wp.quat_rotate(
            wp.transform_get_rotation(body_q[wheel]), wp.vec3(0.0, 1.0, 0.0)
        )
        spin = wp.dot(
            wp.spatial_bottom(body_qd[wheel]) - wp.spatial_bottom(body_qd[root[i]]),
            axis,
        )
        # Motor back EMF limits free-spinning wheels during jumps as well as on the road.
        force[drive_dof[i, k]] = drive * wp.clamp(
            1.0 - spin * wp.sign(drive) * radius / speed_limit, 0.0, 1.0
        )
