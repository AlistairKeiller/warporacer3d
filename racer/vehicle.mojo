"""Terrain-following dynamic bicycle, in metres, seconds and radians.

No contacts or wheel bodies. Surface tilt sets the local frame and gravity.
An implicit two-variable tire update remains stable at zero/reverse speed.
"""
from std.math import cos, sin, sqrt, clamp, pi

comptime SUBSTEPS = 4  # per 60 Hz environment step
comptime DT = Float32(1.0 / 240.0)
comptime GRAVITY = Float32(9.81)
# A 1/10-scale car.
comptime MASS = Float32(3.0)  # kg
comptime INERTIA = Float32(0.11)  # kg m^2 about the vertical axis
comptime AXLE = Float32(0.1651)  # m, half the wheelbase
comptime TRACK = Float32(0.37)  # m, left wheel to right wheel
comptime STEER_RATE = Float32(3.2)  # rad/s at full steering command
comptime STEER_MAX = Float32(0.4189)  # rad, 24 degrees
comptime CORNERING = Float32(55)  # N per unit slip (lateral / forward speed)
comptime CREEP = Float32(0.5)  # m/s; the slip linearisation floors speed here
comptime TORQUE = Float32(0.55)  # N m at the rear axle at full throttle
comptime WHEEL = Float32(0.065)  # m, wheel radius
comptime TOP_SPEED = Float32(5)  # m/s; drive fades to zero here
comptime DRAG = Float32(0.75)  # N s/m, linear rolling and aerodynamic drag


@fieldwise_init
struct Frame(TrivialRegisterPassable):
    """Forward and left unit vectors on the road surface, plus the normal's z."""

    var fx: Float32
    var fy: Float32
    var fz: Float32
    var lx: Float32
    var ly: Float32
    var lz: Float32
    var nz: Float32


def frame(heading: Float32, sx: Float32, sy: Float32) -> Frame:
    """The body frame of a car heading `heading` on a surface with height
    gradient (sx, sy)."""
    var c = cos(heading)
    var s = sin(heading)
    var z = sx * c + sy * s
    var length = sqrt(1 + z * z)
    var nz = 1 / sqrt(1 + sx * sx + sy * sy)
    var fx = c / length
    var fy = s / length
    var fz = z / length
    return Frame(
        fx,
        fy,
        fz,
        nz * (-sy * fz - fy),
        nz * (fx + sx * fz),
        nz * (sy * fx - sx * fy),
        nz,
    )


@fieldwise_init
struct Car(TrivialRegisterPassable):
    var x: Float32
    var y: Float32
    var heading: Float32
    var u: Float32  # forward speed
    var v: Float32  # leftward speed
    var yaw: Float32  # rate
    var steer: Float32  # front wheel angle


def integrate(
    mut car: Car,
    f: Frame,
    steering: Float32,
    throttle: Float32,
    grip: Float32,
    motor: Float32,
):
    """One DT of rear-drive bicycle dynamics in the frame `f`; `grip` and
    `motor` scale the tire limit and the drive force per car."""
    car.steer = clamp(
        car.steer + clamp(steering, -1, 1) * STEER_RATE * DT, -STEER_MAX, STEER_MAX
    )
    var c = cos(car.steer)
    var s = sin(car.steer)
    var coefficient = CORNERING / max(abs(car.u), CREEP)
    var load = 0.5 * MASS * GRAVITY * f.nz * grip  # per axle, friction limit
    var drive = clamp(throttle, -1, 1) * TORQUE / WHEEL * motor
    if drive * car.u > 0:
        drive *= max(Float32(0), 1 - abs(car.u) / TOP_SPEED)
    drive = clamp(drive, -load, load)
    var rear_limit = sqrt(max(Float32(0), load * load - drive * drive))
    # Solve the linear lateral/yaw equations implicitly (a symmetric bicycle).
    var h = DT * coefficient
    var a11 = 1 + h * (c * c + 1) / MASS
    var a12 = h * AXLE * (c * c - 1) / MASS + DT * car.u
    var a21 = h * AXLE * (c * c - 1) / INERTIA
    var a22 = 1 + h * AXLE * AXLE * (c * c + 1) / INERTIA
    var b1 = car.v + h * car.u * s * c / MASS - DT * GRAVITY * f.lz
    var b2 = car.yaw + h * AXLE * car.u * s * c / INERTIA
    var determinant = a11 * a22 - a12 * a21
    var v = (b1 * a22 - b2 * a12) / determinant
    var yaw = (b2 * a11 - b1 * a21) / determinant
    var front = clamp(
        coefficient * (car.u * s - (v + AXLE * yaw) * c), -load, load
    )
    var rear = clamp(-coefficient * (v - AXLE * yaw), -rear_limit, rear_limit)
    var u = car.u + DT * (
        (drive - front * s) / MASS + yaw * v - GRAVITY * f.fz - DRAG * car.u / MASS
    )
    car.v += DT * ((front * c + rear) / MASS - yaw * car.u - GRAVITY * f.lz)
    car.yaw += DT * AXLE * (front * c - rear) / INERTIA
    car.u = u
    car.x += DT * (f.fx * car.u + f.lx * car.v)
    car.y += DT * (f.fy * car.u + f.ly * car.v)
    car.heading += DT * car.yaw * f.nz / max(Float32(0.1), 1 - f.fz * f.fz)
    if car.heading > Float32(pi):
        car.heading -= Float32(2 * pi)
    if car.heading < -Float32(pi):
        car.heading += Float32(2 * pi)
