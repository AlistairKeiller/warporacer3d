"""Terrain-following dynamic bicycle, in metres, seconds and radians.

No contacts or wheel bodies. Surface tilt sets the local frame and gravity.
An implicit two-variable tire update remains stable at zero/reverse speed.
"""
from std.math import cos, sin, sqrt

comptime SUBSTEPS = 4  # per 60 Hz environment step
comptime DT = Float32(1.0 / 240.0)
comptime MASS = Float32(3.0)
comptime AXLE = Float32(0.1651)
comptime INERTIA = Float32(0.11)


@inline(.always)
def clamp(x: Float32, lo: Float32, hi: Float32) -> Float32:
    return min(max(x, lo), hi)


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
    var u: Float32
    var v: Float32
    var yaw: Float32
    var steer: Float32


def integrate(
    mut car: Car,
    f: Frame,
    steering: Float32,
    throttle: Float32,
    grip: Float32,
    motor: Float32,
):
    car.steer = clamp(
        car.steer + clamp(steering, -1, 1) * 3.2 * DT, -0.4189, 0.4189
    )
    var c = cos(car.steer)
    var s = sin(car.steer)
    var coefficient = 55 / max(abs(car.u), 0.5)
    var load = 0.5 * MASS * 9.81 * f.nz * grip
    var drive = clamp(throttle, -1, 1) * 0.55 / 0.065 * motor
    if drive * car.u > 0:
        drive *= max(Float32(0), 1 - abs(car.u) / 5)
    drive = clamp(drive, -load, load)
    var rear_limit = sqrt(max(Float32(0), load * load - drive * drive))
    # Solve the linear lateral/yaw equations implicitly (a symmetric bicycle).
    var h = DT * coefficient
    var a11 = 1 + h * (c * c + 1) / MASS
    var a12 = h * AXLE * (c * c - 1) / MASS + DT * car.u
    var a21 = h * AXLE * (c * c - 1) / INERTIA
    var a22 = 1 + h * AXLE * AXLE * (c * c + 1) / INERTIA
    var b1 = car.v + h * car.u * s * c / MASS - DT * 9.81 * f.lz
    var b2 = car.yaw + h * AXLE * car.u * s * c / INERTIA
    var determinant = a11 * a22 - a12 * a21
    var v = (b1 * a22 - b2 * a12) / determinant
    var yaw = (b2 * a11 - b1 * a21) / determinant
    var front = clamp(
        coefficient * (car.u * s - (v + AXLE * yaw) * c), -load, load
    )
    var rear = clamp(-coefficient * (v - AXLE * yaw), -rear_limit, rear_limit)
    var u = car.u + DT * (
        (drive - front * s) / MASS + yaw * v - 9.81 * f.fz - 0.75 * car.u / MASS
    )
    car.v += DT * ((front * c + rear) / MASS - yaw * car.u - 9.81 * f.lz)
    car.yaw += DT * AXLE * (front * c - rear) / INERTIA
    car.u = u
    car.x += DT * (f.fx * car.u + f.lx * car.v)
    car.y += DT * (f.fy * car.u + f.ly * car.v)
    car.heading += DT * car.yaw * f.nz / max(Float32(0.1), 1 - f.fz * f.fz)
    if car.heading > 3.141592654:
        car.heading -= 6.283185307
    if car.heading < -3.141592654:
        car.heading += 6.283185307
