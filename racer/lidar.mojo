"""A planar scanning lidar like the Hokuyo UST-10LX: 270 degrees, 10 m range.

Beams lie in the car's body plane. Each ray walks the map's coarse cells
(2-D DDA) and tests the triangles listed in every cell it crosses.
"""
from std.math import cos, sin, floor
from std.sys import get_defined_int
from .device import Mat
from .map import Map, cross, dot
from .vehicle import Frame

# Every tenth beam of the UST-10LX's 1081; `-D BEAMS=1081` for all of them.
comptime BEAMS = get_defined_int["BEAMS", 108]()
comptime RANGE = Float32(10)
comptime FOV = Float32(4.71238898)  # 270 degrees
comptime MOUNT_FORWARD = Float32(0.2)
comptime MOUNT_HEIGHT = Float32(0.23)
comptime Vec = SIMD[DType.float32, 4]


def mount(x: Float32, y: Float32, z: Float32, f: Frame) -> Vec:
    var forward = Vec(f.fx, f.fy, f.fz, 0)
    var up = cross(forward, Vec(f.lx, f.ly, f.lz, 0))
    return Vec(x, y, z, 0) + MOUNT_FORWARD * forward + MOUNT_HEIGHT * up


def direction(beam: Int, f: Frame) -> Vec:
    var azimuth = -FOV / 2 + Float32(beam) * (FOV / Float32(BEAMS - 1))
    return cos(azimuth) * Vec(f.fx, f.fy, f.fz, 0) + sin(azimuth) * Vec(
        f.lx, f.ly, f.lz, 0
    )


@inline(.always)
def triangle(d: Mat[1], t: Int, origin: Vec, direction: Vec) -> Float32:
    """Moller-Trumbore distance to the triangle at `t`, or RANGE when missed."""
    var a = Vec(d[0, t], d[0, t + 1], d[0, t + 2], 0)
    var e1 = Vec(d[0, t + 3], d[0, t + 4], d[0, t + 5], 0)
    var e2 = Vec(d[0, t + 6], d[0, t + 7], d[0, t + 8], 0)
    var p = cross(direction, e2)
    var determinant = dot(e1, p)
    if abs(determinant) < 1e-8:
        return RANGE
    var inverse = 1 / determinant
    var relative = origin - a
    var u = dot(relative, p) * inverse
    if u < 0 or u > 1:
        return RANGE
    var q = cross(relative, e1)
    var v = dot(direction, q) * inverse
    var hit = dot(e2, q) * inverse
    return hit if v >= 0 and u + v <= 1 and hit >= 0 else RANGE


def ray(map: Map, d: Mat[1], origin: Vec, direction: Vec) -> Float32:
    var best = RANGE
    var u = (origin[0] - map.x0 + map.cell / 2) / map.coarse
    var v = (origin[1] - map.y0 + map.cell / 2) / map.coarse
    var ix = Int(floor(u))
    var iy = Int(floor(v))
    var step_x = 1 if direction[0] > 0 else -1
    var step_y = 1 if direction[1] > 0 else -1
    var delta_x = abs(map.coarse / direction[0]) if direction[0] != 0 else 1e30
    var delta_y = abs(map.coarse / direction[1]) if direction[1] != 0 else 1e30
    var next_x = (
        (Float32(ix + (step_x + 1) // 2) - u) * map.coarse / direction[0]
    ) if direction[0] != 0 else 1e30
    var next_y = (
        (Float32(iy + (step_y + 1) // 2) - v) * map.coarse / direction[1]
    ) if direction[1] != 0 else 1e30
    var t: Float32 = 0
    while t < best and ix >= 0 and iy >= 0 and ix < map.cnx and iy < map.cny:
        var k = map.starts + iy * map.cnx + ix
        for e in range(Int(d[0, k]), Int(d[0, k + 1])):
            var face = Int(d[0, map.items + e])
            best = min(best, triangle(d, map.triangles + face * 9, origin, direction))
        if next_x < next_y:
            t = next_x
            next_x += delta_x
            ix += step_x
        else:
            t = next_y
            next_y += delta_y
            iy += step_y
    return best
