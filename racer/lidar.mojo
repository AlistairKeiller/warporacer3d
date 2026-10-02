"""A planar scanning lidar like the Hokuyo UST-10LX: 270 degrees, 10 m range.

Beams lie in the car's body plane. A ray first marches through the map's
2-D clearance field in steps that are provably free of walls, obstacles, and
floor (sphere tracing), then walks the coarse cells it is about to enter
(2-D DDA) and tests the triangles listed in every cell it crosses.
"""
from std.math import cos, sin, sqrt, floor
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
comptime POSE = 9  # per-car rows: origin, forward, left
comptime FLOOR_MARGIN = Float32(0.05)  # m; the height grid is within this of the mesh


def pose(p: Mat[POSE], i: Int, origin: Vec, f: Frame):
    """Record car i's beam origin and body-plane basis for `scan`."""
    for k in range(3):
        p[k, i] = origin[k]
    p[3, i], p[4, i], p[5, i] = f.fx, f.fy, f.fz
    p[6, i], p[7, i], p[8, i] = f.lx, f.ly, f.lz


def scan(map: Map, d: Mat[1], p: Mat[POSE], i: Int, beam: Int) -> Float32:
    """Range of one beam of car i from its recorded pose; beams sweep the
    body plane from -FOV / 2 (right) to FOV / 2 (left)."""
    var azimuth = -FOV / 2 + Float32(beam) * (FOV / Float32(BEAMS - 1))
    var origin = Vec(p[0, i], p[1, i], p[2, i], 0)
    var forward = Vec(p[3, i], p[4, i], p[5, i], 0)
    var left = Vec(p[6, i], p[7, i], p[8, i], 0)
    return ray(map, d, origin, cos(azimuth) * forward + sin(azimuth) * left)


def mount(x: Float32, y: Float32, z: Float32, f: Frame) -> Vec:
    var forward = Vec(f.fx, f.fy, f.fz, 0)
    var up = cross(forward, Vec(f.lx, f.ly, f.lz, 0))
    return Vec(x, y, z, 0) + MOUNT_FORWARD * forward + MOUNT_HEIGHT * up


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
    """Distance along the unit `direction` to the first triangle, or RANGE."""
    var start = max(march(map, d, origin, direction) - map.cell, 0)
    return start + walk(map, d, origin + start * direction, direction, RANGE - start)


def march(map: Map, d: Mat[1], origin: Vec, direction: Vec) -> Float32:
    """Sphere tracing: advance by the clearance under the ray (minus the
    bilinear interpolation error) and by the height left above the floor over
    the terrain's steepest slope, so no triangle can lie on the way; stop a
    coarse cell short of anything so the exact walk takes over."""
    var horizontal = max(sqrt(direction[0] * direction[0] + direction[1] * direction[1]), 1e-6)
    var margin = 1.5 * map.cell
    var descent = map.slope * horizontal + abs(direction[2])
    var t: Float32 = 0
    while t < RANGE:
        var p = origin + t * direction
        var step = (map.clearance_at(d, p[0], p[1]) - margin) / horizontal
        if map.slope > 0 or direction[2] != 0:
            var above = p[2] - map.height_at(d, p[0], p[1]) - FLOOR_MARGIN
            step = min(step, above / max(descent, 1e-6))
        if step < 0.5 * map.coarse:
            break
        t += step
    return min(t, RANGE)


def walk(map: Map, d: Mat[1], origin: Vec, direction: Vec, limit: Float32) -> Float32:
    """Exact: the nearest triangle hit within `limit` along the ray, or `limit`."""
    var best = limit
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
    # Large triangles are listed by every cell they cover; the last few
    # faces tested are remembered so the walk does not test them again.
    var recent = SIMD[DType.int32, 4](-1)
    while t < best and ix >= 0 and iy >= 0 and ix < map.cnx and iy < map.cny:
        var k = map.starts + iy * map.cnx + ix
        for e in range(Int(d[0, k]), Int(d[0, k + 1])):
            var face = Int32(d[0, map.items + e])
            if recent.eq(face).reduce_or():
                continue
            recent = recent.shift_right[1]()
            recent[0] = face
            best = min(best, triangle(d, map.triangles + Int(face) * 9, origin, direction))
        if next_x < next_y:
            t = next_x
            next_x += delta_x
            ix += step_x
        else:
            t = next_y
            next_y += delta_y
            iy += step_y
    return best
