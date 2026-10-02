"""Parallel distance lidar: a BVH ray cast per beam over the map triangles."""
from std.math import cos, sin
from std.sys import get_defined_int
from .device import Ptr
from .layout import BEAMS, ROWS, RAYS
from .vehicle import Frame

comptime RANGE = Float32(get_defined_int["RANGE", 10]())
comptime ELEVATION = Float32(0.261799388)
comptime MOUNT_FORWARD = Float32(0.2)
comptime MOUNT_HEIGHT = Float32(0.23)
comptime Vec = SIMD[DType.float32, 4]


@inline(.always)
def xyz(x: Float32, y: Float32, z: Float32) -> Vec:
    var result = Vec(0)
    result[0], result[1], result[2] = x, y, z
    return result


@inline(.always)
def dot(a: Vec, b: Vec) -> Float32:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@inline(.always)
def cross(a: Vec, b: Vec) -> Vec:
    return xyz(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def mount(x: Float32, y: Float32, z: Float32, f: Frame) -> Vec:
    var forward = xyz(f.fx, f.fy, f.fz)
    var up = cross(forward, xyz(f.lx, f.ly, f.lz))
    return xyz(x, y, z) + MOUNT_FORWARD * forward + MOUNT_HEIGHT * up


def direction(beam: Int, f: Frame) -> Vec:
    var azimuth = -2.35619449 + Float32(beam % BEAMS) * (
        4.71238898 / Float32(BEAMS - 1)
    )
    var elevation = Float32(beam // BEAMS - 1) * ELEVATION
    var forward = xyz(f.fx, f.fy, f.fz)
    var left = xyz(f.lx, f.ly, f.lz)
    return cos(elevation) * (
        cos(azimuth) * forward + sin(azimuth) * left
    ) + sin(elevation) * cross(forward, left)


@inline(.always)
def triangle(origin: Vec, direction: Vec, a: Vec, e1: Vec, e2: Vec) -> Float32:
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


def ray(map: Ptr, origin: Vec, direction: Vec) -> Float32:
    var nodes = Int(map[unsafe_offset=11])
    var tree = (
        16
        + 4 * Int(map[unsafe_offset=2]) * Int(map[unsafe_offset=3])
        + 10 * Int(map[unsafe_offset=4])
        + Int(map[unsafe_offset=10])
    )
    var triangles = tree + nodes * 9
    var inverse = Vec(0)
    comptime for axis in range(3):
        inverse[axis] = (
            0 if abs(direction[axis]) < 1e-8 else 1 / direction[axis]
        )
    var distance = RANGE
    var node = 0
    while node < nodes:
        var k = tree + node * 9
        var near: Float32 = 0
        var far = distance
        comptime for axis in range(3):
            var lo = map[unsafe_offset=k + axis] - origin[axis]
            var hi = map[unsafe_offset=k + 3 + axis] - origin[axis]
            if inverse[axis] == 0:
                if lo > 0 or hi < 0:
                    far = -1
            else:
                var a = lo * inverse[axis]
                var b = hi * inverse[axis]
                near = max(near, min(a, b))
                far = min(far, max(a, b))
        if near <= far:
            var count = Int(map[unsafe_offset=k + 8])
            if count == 0:
                node += 1
                continue
            var first = Int(map[unsafe_offset=k + 7])
            for face in range(first, first + count):
                var t = triangles + face * 9
                var a = xyz(
                    map[unsafe_offset=t],
                    map[unsafe_offset=t + 1],
                    map[unsafe_offset=t + 2],
                )
                var e1 = xyz(
                    map[unsafe_offset=t + 3],
                    map[unsafe_offset=t + 4],
                    map[unsafe_offset=t + 5],
                )
                var e2 = xyz(
                    map[unsafe_offset=t + 6],
                    map[unsafe_offset=t + 7],
                    map[unsafe_offset=t + 8],
                )
                distance = min(distance, triangle(origin, direction, a, e1, e2))
        node = Int(map[unsafe_offset=k + 6])
    return distance
