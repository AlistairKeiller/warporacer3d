"""Constant-time support queries and conservative distance-field ray marching."""
from std.math import floor, sqrt, isfinite
from .core import Ptr, clamp
from std.sys import get_defined_int

comptime RANGE = Float32(get_defined_int["RANGE", 10]())


def validate(map: List[Float32]) raises:
    """Check asset sizes and indices before exposing pointers to kernels."""
    if len(map) < 16 or map[0] != 314159 or map[1] != 1:
        raise Error("invalid map; prepare it with prepare.py")
    for value in map:
        if not isfinite(value):
            raise Error("map contains nonfinite values")
    for i in [2, 3, 4, 10]:
        if map[i] < 1 or map[i] > 16000000 or map[i] != floor(map[i]):
            raise Error("invalid map dimensions")
    var cells = Int(map[2]) * Int(map[3])
    var count = Int(map[4])
    var route = 16 + 4 * cells
    var spawns = route + 10 * count
    if cells > 16000000 or len(map) != spawns + Int(map[10]):
        raise Error("invalid or truncated map")
    if (map[5] != 0 and map[5] != 1) or map[8] < 0.0049 or map[8] > 0.1001 or map[9] <= 0:
        raise Error("invalid map geometry")
    for i in range(count):
        var k = route + i * 10
        if (
            map[k + 3] * map[k + 3] + map[k + 4] * map[k + 4] < 1e-12
            or map[k + 6] <= 0
            or map[k + 8] <= 0
        ):
            raise Error("invalid route segment")
    for i in range(spawns, len(map)):
        if map[i] < 0 or map[i] >= Float32(count) or map[i] != floor(map[i]):
            raise Error("invalid spawn index")


@fieldwise_init
struct Surface(TrivialRegisterPassable):
    var clearance: Float32
    var height: Float32
    var sx: Float32
    var sy: Float32


def surface(map: Ptr, x: Float32, y: Float32) -> Surface:
    var nx = Int(map[unsafe_offset=2])
    var ny = Int(map[unsafe_offset=3])
    var cell = map[unsafe_offset=8]
    var fx = (x - map[unsafe_offset=6]) / cell
    var fy = (y - map[unsafe_offset=7]) / cell
    var ix = Int(floor(fx + 0.5))
    var iy = Int(floor(fy + 0.5))
    if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
        return Surface(-1, 0, 0, 0)
    var idx = iy * nx + ix
    var size = nx * ny
    var dx = (fx - Float32(ix)) * cell
    var dy = (fy - Float32(iy)) * cell
    var sx = map[unsafe_offset=16 + 2 * size + idx]
    var sy = map[unsafe_offset=16 + 3 * size + idx]
    return Surface(
        map[unsafe_offset=16 + idx] - sqrt(dx * dx + dy * dy),
        map[unsafe_offset=16 + size + idx] + sx * dx + sy * dy,
        sx,
        sy,
    )


def ray(map: Ptr, x: Float32, y: Float32, dx: Float32, dy: Float32) -> Float32:
    var distance: Float32 = 0
    for _ in range(192):
        var free = surface(map, x + distance * dx, y + distance * dy).clearance
        if free < 0.005:
            return distance
        distance += free
        if distance >= RANGE:
            return RANGE
    # A conservative shortened range at grazing angles; never step over walls.
    return min(distance, RANGE)


def progress(map: Ptr, x: Float32, y: Float32, near: Int32) -> SIMD[DType.float32, 4]:
    var count = Int(map[unsafe_offset=4])
    var start = 16 + 4 * Int(map[unsafe_offset=2]) * Int(map[unsafe_offset=3])
    var best: Float32 = 1e30
    var result = SIMD[DType.float32, 4](0)
    for offset in range(-8, 9):
        var segment = Int(near) + offset
        if map[unsafe_offset=5] > 0:
            segment = (segment % count + count) % count
        elif segment < 0 or segment >= count:
            continue
        var k = start + segment * 10
        var px = map[unsafe_offset=k]
        var py = map[unsafe_offset=k + 1]
        var dx = map[unsafe_offset=k + 3]
        var dy = map[unsafe_offset=k + 4]
        var t = clamp(((x - px) * dx + (y - py) * dy) / (dx * dx + dy * dy), 0, 1)
        var ex = x - px - t * dx
        var ey = y - py - t * dy
        var dist = ex * ex + ey * ey
        if dist < best:
            best = dist
            result[0] = map[unsafe_offset=k + 7] + t * map[unsafe_offset=k + 6]
            result[1] = Float32(segment)
            result[2] = sqrt(dist)
            result[3] = map[unsafe_offset=k + 8]
    return result
