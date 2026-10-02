"""Map compiler: meshes and a route in, one `.wrmap` float32 array out.

Sources are built-in demo ribbons, a YAML file naming OBJ meshes plus a text
route, or a ROS occupancy image (PGM/PNG + YAML). Every road boundary edge
is extruded into a wall so the lidar always has something to see. Grids hold
the road height and a conservative signed clearance; triangles are indexed
by coarse cells for the lidar.
"""
from std.math import floor, ceil, sqrt, cos, sin, atan2, pi
from std.collections import Dict, Deque
from std.os.path import dirname, join
from std.pathlib import Path
from std.python import Python
from layout import TileTensor, Coord, row_major
from layout.numpy import to_numpy, from_numpy
from .map import MAGIC, VERSION, WALL, cross, dot, unit

comptime Vec3 = SIMD[DType.float64, 4]
comptime FAR = Float64(1e20)
comptime COARSE = 8  # lidar cells are this many map cells wide
comptime SPACING = 0.15  # route resampling, metres
comptime FREE = 230  # occupancy pixels at or above this are drivable


@inline(.always)
def corner(a: Vec3, b: Vec3, c: Vec3, i: Int) -> Vec3:
    return a if i == 0 else (b if i == 1 else c)


struct Mesh(Movable):
    """Indexed triangles; `obstacle` parts block but are never driven on."""

    var vertices: List[Vec3]
    var faces: List[Int]  # three vertex indices per triangle
    var obstacle: Bool

    def __init__(
        out self, var vertices: List[Vec3], var faces: List[Int], obstacle: Bool = False
    ):
        self.vertices = vertices^
        self.faces = faces^
        self.obstacle = obstacle

    def corner(self, face: Int, which: Int) -> Vec3:
        return self.vertices[self.faces[3 * face + which]]


def quad(mut mesh: Mesh, a: Vec3, b: Vec3, c: Vec3, d: Vec3):
    var start = len(mesh.vertices)
    for v in [a, b, c, d]:
        mesh.vertices.append(v)
    for index in [0, 1, 2, 0, 2, 3]:
        mesh.faces.append(start + index)


struct Route(Movable):
    """Waypoints with half widths; closed routes wrap from the last to first."""

    var points: List[Vec3]
    var half_width: List[Float64]
    var up: List[Vec3]
    var closed: Bool
    var cumulative: List[Float64]  # distance to each point, then the total

    def __init__(
        out self,
        var points: List[Vec3],
        var half_width: List[Float64],
        var up: List[Vec3],
        closed: Bool,
    ) raises:
        if len(points) < 2:
            raise Error("a route needs at least two points")
        self.closed = closed
        self.cumulative = [0.0]
        for i in range(len(points) if closed else len(points) - 1):
            var d = points[(i + 1) % len(points)] - points[i]
            self.cumulative.append(self.cumulative[i] + sqrt(dot(d, d)))
        if len(up) == 0:
            up = List[Vec3](length=len(points), fill=Vec3(0, 0, 1, 0))
        self.points = points^
        self.half_width = half_width^
        self.up = up^

    def segments(self) -> Int:
        return len(self.cumulative) - 1

    def delta(self, i: Int) -> Vec3:
        return self.points[(i + 1) % len(self.points)] - self.points[i]

    def tangent(self, i: Int) -> Vec3:
        return unit(self.delta(min(i, self.segments() - 1)))

    def resample(self, spacing: Float64) raises -> Route:
        """Points every `spacing` metres along the route (widths interpolated)."""
        var total = self.cumulative[self.segments()]
        var count = max(2, Int(ceil(total / spacing)) + 1)
        var points = List[Vec3]()
        var widths = List[Float64]()
        var segment = 0
        for k in range(count - 1 if self.closed else count):
            var s = total * Float64(k) / Float64(count - 1)
            while segment + 1 < self.segments() and self.cumulative[segment + 1] < s:
                segment += 1
            var length = self.cumulative[segment + 1] - self.cumulative[segment]
            var t = min(max((s - self.cumulative[segment]) / length, 0), 1)
            var next = (segment + 1) % len(self.points)
            points.append(self.points[segment] + self.delta(segment) * t)
            widths.append(
                self.half_width[segment]
                + (self.half_width[next] - self.half_width[segment]) * t
            )
        return Route(points^, widths^, List[Vec3](), self.closed)


struct Track(Movable):
    var parts: List[Mesh]
    var route: Route

    def __init__(out self, var parts: List[Mesh], var route: Route):
        self.parts = parts^
        self.route = route^


def ribbon(route: Route) -> Mesh:
    """Two triangles per segment between the left and right route edges."""
    var vertices = List[Vec3]()
    for i in range(len(route.points)):
        var left = unit(cross(route.up[i], route.tangent(i))) * route.half_width[i]
        vertices.append(route.points[i] + left)
        vertices.append(route.points[i] - left)
    var faces = List[Int]()
    for i in range(route.segments()):
        var j = (i + 1) % len(route.points)
        for index in [2 * i, 2 * i + 1, 2 * j + 1, 2 * i, 2 * j + 1, 2 * j]:
            faces.append(index)
    return Mesh(vertices^, faces^)


def demo(kind: String) raises -> Track:
    """`flat`, `ramp`, or `bank`: a 1.3 m half-width ribbon around a circle."""
    if kind != "flat" and kind != "ramp" and kind != "bank":
        raise Error("unknown demo map: " + kind)
    var points = List[Vec3]()
    var up = List[Vec3]()
    var widths = List[Float64]()
    for i in range(128):
        var t = 2 * pi * Float64(i) / 128
        var z = 0.7 * (1 - cos(2 * t)) if kind == "ramp" else 0.0
        points.append(Vec3(6 * cos(t), 6 * sin(t), z, 0))
        if kind == "bank":
            up.append(Vec3(-0.25 * cos(t), -0.25 * sin(t), 1, 0))
        widths.append(1.3)
    var route = Route(points^, widths^, up^, True)
    var parts = List[Mesh]()
    parts.append(ribbon(route))
    return Track(parts^, route^)


def load_obj(path: String, obstacle: Bool) raises -> Mesh:
    """Wavefront OBJ: `v x y z` and `f a b c ...` (polygons fan-triangulated)."""
    var vertices = List[Vec3]()
    var faces = List[Int]()
    for line in Path(path).read_text().splitlines():
        var words = line.split()
        if len(words) >= 4 and words[0] == "v":
            vertices.append(
                Vec3(Float64(words[1]), Float64(words[2]), Float64(words[3]), 0)
            )
        elif len(words) >= 4 and words[0] == "f":
            var polygon = List[Int]()
            for i in range(1, len(words)):
                var index = Int(words[i].split("/")[0])
                polygon.append(index - 1 if index > 0 else len(vertices) + index)
            for i in range(1, len(polygon) - 1):
                for index in [polygon[0], polygon[i], polygon[i + 1]]:
                    faces.append(index)
    return Mesh(vertices^, faces^, obstacle)


def load_route(path: String, closed: Bool) raises -> Route:
    """Text route: one `x y z [half_width]` per line (commas allowed)."""
    var points = List[Vec3]()
    var widths = List[Float64]()
    for line in Path(path).read_text().splitlines():
        var words = String(line).replace(",", " ").split("#")[0].split()
        if len(words) < 3:
            continue
        points.append(
            Vec3(Float64(words[0]), Float64(words[1]), Float64(words[2]), 0)
        )
        widths.append(Float64(words[3]) if len(words) > 3 else 1.5)
    return Route(points^, widths^, List[Vec3](), closed)


def parse_yaml(path: String) raises -> Dict[String, String]:
    """Flat `key: value` pairs (comments and nesting are ignored)."""
    var result = Dict[String, String]()
    for line in Path(path).read_text().splitlines():
        var text = String(line).split("#")[0]
        var colon = text.find(":")
        if colon > 0 and not text.startswith(" "):
            result[String(text[byte=0:colon].strip())] = String(
                text[byte = colon + 1 :].strip()
            )
    return result^


def load_track(path: String) raises -> Track:
    if path == "flat" or path == "ramp" or path == "bank":
        return demo(path)
    var meta = parse_yaml(path)
    var folder = dirname(path)
    if "image" in meta:
        return image_track(folder, meta)
    var closed = meta.get("closed", "true") != "false"
    var parts = List[Mesh]()
    for name in meta["mesh"].split(","):
        parts.append(load_obj(join(folder, String(name.strip())), False))
    if "obstacles" in meta:
        for name in meta["obstacles"].split(","):
            parts.append(load_obj(join(folder, String(name.strip())), True))
    return Track(parts^, load_route(join(folder, meta["route"]), closed))


# ===-------------------------------------------------------------------=== #
# Grids and rasterization.
# ===-------------------------------------------------------------------=== #


@fieldwise_init
struct Grid(TrivialRegisterPassable):
    """Cells centred at (x0 + ix * cell, y0 + iy * cell)."""

    var nx: Int
    var ny: Int
    var x0: Float64
    var y0: Float64
    var cell: Float64

    def cells(self, a: Vec3, b: Vec3, c: Vec3, conservative: Bool) -> List[Int]:
        """Cells whose centre (or, conservatively, whose square) lies in the
        xy projection of the triangle."""
        var lo = min(min(a, b), c)
        var hi = max(max(a, b), c)
        var x0 = max(Int(floor((lo[0] - self.x0) / self.cell)) - 1, 0)
        var y0 = max(Int(floor((lo[1] - self.y0) / self.cell)) - 1, 0)
        var x1 = min(Int(ceil((hi[0] - self.x0) / self.cell)) + 2, self.nx)
        var y1 = min(Int(ceil((hi[1] - self.y0) / self.cell)) + 2, self.ny)
        var sign: Float64 = 1 if cross(b - a, c - a)[2] >= 0 else -1
        var result = List[Int]()
        for iy in range(y0, y1):
            for ix in range(x0, x1):
                var x = self.x0 + Float64(ix) * self.cell
                var y = self.y0 + Float64(iy) * self.cell
                var inside = True
                for edge in range(3):
                    var p = corner(a, b, c, edge)
                    var q = corner(a, b, c, (edge + 1) % 3)
                    var dx = q[0] - p[0]
                    var dy = q[1] - p[1]
                    var margin = (
                        self.cell * 0.5 * (abs(dx) + abs(dy)) if conservative else 0
                    )
                    if sign * (dx * (y - p[1]) - dy * (x - p[0])) < -margin - 1e-9:
                        inside = False
                        break
                if inside:
                    result.append(iy * self.nx + ix)
        return result^


def distance_transform(free: List[Bool], width: Int, height: Int) -> List[Float64]:
    """Squared distance (in cells) from every cell to the nearest non-free
    cell: Felzenszwalb & Huttenlocher, columns then rows."""
    var longest = max(width, height)
    var squared = List[Float64](length=width * height, fill=0)
    var f = List[Float64](length=longest, fill=0)
    var d = List[Float64](length=longest, fill=0)
    var v = List[Int](length=longest, fill=0)
    var z = List[Float64](length=longest + 1, fill=0)
    for x in range(width):
        for y in range(height):
            f[y] = FAR if free[y * width + x] else 0
        transform_1d(f, height, d, v, z)
        for y in range(height):
            squared[y * width + x] = d[y]
    for y in range(height):
        for x in range(width):
            f[x] = squared[y * width + x]
        transform_1d(f, width, d, v, z)
        for x in range(width):
            squared[y * width + x] = d[x]
    return squared^


def transform_1d(
    f: List[Float64], n: Int, mut d: List[Float64], mut v: List[Int], mut z: List[Float64]
):
    var k = 0
    v[0] = 0
    z[0] = -FAR
    z[1] = FAR
    for q in range(1, n):
        var s = intersection(f, q, v[k])
        while s <= z[k]:
            k -= 1
            s = intersection(f, q, v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = FAR
    k = 0
    for q in range(n):
        while z[k + 1] < Float64(q):
            k += 1
        d[q] = Float64((q - v[k]) * (q - v[k])) + f[v[k]]


@inline(.always)
def intersection(f: List[Float64], q: Int, p: Int) -> Float64:
    return ((f[q] + Float64(q * q)) - (f[p] + Float64(p * p))) / Float64(2 * q - 2 * p)


def edge_key(p: Vec3, q: Vec3) -> SIMD[DType.int64, 4]:
    """Order-independent key of an edge by its xy endpoints, quantized to 1 um."""
    var a = (p * 1e6).cast[DType.int64]()
    var b = (q * 1e6).cast[DType.int64]()
    if a[0] > b[0] or (a[0] == b[0] and a[1] > b[1]):
        a, b = b, a
    return SIMD[DType.int64, 4](a[0], a[1], b[0], b[1])


# ===-------------------------------------------------------------------=== #
# The compiler.
# ===-------------------------------------------------------------------=== #


def compile(track: Track, resolution: Float64 = 0.025) raises -> List[Float32]:
    var lo = Vec3(FAR, FAR, FAR, 0)
    var hi = -lo
    for p in range(len(track.parts)):
        for v in track.parts[p].vertices:
            lo = min(lo, v)
            hi = max(hi, v)
    var grid = Grid(
        Int(ceil((hi[0] - lo[0] + 2) / resolution)) + 1,
        Int(ceil((hi[1] - lo[1] + 2) / resolution)) + 1,
        floor((lo[0] - 1) / resolution) * resolution,
        floor((lo[1] - 1) / resolution) * resolution,
        resolution,
    )
    var cells = grid.nx * grid.ny
    var height = List[Float32](length=cells, fill=0)
    var road = List[Bool](length=cells, fill=False)
    var corners = List[Vec3]()  # a, b, c of every collidable triangle
    var edges = Dict[SIMD[DType.int64, 4], Int]()
    var boundary = List[Vec3]()
    for p in range(len(track.parts)):
        ref part = track.parts[p]
        for face in range(len(part.faces) // 3):
            var a = part.corner(face, 0)
            var b = part.corner(face, 1)
            var c = part.corner(face, 2)
            var normal = unit(cross(b - a, c - a))
            for v in [a, b, c]:
                corners.append(v)
            if part.obstacle or abs(normal[2]) < 0.7:
                continue  # walls and steep faces only block
            # Drivable: sample the plane at every cell centre inside the face.
            for k in grid.cells(a, b, c, False):
                var x = grid.x0 + Float64(k % grid.nx) * resolution
                var y = grid.y0 + Float64(k // grid.nx) * resolution
                height[k] = Float32(
                    a[2] - (normal[0] * (x - a[0]) + normal[1] * (y - a[1])) / normal[2]
                )
                road[k] = True
            for edge in range(3):
                var p = corner(a, b, c, edge)
                var q = corner(a, b, c, (edge + 1) % 3)
                var key = edge_key(p, q)
                edges[key] = edges.get(key, 0) + 1
                boundary.append(p)
                boundary.append(q)
    # Shared edges cancel; the surviving edges bound the road and get walls.
    var walls = Mesh(List[Vec3](), List[Int](), True)
    for i in range(0, len(boundary), 2):
        if edges[edge_key(boundary[i], boundary[i + 1])] == 1:
            var up = Vec3(0, 0, WALL, 0)
            quad(walls, boundary[i], boundary[i + 1], boundary[i + 1] + up, boundary[i] + up)
    for face in range(len(walls.faces) // 3):
        for which in range(3):
            corners.append(walls.corner(face, which))
    # Clearance: distance to the nearest cell that is off the road or that
    # any wall or obstacle overlaps (so even sub-cell walls block).
    var free = road.copy()
    var triangles = len(corners) // 3
    for t in range(triangles):
        var a = corners[3 * t]
        var b = corners[3 * t + 1]
        var c = corners[3 * t + 2]
        if abs(unit(cross(b - a, c - a))[2]) < 0.7:
            for k in grid.cells(a, b, c, True):
                free[k] = False
    var squared = distance_transform(free, grid.nx, grid.ny)
    var clearance = List[Float32](length=cells, fill=Float32(-resolution))
    for k in range(cells):
        if free[k]:
            clearance[k] = Float32((sqrt(squared[k]) - 0.7071) * resolution)
    # Route segments and the spawn table.
    var route = track.route.resample(SPACING)
    var segments = List[Float32]()
    var spawns = List[Float32]()
    for i in range(route.segments()):
        var p = route.points[i]
        var d = route.delta(i)
        for value in [
            p[0], p[1], p[2], d[0], d[1], d[2], route.cumulative[i], route.half_width[i], atan2(d[1], d[0]),
        ]:
            segments.append(Float32(value))
        var middle = p + d * 0.5
        var ix = Int(floor((middle[0] - grid.x0) / resolution + 0.5))
        var iy = Int(floor((middle[1] - grid.y0) / resolution + 0.5))
        var ending = not route.closed and route.cumulative[i + 1] > route.cumulative[route.segments()] - 0.5
        if ix >= 0 and iy >= 0 and ix < grid.nx and iy < grid.ny and not ending:
            if clearance[iy * grid.nx + ix] > 0.3 and route.half_width[i] > 0.2:
                spawns.append(Float32(i))
    if len(spawns) == 0:
        raise Error("the route has no segment with vehicle clearance")
    # Coarse cells list the triangles the lidar must test.
    var coarse = Grid(
        (grid.nx + COARSE - 1) // COARSE,
        (grid.ny + COARSE - 1) // COARSE,
        grid.x0 - resolution / 2 + COARSE * resolution / 2,
        grid.y0 - resolution / 2 + COARSE * resolution / 2,
        COARSE * resolution,
    )
    var lists = List[List[Int]](length=coarse.nx * coarse.ny, fill=List[Int]())
    for t in range(triangles):
        for k in coarse.cells(corners[3 * t], corners[3 * t + 1], corners[3 * t + 2], True):
            lists[k].append(t)
    var starts = List[Float32]()
    var items = List[Float32]()
    for k in range(len(lists)):
        starts.append(Float32(len(items)))
        for t in lists[k]:
            items.append(Float32(t))
    starts.append(Float32(len(items)))
    var result = List[Float32]()
    for value in [
        Float64(MAGIC), Float64(VERSION), Float64(grid.nx), Float64(grid.ny),
        grid.x0, grid.y0, resolution, 1.0 if route.closed else 0.0,
        Float64(route.segments()), route.cumulative[route.segments()], Float64(len(spawns)),
        Float64(coarse.nx), Float64(coarse.ny), coarse.cell, Float64(triangles), Float64(len(items)),
    ]:
        result.append(Float32(value))
    result.extend(clearance^)
    result.extend(height^)
    result.extend(segments^)
    result.extend(spawns^)
    result.extend(starts^)
    result.extend(items^)
    for t in range(triangles):
        var a = corners[3 * t]
        for v in [a, corners[3 * t + 1] - a, corners[3 * t + 2] - a]:
            for axis in range(3):
                result.append(Float32(v[axis]))
    return result^


# ===-------------------------------------------------------------------=== #
# ROS occupancy images: free pixels become floor, their borders become walls,
# and the route is the longest skeleton loop near the world origin.
# ===-------------------------------------------------------------------=== #


def load_image(path: String, mut pixels: List[UInt8]) raises -> Tuple[Int, Int]:
    """8-bit grey pixels, row-major from the top, from a PGM or PNG file."""
    var raw = Path(path).read_bytes()
    if raw[0] == 0x50 and (raw[1] == 0x35 or raw[1] == 0x32):
        return decode_pgm(raw, pixels)
    if raw[0] == 0x89 and raw[1] == 0x50:
        return decode_png(raw, pixels)
    raise Error("unsupported image format (use PGM or PNG): " + path)


def decode_pgm(raw: List[UInt8], mut pixels: List[UInt8]) raises -> Tuple[Int, Int]:
    """Binary (P5) or ASCII (P2) portable greymaps."""
    var fields = List[Int]()
    var i = 2
    var binary = raw[1] == 0x35
    while len(fields) < (3 if binary else 1 << 30) and i < len(raw):
        if raw[i] == 0x23:  # comment
            while raw[i] != 0x0A:
                i += 1
        elif raw[i] >= 0x30 and raw[i] <= 0x39:
            var value = 0
            while raw[i] >= 0x30 and raw[i] <= 0x39:
                value = value * 10 + Int(raw[i]) - 0x30
                i += 1
            fields.append(value)
        else:
            i += 1
    var width = fields[0]
    var height = fields[1]
    for k in range(width * height):
        var value = Int(raw[i + 1 + k]) if binary else fields[3 + k]
        pixels.append(UInt8(value * 255 // fields[2]))
    return (width, height)


def decode_png(raw: List[UInt8], mut pixels: List[UInt8]) raises -> Tuple[Int, Int]:
    """Non-interlaced 8-bit PNG (grey, grey+alpha, RGB, or RGBA)."""
    var width = 0
    var height = 0
    var channels = 0
    var deflated = List[UInt8]()
    var at = 8
    while at + 8 <= len(raw):
        var length = 0
        for k in range(4):
            length = length * 256 + Int(raw[at + k])
        var kind = String(StringSlice(unsafe_from_utf8=Span(raw)[at + 4 : at + 8]))
        var body = at + 8
        if kind == "IHDR":
            for k in range(4):
                width = width * 256 + Int(raw[body + k])
                height = height * 256 + Int(raw[body + 4 + k])
            if raw[body + 8] != 8 or raw[body + 12] != 0:
                raise Error("only 8-bit non-interlaced PNG files are supported")
            var colour = Int(raw[body + 9])
            channels = 1 if colour == 0 else (3 if colour == 2 else (2 if colour == 4 else 4))
        elif kind == "IDAT":
            deflated.extend(Span(raw)[body : body + length])
        at = body + length + 4
    var packed = to_numpy(TileTensor(Span(deflated), row_major(Coord(len(deflated)))))
    var array = Python.import_module("numpy").frombuffer(
        Python.import_module("zlib").decompress(packed.tobytes()), dtype="uint8"
    )
    var inflated = from_numpy[DType.uint8, 1](array)
    # Undo the per-row filters in place (None, Sub, Up, Average, Paeth).
    var stride = width * channels
    var rows = List[UInt8](length=height * stride, fill=0)
    for row in range(height):
        var kind = Int(inflated[row * (stride + 1)])
        var line = row * (stride + 1) + 1
        for i in range(stride):
            var a = Int(rows[row * stride + i - channels]) if i >= channels else 0
            var b = Int(rows[(row - 1) * stride + i]) if row > 0 else 0
            var c = Int(rows[(row - 1) * stride + i - channels]) if row > 0 and i >= channels else 0
            var x = Int(inflated[line + i])
            if kind == 1:
                x += a
            elif kind == 2:
                x += b
            elif kind == 3:
                x += (a + b) // 2
            elif kind == 4:
                var p = a + b - c
                x += a if abs(p - a) <= abs(p - b) and abs(p - a) <= abs(p - c) else (
                    b if abs(p - b) <= abs(p - c) else c
                )
            rows[row * stride + i] = UInt8(x & 255)
        for col in range(width):
            var k = row * stride + col * channels
            if channels >= 3:
                pixels.append(
                    UInt8(
                        (Int(rows[k]) * 19595 + Int(rows[k + 1]) * 38470 + Int(rows[k + 2]) * 7471 + 0x8000)
                        >> 16
                    )
                )
            else:
                pixels.append(rows[k])
    return (width, height)


def neighbours(skeleton: List[Bool], width: Int, height: Int, i: Int) -> List[Int]:
    var result = List[Int]()
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            var y = i // width + dy
            var x = i % width + dx
            if (dy != 0 or dx != 0) and y >= 0 and y < height and x >= 0 and x < width:
                if skeleton[y * width + x]:
                    result.append(y * width + x)
    return result^


def skeletonize(mut skeleton: List[Bool], width: Int, height: Int):
    """Zhang-Suen thinning to a one-pixel-wide skeleton, then every branch
    with a free end is eroded so only loops remain."""
    var remove = List[Int]()
    var changed = True
    while changed:
        changed = False
        for pass_index in range(2):
            remove.clear()
            for i in range(width + 1, width * (height - 1) - 1):
                if not skeleton[i] or i % width == 0 or i % width == width - 1:
                    continue
                var p = SIMD[DType.bool, 8](
                    skeleton[i - width], skeleton[i - width + 1], skeleton[i + 1],
                    skeleton[i + width + 1], skeleton[i + width], skeleton[i + width - 1],
                    skeleton[i - 1], skeleton[i - width - 1],
                )
                var count = Int(p.cast[DType.int32]().reduce_add())
                var transitions = 0
                for k in range(8):
                    transitions += Int(not p[k] and p[(k + 1) % 8])
                var first = (p[0] and p[2] and p[4]) if pass_index == 0 else (p[0] and p[2] and p[6])
                var second = (p[2] and p[4] and p[6]) if pass_index == 0 else (p[0] and p[4] and p[6])
                if count >= 2 and count <= 6 and transitions == 1 and not first and not second:
                    remove.append(i)
            for i in remove:
                skeleton[i] = False
                changed = True
    var tips = List[Int]()
    while True:
        tips.clear()
        for i in range(width * height):
            if skeleton[i] and len(neighbours(skeleton, width, height, i)) <= 1:
                tips.append(i)
        if len(tips) == 0:
            return
        for i in tips:
            skeleton[i] = False


def longest_loop(
    skeleton: List[Bool], width: Int, height: Int, row: Float64, col: Float64
) raises -> List[Int]:
    """The longest closed pixel loop through a skeleton point near (row, col):
    for each of the nearest seeds, BFS between pairs of its neighbours."""
    var candidates = List[Int]()
    var keys = List[Float64]()
    for i in range(width * height):
        if skeleton[i]:
            candidates.append(i)
            keys.append((Float64(i // width) - row) ** 2 + (Float64(i % width) - col) ** 2)
    var best = List[Int]()
    for _ in range(min(32, len(candidates))):
        var nearest = 0
        for c in range(len(candidates)):
            if keys[c] < keys[nearest]:
                nearest = c
        var start = candidates[nearest]
        keys[nearest] = FAR
        var around = neighbours(skeleton, width, height, start)
        for a in range(len(around)):
            for b in range(a + 1, len(around)):
                var parent = Dict[Int, Int]()
                parent[around[a]] = around[a]
                var queue = Deque[Int]()
                queue.append(around[a])
                while len(queue) > 0 and around[b] not in parent:
                    var u = queue.popleft()
                    for v in neighbours(skeleton, width, height, u):
                        if v != start and v not in parent:
                            parent[v] = u
                            queue.append(v)
                if around[b] not in parent:
                    continue
                var path = List[Int]()
                path.append(start)
                var node = around[b]
                while node != around[a]:
                    path.append(node)
                    node = parent[node]
                path.append(around[a])
                if len(path) > len(best):
                    best = path^
    if len(best) == 0:
        raise Error("could not extract a closed centerline loop from the image")
    return best^


def image_track(folder: String, meta: Dict[String, String]) raises -> Track:
    var pixels = List[UInt8]()
    var w, h = load_image(join(folder, meta["image"]), pixels)
    var resolution = Float64(meta["resolution"])
    var origin = meta["origin"].replace("[", " ").replace("]", " ").replace(",", " ").split()
    var ox = Float64(origin[0])
    var oy = Float64(origin[1])
    var free = List[Bool](capacity=w * h)
    for value in pixels:
        free.append(Int(value) >= FREE)
    var squared = distance_transform(free, w, h)
    var skeleton = free.copy()
    skeletonize(skeleton, w, h)
    var loop = longest_loop(skeleton, w, h, Float64(h - 1) + oy / resolution, -ox / resolution)
    # World coordinates, circular moving-average smoothing, then a route whose
    # half widths come from the distance to the walls.
    var count = len(loop)
    var points = List[Vec3]()
    var widths = List[Float64]()
    for k in range(count):
        var x: Float64 = 0
        var y: Float64 = 0
        for offset in range(-25, 26):
            var i = loop[((k + offset) % count + count) % count]
            x += ox + Float64(i % w) * resolution
            y += oy + Float64(h - 1 - i // w) * resolution
        points.append(Vec3(x / 51, y / 51, 0, 0))
        var i = loop[k]
        widths.append(max(sqrt(squared[i]) * resolution, 0.1))
    var route = Route(points^, widths^, List[Vec3](), True)
    # The floor is one quad; every free/occupied pixel edge is a wall.
    var x0 = ox - resolution / 2
    var y0 = oy - resolution / 2
    var floor = Mesh(List[Vec3](), List[Int]())
    quad(
        floor,
        Vec3(x0, y0, 0, 0),
        Vec3(x0 + Float64(w) * resolution, y0, 0, 0),
        Vec3(x0 + Float64(w) * resolution, y0 + Float64(h) * resolution, 0, 0),
        Vec3(x0, y0 + Float64(h) * resolution, 0, 0),
    )
    var walls = Mesh(List[Vec3](), List[Int](), True)
    for r in range(h):
        for c in range(w):
            if not free[r * w + c]:
                continue
            for side in range(4):
                var dr = [0, 0, 1, -1][side]
                var dc = [1, -1, 0, 0][side]
                var nr = r + dr
                var nc = c + dc
                if nr >= 0 and nr < h and nc >= 0 and nc < w and free[nr * w + nc]:
                    continue
                var centre = Vec3(
                    ox + (Float64(c) + Float64(dc) / 2) * resolution,
                    oy + (Float64(h - 1 - r) - Float64(dr) / 2) * resolution,
                    0,
                    0,
                )
                var along = Vec3(Float64(dr), Float64(dc), 0, 0) * (resolution / 2)
                var up = Vec3(0, 0, WALL, 0)
                quad(walls, centre - along, centre + along, centre + along + up, centre - along + up)
    var parts = List[Mesh]()
    parts.append(floor^)
    parts.append(walls^)
    return Track(parts^, route^)
