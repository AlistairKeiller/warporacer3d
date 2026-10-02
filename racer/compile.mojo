"""Compile a track into the float32 map the kernels consume.

Layout (all float32): a 16-value header; four `ny x nx` grids (clearance,
height, x-slope, y-slope); route segments (10 values each); spawn segment
indices; BVH nodes (9 values each); triangles as (a, b - a, c - a).
"""
from std.math import floor, ceil, sqrt, atan2
from std.collections import Dict
from std.pathlib import Path
from .track import (
    Track,
    Mesh,
    Route,
    Vec3,
    vec3,
    cross,
    dot,
    norm,
    unit,
    demo_track,
    load_obj,
    load_route,
)
from .image import image_track

comptime HEADER = 16
comptime MAGIC = 314159
comptime VERSION = 2
comptime LEAF = 8
comptime FAR = Float64(1e20)


struct Grid(Movable):
    var nx: Int
    var ny: Int
    var origin_x: Float64
    var origin_y: Float64
    var resolution: Float64
    var height: List[Float32]
    var sx: List[Float32]
    var sy: List[Float32]
    var road: List[Bool]
    var blocked: List[Bool]

    def __init__(out self, lo: Vec3, hi: Vec3, resolution: Float64) raises:
        self.resolution = resolution
        self.origin_x = floor((lo[0] - 1.0) / resolution) * resolution
        self.origin_y = floor((lo[1] - 1.0) / resolution) * resolution
        self.nx = Int(ceil((hi[0] + 1.0 - self.origin_x) / resolution)) + 1
        self.ny = Int(ceil((hi[1] + 1.0 - self.origin_y) / resolution)) + 1
        if self.nx * self.ny > 16_000_000:
            raise Error("map exceeds 16 million cells; increase resolution")
        var cells = self.nx * self.ny
        self.height = List[Float32](length=cells, fill=0)
        self.sx = List[Float32](length=cells, fill=0)
        self.sy = List[Float32](length=cells, fill=0)
        self.road = List[Bool](length=cells, fill=False)
        self.blocked = List[Bool](length=cells, fill=False)

    def span(
        self, lo_x: Float64, lo_y: Float64, hi_x: Float64, hi_y: Float64
    ) -> Tuple[Int, Int, Int, Int]:
        """Cell index range [x0, x1) x [y0, y1) covering a box with margins."""
        var x0 = max(
            Int(floor((lo_x - self.origin_x) / self.resolution)) - 1, 0
        )
        var y0 = max(
            Int(floor((lo_y - self.origin_y) / self.resolution)) - 1, 0
        )
        var x1 = min(
            Int(ceil((hi_x - self.origin_x) / self.resolution)) + 2, self.nx
        )
        var y1 = min(
            Int(ceil((hi_y - self.origin_y) / self.resolution)) + 2, self.ny
        )
        return (x0, y0, x1, y1)

    def x(self, ix: Int) -> Float64:
        return self.origin_x + Float64(ix) * self.resolution

    def y(self, iy: Int) -> Float64:
        return self.origin_y + Float64(iy) * self.resolution


def rasterize_obstacle(mut grid: Grid, a: Vec3, b: Vec3, c: Vec3, normal: Vec3):
    """Block every cell whose square overlaps the projected face, so even
    sub-cell vertical walls survive."""
    var x0, y0, x1, y1 = grid.span(
        min(min(a[0], b[0]), c[0]),
        min(min(a[1], b[1]), c[1]),
        max(max(a[0], b[0]), c[0]),
        max(max(a[1], b[1]), c[1]),
    )
    var sign: Float64 = 1 if normal[2] >= 0 else -1
    for iy in range(y0, y1):
        for ix in range(x0, x1):
            var xx = grid.x(ix)
            var yy = grid.y(iy)
            var inside = True
            for edge in range(3):
                var p = a if edge == 0 else (b if edge == 1 else c)
                var q = b if edge == 0 else (c if edge == 1 else a)
                var dx = q[0] - p[0]
                var dy = q[1] - p[1]
                var margin = grid.resolution * 0.5 * (abs(dx) + abs(dy))
                if (
                    sign * (dx * (yy - p[1]) - dy * (xx - p[0]))
                    < -margin - 1e-9
                ):
                    inside = False
                    break
            if inside:
                grid.blocked[iy * grid.nx + ix] = True


def rasterize_road(
    mut grid: Grid, a: Vec3, b: Vec3, c: Vec3, normal: Vec3
) raises:
    var x0, y0, x1, y1 = grid.span(
        min(min(a[0], b[0]), c[0]),
        min(min(a[1], b[1]), c[1]),
        max(max(a[0], b[0]), c[0]),
        max(max(a[1], b[1]), c[1]),
    )
    var gx = -normal[0] / normal[2]
    var gy = -normal[1] / normal[2]
    for iy in range(y0, y1):
        for ix in range(x0, x1):
            var xx = grid.x(ix)
            var yy = grid.y(iy)
            var inside = True
            for edge in range(3):
                var p = a if edge == 0 else (b if edge == 1 else c)
                var q = b if edge == 0 else (c if edge == 1 else a)
                if (q[0] - p[0]) * (yy - p[1]) - (q[1] - p[1]) * (
                    xx - p[0]
                ) < -1e-8:
                    inside = False
                    break
            if not inside:
                continue
            var index = iy * grid.nx + ix
            var zz = a[2] + gx * (xx - a[0]) + gy * (yy - a[1])
            if (
                grid.road[index]
                and abs(Float64(grid.height[index]) - zz) > 0.01
            ):
                raise Error(
                    "overlapping road heights are unsupported (no bridges)"
                )
            grid.height[index] = Float32(zz)
            grid.sx[index] = Float32(gx)
            grid.sy[index] = Float32(gy)
            grid.road[index] = True


def rasterize_edge(mut grid: Grid, a: Vec3, b: Vec3):
    """Block every cell whose square intersects a road boundary edge."""
    var x0, y0, x1, y1 = grid.span(
        min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])
    )
    var dx = b[0] - a[0]
    var dy = b[1] - a[1]
    var margin = grid.resolution * 0.5 * (abs(dx) + abs(dy)) + 1e-9
    for iy in range(y0, y1):
        for ix in range(x0, x1):
            var xx = grid.x(ix)
            var yy = grid.y(iy)
            if abs(dx * (yy - a[1]) - dy * (xx - a[0])) <= margin:
                grid.blocked[iy * grid.nx + ix] = True


def edge_key(p: Vec3, q: Vec3) -> String:
    """Order-independent key of an edge by its endpoints, quantized to 1 um."""
    var a = SIMD[DType.int64, 4](0)
    var b = SIMD[DType.int64, 4](0)
    for axis in range(2):
        a[axis] = Int64(floor(p[axis] * 1e6 + 0.5))
        b[axis] = Int64(floor(q[axis] * 1e6 + 0.5))
    if a[0] > b[0] or (a[0] == b[0] and a[1] > b[1]):
        a, b = b, a
    return String(a[0], ",", a[1], ",", b[0], ",", b[1])


def distance_transform(grid: Grid) -> List[Float32]:
    """Conservative clearance: Euclidean distance to the nearest blocked or
    off-road cell centre, minus half a cell diagonal; negative off the road."""
    var nx = grid.nx
    var ny = grid.ny
    var longest = max(nx, ny)
    var squared = List[Float64](length=nx * ny, fill=0)
    var f = List[Float64](length=longest, fill=0)
    var d = List[Float64](length=longest, fill=0)
    var v = List[Int](length=longest, fill=0)
    var z = List[Float64](length=longest + 1, fill=0)
    # Felzenszwalb & Huttenlocher: columns, then rows.
    for ix in range(nx):
        for iy in range(ny):
            var index = iy * nx + ix
            f[iy] = FAR if grid.road[index] and not grid.blocked[index] else 0
        transform_1d(f, ny, d, v, z)
        for iy in range(ny):
            squared[iy * nx + ix] = d[iy]
    for iy in range(ny):
        for ix in range(nx):
            f[ix] = squared[iy * nx + ix]
        transform_1d(f, nx, d, v, z)
        for ix in range(nx):
            squared[iy * nx + ix] = d[ix]
    var result = List[Float32](length=nx * ny, fill=0)
    for index in range(nx * ny):
        if grid.road[index] and not grid.blocked[index]:
            result[index] = Float32(
                sqrt(squared[index]) * grid.resolution
                - grid.resolution / sqrt(2.0)
            )
        else:
            result[index] = Float32(-grid.resolution)
    return result^


def transform_1d(
    f: List[Float64],
    n: Int,
    mut d: List[Float64],
    mut v: List[Int],
    mut z: List[Float64],
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
        var dq = Float64(q - v[k])
        d[q] = dq * dq + f[v[k]]


@inline(.always)
def intersection(f: List[Float64], q: Int, p: Int) -> Float64:
    return ((f[q] + Float64(q * q)) - (f[p] + Float64(p * p))) / Float64(
        2 * q - 2 * p
    )


def nth_element(
    mut order: List[Int], lo: Int, hi: Int, nth: Int, keys: List[Float64]
):
    """Partially sort order[lo:hi] by key so order[nth] is in sorted place."""
    var left = lo
    var right = hi - 1
    while right > left:
        var pivot = keys[order[(left + right) // 2]]
        var i = left
        var j = right
        while i <= j:
            while keys[order[i]] < pivot:
                i += 1
            while keys[order[j]] > pivot:
                j -= 1
            if i <= j:
                order[i], order[j] = order[j], order[i]
                i += 1
                j -= 1
        if nth <= j:
            right = j
        elif nth >= i:
            left = i
        else:
            return


struct Tree(Movable):
    """A flat BVH over every triangle of every part."""

    var corners: List[Vec3]  # a, b, c per triangle
    var lo: List[Vec3]
    var hi: List[Vec3]
    var centre: List[Vec3]
    var nodes: List[Float32]
    var order: List[Int]

    def __init__(out self, track: Track):
        self.corners = List[Vec3]()
        self.lo = List[Vec3]()
        self.hi = List[Vec3]()
        self.centre = List[Vec3]()
        for i in range(len(track.parts)):
            ref part = track.parts[i]
            for face in range(part.count()):
                var a = part.corner(face, 0)
                var b = part.corner(face, 1)
                var c = part.corner(face, 2)
                self.corners.append(a)
                self.corners.append(b)
                self.corners.append(c)
                self.lo.append(min(min(a, b), c))
                self.hi.append(max(max(a, b), c))
                self.centre.append((min(min(a, b), c) + max(max(a, b), c)) / 2)
        self.nodes = List[Float32]()
        self.order = List[Int]()
        var indices = List[Int](capacity=len(self.centre))
        for i in range(len(self.centre)):
            indices.append(i)
        self.split(indices, 0, len(indices))

    def split(mut self, mut indices: List[Int], lo: Int, hi: Int):
        var node = len(self.nodes)
        for _ in range(9):
            self.nodes.append(0)
        var first = 0
        var count = 0
        var box_lo = vec3(1e300, 1e300, 1e300)
        var box_hi = -box_lo
        for i in range(lo, hi):
            box_lo = min(box_lo, self.lo[indices[i]])
            box_hi = max(box_hi, self.hi[indices[i]])
        if hi - lo <= LEAF:
            first = len(self.order)
            count = hi - lo
            for i in range(lo, hi):
                self.order.append(indices[i])
        else:
            var extent = vec3(0, 0, 0)
            var c_lo = vec3(1e300, 1e300, 1e300)
            var c_hi = -c_lo
            for i in range(lo, hi):
                c_lo = min(c_lo, self.centre[indices[i]])
                c_hi = max(c_hi, self.centre[indices[i]])
            extent = c_hi - c_lo
            var axis = 0
            if extent[1] > extent[axis]:
                axis = 1
            if extent[2] > extent[axis]:
                axis = 2
            var keys = List[Float64](capacity=len(self.centre))
            for i in range(len(self.centre)):
                keys.append(self.centre[i][axis])
            var middle = lo + (hi - lo) // 2
            nth_element(indices, lo, hi, middle, keys)
            self.split(indices, lo, middle)
            self.split(indices, middle, hi)
        for axis in range(3):
            self.nodes[node + axis] = Float32(box_lo[axis])
            self.nodes[node + 3 + axis] = Float32(box_hi[axis])
        self.nodes[node + 6] = Float32(len(self.nodes) // 9)
        self.nodes[node + 7] = Float32(first)
        self.nodes[node + 8] = Float32(count)


def compile_track(
    track: Track, resolution: Float64 = 0.025
) raises -> List[Float32]:
    if not (resolution >= 0.005 and resolution <= 0.1):
        raise Error("resolution must be between 0.005 and 0.1 metres")
    var lo, hi = track.bounds()
    var grid = Grid(lo, hi, resolution)
    var edges = Dict[String, Int]()
    var road_edges = List[Vec3]()
    for i in range(len(track.parts)):
        ref part = track.parts[i]
        for face in range(part.count()):
            var a = part.corner(face, 0)
            var b = part.corner(face, 1)
            var c = part.corner(face, 2)
            var normal = cross(b - a, c - a)
            if not part.obstacle and normal[2] <= 1e-8:
                continue  # Solid road sides and undersides are not drivable.
            if part.obstacle:
                rasterize_obstacle(grid, a, b, c, normal)
                continue
            var slope = normal[2] / norm(normal)
            if slope < 0.1:
                continue
            if slope < 0.7:
                raise Error("road slope exceeds the supported 45 degrees")
            rasterize_road(grid, a, b, c, normal)
            # Shared edges cancel; the surviving edges bound the road.
            for edge in range(3):
                var p = a if edge == 0 else (b if edge == 1 else c)
                var q = b if edge == 0 else (c if edge == 1 else a)
                var key = edge_key(p, q)
                edges[key] = edges.get(key, 0) + 1
                road_edges.append(p)
                road_edges.append(q)
    # Preserve sub-cell road holes/edges, just as we preserve thin walls:
    # every boundary edge blocks each cell whose square it crosses.
    for i in range(0, len(road_edges), 2):
        if edges[edge_key(road_edges[i], road_edges[i + 1])] == 1:
            rasterize_edge(grid, road_edges[i], road_edges[i + 1])
    var free = 0
    for index in range(grid.nx * grid.ny):
        if grid.road[index] and not grid.blocked[index]:
            free += 1
    if free == 0:
        raise Error("map has no drivable surface")
    var distance = distance_transform(grid)
    # Uniform arc-length samples bound the runtime's local projection search,
    # independent of how densely the source route was authored.
    var samples = max(4, Int(ceil(track.route.total / 0.15)) + 1)
    var distances = List[Float64](capacity=samples)
    for i in range(samples - 1 if track.route.closed else samples):
        distances.append(track.route.total * Float64(i) / Float64(samples - 1))
    var route = track.route.sample(distances)
    var count = route.segments()
    var segments = List[Float32](capacity=count * 10)
    var spawns = List[Float32]()
    for i in range(count):
        var p = route.points[i]
        var d = route.delta[i]
        segments.append(Float32(p[0]))
        segments.append(Float32(p[1]))
        segments.append(Float32(p[2]))
        segments.append(Float32(d[0]))
        segments.append(Float32(d[1]))
        segments.append(Float32(d[2]))
        segments.append(Float32(route.lengths[i]))
        segments.append(Float32(route.distance[i]))
        segments.append(Float32(route.half_width[i]))
        segments.append(Float32(atan2(d[1], d[0])))
        if spawn_ok(grid, distance, route, i):
            spawns.append(Float32(i))
    if len(spawns) == 0:
        raise Error(
            "route has no spawn with vehicle clearance and road support"
        )
    var tree = Tree(track)
    var result = List[Float32](capacity=HEADER + 4 * grid.nx * grid.ny)
    result.append(MAGIC)
    result.append(VERSION)
    result.append(Float32(grid.nx))
    result.append(Float32(grid.ny))
    result.append(Float32(count))
    result.append(Float32(1 if route.closed else 0))
    result.append(Float32(grid.origin_x))
    result.append(Float32(grid.origin_y))
    result.append(Float32(resolution))
    result.append(Float32(route.total))
    result.append(Float32(len(spawns)))
    result.append(Float32(len(tree.nodes) // 9))
    result.append(Float32(len(tree.order)))
    for _ in range(HEADER - 13):
        result.append(0)
    result.extend(distance^)
    result.extend(Span(grid.height))
    result.extend(Span(grid.sx))
    result.extend(Span(grid.sy))
    result.extend(segments^)
    result.extend(spawns^)
    result.extend(Span(tree.nodes))
    for face in tree.order:
        var a = tree.corners[3 * face]
        var e1 = tree.corners[3 * face + 1] - a
        var e2 = tree.corners[3 * face + 2] - a
        for axis in range(3):
            result.append(Float32(a[axis]))
        for axis in range(3):
            result.append(Float32(e1[axis]))
        for axis in range(3):
            result.append(Float32(e2[axis]))
    return result^


def spawn_ok(
    grid: Grid, distance: List[Float32], route: Route, i: Int
) raises -> Bool:
    """Validate the same three chassis circles as the runtime, including its
    conservative lookup and rounding. A spawn must survive its first step."""
    var centre = route.points[i] + route.delta[i] * 0.5
    var d = route.delta[i]
    var horizontal = sqrt(d[0] * d[0] + d[1] * d[1])
    if horizontal < 1e-7:
        raise Error(
            "route contains a vertical or degenerate horizontal segment"
        )
    var px = (centre[0] - grid.origin_x) / grid.resolution
    var py = (centre[1] - grid.origin_y) / grid.resolution
    var ix = Int(floor(px + 0.5))
    var iy = Int(floor(py + 0.5))
    if ix < 0 or ix >= grid.nx or iy < 0 or iy >= grid.ny:
        return False
    var index = iy * grid.nx + ix
    var offset_x = centre[0] - grid.origin_x - Float64(ix) * grid.resolution
    var offset_y = centre[1] - grid.origin_y - Float64(iy) * grid.resolution
    var ground = (
        Float64(grid.height[index])
        + Float64(grid.sx[index]) * offset_x
        + Float64(grid.sy[index]) * offset_y
    )
    if abs(ground - centre[2]) >= 0.05 or route.half_width[i] <= 0.2:
        return False
    if (
        not route.closed
        and route.distance[i] + 0.5 * route.lengths[i] >= route.total - 0.3
    ):
        return False
    var fx = d[0] / horizontal
    var fy = d[1] / horizontal
    var tilt = Float64(grid.sx[index]) * fx + Float64(grid.sy[index]) * fy
    fx /= sqrt(1 + tilt * tilt)
    fy /= sqrt(1 + tilt * tilt)
    for end in range(-1, 2):
        var qx = (
            centre[0] + Float64(end) * 0.15 * fx - grid.origin_x
        ) / grid.resolution
        var qy = (
            centre[1] + Float64(end) * 0.15 * fy - grid.origin_y
        ) / grid.resolution
        var jx = Int(floor(qx + 0.5))
        var jy = Int(floor(qy + 0.5))
        if jx < 0 or jx >= grid.nx or jy < 0 or jy >= grid.ny:
            return False
        var clearance = (
            Float64(distance[jy * grid.nx + jx])
            - sqrt(
                (qx - Float64(jx)) * (qx - Float64(jx))
                + (qy - Float64(jy)) * (qy - Float64(jy))
            )
            * grid.resolution
        )
        if clearance <= 0.25:
            return False
    return True


# ===-------------------------------------------------------------------=== #
# Map files: built-in demos, YAML with OBJ meshes + a text route, or images.
# ===-------------------------------------------------------------------=== #


def parse_yaml(path: String) raises -> Dict[String, String]:
    """Flat `key: value` pairs; `parts:` list items join as `mesh|obstacle;...`.
    """
    var result = Dict[String, String]()
    var parts = String()
    var mesh = String()
    var obstacle = String("false")
    var in_parts = False
    for raw in Path(path).read_text().splitlines():
        var line = String(raw)
        var hash = line.find("#")
        if hash >= 0:
            var cut = String(line[byte=0:hash])
            line = cut^
        if line.strip().byte_length() == 0:
            continue
        var indented = line.startswith(" ") or line.startswith("-")
        if in_parts and indented:
            var item = String(line.strip())
            if item.startswith("-"):
                if mesh.byte_length() > 0:
                    parts += mesh + "|" + obstacle + ";"
                mesh = String()
                obstacle = String("false")
                var rest = String(item[byte=1:].strip())
                item = rest^
            var colon = item.find(":")
            if colon >= 0:
                var key = String(item[byte=0:colon].strip())
                var value = String(item[byte = colon + 1 :].strip())
                if key == "mesh":
                    mesh = value
                elif key == "obstacle":
                    obstacle = value
            continue
        if in_parts:
            if mesh.byte_length() > 0:
                parts += mesh + "|" + obstacle + ";"
            mesh = String()
            in_parts = False
        var colon = line.find(":")
        if colon < 0:
            continue
        var key = String(line[byte=0:colon].strip())
        var value = String(line[byte = colon + 1 :].strip())
        if key == "parts":
            in_parts = True
        else:
            result[key] = value
    if in_parts and mesh.byte_length() > 0:
        parts += mesh + "|" + obstacle + ";"
    if parts.byte_length() > 0:
        result["parts"] = parts
    return result^


def directory(path: String) -> String:
    var slash = path.rfind("/")
    return String(path[byte = 0 : slash + 1]) if slash >= 0 else String()


def stem(path: String) -> String:
    var pieces = path.split("/")
    var name = String(pieces[len(pieces) - 1])
    var dot = name.rfind(".")
    return String(name[byte=0:dot]) if dot > 0 else name


def load_track(path: String) raises -> Track:
    if path == "flat" or path == "ramp" or path == "bank":
        return demo_track(path)
    var meta = parse_yaml(path)
    var folder = directory(path)
    if "demo" in meta:
        return demo_track(meta["demo"])
    if "image" in meta:
        return image_track(path, meta)
    if "route" not in meta:
        raise Error("map YAML needs `route:` (and `mesh:` or `parts:`)")
    var closed = True
    if "closed" in meta:
        closed = meta["closed"].lower() != "false"
    var route = load_route(folder + meta["route"], closed)
    var parts = List[Mesh]()
    var specs = String()
    if "parts" in meta:
        specs = meta["parts"]
    elif "mesh" in meta:
        specs = meta["mesh"] + "|false;"
    else:
        raise Error("map YAML needs `mesh:` or `parts:`")
    for spec in specs.split(";"):
        if spec.byte_length() == 0:
            continue
        var fields = spec.split("|")
        parts.append(
            load_obj(
                folder + String(fields[0]), String(fields[1]).lower() == "true"
            )
        )
    return Track(parts^, route^, stem(path))
