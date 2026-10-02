"""Static triangle maps and authored 3D routes; independent of the physics.

A track is a list of mesh parts (road or obstacle) plus a route of 3D points
with half widths. Built-in demos ("flat", "ramp", "bank") are ribbons around
a circle. YAML maps reference OBJ meshes and a plain-text route, or a ROS
occupancy image (see image.mojo).
"""
from std.math import sqrt, cos, sin, pi
from std.pathlib import Path

comptime Vec3 = SIMD[DType.float64, 4]


@inline(.always)
def vec3(x: Float64, y: Float64, z: Float64) -> Vec3:
    var v = Vec3(0)
    v[0], v[1], v[2] = x, y, z
    return v


@inline(.always)
def cross(a: Vec3, b: Vec3) -> Vec3:
    return vec3(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@inline(.always)
def dot(a: Vec3, b: Vec3) -> Float64:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@inline(.always)
def norm(a: Vec3) -> Float64:
    return sqrt(dot(a, a))


@inline(.always)
def unit(a: Vec3) -> Vec3:
    return a / max(norm(a), 1e-9)


struct Mesh(Movable):
    """Indexed triangles; `obstacle` parts block but are never driven on."""

    var vertices: List[Vec3]
    var triangles: List[Int]  # three vertex indices per face
    var obstacle: Bool

    def __init__(
        out self,
        var vertices: List[Vec3],
        var triangles: List[Int],
        obstacle: Bool = False,
    ) raises:
        if len(vertices) == 0 or len(triangles) == 0 or len(triangles) % 3 != 0:
            raise Error("mesh needs vertices and whole triangles")
        for index in triangles:
            if index < 0 or index >= len(vertices):
                raise Error("mesh contains invalid vertex indices")
        for v in vertices:
            for axis in range(3):
                if not (v[axis] == v[axis]) or abs(v[axis]) > 1e9:
                    raise Error("mesh contains nonfinite vertices")
        for f in range(len(triangles) // 3):
            var a = vertices[triangles[3 * f]]
            var b = vertices[triangles[3 * f + 1]]
            var c = vertices[triangles[3 * f + 2]]
            if norm(cross(b - a, c - a)) < 1e-9:
                raise Error("mesh contains degenerate triangles")
        self.vertices = vertices^
        self.triangles = triangles^
        self.obstacle = obstacle

    def corner(self, face: Int, which: Int) -> Vec3:
        return self.vertices[self.triangles[3 * face + which]]

    def count(self) -> Int:
        return len(self.triangles) // 3


struct Route(Movable):
    """Waypoints with half widths; closed routes wrap from the last to first."""

    var points: List[Vec3]
    var up: List[Vec3]
    var half_width: List[Float64]
    var closed: Bool
    var delta: List[Vec3]
    var lengths: List[Float64]
    var distance: List[Float64]  # cumulative, len(points) + 1 when closed
    var total: Float64

    def __init__(
        out self,
        var points: List[Vec3],
        var half_width: List[Float64],
        var up: List[Vec3],
        closed: Bool,
    ) raises:
        if len(points) < 2 or len(half_width) != len(points):
            raise Error("route needs at least two points with widths")
        var segments = len(points) if closed else len(points) - 1
        self.delta = List[Vec3](capacity=segments)
        self.lengths = List[Float64](capacity=segments)
        self.distance = List[Float64](capacity=segments + 1)
        self.distance.append(0)
        for i in range(segments):
            var d = points[(i + 1) % len(points)] - points[i]
            var length = norm(d)
            if length < 1e-5:
                raise Error("route needs distinct consecutive points")
            self.delta.append(d)
            self.lengths.append(length)
            self.distance.append(self.distance[i] + length)
        for w in half_width:
            if not (w > 0):
                raise Error("route needs positive widths")
        self.total = self.distance[segments]
        if len(up) == 0:
            for _ in range(len(points)):
                up.append(vec3(0, 0, 1))
        if len(up) != len(points):
            raise Error("route up vectors must match the points")
        for i in range(len(points)):
            var tangent = unit(self.delta[min(i, segments - 1)])
            var u = up[i] - tangent * dot(up[i], tangent)
            if norm(u) < 0.9:
                raise Error(
                    "route up vectors must not be parallel to the route"
                )
            up[i] = unit(u)
        self.points = points^
        self.half_width = half_width^
        self.up = up^
        self.closed = closed

    def segments(self) -> Int:
        return len(self.delta)

    def tangent(self, i: Int) -> Vec3:
        return unit(self.delta[min(i, self.segments() - 1)])

    def sample(self, distances: List[Float64]) raises -> Route:
        """Interpolate points and widths at arc lengths along this route."""
        var points = List[Vec3](capacity=len(distances))
        var widths = List[Float64](capacity=len(distances))
        var segment = 0
        for s in distances:
            while (
                segment + 1 < self.segments() and self.distance[segment + 1] < s
            ):
                segment += 1
            var t = (s - self.distance[segment]) / self.lengths[segment]
            t = min(max(t, 0.0), 1.0)
            var next = (segment + 1) % len(self.points)
            points.append(self.points[segment] + self.delta[segment] * t)
            widths.append(
                self.half_width[segment]
                + (self.half_width[next] - self.half_width[segment]) * t
            )
        return Route(points^, widths^, List[Vec3](), self.closed)


struct Track(Movable):
    var parts: List[Mesh]
    var route: Route
    var name: String

    def __init__(
        out self, var parts: List[Mesh], var route: Route, name: String
    ) raises:
        if len(parts) == 0:
            raise Error("track requires at least one mesh part")
        self.parts = parts^
        self.route = route^
        self.name = name

    def bounds(self) -> Tuple[Vec3, Vec3]:
        var lo = vec3(1e300, 1e300, 1e300)
        var hi = -lo
        for i in range(len(self.parts)):
            for v in self.parts[i].vertices:
                lo = min(lo, v)
                hi = max(hi, v)
        return (lo, hi)


def ribbon(route: Route) raises -> Mesh:
    """Two triangles per segment between the left and right route edges."""
    var vertices = List[Vec3](capacity=2 * len(route.points))
    for i in range(len(route.points)):
        var left = (
            unit(cross(route.up[i], route.tangent(i))) * route.half_width[i]
        )
        vertices.append(route.points[i] + left)
        vertices.append(route.points[i] - left)
    var faces = List[Int](capacity=6 * route.segments())
    for i in range(route.segments()):
        var j = (i + 1) % len(route.points)
        faces.append(2 * i)
        faces.append(2 * i + 1)
        faces.append(2 * j + 1)
        faces.append(2 * i)
        faces.append(2 * j + 1)
        faces.append(2 * j)
    return Mesh(vertices^, faces^)


def demo_track(kind: String) raises -> Track:
    """Small, editable examples; no special physics paths for procedural maps.
    """
    if kind != "flat" and kind != "ramp" and kind != "bank":
        raise Error("unknown demo map: " + kind)
    var points = List[Vec3]()
    var up = List[Vec3]()
    var widths = List[Float64]()
    for i in range(128):
        var t = 2 * pi * Float64(i) / 128
        var z = 0.7 * (1 - cos(2 * t)) if kind == "ramp" else 0.0
        points.append(vec3(6 * cos(t), 6 * sin(t), z))
        if kind == "bank":
            up.append(vec3(-0.25 * cos(t), -0.25 * sin(t), 1))
        widths.append(1.3)
    var route = Route(points^, widths^, up^, True)
    var parts = List[Mesh]()
    parts.append(ribbon(route))
    return Track(parts^, route^, kind)


def load_obj(path: String, obstacle: Bool = False) raises -> Mesh:
    """Wavefront OBJ: `v x y z` and `f a b c ...` (polygons fan-triangulated).
    """
    var vertices = List[Vec3]()
    var faces = List[Int]()
    for line in Path(path).read_text().splitlines():
        var words = line.split()
        if len(words) == 0:
            continue
        if words[0] == "v" and len(words) >= 4:
            vertices.append(
                vec3(
                    Float64(String(words[1])),
                    Float64(String(words[2])),
                    Float64(String(words[3])),
                )
            )
        elif words[0] == "f" and len(words) >= 4:
            var polygon = List[Int]()
            for i in range(1, len(words)):
                var token = String(words[i].split("/")[0])
                var index = Int(token)
                polygon.append(
                    index - 1 if index > 0 else len(vertices) + index
                )
            for i in range(1, len(polygon) - 1):
                faces.append(polygon[0])
                faces.append(polygon[i])
                faces.append(polygon[i + 1])
    return Mesh(vertices^, faces^, obstacle)


def load_route(path: String, closed: Bool) raises -> Route:
    """Text route: one `x y z [half_width]` per line (commas allowed)."""
    var points = List[Vec3]()
    var widths = List[Float64]()
    for raw in Path(path).read_text().splitlines():
        var line = String(raw).replace(",", " ")
        var hash = line.find("#")
        if hash >= 0:
            var cut = String(line[byte=0:hash])
            line = cut^
        var words = line.split()
        if len(words) < 3:
            continue
        points.append(
            vec3(
                Float64(String(words[0])),
                Float64(String(words[1])),
                Float64(String(words[2])),
            )
        )
        widths.append(Float64(String(words[3])) if len(words) > 3 else 1.5)
    return Route(points^, widths^, List[Vec3](), closed)
