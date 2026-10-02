"""Compiled map as seen by the kernels: grids, route, spawns, and triangles.

`.wrmap` is one float32 array (see compile.mojo for the writer):
  header[16], clearance[ny*nx], height[ny*nx], route segments[9 each],
  spawn segment indices, coarse-cell triangle lists (CSR), triangles (a, b-a, c-a).
"""
from std.math import floor, sqrt, clamp
from .device import Mat

comptime MAGIC = 314159
comptime VERSION = 3
comptime HEADER = 16
comptime WALL = 0.5  # metres; every road boundary edge becomes a wall this tall
# Route segment record: fields of the SEGMENT floats per segment.
comptime SEGMENT = 9
comptime POSITION = 0  # x, y, z of the segment start
comptime DELTA = 3  # dx, dy, dz to the next segment
comptime DISTANCE = 6  # along the route to the segment start
comptime HALF_WIDTH = 7
comptime COURSE = 8  # heading of the segment


# 3-vectors are 4-lane SIMD with a zero fourth lane, in float32 for kernels
# and float64 for the compiler.
@inline(.always)
def cross[dt: DType](a: SIMD[dt, 4], b: SIMD[dt, 4]) -> SIMD[dt, 4]:
    return a.shuffle[1, 2, 0, 3]() * b.shuffle[2, 0, 1, 3]() - a.shuffle[
        2, 0, 1, 3
    ]() * b.shuffle[1, 2, 0, 3]()


@inline(.always)
def dot[dt: DType](a: SIMD[dt, 4], b: SIMD[dt, 4]) -> Scalar[dt]:
    return (a * b).reduce_add()


@inline(.always)
def unit[dt: DType](a: SIMD[dt, 4]) -> SIMD[dt, 4]:
    return a / max(sqrt(dot(a, a)), 1e-9)


@fieldwise_init
struct Surface(TrivialRegisterPassable):
    var clearance: Float32  # signed distance to the nearest wall or road edge
    var height: Float32
    var sx: Float32  # d height / dx
    var sy: Float32


@fieldwise_init
struct Patch(TrivialRegisterPassable):
    """The grid cell square around a point and its bilinear weights."""

    var inside: Bool
    var k: Int  # cell index of the lower-left corner
    var tx: Float32  # position inside the square, 0..1
    var ty: Float32
    var weights: SIMD[DType.float32, 4]  # of corners 00, 10, 01, 11


@fieldwise_init
struct Projection(TrivialRegisterPassable):
    """A point projected onto the route."""

    var distance: Float32  # along the route
    var segment: Float32  # index, as the state stores it
    var offset: Float32  # lateral, unsigned
    var half_width: Float32  # of the road there


struct Map(TrivialRegisterPassable):
    """Header fields and region offsets; the map floats themselves (`d`) are
    passed alongside because kernel-visible structs cannot hold views."""

    var nx: Int
    var ny: Int
    var x0: Float32
    var y0: Float32
    var cell: Float32
    var closed: Bool
    var segments: Int
    var total: Float32
    var spawns: Int
    var cnx: Int
    var cny: Int
    var coarse: Float32
    var height: Int
    var route: Int
    var spawn: Int
    var starts: Int
    var items: Int
    var triangles: Int
    var slope: Float32  # bound on |grad height| over the free road, for the lidar

    def __init__(out self, data: List[Float32]) raises:
        if len(data) < HEADER or data[0] != MAGIC or data[1] != VERSION:
            raise Error("not a version " + String(VERSION) + " .wrmap; re-run prepare")
        self.nx = Int(data[2])
        self.ny = Int(data[3])
        self.x0 = data[4]
        self.y0 = data[5]
        self.cell = data[6]
        self.closed = data[7] > 0
        self.segments = Int(data[8])
        self.total = data[9]
        self.spawns = Int(data[10])
        self.cnx = Int(data[11])
        self.cny = Int(data[12])
        self.coarse = data[13]
        var cells = self.nx * self.ny
        self.height = HEADER + cells
        self.route = self.height + cells
        self.spawn = self.route + SEGMENT * self.segments
        self.starts = self.spawn + self.spawns
        self.items = self.starts + self.cnx * self.cny + 1
        self.triangles = self.items + Int(data[15])
        # The steepest height change between neighbouring free cells, as a
        # 2-D Lipschitz bound (times sqrt 2 for the diagonal) on the terrain.
        var steepest: Float32 = 0
        for k in range(cells):
            if data[HEADER + k] <= 0:
                continue
            for other in [k + 1, k + self.nx]:
                if other < cells and data[HEADER + other] > 0:
                    steepest = max(steepest, abs(data[self.height + other] - data[self.height + k]))
        self.slope = steepest / self.cell * sqrt(Float32(2))

    def patch(self, x: Float32, y: Float32) -> Patch:
        var fx = (x - self.x0) / self.cell
        var fy = (y - self.y0) / self.cell
        var ix = Int(floor(fx))
        var iy = Int(floor(fy))
        var tx = fx - Float32(ix)
        var ty = fy - Float32(iy)
        return Patch(
            ix >= 0 and iy >= 0 and ix + 1 < self.nx and iy + 1 < self.ny,
            iy * self.nx + ix,
            tx,
            ty,
            SIMD[DType.float32, 4]((1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty),
        )

    @inline(.always)
    def corners(self, d: Mat[1], grid: Int, p: Patch) -> SIMD[DType.float32, 4]:
        """The four corners (00, 10, 01, 11) of patch `p` in the grid at `grid`."""
        var k = grid + p.k
        return SIMD[DType.float32, 4](
            d[0, k], d[0, k + 1], d[0, k + self.nx], d[0, k + self.nx + 1]
        )

    def surface(self, d: Mat[1], x: Float32, y: Float32) -> Surface:
        """Bilinear clearance and height; the slope is the patch gradient."""
        var p = self.patch(x, y)
        if not p.inside:
            return Surface(-1, 0, 0, 0)
        var c = self.corners(d, HEADER, p)
        var h = self.corners(d, self.height, p)
        return Surface(
            dot(c, p.weights),
            dot(h, p.weights),
            ((h[1] - h[0]) * (1 - p.ty) + (h[3] - h[2]) * p.ty) / self.cell,
            ((h[2] - h[0]) * (1 - p.tx) + (h[3] - h[1]) * p.tx) / self.cell,
        )

    @inline(.always)
    def sample(self, d: Mat[1], grid: Int, x: Float32, y: Float32, outside: Float32) -> Float32:
        """One grid bilinearly (half the loads of `surface`), `outside` off it."""
        var p = self.patch(x, y)
        return dot(self.corners(d, grid, p), p.weights) if p.inside else outside

    def height_at(self, d: Mat[1], x: Float32, y: Float32) -> Float32:
        return self.sample(d, self.height, x, y, 0)

    def clearance_at(self, d: Mat[1], x: Float32, y: Float32) -> Float32:
        return self.sample(d, HEADER, x, y, -1)

    @inline(.always)
    def segment(self, d: Mat[1], i: Int, field: Int) -> Float32:
        return d[0, self.route + i * SEGMENT + field]

    def progress(self, d: Mat[1], x: Float32, y: Float32, near: Int) -> Projection:
        """Project (x, y) onto the route, searching 8 segments either side of
        segment `near`."""
        var best: Float32 = 1e30
        var result = Projection(0, 0, 0, 0)
        for offset in range(-8, 9):
            var i = near + offset
            if self.closed:
                i = (i % self.segments + self.segments) % self.segments
            elif i < 0 or i >= self.segments:
                continue
            var px = self.segment(d, i, POSITION)
            var py = self.segment(d, i, POSITION + 1)
            var dx = self.segment(d, i, DELTA)
            var dy = self.segment(d, i, DELTA + 1)
            var t = ((x - px) * dx + (y - py) * dy) / (dx * dx + dy * dy)
            t = clamp(t, 0, 1)
            var ex = x - px - t * dx
            var ey = y - py - t * dy
            var dist = ex * ex + ey * ey
            if dist < best:
                best = dist
                var dz = self.segment(d, i, DELTA + 2)
                result.distance = self.segment(d, i, DISTANCE) + t * sqrt(
                    dx * dx + dy * dy + dz * dz
                )
                result.segment = Float32(i)
                result.offset = sqrt(dist)
                result.half_width = self.segment(d, i, HALF_WIDTH)
        return result
