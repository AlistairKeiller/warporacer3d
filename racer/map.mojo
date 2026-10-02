"""Compiled map as seen by the kernels: grids, route, spawns, and triangles.

`.wrmap` is one float32 array (see compile.mojo for the writer):
  header[16], clearance[ny*nx], height[ny*nx], route segments[9 each],
  spawn segment indices, coarse-cell triangle lists (CSR), triangles (a, b-a, c-a).
"""
from std.math import floor, sqrt
from .device import Mat

comptime MAGIC = 314159
comptime VERSION = 3
comptime HEADER = 16
comptime SEGMENT = 9  # px, py, pz, dx, dy, dz, cumulative distance, half width, heading
comptime WALL = 0.5  # metres; every road boundary edge becomes a wall this tall


@fieldwise_init
struct Surface(TrivialRegisterPassable):
    var clearance: Float32  # signed distance to the nearest wall or road edge
    var height: Float32
    var sx: Float32  # d height / dx
    var sy: Float32


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

    def surface(self, d: Mat[1], x: Float32, y: Float32) -> Surface:
        """Bilinear clearance and height; the slope is the patch gradient."""
        var fx = (x - self.x0) / self.cell
        var fy = (y - self.y0) / self.cell
        var ix = Int(floor(fx))
        var iy = Int(floor(fy))
        if ix < 0 or iy < 0 or ix + 1 >= self.nx or iy + 1 >= self.ny:
            return Surface(-1, 0, 0, 0)
        var tx = fx - Float32(ix)
        var ty = fy - Float32(iy)
        var k = iy * self.nx + ix
        var h00 = d[0, self.height + k]
        var h10 = d[0, self.height + k + 1]
        var h01 = d[0, self.height + k + self.nx]
        var h11 = d[0, self.height + k + self.nx + 1]
        var c00 = d[0, HEADER + k]
        var c10 = d[0, HEADER + k + 1]
        var c01 = d[0, HEADER + k + self.nx]
        var c11 = d[0, HEADER + k + self.nx + 1]
        var w00 = (1 - tx) * (1 - ty)
        var w10 = tx * (1 - ty)
        var w01 = (1 - tx) * ty
        var w11 = tx * ty
        return Surface(
            c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11,
            h00 * w00 + h10 * w10 + h01 * w01 + h11 * w11,
            ((h10 - h00) * (1 - ty) + (h11 - h01) * ty) / self.cell,
            ((h01 - h00) * (1 - tx) + (h11 - h10) * tx) / self.cell,
        )

    @inline(.always)
    def segment(self, d: Mat[1], i: Int, field: Int) -> Float32:
        return d[0, self.route + i * SEGMENT + field]

    def progress(
        self, d: Mat[1], x: Float32, y: Float32, near: Int
    ) -> SIMD[DType.float32, 4]:
        """Project onto the route near segment `near`: (distance along the
        route, segment, lateral offset, half width)."""
        var best: Float32 = 1e30
        var result = SIMD[DType.float32, 4](0)
        for offset in range(-8, 9):
            var i = near + offset
            if self.closed:
                i = (i % self.segments + self.segments) % self.segments
            elif i < 0 or i >= self.segments:
                continue
            var px = self.segment(d, i, 0)
            var py = self.segment(d, i, 1)
            var dx = self.segment(d, i, 3)
            var dy = self.segment(d, i, 4)
            var t = ((x - px) * dx + (y - py) * dy) / (dx * dx + dy * dy)
            t = min(max(t, 0), 1)
            var ex = x - px - t * dx
            var ey = y - py - t * dy
            var dist = ex * ex + ey * ey
            if dist < best:
                best = dist
                var dz = self.segment(d, i, 5)
                result[0] = self.segment(d, i, 6) + t * sqrt(
                    dx * dx + dy * dy + dz * dz
                )
                result[1] = Float32(i)
                result[2] = sqrt(dist)
                result[3] = self.segment(d, i, 7)
        return result

