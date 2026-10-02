"""ROS occupancy images (PGM or PNG + YAML) as closed-loop tracks.

Pixels at or above `FREE` are drivable floor; every free/occupied pixel edge
becomes a 0.6 m wall. The route is the longest skeleton loop through the
free space near the world origin, Savitzky-Golay smoothed and resampled.
PNG files are inflated with the system zlib.
"""
from std.math import sqrt, floor
from std.collections import Dict, Deque
from std.ffi import OwnedDLHandle, c_int
from std.pathlib import Path
from std.sys.info import CompilationTarget
from .track import Track, Mesh, Route, Vec3, vec3

comptime FREE = 230  # image value at/above which a pixel is drivable
comptime HALF_WINDOW = 25  # Savitzky-Golay half window (51 waypoints)
comptime WALL_HEIGHT = 0.6
comptime FAR = Float64(1e20)


struct Image(Movable):
    var width: Int
    var height: Int
    var pixels: List[UInt8]  # 8-bit grey, row-major from the top

    def __init__(
        out self, width: Int, height: Int, var pixels: List[UInt8]
    ) raises:
        if width < 2 or height < 2 or len(pixels) != width * height:
            raise Error("image dimensions do not match its data")
        self.width = width
        self.height = height
        self.pixels = pixels^


def load_image(path: String) raises -> Image:
    var raw = Path(path).read_bytes()
    if len(raw) >= 8 and raw[0] == 0x89 and raw[1] == 0x50:
        return decode_png(raw)
    if len(raw) >= 2 and raw[0] == 0x50 and (raw[1] == 0x35 or raw[1] == 0x32):
        return decode_pgm(raw)
    raise Error("unsupported image format (use PGM or PNG): " + path)


def decode_pgm(raw: List[UInt8]) raises -> Image:
    """Binary (P5) or ASCII (P2) portable greymaps up to 255 levels."""
    var binary = raw[1] == 0x35
    var fields = List[Int]()
    var i = 2
    while len(fields) < 3:
        while i < len(raw) and (
            raw[i] == 0x20 or raw[i] == 0x0A or raw[i] == 0x0D or raw[i] == 0x09
        ):
            i += 1
        if i < len(raw) and raw[i] == 0x23:
            while i < len(raw) and raw[i] != 0x0A:
                i += 1
            continue
        var value = 0
        var digits = 0
        while i < len(raw) and raw[i] >= 0x30 and raw[i] <= 0x39:
            value = value * 10 + Int(raw[i]) - 0x30
            i += 1
            digits += 1
        if digits == 0:
            raise Error("malformed PGM header")
        fields.append(value)
    var width = fields[0]
    var height = fields[1]
    if fields[2] < 1 or fields[2] > 255:
        raise Error("PGM must use at most 255 grey levels")
    var pixels = List[UInt8](capacity=width * height)
    if binary:
        i += 1  # single whitespace after the header
        if len(raw) < i + width * height:
            raise Error("truncated PGM")
        for k in range(width * height):
            pixels.append(UInt8(Int(raw[i + k]) * 255 // fields[2]))
    else:
        for _ in range(width * height):
            while i < len(raw) and not (raw[i] >= 0x30 and raw[i] <= 0x39):
                i += 1
            var value = 0
            while i < len(raw) and raw[i] >= 0x30 and raw[i] <= 0x39:
                value = value * 10 + Int(raw[i]) - 0x30
                i += 1
            pixels.append(UInt8(value * 255 // fields[2]))
    return Image(width, height, pixels^)


@inline(.always)
def be32(raw: List[UInt8], at: Int) -> Int:
    return (
        (Int(raw[at]) << 24)
        | (Int(raw[at + 1]) << 16)
        | (Int(raw[at + 2]) << 8)
        | Int(raw[at + 3])
    )


def decode_png(raw: List[UInt8]) raises -> Image:
    """Non-interlaced PNG of any colour type and bit depth, to 8-bit grey."""
    var at = 8
    var width = 0
    var height = 0
    var depth = 0
    var colour = 0
    var palette = List[UInt8]()
    var deflated = List[UInt8]()
    while at + 8 <= len(raw):
        var length = be32(raw, at)
        var kind = String(
            StringSlice(unsafe_from_utf8=Span(raw)[at + 4 : at + 8])
        )
        var body = at + 8
        if body + length + 4 > len(raw):
            raise Error("truncated PNG")
        if kind == "IHDR":
            width = be32(raw, body)
            height = be32(raw, body + 4)
            depth = Int(raw[body + 8])
            colour = Int(raw[body + 9])
            if raw[body + 12] != 0:
                raise Error("interlaced PNG files are not supported")
        elif kind == "PLTE":
            for i in range(length):
                palette.append(raw[body + i])
        elif kind == "IDAT":
            for i in range(length):
                deflated.append(raw[body + i])
        elif kind == "IEND":
            break
        at = body + length + 4
    if width == 0 or height == 0 or len(deflated) == 0:
        raise Error("PNG is missing its header or image data")
    var channels = 1
    if colour == 2:
        channels = 3
    elif colour == 4:
        channels = 2
    elif colour == 6:
        channels = 4
    elif colour != 0 and colour != 3:
        raise Error("unsupported PNG colour type")
    var stride = (width * channels * depth + 7) // 8
    var bpp = max(1, channels * depth // 8)
    var expected = height * (stride + 1)
    var inflated = List[UInt8](length=expected, fill=0)
    var size = UInt64(expected)
    var zlib = OwnedDLHandle(
        "libz.1.dylib" if CompilationTarget.is_macos() else "libz.so.1"
    )
    var uncompress = zlib.get_function[c_int]("uncompress")
    if (
        uncompress(
            inflated.unsafe_ptr(),
            Pointer(to=size),
            deflated.unsafe_ptr(),
            UInt64(len(deflated)),
        )
        != 0
        or Int(size) != expected
    ):
        raise Error("PNG image data failed to inflate")
    # Undo the per-row filters in place (None, Sub, Up, Average, Paeth).
    var previous = List[UInt8](length=stride, fill=0)
    var pixels = List[UInt8](capacity=width * height)
    for row in range(height):
        var start = row * (stride + 1)
        var kind = inflated[start]
        var line = start + 1
        for i in range(stride):
            var a = Int(inflated[line + i - bpp]) if i >= bpp else 0
            var b = Int(previous[i])
            var c = Int(previous[i - bpp]) if i >= bpp else 0
            var x = Int(inflated[line + i])
            var value = x
            if kind == 1:
                value = x + a
            elif kind == 2:
                value = x + b
            elif kind == 3:
                value = x + (a + b) // 2
            elif kind == 4:
                var p = a + b - c
                var pa = abs(p - a)
                var pb = abs(p - b)
                var pc = abs(p - c)
                value = x + (
                    a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                )
            elif kind != 0:
                raise Error("unknown PNG filter")
            inflated[line + i] = UInt8(value & 255)
        for i in range(stride):
            previous[i] = inflated[line + i]
        for col in range(width):
            pixels.append(
                grey(inflated, line, col, channels, depth, colour, palette)
            )
    return Image(width, height, pixels^)


@inline(.always)
def luma(r: Int, g: Int, b: Int) -> UInt8:
    """ITU-R 601 grey, rounded exactly like PIL's `convert("L")`."""
    return UInt8((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16)


def grey(
    data: List[UInt8],
    line: Int,
    col: Int,
    channels: Int,
    depth: Int,
    colour: Int,
    palette: List[UInt8],
) raises -> UInt8:
    """One pixel of a decoded PNG row as 8-bit grey."""
    var samples = SIMD[DType.int32, 4](0)
    for channel in range(channels):
        var index = col * channels + channel
        var value: Int
        if depth == 8:
            value = Int(data[line + index])
        elif depth == 16:
            value = Int(data[line + index * 2])
        else:
            var bit = index * depth
            value = (Int(data[line + bit // 8]) >> (8 - depth - bit % 8)) & (
                (1 << depth) - 1
            )
            if colour != 3:
                value = value * 255 // ((1 << depth) - 1)
        samples[channel] = Int32(value)
    if colour == 3:
        var entry = Int(samples[0]) * 3
        if entry + 2 >= len(palette):
            raise Error("PNG palette index out of range")
        return luma(
            Int(palette[entry]),
            Int(palette[entry + 1]),
            Int(palette[entry + 2]),
        )
    if colour == 2 or colour == 6:
        return luma(Int(samples[0]), Int(samples[1]), Int(samples[2]))
    return UInt8(Int(samples[0]))


# ===-------------------------------------------------------------------=== #
# Free-space analysis: distance to walls, skeleton, and the centreline loop.
# ===-------------------------------------------------------------------=== #


def squared_distance(
    free: List[Bool], width: Int, height: Int
) -> List[Float64]:
    """Squared pixel distance from each free pixel to the nearest other pixel
    (Felzenszwalb & Huttenlocher, columns then rows)."""
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


def skeletonize(mut skeleton: List[Bool], width: Int, height: Int):
    """Zhang-Suen thinning to a one-pixel-wide, 8-connected skeleton."""
    var remove = List[Int]()
    while True:
        var changed = False
        for pass_index in range(2):
            remove.clear()
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    var i = y * width + x
                    if not skeleton[i]:
                        continue
                    var p2 = skeleton[i - width]
                    var p3 = skeleton[i - width + 1]
                    var p4 = skeleton[i + 1]
                    var p5 = skeleton[i + width + 1]
                    var p6 = skeleton[i + width]
                    var p7 = skeleton[i + width - 1]
                    var p8 = skeleton[i - 1]
                    var p9 = skeleton[i - width - 1]
                    var count = (
                        Int(p2)
                        + Int(p3)
                        + Int(p4)
                        + Int(p5)
                        + Int(p6)
                        + Int(p7)
                        + Int(p8)
                        + Int(p9)
                    )
                    if count < 2 or count > 6:
                        continue
                    var transitions = (
                        Int(not p2 and p3)
                        + Int(not p3 and p4)
                        + Int(not p4 and p5)
                        + Int(not p5 and p6)
                        + Int(not p6 and p7)
                        + Int(not p7 and p8)
                        + Int(not p8 and p9)
                        + Int(not p9 and p2)
                    )
                    if transitions != 1:
                        continue
                    var first = (p2 and p4 and p6) if pass_index == 0 else (
                        p2 and p4 and p8
                    )
                    var second = (p4 and p6 and p8) if pass_index == 0 else (
                        p2 and p6 and p8
                    )
                    if first or second:
                        continue
                    remove.append(i)
            for i in remove:
                skeleton[i] = False
                changed = True
        if not changed:
            return


def neighbours(
    skeleton: List[Bool], width: Int, height: Int, i: Int
) -> List[Int]:
    var result = List[Int](capacity=8)
    var y = i // width
    var x = i % width
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            var ny = y + dy
            var nx = x + dx
            if (
                ny >= 0
                and ny < height
                and nx >= 0
                and nx < width
                and skeleton[ny * width + nx]
            ):
                result.append(ny * width + nx)
    return result^


def prune_spurs(mut skeleton: List[Bool], width: Int, height: Int):
    """Erode every branch with a free end until only loops remain."""
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


def largest_component(mut skeleton: List[Bool], width: Int, height: Int) raises:
    var label = List[Int](length=width * height, fill=0)
    var sizes = List[Int]()
    sizes.append(0)
    var queue = Deque[Int]()
    for seed in range(width * height):
        if not skeleton[seed] or label[seed] != 0:
            continue
        var current = len(sizes)
        sizes.append(0)
        label[seed] = current
        queue.append(seed)
        while len(queue) > 0:
            var i = queue.popleft()
            sizes[current] += 1
            for j in neighbours(skeleton, width, height, i):
                if label[j] == 0:
                    label[j] = current
                    queue.append(j)
    if len(sizes) == 1:
        raise Error("empty skeleton after pruning; is the track a closed loop?")
    var best = 1
    for c in range(2, len(sizes)):
        if sizes[c] > sizes[best]:
            best = c
    for i in range(width * height):
        skeleton[i] = label[i] == best


def shortest_path(
    skeleton: List[Bool],
    width: Int,
    height: Int,
    start: Int,
    source: Int,
    target: Int,
) raises -> List[Int]:
    """BFS path source -> target avoiding `start`, or empty if unreachable."""
    var parent = Dict[Int, Int]()
    parent[source] = source
    var queue = Deque[Int]()
    queue.append(source)
    while len(queue) > 0 and target not in parent:
        var u = queue.popleft()
        for v in neighbours(skeleton, width, height, u):
            if v != start and v not in parent:
                parent[v] = u
                queue.append(v)
    var path = List[Int]()
    if target not in parent:
        return path^
    var node = target
    while node != source:
        path.append(node)
        node = parent[node]
    path.append(source)
    return path^


def longest_loop(
    skeleton: List[Bool],
    width: Int,
    height: Int,
    origin_row: Float64,
    origin_col: Float64,
) raises -> List[Int]:
    """Longest closed pixel loop through skeleton points near the world origin.

    Trying several seeds and every pair of a seed's neighbours keeps this
    robust when a seed lands on a junction or articulation point."""
    var candidates = List[Int]()
    var keys = List[Float64]()
    for i in range(width * height):
        if skeleton[i]:
            var dy = Float64(i // width) - origin_row
            var dx = Float64(i % width) - origin_col
            candidates.append(i)
            keys.append(dy * dy + dx * dx)
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
                var path = shortest_path(
                    skeleton, width, height, start, around[a], around[b]
                )
                if len(path) > 0 and len(path) + 1 > len(best):
                    best.clear()
                    best.append(start)
                    for k in range(len(path) - 1, -1, -1):
                        best.append(path[k])
    if len(best) == 0:
        raise Error("could not extract a closed centerline loop")
    return best^


def smoothing_weight(offset: Int) -> Float64:
    """Savitzky-Golay (cubic, window 2m+1) smoothing weight at `offset`."""
    var m = Float64(HALF_WINDOW)
    var i = Float64(offset)
    return (3 * (3 * m * m + 3 * m - 1) - 15 * i * i) / (
        (2 * m - 1) * (2 * m + 1) * (2 * m + 3)
    )


def image_track(path: String, meta: Dict[String, String]) raises -> Track:
    var folder = String()
    var slash = path.rfind("/")
    if slash >= 0:
        folder = String(path[byte = 0 : slash + 1])
    var image = load_image(folder + meta["image"])
    var resolution = Float64(meta["resolution"])
    var origin = (
        meta["origin"]
        .replace("[", " ")
        .replace("]", " ")
        .replace(",", " ")
        .split()
    )
    if len(origin) < 2:
        raise Error("map YAML origin must be [x, y, yaw]")
    var ox = Float64(String(origin[0]))
    var oy = Float64(String(origin[1]))
    var w = image.width
    var h = image.height
    var free = List[Bool](capacity=w * h)
    for value in image.pixels:
        free.append(Int(value) >= FREE)
    var squared = squared_distance(free, w, h)
    var skeleton = free.copy()
    skeletonize(skeleton, w, h)
    prune_spurs(skeleton, w, h)
    largest_component(skeleton, w, h)
    var loop = longest_loop(
        skeleton,
        w,
        h,
        Float64(h - 1) - (0 - oy) / resolution,
        (0 - ox) / resolution,
    )
    # World coordinates, then circular Savitzky-Golay smoothing.
    var count = len(loop)
    var raw_x = List[Float64](capacity=count)
    var raw_y = List[Float64](capacity=count)
    for i in loop:
        raw_x.append(ox + Float64(i % w) * resolution)
        raw_y.append(oy + Float64(h - 1 - i // w) * resolution)
    var xs = List[Float64](capacity=count)
    var ys = List[Float64](capacity=count)
    for k in range(count):
        var sx: Float64 = 0
        var sy: Float64 = 0
        for offset in range(-HALF_WINDOW, HALF_WINDOW + 1):
            var weight = smoothing_weight(offset)
            var j = ((k + offset) % count + count) % count
            sx += weight * raw_x[j]
            sy += weight * raw_y[j]
        xs.append(sx)
        ys.append(sy)
    # Resample the closed loop every 0.15 m.
    var cumulative = List[Float64](capacity=count + 1)
    cumulative.append(0)
    for k in range(count):
        var j = (k + 1) % count
        cumulative.append(
            cumulative[k] + sqrt((xs[j] - xs[k]) ** 2 + (ys[j] - ys[k]) ** 2)
        )
    var total = cumulative[count]
    var samples = max(3, Int(total / 0.15))
    var points = List[Vec3](capacity=samples)
    var widths = List[Float64](capacity=samples)
    var segment = 0
    for n in range(samples):
        var s = total * Float64(n) / Float64(samples)
        while segment + 1 < count and cumulative[segment + 1] < s:
            segment += 1
        var j = (segment + 1) % count
        var span = cumulative[segment + 1] - cumulative[segment]
        var t = (s - cumulative[segment]) / span if span > 0 else 0.0
        var x = xs[segment] + (xs[j] - xs[segment]) * t
        var y = ys[segment] + (ys[j] - ys[segment]) * t
        points.append(vec3(x, y, 0))
        var col = min(max(Int((x - ox) / resolution), 0), w - 1)
        var row = min(
            max(Int(Float64(h - 1) - (y - oy) / resolution), 0), h - 1
        )
        widths.append(max(sqrt(squared[row * w + col]) * resolution, 0.1))
    var route = Route(points^, widths^, List[Vec3](), True)
    # Floor quad and one 0.6 m wall quad per free/occupied pixel edge.
    var x0 = ox - resolution / 2
    var y0 = oy - resolution / 2
    var x1 = x0 + Float64(w) * resolution
    var y1 = y0 + Float64(h) * resolution
    var floor_vertices = List[Vec3]()
    floor_vertices.append(vec3(x0, y0, 0))
    floor_vertices.append(vec3(x1, y0, 0))
    floor_vertices.append(vec3(x1, y1, 0))
    floor_vertices.append(vec3(x0, y1, 0))
    var floor_faces = List[Int]()
    for index in [0, 1, 2, 0, 2, 3]:
        floor_faces.append(index)
    var vertices = List[Vec3]()
    var faces = List[Int]()
    for r in range(h):
        for c in range(w):
            if not free[r * w + c]:
                continue
            for direction in range(4):
                var dr = 0
                var dc = 0
                if direction == 0:
                    dc = 1
                elif direction == 1:
                    dc = -1
                elif direction == 2:
                    dr = 1
                else:
                    dr = -1
                var nr = r + dr
                var nc = c + dc
                var open = (
                    nr >= 0
                    and nr < h
                    and nc >= 0
                    and nc < w
                    and free[nr * w + nc]
                )
                if open:
                    continue
                var x = ox + (Float64(c) + Float64(dc) / 2) * resolution
                var y = oy + (Float64(h - 1 - r) - Float64(dr) / 2) * resolution
                var tangent = vec3(Float64(dr), Float64(dc), 0) * (
                    resolution / 2
                )
                var a = vec3(x, y, 0) - tangent
                var b = vec3(x, y, 0) + tangent
                var start = len(vertices)
                vertices.append(a)
                vertices.append(b)
                vertices.append(b + vec3(0, 0, WALL_HEIGHT))
                vertices.append(a + vec3(0, 0, WALL_HEIGHT))
                # This winding points into free space for all four directions.
                for index in [0, 2, 1, 0, 3, 2]:
                    faces.append(start + index)
    var parts = List[Mesh]()
    parts.append(Mesh(floor_vertices^, floor_faces^))
    parts.append(Mesh(vertices^, faces^, True))
    var pieces = path.split("/")
    var name = String(pieces[len(pieces) - 1])
    var dot = name.rfind(".")
    if dot > 0:
        var short = String(name[byte=0:dot])
        name = short^
    return Track(parts^, route^, name)
