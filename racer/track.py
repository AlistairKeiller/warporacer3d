"""Tracks, packed into one float32 array that the simulation, the trainer and the
viewer all read.

A track is a closed route of evenly spaced waypoints (x, y, z, heading) plus
layered grids over the same xy extent: for every layer, clearance (signed metres
to the nearest wall), height, and the nearest waypoint of every cell. One layer
is enough unless the route crosses over itself; then the raised pass is layer 1
and the car picks the drivable layer nearest its own z.

Sources: `flat`, `ramp`, `bank`, `bridge` (built-in ribbons), or a YAML map with
either a ROS occupancy image (SLAM output; free pixels >= 230, optional `ramps`
boxes laid on the floor) or a `route` text file (x y z per line) with `half_width`
and `bank`.

Array layout: header[12] = nx, ny, x0, y0, cell, waypoints, spacing, layers,
steepest slope, 0, 0, 0; then per layer clearance[ny*nx], height[ny*nx],
waypoint[ny*nx]; then route[waypoints*4]. Cell (ix, iy) is centred at
(x0 + ix*cell, y0 + iy*cell); rows run south to north.
"""

from collections import deque
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter, label
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from skimage.morphology import skeletonize

SPACING = 0.15  # metres between waypoints
FREE = 230  # image value at or above which a pixel is drivable
ADJ = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c]


def load(name):
    if name in ("flat", "ramp", "bank", "bridge"):
        t = np.linspace(0, 2 * np.pi, 160, endpoint=False)
        if name == "bridge":  # a figure eight whose second pass crosses 0.9 m above the first
            z = 0.9 * np.clip(np.cos((t - np.pi) * 1.4), 0, 1) ** 2
            return ribbon(np.column_stack([7 * np.sin(t), 3.5 * np.sin(2 * t), z]), 1.1, 0, 0.025)
        z = 0.7 * (1 - np.cos(2 * t)) if name == "ramp" else 0 * t
        return ribbon(np.column_stack([6 * np.cos(t), 6 * np.sin(t), z]), 1.3, 0.25 * (name == "bank"), 0.025)
    meta = yaml.safe_load(Path(name).read_text())
    cell = float(meta.get("resolution", 0.025))
    if "route" in meta:
        route = np.loadtxt(Path(name).parent / meta["route"])[:, :3]
        return ribbon(route, meta.get("half_width", 1.3), meta.get("bank", 0), cell, meta.get("ramps", []))
    image = np.asarray(Image.open(Path(name).parent / meta["image"]).convert("L"))
    free = np.flipud(image >= FREE)  # image rows run from the north
    x0, y0 = meta["origin"][:2]
    loop = longest_loop(skeleton(free), (-y0 / cell, -x0 / cell))
    xy = savgol_filter(loop[:, ::-1] * cell + (x0, y0), 51, 3, axis=0, mode="wrap")
    route = resample(np.column_stack([xy, 0 * xy[:, 0]]))
    waypoint = KDTree(route[:, :2]).query(centres(free.shape, x0, y0, cell))[1].reshape(free.shape)
    clearance = (distance_transform_edt(free) - distance_transform_edt(~free) - 0.5) * cell
    return pack(clearance[None], np.zeros(free.shape)[None], waypoint[None], route, x0, y0, cell, 0, meta.get("ramps", []))


def ribbon(route, half_width, bank, cell, ramps=()):
    """A road `half_width` either side of `route`, rising `bank` metres per metre
    outward. Clearance is the analytic distance to the ribbon's edges. Where the
    route crosses over itself, the upper pass and the whole hump it belongs to
    form layer 1, so every layer is a continuous road."""
    route = resample(route)
    heading = headings(route)
    grade = np.gradient(route[:, 2], arc(route)[:-1])
    lo = route[:, :2].min(0) - 2
    shape = tuple(reversed((np.ceil((route[:, :2].max(0) + 2 - lo) / cell).astype(int) + 1).tolist()))
    xy = centres(shape, *lo, cell)
    tree = KDTree(route[:, :2])
    pairs = []  # (cell, nearest waypoint) for every pass of the route over the cell
    upper = np.zeros(len(route), bool)  # waypoints of a pass that crosses above another
    for k, near in enumerate(tree.query_ball_point(xy, half_width + 0.5, return_sorted=True)):
        if not near:
            continue
        near = np.array(near)  # runs of consecutive waypoint indices are separate passes
        passes = np.split(near, np.nonzero(np.diff(near) > 2)[0] + 1)
        if len(passes) > 1 and near[0] == 0 and near[-1] == len(route) - 1:
            passes[0] = np.concatenate([passes.pop(), passes[0]])
        nearest = [p[np.linalg.norm(route[p, :2] - xy[k], axis=1).argmin()] for p in passes]
        nearest.sort(key=lambda w: route[w, 2])
        if route[nearest[-1], 2] - route[nearest[0], 2] < 0.3:  # same level: one road, the nearest pass
            nearest = [min(nearest, key=lambda w: np.linalg.norm(route[w, :2] - xy[k]))]
        upper[nearest[1:]] = True
        pairs += [(k, w) for w in nearest]
    pairs = np.array(pairs)
    high = route[:, 2] > route[:, 2].min() + 0.1
    while True:  # extend the upper layer along the route while the road stays raised
        grown = upper | ((np.roll(upper, 1) | np.roll(upper, -1)) & high)
        if (grown == upper).all():
            break
        upper = grown
    layers = [pairs[upper[pairs[:, 1]] == l] for l in (False, True) if (upper[pairs[:, 1]] == l).any()]
    clearance = np.full((len(layers),) + shape, -10.0)
    height = np.zeros(clearance.shape)
    waypoint = np.tile(tree.query(xy)[1].reshape(shape), (len(layers), 1, 1))  # off the road: the nearest
    for layer, (cells, w) in enumerate(l.T for l in layers):
        dx, dy = xy[cells, 0] - route[w, 0], xy[cells, 1] - route[w, 1]
        along, left = dx * np.cos(heading[w]) + dy * np.sin(heading[w]), -dx * np.sin(heading[w]) + dy * np.cos(heading[w])
        clearance.reshape(len(layers), -1)[layer, cells] = half_width - abs(left)
        height.reshape(len(layers), -1)[layer, cells] = route[w, 2] + grade[w] * along - bank * left
        waypoint.reshape(len(layers), -1)[layer, cells] = w
    return pack(clearance, height, waypoint, route, *lo, cell, np.hypot(abs(grade).max(), bank), ramps)


def pack(clearance, height, waypoint, route, x0, y0, cell, slope, ramps):
    """`slope` bounds the terrain gradient (the lidar's march step relies on it)."""
    ny, nx = clearance.shape[1:]
    xy = centres((ny, nx), x0, y0, cell)
    for ramp in ramps:  # a box with a sloped top: rise over `rise` metres, flat, then descend
        c, s = np.cos(ramp["heading"]), np.sin(ramp["heading"])
        u = (xy[:, 0] - ramp["center"][0]) * c + (xy[:, 1] - ramp["center"][1]) * s + ramp["length"] / 2
        v = -(xy[:, 0] - ramp["center"][0]) * s + (xy[:, 1] - ramp["center"][1]) * c
        rise = ramp.get("rise", ramp["length"] / 2)
        top = ramp["height"] * np.clip(np.minimum(u, ramp["length"] - u) / rise, 0, 1)
        height += np.where(abs(v) < ramp["width"] / 2, top, 0).reshape(ny, nx)
        slope += ramp["height"] / rise
    for l in range(len(clearance)):  # off the road, heights continue from the nearest road cell
        iy, ix = distance_transform_edt(clearance[l] <= 0, return_indices=True)[1]
        height[l] = gaussian_filter(height[l, iy, ix], 1.5)  # the fill and the blur hide rasterization steps
    header = [nx, ny, x0, y0, cell, len(route), arc(route)[-1] / len(route), len(clearance), slope, 0, 0, 0]
    grids = np.stack([clearance, height, waypoint], 1).ravel()
    return np.concatenate([header, grids, np.column_stack([route, headings(route)]).ravel()]).astype(np.float32)


def centres(shape, x0, y0, cell):
    x, y = np.meshgrid(x0 + cell * np.arange(shape[1]), y0 + cell * np.arange(shape[0]))
    return np.column_stack([x.ravel(), y.ravel()])


def arc(route):
    """Cumulative distance to each point and back around to the first."""
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.roll(route, -1, 0) - route, axis=1))])


def resample(route):
    s = arc(route)
    even = np.linspace(0, s[-1], max(2, round(s[-1] / SPACING)), endpoint=False)
    closed = np.vstack([route, route[:1]])
    return np.column_stack([np.interp(even, s, closed[:, i]) for i in range(3)])


def headings(route):
    d = np.roll(route, -1, 0) - route
    return np.arctan2(d[:, 1], d[:, 0])


def skeleton(free):
    """One-pixel-wide medial axis of the free space with every dead-end branch removed."""
    skel = skeletonize(free)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    while True:
        tips = skel & (convolve(skel.astype(int), kernel, mode="constant") <= 1)
        if not tips.any():
            break
        skel &= ~tips
    labels, n = label(skel, structure=np.ones((3, 3)))
    if n == 0:
        raise RuntimeError("no closed centerline loop in the image")
    return labels == np.bincount(labels.ravel())[1:].argmax() + 1


def neighbours(skel, p):
    h, w = skel.shape
    return [(p[0] + r, p[1] + c) for r, c in ADJ if 0 <= p[0] + r < h and 0 <= p[1] + c < w and skel[p[0] + r, p[1] + c]]


def path(skel, start, src, dst):
    """Shortest 8-connected path src -> dst avoiding start, or None."""
    parent, queue = {src: src}, deque([src])
    while queue and dst not in parent:
        u = queue.popleft()
        for v in neighbours(skel, u):
            if v not in parent and v != start:
                parent[v] = u
                queue.append(v)
    if dst not in parent:
        return None
    walk = [dst]
    while walk[-1] != src:
        walk.append(parent[walk[-1]])
    return walk


def longest_loop(skel, origin):
    """Longest closed pixel loop through one of the 32 skeleton points nearest `origin` (row, col):
    for every pair of a seed's neighbours, the shortest path between them that avoids the seed."""
    points = np.argwhere(skel)
    best = []
    for k in np.argsort(((points - origin) ** 2).sum(1))[:32]:
        start = tuple(points[k].tolist())
        around = neighbours(skel, start)
        for i in range(len(around)):
            for j in range(i + 1, len(around)):
                walk = path(skel, start, around[i], around[j])
                if walk and len(walk) + 1 > len(best):
                    best = [start] + walk[::-1]
    if not best:
        raise RuntimeError("could not extract a closed centerline loop")
    return np.array(best)
