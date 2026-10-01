"""Compile static flat/ramp maps into the distance field consumed by Mojo.

Python is only used here, before training. The runtime loads one float32 file.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely import Polygon, union_all

from warporacer.track import Route, Track

HEADER = 16
MAGIC = 314159
VERSION = 1


def compile_track(track, resolution=0.025):
    if not np.isfinite(resolution) or not 0.005 <= resolution <= 0.1:
        raise ValueError("resolution must be between 0.005 and 0.1 metres")
    lo, hi = track.bounds
    origin = np.floor((lo[:2] - 1.0) / resolution) * resolution
    shape = np.ceil((hi[:2] + 1.0 - origin) / resolution).astype(int) + 1
    nx, ny = map(int, shape)
    if nx * ny > 16_000_000:
        raise ValueError("map exceeds 16 million cells; increase resolution")
    height = np.zeros((ny, nx), np.float32)
    sx, sy = height.copy(), height.copy()
    road = np.zeros((ny, nx), bool)
    blocked = road.copy()
    road_polygons = []
    for part in track.parts:
        for triangle in part.vertices[part.triangles]:
            a, b, c = triangle.astype(np.float64)
            normal = np.cross(b - a, c - a)
            if not part.obstacle and normal[2] <= 1e-8:
                continue  # Solid road sides and undersides are not drivable.
            start = np.maximum(
                np.floor((triangle[:, :2].min(0) - origin) / resolution).astype(int)
                - 1,
                0,
            )
            stop = np.minimum(
                np.ceil((triangle[:, :2].max(0) - origin) / resolution).astype(int) + 2,
                shape,
            )
            xx, yy = np.meshgrid(
                origin[0] + np.arange(start[0], stop[0]) * resolution,
                origin[1] + np.arange(start[1], stop[1]) * resolution,
            )
            sl = np.s_[start[1] : stop[1], start[0] : stop[0]]
            if part.obstacle:
                # Rasterize projected faces, including zero-width vertical walls.
                # Cell-box overlap is conservative: even sub-cell walls survive.
                inside = np.ones(xx.shape, bool)
                for p, q in ((a, b), (b, c), (c, a)):
                    dx, dy = q[:2] - p[:2]
                    cross = dx * (yy - p[1]) - dy * (xx - p[0])
                    margin = resolution * 0.5 * (abs(dx) + abs(dy))
                    sign = 1 if normal[2] >= 0 else -1
                    inside &= sign * cross >= -margin - 1e-9
                blocked[sl] |= inside
                continue
            if normal[2] / np.linalg.norm(normal) < 0.1:
                continue
            if normal[2] / np.linalg.norm(normal) < 0.7:
                raise ValueError("road slope exceeds the supported 45 degrees")
            inside = np.ones(xx.shape, bool)
            for p, q in ((a, b), (b, c), (c, a)):
                dx, dy = q[:2] - p[:2]
                inside &= dx * (yy - p[1]) - dy * (xx - p[0]) >= -1e-8
            gx, gy = -normal[:2] / normal[2]
            zz = a[2] + gx * (xx - a[0]) + gy * (yy - a[1])
            if np.any(inside & road[sl] & (abs(height[sl] - zz) > 0.01)):
                raise ValueError(
                    "overlapping road heights are unsupported (no bridges)"
                )
            height[sl][inside], sx[sl][inside], sy[sl][inside] = zz[inside], gx, gy
            road[sl] |= inside
            road_polygons.append(Polygon(triangle[:, :2]))
    # Preserve sub-cell road holes/edges, just as we preserve thin walls.
    # Shared triangle edges disappear; actual boundaries occupy every cell
    # whose square intersects them. No generated floor spans a gap.
    footprint = union_all(road_polygons)
    polygons = footprint.geoms if hasattr(footprint, "geoms") else [footprint]
    boundaries = []
    for polygon in polygons:
        for ring in (polygon.exterior, *polygon.interiors):
            vertices = np.asarray(ring.coords)
            boundaries.extend(itertools.pairwise(vertices))
    for a, b in boundaries:
        start = np.maximum(
            np.floor((np.minimum(a[:2], b[:2]) - origin) / resolution).astype(int) - 1,
            0,
        )
        stop = np.minimum(
            np.ceil((np.maximum(a[:2], b[:2]) - origin) / resolution).astype(int) + 2,
            shape,
        )
        xx, yy = np.meshgrid(
            origin[0] + np.arange(start[0], stop[0]) * resolution,
            origin[1] + np.arange(start[1], stop[1]) * resolution,
        )
        dx, dy = b[:2] - a[:2]
        intersect = (
            abs(dx * (yy - a[1]) - dy * (xx - a[0]))
            <= resolution * 0.5 * (abs(dx) + abs(dy)) + 1e-9
        )
        blocked[start[1] : stop[1], start[0] : stop[0]] |= intersect
    free = road & ~blocked
    if not free.any():
        raise ValueError("map has no drivable surface")
    # Distance to occupied cell centres minus a half diagonal is a conservative
    # lower bound on clearance. Runtime subtracts the query's centre offset too.
    distance = distance_transform_edt(free) * resolution - resolution / np.sqrt(2)
    distance[~free] = -resolution
    authored = track.route
    # Uniform arc-length samples bound the runtime's local projection search,
    # independent of how densely the source route was authored.
    distances = np.linspace(
        0, authored.total, max(4, int(np.ceil(authored.total / 0.15)) + 1)
    )
    points = authored.points
    widths = authored.half_width
    if authored.closed:
        distances = distances[:-1]
        points = np.vstack((points, points[0]))
        widths = np.r_[widths, widths[0]]
    route = Route(
        np.column_stack(
            [np.interp(distances, authored.distance, points[:, i]) for i in range(3)]
        ),
        half_width=np.interp(distances, authored.distance, widths),
        closed=authored.closed,
    )
    count = len(route.delta)
    segments = np.column_stack(
        (
            route.points[:count],
            route.delta,
            route.lengths,
            route.distance[:count],
            route.half_width[:count],
            np.arctan2(route.delta[:, 1], route.delta[:, 0]),
        )
    ).astype(np.float32)
    centres = route.points[:count] + 0.5 * route.delta
    cells = np.stack((distance, height, sx, sy)).astype(np.float32)
    pixel = (centres[:, :2] - origin) / resolution
    ix, iy = np.floor(pixel + 0.5).astype(int).T
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy = np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)
    offset = centres[:, :2] - origin - np.column_stack((ix, iy)) * resolution
    ground_z = height[iy, ix] + sx[iy, ix] * offset[:, 0] + sy[iy, ix] * offset[:, 1]
    valid &= abs(ground_z - centres[:, 2]) < 0.05
    valid &= route.half_width[:count] > 0.2
    if not route.closed:
        valid &= route.distance[:count] + 0.5 * route.lengths < route.total - 0.3
    # Validate the same three chassis circles as the runtime, including its
    # conservative lookup and rounding. A spawn must survive its first step.
    horizontal = np.linalg.norm(route.delta[:, :2], axis=1, keepdims=True)
    if np.any(horizontal < 1e-7):
        raise ValueError("route contains a vertical or degenerate horizontal segment")
    forward = route.delta[:, :2] / horizontal
    tilt = sx[iy, ix] * forward[:, 0] + sy[iy, ix] * forward[:, 1]
    forward /= np.sqrt(1 + tilt[:, None] ** 2)
    queries = (
        centres[:, None, :2]
        + np.array([-0.15, 0, 0.15])[None, :, None] * forward[:, None]
    )
    pixel = (queries - origin) / resolution
    index = np.floor(pixel + 0.5).astype(int)
    valid &= ((index >= 0) & (index < shape)).all(axis=(1, 2))
    index = np.clip(index, 0, shape - 1)
    clearance = distance[index[:, :, 1], index[:, :, 0]] - np.linalg.norm(
        (pixel - index) * resolution, axis=2
    )
    valid &= (clearance > 0.25).all(axis=1)
    spawns = np.flatnonzero(valid).astype(np.float32)
    if not len(spawns):
        raise ValueError("route has no spawn with vehicle clearance and road support")
    header = np.zeros(HEADER, np.float32)
    header[:11] = (
        MAGIC,
        VERSION,
        nx,
        ny,
        count,
        route.closed,
        *origin,
        resolution,
        route.total,
        len(spawns),
    )
    return np.concatenate((header, cells.ravel(), segments.ravel(), spawns)).astype(
        "<f4"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resolution", type=float, default=0.025)
    args = parser.parse_args()
    data = compile_track(Track.load(args.map), args.resolution)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(args.output)
    print(
        f"{args.output}: {int(data[2])} × {int(data[3])} cells, {int(data[4])} segments, {data.nbytes / 1e6:.1f} MB"
    )


if __name__ == "__main__":
    main()
