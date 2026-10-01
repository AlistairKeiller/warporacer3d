"""Static triangle maps and authored 3D routes; independent of the physics engine."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from yaml import safe_load


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


@dataclass
class Route:
    points: np.ndarray
    up: np.ndarray | tuple = (0, 0, 1)
    half_width: np.ndarray | float = 1.5
    closed: bool = True

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or len(self.points) < 2:
            raise ValueError("Route points must have shape (N, 3), with N >= 2")
        self.up = np.broadcast_to(self.up, self.points.shape).astype(np.float32).copy()
        self.half_width = (
            np.broadcast_to(self.half_width, (len(self.points),))
            .astype(np.float32)
            .copy()
        )
        end = np.roll(self.points, -1, axis=0) if self.closed else self.points[1:]
        self.delta = end - self.points[: len(end)]
        self.lengths = np.linalg.norm(self.delta, axis=1)
        if not all(
            np.isfinite(a).all() for a in (self.points, self.up, self.half_width)
        ):
            raise ValueError("Route contains non-finite values")
        if np.any(self.lengths < 1e-5) or np.any(self.half_width <= 0):
            raise ValueError(
                "Route needs distinct consecutive points and positive widths"
            )
        tangent = unit(self.delta)
        if not self.closed:
            tangent = np.vstack([tangent, tangent[-1]])
        self.up = unit(
            self.up - tangent * (self.up * tangent).sum(axis=1, keepdims=True)
        )
        if np.any(np.linalg.norm(self.up, axis=1) < 0.9):
            raise ValueError("Route up vectors must not be parallel to the route")
        self.tangent = tangent
        self.distance = np.r_[0, np.cumsum(self.lengths)].astype(np.float32)
        self.total = float(self.distance[-1])


@dataclass
class MeshPart:
    vertices: np.ndarray
    triangles: np.ndarray
    obstacle: bool = False

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float32)
        faces = np.asarray(self.triangles)
        if faces.dtype.kind not in "iu":
            raise ValueError("Triangle indices must be integers")
        faces = faces.reshape(-1, 3)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or not len(faces):
            raise ValueError("Mesh needs vertices (N, 3) and triangles (M, 3)")
        if (
            not np.isfinite(self.vertices).all()
            or faces.min() < 0
            or faces.max() >= len(self.vertices)
            or faces.max() > np.iinfo(np.int32).max
        ):
            raise ValueError("Mesh contains invalid vertices or indices")
        self.triangles = faces.astype(np.int32)
        a, b, c = self.vertices[self.triangles].transpose(1, 0, 2)
        if np.any(np.linalg.norm(np.cross(b - a, c - a), axis=1) < 1e-9):
            raise ValueError("Mesh contains degenerate triangles")


@dataclass
class Track:
    parts: list[MeshPart]
    route: Route
    name: str = "track"

    def __post_init__(self):
        if not self.parts:
            raise ValueError("Track requires at least one mesh part")

    @classmethod
    def load(cls, path):
        if str(path) in ("flat", "ramp", "bank"):
            return demo_track(str(path))
        path = Path(path)
        meta = safe_load(path.read_text())
        if "image" in meta:
            return _image_track(path)
        if "demo" in meta:
            return demo_track(meta["demo"])
        with np.load(path.parent / meta["route"], allow_pickle=False) as data:
            route = Route(
                data["points"],
                data.get("up", (0, 0, 1)),
                data.get("half_width", 1.5),
                meta.get("closed", True),
            )
        files = meta.get("parts")
        if files is None:
            files = [{"mesh": meta["mesh"]}]
        parts = []
        for part in files:
            file = path.parent / part["mesh"]
            if file.suffix == ".npz":
                with np.load(file, allow_pickle=False) as data:
                    vertices, faces = data["vertices"], data["triangles"]
            else:
                import trimesh

                mesh = trimesh.load_scene(
                    file
                ).to_mesh()  # Bake every scene-node transform.
                vertices, faces = mesh.vertices, mesh.faces
            parts.append(MeshPart(vertices, faces, part.get("obstacle", False)))
        return cls(parts, route, path.stem)

    @cached_property
    def bounds(self):
        vertices = np.concatenate([p.vertices for p in self.parts])
        return np.stack([vertices.min(axis=0), vertices.max(axis=0)])


def ribbon(route):
    left = unit(np.cross(route.up, route.tangent)) * route.half_width[:, None]
    vertices = np.stack([route.points + left, route.points - left], axis=1).reshape(
        -1, 3
    )
    faces = []
    for i in range(len(route.lengths)):
        j = (i + 1) % len(route.points)
        faces.extend([(2 * i, 2 * i + 1, 2 * j + 1), (2 * i, 2 * j + 1, 2 * j)])
    return MeshPart(vertices, faces)


def demo_track(kind="ramp"):
    """Small, editable examples; no special physics paths for procedural maps."""
    if kind in ("flat", "ramp", "bank"):
        t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        z = 0.7 * (1 - np.cos(2 * t)) if kind == "ramp" else np.zeros_like(t)
        points = np.column_stack([6 * np.cos(t), 6 * np.sin(t), z])
        up = (
            np.column_stack([-0.25 * np.cos(t), -0.25 * np.sin(t), np.ones_like(t)])
            if kind == "bank"
            else (0, 0, 1)
        )
        route = Route(points, up, 1.3)
    else:
        raise ValueError(f"Unknown demo map: {kind}")
    return Track([ribbon(route)], route, kind)


def _image_track(path):
    from occupancy import OCC_THRESH, ImageMap

    image = ImageMap(path)
    xy = image.centerline
    length = np.r_[
        0, np.cumsum(np.linalg.norm(np.diff(np.vstack([xy, xy[0]]), axis=0), axis=1))
    ]
    s = np.linspace(0, length[-1], max(3, int(length[-1] / 0.15)), endpoint=False)
    points = np.column_stack(
        [np.interp(s, length, np.r_[xy[:, k], xy[0, k]]) for k in range(2)]
        + [np.zeros(len(s))]
    )
    col, row = image.world_to_px(points[:, 0], points[:, 1])
    widths = (
        image.edt[
            np.clip(row.astype(int), 0, image.h - 1),
            np.clip(col.astype(int), 0, image.w - 1),
        ]
        * image.res
    )
    route = Route(points, half_width=np.maximum(widths, 0.1))
    x0, y0 = image.ox - image.res / 2, image.oy - image.res / 2
    x1, y1 = x0 + image.w * image.res, y0 + image.h * image.res
    floor = MeshPart(
        [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)], [(0, 1, 2), (0, 2, 3)]
    )
    free = np.pad(image.image >= OCC_THRESH, 1)
    vertices, faces = [], []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        adjacent = free[1 + dr : 1 + dr + image.h, 1 + dc : 1 + dc + image.w]
        rows, cols = np.nonzero(free[1:-1, 1:-1] & ~adjacent)
        for r, c in zip(rows, cols):
            x, y = (
                image.ox + (c + dc / 2) * image.res,
                image.oy + (image.h - 1 - r - dr / 2) * image.res,
            )
            tangent = np.array([dr, dc, 0]) * image.res / 2
            a = np.array([x, y, 0]) - tangent
            b = np.array([x, y, 0]) + tangent
            start = len(vertices)
            vertices.extend([a, b, b + [0, 0, 0.6], a + [0, 0, 0.6]])
            # This winding points into free space for all four edge directions.
            faces.extend([(start, start + 2, start + 1), (start, start + 3, start + 2)])
    walls = MeshPart(vertices, faces, obstacle=True)
    return Track([floor, walls], route, path.stem)
