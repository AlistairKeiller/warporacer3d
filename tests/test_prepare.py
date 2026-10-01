"""Geometry checks for the offline distance-field compiler."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare import compile_track
from track import MeshPart, Route, Track, demo_track, ribbon


class PrepareTests(unittest.TestCase):
    def test_mesh_indices_cannot_wrap_into_valid_triangles(self):
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        for dtype in (np.int64, np.uint64):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                MeshPart(vertices, np.array([[2**32, 1, 2]], dtype=dtype))

    def test_mesh_import_bakes_scene_transforms(self):
        import trimesh

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route = Route([[0, 0, 0], [6, 0, 0]], closed=False)
            np.savez(
                root / "route.npz", points=route.points, half_width=route.half_width
            )
            mesh = ribbon(route)
            np.savez(
                root / "road.npz", vertices=mesh.vertices, triangles=mesh.triangles
            )
            (root / "map.yaml").write_text(
                "mesh: road.npz\nroute: route.npz\nclosed: false\n"
            )
            loaded = Track.load(root / "map.yaml")
            np.testing.assert_array_equal(loaded.parts[0].vertices, mesh.vertices)
            scene = trimesh.Scene()
            transform = np.eye(4)
            transform[2, 3] = 3
            scene.add_geometry(
                trimesh.Trimesh(mesh.vertices, mesh.triangles), transform=transform
            )
            scene.export(root / "road.glb")
            (root / "map.yaml").write_text(
                "mesh: road.glb\nroute: route.npz\nclosed: false\n"
            )
            self.assertAlmostEqual(float(Track.load(root / "map.yaml").bounds[1, 2]), 3)

    def test_demo_maps(self):
        for kind in ("flat", "ramp", "bank"):
            data = compile_track(Track.load(kind), 0.05)
            nx, ny, segments, spawns = map(int, data[[2, 3, 4, 10]])
            self.assertEqual(
                len(data),
                16
                + 4 * nx * ny
                + 10 * segments
                + spawns
                + 9 * sum(map(int, data[11:13])),
            )
            self.assertTrue(np.isfinite(data).all())
            self.assertGreater(spawns, 0)
            fields = data[16 : 16 + 4 * nx * ny].reshape(4, ny, nx)
            if kind != "flat":
                self.assertGreater(abs(fields[2:4]).max(), 0.05)
            else:
                np.testing.assert_array_equal(fields[2:4], 0)

    def test_bridges_rejected(self):
        t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        route = Route(
            np.column_stack([8 * np.sin(t), 4 * np.sin(2 * t), 1 + np.cos(t)]),
            half_width=0.85,
        )
        with self.assertRaisesRegex(ValueError, "overlapping"):
            compile_track(Track([ribbon(route)], route), 0.05)

    def test_invalid_resolution(self):
        for resolution in (0, np.nan, 1):
            with self.assertRaises(ValueError):
                compile_track(demo_track("flat"), resolution)

    def test_route_outside_road_rejected(self):
        road = demo_track("flat")
        road.route = Route(np.array([[100, 100, 0], [101, 100, 0]]), closed=False)
        with self.assertRaisesRegex(ValueError, "spawn"):
            compile_track(road, 0.05)

    def test_subcell_wall_survives(self):
        road = MeshPart(
            np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
        wall = MeshPart(
            np.array([[0.011, -1, 0], [0.011, 1, 0], [0.011, 1, 1], [0.011, -1, 1]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
            obstacle=True,
        )
        track = Track(
            [road, wall], Route(np.array([[-1, -1, 0], [-1, 1, 0]]), closed=False)
        )
        data = compile_track(track, 0.05)
        nx, ny = map(int, data[2:4])
        field = data[16 : 16 + nx * ny].reshape(ny, nx)
        x = round((0.011 - data[6]) / data[8])
        y = round((0 - data[7]) / data[8])
        self.assertLess(field[y, x], 0)


if __name__ == "__main__":
    unittest.main()
