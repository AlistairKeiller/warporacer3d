"""Geometry checks for the offline distance-field compiler."""

import unittest

import numpy as np

from prepare import compile_track
from warporacer.track import MeshPart, Route, Track, demo_track


class PrepareTests(unittest.TestCase):
    def test_flat_and_ramp_roundtrip(self):
        for kind in ("flat", "ramp"):
            data = compile_track(demo_track(kind), 0.05)
            nx, ny, segments, spawns = map(int, data[[2, 3, 4, 10]])
            self.assertEqual(len(data), 16 + 4 * nx * ny + 10 * segments + spawns)
            self.assertTrue(np.isfinite(data).all())
            self.assertGreater(spawns, 0)
            fields = data[16 : 16 + 4 * nx * ny].reshape(4, ny, nx)
            if kind == "ramp":
                self.assertGreater(abs(fields[2:4]).max(), 0.05)
            else:
                np.testing.assert_array_equal(fields[2:4], 0)

    def test_bridges_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlapping"):
            compile_track(demo_track("overpass"), 0.05)

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
