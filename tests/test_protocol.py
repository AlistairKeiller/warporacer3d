"""Drive the native viewer protocol through a thin wall and an open finish."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare import compile_track
from track import MeshPart, Route, Track
from viewer import build


class ProtocolTests(unittest.TestCase):
    def test_wall_lidar_finish_and_reset(self):
        road = MeshPart(
            np.array([[-4, -3, 0], [4, -3, 0], [4, 3, 0], [-4, 3, 0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
        wall = MeshPart(
            np.array(
                [[0.011, -3, 0], [0.011, 3, 0], [0.011, 3, 0.6], [0.011, -3, 0.6]]
            ),
            np.array([[0, 1, 2], [0, 2, 3]]),
            obstacle=True,
        )
        track = Track(
            [road, wall], Route(np.array([[-3, 0, 0], [3, 0, 0]]), closed=False)
        )
        binary, n = build(cpu_only=True), 32
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "wall.wrmap"
            compile_track(track).tofile(asset)
            with subprocess.Popen(
                [str(binary), "serve", str(asset), "cpu", str(n), "1", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            ) as runtime:
                try:
                    while True:
                        line = runtime.stdout.readline()
                        self.assertIsNone(runtime.poll(), "runtime failed to start")
                        if line.startswith("{"):
                            sensor = json.loads(line)
                            if sensor.get("ready"):
                                break
                    state_size, obs_size = sensor["state"], sensor["obs"]

                    def step(throttle):
                        runtime.stdin.write(f"0 {throttle} 0 0\n")
                        runtime.stdin.flush()
                        data = np.array(json.loads(runtime.stdout.readline())).reshape(
                            -1
                        )
                        self.assertTrue(np.isfinite(data).all())
                        return (
                            data[: n * state_size].reshape(n, state_size),
                            data[n * state_size : n * (state_size + obs_size)].reshape(
                                n, obs_size
                            ),
                            data[-n * 2 :].reshape(n, 2),
                        )

                    state, observation, result = step(0)
                    np.testing.assert_array_equal(result[:, 1], 0)
                    np.testing.assert_array_equal(
                        state[:, 9], 1
                    )  # every baked spawn is safe
                    left = state[:, 0] < -0.5
                    self.assertTrue(left.any())
                    beams = sensor["beams"]
                    middle = beams // 2
                    angle = -3 * np.pi / 4 + middle * (3 * np.pi / 2) / (beams - 1)
                    ranges = (
                        observation[:, 8:].reshape(n, sensor["rows"], beams)
                        * sensor["range"]
                    )
                    np.testing.assert_allclose(
                        ranges[left, 1, middle],
                        (0.011 - state[left, 0] - sensor["mount"][0]) / np.cos(angle),
                        atol=1e-4,
                    )
                    right = (state[:, 0] > 0.5) & (state[:, 0] < 2.5)
                    self.assertTrue(right.any())
                    np.testing.assert_allclose(
                        ranges[right, 0, middle],
                        sensor["mount"][2] / np.sin(sensor["elevation"]),
                        atol=1e-5,
                    )
                    np.testing.assert_allclose(
                        ranges[right, 1:, middle], sensor["range"]
                    )
                    above_wall = state[:, 0] < -2
                    self.assertTrue(above_wall.any())
                    np.testing.assert_allclose(
                        ranges[above_wall, 2, middle], sensor["range"]
                    )
                    reasons = set()
                    for _ in range(600):
                        state, _, result = step(100)
                        reasons.update(result[:, 1])
                        ended = result[:, 1] > 0
                        self.assertTrue(
                            (state[ended, 8] == 0).all(),
                            "terminal cars must respawn immediately",
                        )
                    self.assertIn(1, reasons, "wall impact must terminate")
                    self.assertIn(3, reasons, "open route must finish")
                finally:
                    runtime.terminate()
                    runtime.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
