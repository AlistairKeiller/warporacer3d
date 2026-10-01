"""Drive the native viewer protocol through a thin wall and an open finish."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app import build
from prepare import compile_track
from warporacer.track import MeshPart, Route, Track


class ProtocolTests(unittest.TestCase):
    def test_wall_lidar_finish_and_reset(self):
        road = MeshPart(
            np.array([[-4, -3, 0], [4, -3, 0], [4, 3, 0], [-4, 3, 0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
        wall = MeshPart(
            np.array([[0.011, -3, 0], [0.011, 3, 0], [0.011, 3, 1], [0.011, -3, 1]]),
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
                    while '"ready":true' not in runtime.stdout.readline():
                        self.assertIsNone(runtime.poll(), "runtime failed to start")

                    def step(throttle):
                        runtime.stdin.write(f"0 {throttle} 0 0\n")
                        runtime.stdin.flush()
                        data = np.array(json.loads(runtime.stdout.readline())).reshape(
                            -1
                        )
                        self.assertTrue(np.isfinite(data).all())
                        return (
                            data[: n * 14].reshape(n, 14),
                            data[n * 14 : n * 86].reshape(n, 72),
                            data[-n * 2 :].reshape(n, 2),
                        )

                    state, observation, result = step(0)
                    np.testing.assert_array_equal(result[:, 1], 0)
                    np.testing.assert_array_equal(
                        state[:, 9], 1
                    )  # every baked spawn is safe
                    left = state[:, 0] < -0.5
                    self.assertTrue(left.any())
                    # The forward-facing beams must stop at the 1.1 cm-offset wall.
                    self.assertTrue(
                        (
                            observation[left, 8 + 32] * 10 < abs(state[left, 0]) + 0.03
                        ).all()
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
