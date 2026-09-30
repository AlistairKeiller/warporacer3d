"""Newton's 3D viewer with a small keyboard-driving wrapper."""

import time

import numpy as np
import torch
from newton.viewer import ViewerGL


class Viewer:
    def __init__(self, env, headless=False, width=1280, height=720):
        self.env = env
        self.viewer = ViewerGL(width=width, height=height, headless=headless)
        self.time = 0.0
        self.reset()

    def reset(self):
        self.batch = self.env.batches[0]
        self.viewer.set_model(self.batch.model)
        self.viewer.set_visible_worlds(list(range(min(64, self.batch.n))))
        low, high = self.batch.track.bounds
        center = (low + high) / 2
        size = max(float(np.max(high - low)), 3.0)
        self.viewer.set_camera(
            pos=(center + np.array([0, -0.8 * size, 0.7 * size])).tolist(),
            pitch=-40,
            yaw=90,
        )

    def render(self):
        self.time += self.env.config.dt
        with self.env.scope():
            self.viewer.begin_frame(self.time)
            self.viewer.log_state(self.batch.state)
            self.viewer.end_frame()

    def screenshot(self):
        return self.viewer.get_frame().numpy()

    def interactive(self):
        actions = torch.zeros(
            (self.env.num_envs, self.env.act_dim), device=self.env.torch_device
        )
        deadline = time.perf_counter()
        try:
            while self.viewer.is_running():
                actions.zero_()
                actions[0, 0] = float(self.viewer.is_key_down("J")) - float(
                    self.viewer.is_key_down("L")
                )
                actions[0, 1] = float(self.viewer.is_key_down("I")) - float(
                    self.viewer.is_key_down("K")
                )
                self.env.step(actions)
                self.render()
                deadline += self.env.config.dt
                time.sleep(max(0, deadline - time.perf_counter()))
        finally:
            self.close()

    def close(self):
        self.viewer.close()
