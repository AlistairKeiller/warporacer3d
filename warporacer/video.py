"""Record a separate evaluation world without changing training state or RNG."""

from pathlib import Path

import imageio.v2 as imageio
import torch

from warporacer.sim import Env
from warporacer.viewer import Viewer


@torch.no_grad()
def record_rollout(env, agent, obs_rms, num_steps, out_path):
    evaluation = Env(
        env.tracks[0],
        1,
        seed=env.seed,
        device=env.device,
        car=env.car,
        config=env.config,
    )
    viewer = Viewer(evaluation, headless=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(
            out_path, fps=round(1 / env.config.dt), macro_block_size=2
        ) as writer:
            for _ in range(num_steps):
                action = agent.actor(obs_rms.normalize(evaluation.obs))
                evaluation.step(action)
                viewer.render()
                writer.append_data(viewer.screenshot())
    finally:
        viewer.close()
