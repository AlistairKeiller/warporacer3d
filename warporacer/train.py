"""Train a GPU PPO policy on mesh maps or legacy ROS images."""

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import warp as wp
from typer import run

from warporacer.agent import Agent
from warporacer.ppo import PPO
from warporacer.sim import Env, SimConfig
from warporacer.track import Track

_track_cache = {}


def load_tracks(paths, k, rng):
    tracks = []
    for path in rng.permutation(paths):
        if path not in _track_cache:
            try:
                _track_cache[path] = Track.load(path)
            except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
                print(f"[maps] skipping {path.name}: {error}")
                _track_cache[path] = None
        if _track_cache[path] is not None:
            tracks.append(_track_cache[path])
        if len(tracks) == k:
            return tracks
    if not tracks:
        raise RuntimeError("No loadable maps")
    return tracks


def main(
    maps: Path,
    num_envs: int = 256,
    iterations: int = 2000,
    seed: int = 0,
    log_dir: Path = Path("logs/3d"),
    device: str = "",
    record_every: int = 0,
    record_steps: int = 1800,
    max_active_maps: int = 8,
    switch_map_iter: int = 0,
    substeps: int = SimConfig.substeps,
    solver_iterations: int = SimConfig.iterations,
    use_graph: bool = True,
    rollouts: int = 24,
    lidar_beams: int = 108,
    max_steps: int = 10_000,
    live_viewer: bool = False,
    interactive: bool = False,
    use_wandb: bool = True,
):
    wp.init()
    torch.manual_seed(seed)
    dev = wp.get_device(device or None)
    if dev.is_cuda:
        torch.set_float32_matmul_precision("high")  # Use TF32 for the policy MLP.
    print(
        f"[device] {dev}; {'GPU physics and learning' if dev.is_cuda else 'CPU debug mode'}"
    )
    paths = sorted(maps.glob("*.yaml")) if maps.is_dir() else [maps]
    if not paths:
        raise FileNotFoundError(f"No map YAMLs under {maps}")
    if num_envs <= 0 or max_active_maps <= 0 or iterations < 0:
        raise ValueError(
            "Environment/map counts must be positive; iterations must be nonnegative"
        )
    rng = np.random.default_rng(seed)
    cfg = SimConfig(
        substeps=substeps,
        iterations=solver_iterations,
        use_graph=use_graph,
        lidar_beams=lidar_beams,
        max_steps=max_steps,
    )
    env = Env(
        load_tracks(paths, 1 if interactive else min(max_active_maps, num_envs), rng),
        1 if interactive else num_envs,
        seed=seed,
        device=dev,
        config=cfg,
    )
    if interactive:
        from warporacer.viewer import Viewer

        Viewer(env).interactive()
        return
    if live_viewer:
        from warporacer.viewer import Viewer

        env.viewer = Viewer(env)
    agent = Agent(env.obs_dim, env.act_dim).to(env.torch_device)
    ppo = PPO(env, agent, rollouts=rollouts)
    log_dir.mkdir(parents=True, exist_ok=True)
    run = None
    if use_wandb:
        import wandb

        run = wandb.init(
            project="warporacer3d",
            name=f"3d_seed{seed}_n{num_envs}",
            config={
                "maps": str(maps),
                "num_envs": num_envs,
                "seed": seed,
                "sim": asdict(cfg),
                "car": asdict(env.car),
            },
        )
    last = time.perf_counter()
    try:
        for it in range(iterations):
            log = ppo.iterate()
            now = time.perf_counter()
            log["sps"] = int(ppo.batch_size / (now - last))
            last = now
            if run is not None:
                run.log(log, step=ppo.global_step)
            if it % 10 == 0:
                print(
                    f"[it {it:4d}] step={ppo.global_step} sps={log['sps']} kl={log['approx_kl']:.4f} lr={log['lr']:.2e}"
                )
            if record_every > 0 and (it + 1) % record_every == 0:
                from warporacer.video import record_rollout

                record_rollout(
                    env,
                    agent,
                    ppo.obs_rms,
                    record_steps,
                    log_dir / f"rollout_{it + 1:06d}.mp4",
                )
            if (
                switch_map_iter > 0
                and (it + 1) % switch_map_iter == 0
                and len(paths) > 1
            ):
                env.rotate(load_tracks(paths, min(max_active_maps, num_envs), rng))
                ppo.reset_env_stats()
                if env.viewer is not None:
                    env.viewer.reset()
        torch.save(
            {
                "format_version": 2,
                "agent": agent.state_dict(),
                "obs_dim": env.obs_dim,
                "act_dim": env.act_dim,
                "car": asdict(env.car),
                "sim": asdict(cfg),
                "obs_mean": ppo.obs_rms.mean.cpu(),
                "obs_var": ppo.obs_rms.var.cpu(),
                "obs_count": ppo.obs_rms.count.cpu(),
            },
            log_dir / "agent_final.pt",
        )
        print(f"[checkpoint] {log_dir / 'agent_final.pt'}")
    finally:
        if env.viewer is not None:
            env.viewer.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    run(main)
