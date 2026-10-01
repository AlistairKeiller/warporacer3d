"""Measure batched environment stepping, excluding construction and compilation."""

import argparse
import cProfile
import gc
import json
import pstats
import time
from pathlib import Path

import torch
import warp as wp

from warporacer.sim import Env, SimConfig
from warporacer.track import Track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="maps/3d/ramp.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 32, 128, 512])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--substeps", type=int, default=SimConfig.substeps)
    parser.add_argument("--solver-iterations", type=int, default=SimConfig.iterations)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--no-use-graph", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.steps, args.warmup, *args.num_envs) <= 0:
        parser.error("Environment counts and step counts must be positive")
    config = SimConfig(
        substeps=args.substeps,
        iterations=args.solver_iterations,
        use_graph=not args.no_use_graph,
    )
    track = Track.load(args.map)
    results = []
    for n in args.num_envs:
        env = Env(track, n, device=args.device, config=config)
        actions = torch.zeros((n, 2), device=env.torch_device)
        actions[:, 1] = 0.6
        for _ in range(args.warmup):
            env.step(actions)
        wp.synchronize_device(env.device)
        profiler = cProfile.Profile() if args.profile else None
        if profiler:
            profiler.enable()
        start = time.perf_counter()
        for _ in range(args.steps):
            env.step(actions)
        wp.synchronize_device(env.device)
        elapsed = time.perf_counter() - start
        if profiler:
            profiler.disable()
            pstats.Stats(profiler).sort_stats("cumtime").print_stats(12)
        if not torch.isfinite(env.obs).all():
            raise RuntimeError("Nonfinite observations during benchmark")
        result = {
            "device": str(env.device),
            "num_envs": n,
            "substeps": config.substeps,
            "iterations": config.iterations,
            "use_graph": config.use_graph,
            "step_ms": elapsed * 1000 / args.steps,
            "env_steps_per_second": n * args.steps / elapsed,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        del env
        gc.collect()
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
