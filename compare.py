"""Synchronized Mac comparison on the same 6 m-radius, 2.6 m-wide flat circuit.

The original 2D racer uses its own virtual environment and public step API.
Mojo runs both default sensing and the original's 108 beams / 20 m range.
Construction, compilation, and viewer I/O are excluded from timings.
"""

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def original(root, counts, steps, repeats, training):
    sys.path.insert(0, str(root))
    from types import SimpleNamespace

    import numpy as np
    import torch
    import warp as wp
    from scipy.ndimage import distance_transform_edt

    from warporacer.sim import Env

    cell, size, origin = 0.025, 665, -8.3
    x = origin + np.arange(size) * cell
    y = origin + np.arange(size - 1, -1, -1) * cell
    radius = np.hypot(x[None, :], y[:, None])
    free = (radius > 4.7) & (radius < 7.3)
    angle = np.arange(252) * 2 * np.pi / 252
    route = np.column_stack((6 * np.cos(angle), 6 * np.sin(angle)))
    theta = np.mod(np.arctan2(y[:, None], x[None, :]), 2 * np.pi)
    track = SimpleNamespace(
        h=size,
        w=size,
        ox=origin,
        oy=origin,
        res=cell,
        edt=(distance_transform_edt(free) * cell).astype(np.float32),
        centerline=route.astype(np.float32),
        angles=(angle + np.pi / 2).astype(np.float32),
        lut=(np.rint(theta * 252 / (2 * np.pi)).astype(np.int32) % 252),
    )

    def synchronize():
        wp.synchronize_device("cpu")
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

    results = []
    for n in counts:
        env = Env(track, n, device="cpu")
        actions = torch.empty((n, 2), device=env.torch_device)
        actions[:, 0], actions[:, 1] = 0.1, 0.7
        if training:
            from warporacer.agent import Agent
            from warporacer.ppo import PPO

            torch.manual_seed(42)
            learner = PPO(env, Agent().to(env.torch_device))
            run = learner.iterate
            samples = n * learner.T
        else:
            run = partial(env.step, actions)
            samples = n
        for _ in range(2 if training else 30):
            run()
        timings = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            for _ in range(training or steps):
                run()
            synchronize()
            timings.append(time.perf_counter() - start)
        elapsed = median(timings)
        if not torch.isfinite(env.obs).all():
            raise RuntimeError("Original racer produced nonfinite observations")
        if training and not all(
            torch.isfinite(p).all().item() for p in learner.agent.parameters()
        ):
            raise RuntimeError("Original PPO produced nonfinite parameters")
        results.append(
            {
                "runtime": "warporacer",
                "device": "cpu + " + str(env.torch_device),
                "cars": n,
                "beams": 108,
                "range": 20,
                "step_ms": elapsed * 1000 / (training or steps),
                "transitions_s": samples * (training or steps) / elapsed,
            }
        )
        env._pool.shutdown()
    print("RESULT " + json.dumps(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ROOT.parent / "warporacer")
    parser.add_argument("--cars", type=int, nargs="+", default=[1, 32, 128, 1024])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--train-iterations",
        type=int,
        default=0,
        help="Compare full PPO training instead of stepping",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "build/comparison.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.original = args.original.resolve()
    if min(args.steps, args.repeats, *args.cars) < 1:
        parser.error("cars, steps, and repeats must be positive")
    if args.train_iterations < 0:
        parser.error("training iterations cannot be negative")
    if args.worker:
        original(
            args.original.resolve(),
            args.cars,
            args.steps,
            args.repeats,
            args.train_iterations,
        )
        return
    from app import build
    from prepare import compile_track
    from warporacer.track import demo_track

    binary = build()
    matched = ROOT / "build/racer-matched"
    subprocess.run(
        [
            str(ROOT / ".venv/bin/mojo"),
            "build",
            "main.mojo",
            "-D",
            "BEAMS=108",
            "-D",
            "RANGE=20",
            "-o",
            str(matched),
        ],
        cwd=ROOT,
        check=True,
    )
    map_path = ROOT / "build/comparison.wrmap"
    compile_track(demo_track("flat")).tofile(map_path)
    command = [
        str(args.original / ".venv/bin/python"),
        str(Path(__file__).resolve()),
        "--worker",
        "--original",
        str(args.original.resolve()),
        "--steps",
        str(args.steps),
        "--repeats",
        str(args.repeats),
        "--train-iterations",
        str(args.train_iterations),
        "--cars",
        *map(str, args.cars),
    ]
    output = subprocess.check_output(command, text=True)
    results = json.loads(
        next(line[7:] for line in output.splitlines() if line.startswith("RESULT "))
    )
    for executable, beams, reach in ((binary, 64, 10), (matched, 108, 20)):
        for device in ("cpu", "gpu"):
            for n in args.cars:
                timings = []
                for _ in range(args.repeats):
                    output = subprocess.check_output(
                        [
                            str(executable),
                            "train" if args.train_iterations else "benchmark",
                            str(map_path),
                            device,
                            str(n),
                            str(args.train_iterations or args.steps),
                            str(ROOT / "build/comparison-policy.wrppo"),
                        ],
                        text=True,
                    )
                    timing = re.search(
                        r"; ([\d.eE+-]+) training transitions/s"
                        if args.train_iterations
                        else r"cars; ([\d.eE+-]+) ms/step;",
                        output,
                    )
                    if timing is None:
                        raise RuntimeError(output)
                    timings.append(
                        n * 32 * 1000 / float(timing[1])
                        if args.train_iterations
                        else float(timing[1])
                    )
                elapsed = median(timings)
                results.append(
                    {
                        "runtime": "mojo",
                        "device": device,
                        "cars": n,
                        "beams": beams,
                        "range": reach,
                        "step_ms": elapsed,
                        "transitions_s": n
                        * (32 if args.train_iterations else 1)
                        * 1000
                        / elapsed,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "steps": args.steps,
                "repeats": args.repeats,
                "mode": "train" if args.train_iterations else "step",
                "train_iterations": args.train_iterations,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    for row in results:
        unit = "ms/iteration" if args.train_iterations else "ms/step"
        print(
            f"{row['runtime']:10s} {row['device']:10s} {row['cars']:5d} cars {row['beams']:3d} beams: "
            f"{row['step_ms']:8.3f} {unit} {row['transitions_s']:12,.0f} transitions/s"
        )
    print("Saved", args.output)


if __name__ == "__main__":
    main()
