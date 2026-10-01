# warporacer3d

3D racing with Newton rigid-body physics, Warp controls and sensing, and PyTorch
PPO. Cars have physical wheels, suspension, steering, and full six-degree-of-freedom
motion. Maps are triangle meshes with authored 3D routes, so ramps, banking,
bridges, tunnels, and gaps use the same interface. Existing ROS image maps also work.

Physics, lidar, race logic, resets, and learning run on an NVIDIA CUDA GPU.
CPU execution is available for debugging. Controls run at 60 Hz with 16 substeps
and 12 XPBD iterations, selected through batched driving and landing tests.
Graph replay removes Python kernel-launch overhead on CPU and CUDA.

Run the following commands from the `warporacer3d` directory. The Python package
remains `warporacer`, so the existing imports still work.

```bash
# Train on a ramp map; W&B is enabled unless explicitly disabled.
uv run python main.py maps/3d/ramp.yaml --device cuda:0 --no-use-wandb

# Train across maps, rotating the active pool every 20 iterations.
uv run python main.py maps/3d/ --switch-map-iter 20 --no-use-wandb

# Drive manually: I/K forward/reverse, J/L change steering angle.
uv run --extra viz python main.py maps/3d/ramp.yaml --interactive --device cpu

# Watch training, or periodically record a separate evaluation car.
uv run --extra viz python main.py maps/3d/ --live-viewer --no-use-wandb
uv run --extra viz python main.py maps/3d/ramp.yaml --record-every 20 --no-use-wandb

# Small CPU smoke run; checkpoints go to logs/3d/agent_final.pt.
uv run python main.py maps/3d/ramp.yaml --device cpu --num-envs 2 --iterations 1 --rollouts 4 --no-use-wandb
```

Viewer and MP4 recording require the `viz` extra and a working OpenGL context.
The live viewer displays up to 64 cars on the first active map. Examples in
`maps/3d/` cover flat roads, ramps, banking, an overpass, and a ramp-gap landing.
Use `maps/my_map.yaml` or `maps/` for legacy occupancy-image maps.

## Custom maps

Use metres, Z up, and counterclockwise triangles when viewed from outside the
surface. Road tops must face up; give walls and obstacles the sides that cars can
hit. Mesh contacts and lidar reject back faces. Scene-node transforms in OBJ/GLB
imports are baked into the geometry; author the assets in metres.

A map YAML names geometry and its driving route:

```yaml
mesh: road.glb             # OBJ, GLB, or NPZ geometry
route: route.npz
closed: true
```

Geometry NPZ files contain `vertices[N,3]` and integer `triangles[M,3]`. Route NPZ
files contain `points[N,3]`, optional `up[N,3]` (default world Z), and optional
`half_width[N]` (default 1.5 metres). An open route needs `closed: false`; a closed
route must omit the repeated endpoint. Up vectors describe banking and spawn
orientation. Keep the route on the road surface, with enough segments to describe
its curvature and provide useful spawn locations.

Parts can distinguish drivable road from obstacles:

```yaml
parts:
  - mesh: road.glb
  - mesh: barriers.glb
    obstacle: true
route: route.npz
closed: false
```

`Route`, `MeshPart`, and `Track` can also be constructed directly in Python.
`ribbon(route)` generates a solid road from a route; `demo_track()` demonstrates
ramps, banks, overlapping levels, and disconnected landing geometry. Spawns are
segment midpoints with four-wheel support checked against the mesh. Invalid
routes or maps without supported spawns raise an error.

Geometry determines contact and sensing; the route determines direction, progress,
width, and episode completion. A junction can have several map manifests sharing
one mesh and following different routes. Progress follows nearby route segments
in arc length and 3D space, preserving the intended branch through crossings.
There is no generated floor under gaps in mesh maps.

## Environment interface

```python
import torch
from warporacer.sim import Env, SimConfig
from warporacer.track import Track

track = Track.load("maps/3d/ramp.yaml")
env = Env(track, num_envs=256, device="cuda:0", config=SimConfig())
actions = torch.zeros((env.num_envs, env.act_dim), device=env.torch_device)
obs, reward, done = env.step(actions)
```

Actions are normalized steering-target rate and forward/reverse motor torque.
Steering targets have angle and slew limits. Motors have bounded torque and a
speed curve; body motion follows Newton's dynamics. Wheel friction and motor
strength vary by ±15% on each reset. Tires use contact friction, with a simple
linear chassis drag; this is a simplified car rather than a calibrated tire model.

Default observations have 338 values:

| Indices | Values |
| --- | --- |
| `0` | Measured mean front steering angle, radians |
| `1:4` | Chassis-frame linear velocity, m/s |
| `4:7` | Chassis-frame angular velocity, rad/s |
| `7:10` | Unit gravity direction in the chassis frame |
| `10:14` | Front-left, front-right, rear-left, rear-right contact flags |
| `14:` | Lidar ranges, metres; three elevation rows of 108 beams |

`SimConfig` configures lidar mount, elevations, field of view, beam count, and
range. Every ray follows the complete chassis pose and queries only its map.
The policy derives its input size from the environment.

Outputs are persistent Torch views over Warp buffers. Copy them when retaining
history. Finished cars reset automatically, so terminal rewards/dones accompany
the next episode's initial observation. `env.reason` contains `0` running, `1`
collision, `2` lost/off-course, `3` prolonged rollover, `4` timeout, or `5` open-route
finish. Airborne motion is allowed. `env.observe()` does not advance time or RNG;
`env.reset(mask)` resets selected cars; `env.rotate(tracks)` replaces the map pool
while preserving output allocations. Map import and model construction run on
the host; stepping uses device arrays and a shared CUDA stream with PyTorch.

The environment owns a Warp blocking stream, shared with Torch during its calls.
GPU event waits order inputs and outputs with the caller's current Torch stream,
including non-default streams. This also protects Newton's temporary buffers.
When reading environment buffers on a new Torch stream before calling an
environment method, use `with env.scope():` to establish the same ordering.

The first step captures a Warp graph; later steps replay it using the same
device buffers. Odd substep counts use two graphs for alternating state buffers.
Resets remain inside the graph; changing maps discards the graphs and captures
the new geometry on the next step. Use `--no-use-graph` for ordinary launches.
PPO reuses rollout buffers, transfers episode statistics once per rollout, and
uses fused Adam and TF32 policy matrix multiplication on CUDA. Distribution
validation is disabled and KL transfers happen once per epoch to avoid GPU
synchronization inside those loops.

Checkpoints use format version 2 and include policy weights, observation
normalization, dimensions, and vehicle/simulation configuration. Old 2D policies
have incompatible observation dimensions.

## Code and checks

| Module | Responsibility |
| --- | --- |
| `track.py`, `legacy.py` | Mesh/route data, import, procedural roads, image adapter |
| `vehicle.py` | Car construction and motor controls |
| `sim.py` | Per-map Newton worlds, GPU episodes, resets, progress, and sensing |
| `agent.py`, `ppo.py`, `train.py` | Policy, PPO, and CLI |
| `viewer.py`, `video.py` | Newton viewer and isolated evaluation recordings |

[ARCHITECTURE.md](ARCHITECTURE.md) records the design and the pure Warp comparison.
Local `../warp` and `../newton` sources served as API references. Runtime packages
are pinned through `uv.lock`; the source directories are not runtime dependencies.

```bash
uv run python -m unittest discover -s tests -v
RACER_DEVICE=cuda:0 uv run python -m unittest discover -s tests -v
```

The behavior suite covers driving, steering, suspension, slopes, banking,
takeoff/landing, wall contacts, overpasses, route crossings, resets, map changes,
mesh import (including integer-overflow rejection), legacy maps, and a PPO update.
Drive/coast and jump checks each run 32 randomized cars; a 128-car check compares
world isolation, and graph replay is compared with ordinary stepping and resets.
The CUDA test exercises delayed action production and observation/reset reads
across two Torch streams; it skips when CUDA is unavailable. CPU checks and
OpenGL recording were run locally on macOS. CUDA execution still needs validation
on an NVIDIA machine.

## Benchmarking

Measure simulation throughput without the viewer or PPO. Compilation and graph
capture are warmed up; device synchronization includes all GPU work in the timer.

```bash
uv run python benchmark.py --device cuda:0 --num-envs 512 1024 2048 4096
uv run python benchmark.py --device cuda:0 --num-envs 512 --no-use-graph
uv run python main.py maps/3d/ --device cuda:0 --num-envs 4096 --no-use-wandb
```

For CPU measurements use `--device cpu`. `--profile` reports Python overhead;
`--substeps` and `--solver-iterations` allow explicit physics comparisons.

Local ARM CPU timings on `maps/3d/ramp.yaml`, with 338 observations and no viewer
or PPO (three warmup steps, twelve timed steps):

| Parallel cars | Previous 3D step (ms) | Optimized step (ms) | Speedup |
| --- | ---: | ---: | ---: |
| 1 | 58.39 | 2.08 | 28.1× |
| 32 | 175.80 | 38.94 | 4.5× |
| 128 | 529.96 | 153.53 | 3.5× |
| 512 | 2003.99 | 623.86 | 3.2× |

These compare the previous 24-substep/32-iteration 3D implementation with the new
defaults. They are CPU measurements; GPU throughput remains to be measured.
Use `--substeps 24 --solver-iterations 32` to retain the earlier solver budget.
