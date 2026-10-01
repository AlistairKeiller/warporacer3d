# Cross-platform GPU decision

Investigated and implemented on 2026-10-01. The chosen runtime is **pure Mojo
simulation and PPO**, using MAX for portable CPU/GPU matrix multiplication.
Python remains an offline importer and local viewer host.

## Why this model

With flat roads and ramps as the terrain scope, a seven-state terrain-following
dynamic bicycle is enough to model steering, slip, grip limits, and gravity on
slopes. Height/pitch/roll follow the surface; leaving it terminates the episode.
This removes eleven physical bodies per car, suspension constraints, general
collision detection, and the 16-substep/12-iteration Newton XPBD solve.

The CommonRoad MB source is a 29-state ODE, rather than a contact solver, but its
vertical/suspension equations assume an implicit flat road. It accepts steering
rate and acceleration, with no terrain height or normal input. Adapting and
calibrating it adds work that this terrain scope does not need.
[Source inspected](https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/PYTHON/vehiclemodels/vehicle_dynamics_mb.py?ref_type=heads).

For suspension, takeoff, or landing, use the retained Newton runtime, or later
consider one rigid chassis with four spring–damper raycast wheels. Those features
are outside the simplified runtime's assumptions. The nonplanar model reference
also makes continuous support and small road curvature assumptions explicit.
[Nonplanar vehicle models](https://arxiv.org/abs/2104.08427).

The original 2D `warporacer` already has distance-field lidar and a small bicycle
model. Reuse the idea, while accounting for slope gravity, tangent velocities,
conservative thin-wall/edge queries, and locally continuous route projection.
[BENCHMARKS.md](BENCHMARKS.md) compares actual runtimes on the same Mac.

## Platform alternatives

| Candidate | Decision for this project |
| --- | --- |
| Mojo + MAX | Chosen: CPU/Metal/CUDA/HIP from shared source, portable GEMM, and a pure Mojo learner |
| Quadrants / Taichi | Reasonable if retaining Python kernels; would still need a learner and device interop |
| SlangPy | Metal device creation worked locally; owning shaders plus learning plumbing adds another language |
| Genesis | Metal initialization worked locally; its contact simulator is more than the revised problem needs |
| WebGPU / wgpu | Broad deployment appeal; a custom learner and GPU plumbing would increase owned code |
| JAX / MJX | Less direct for the Mac-first requirement; Apple support depends on a separate experimental plugin |

Primary platform references:
[Mojo requirements](https://mojolang.org/docs/requirements/),
[Quadrants](https://github.com/Genesis-Embodied-AI/quadrants),
[Taichi backends](https://docs.taichi-lang.org/docs/hello_world),
[SlangPy](https://slangpy.shader-slang.org/en/latest/),
[Genesis installation](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html),
[wgpu](https://wgpu.rs/), and
[Apple JAX Metal](https://developer.apple.com/metal/jax/).
Only Mojo was benchmarked as a full replacement here; alternative initialization
checks establish availability, not their relative speed.

## Toolchain checks and fixes

The local machine is an Apple M5 Pro (18 CPU cores, 20 GPU cores, 48 GB), macOS
27.0, Xcode 27.0. `xcodebuild -showComponent MetalToolchain -json` now reports
**installed**, build 27A266a, toolchain `com.apple.dt.toolchain.Metal.32023.921.5`.
Both native GPU kernels and MAX GPU matmul execute successfully. Mojo 1.1.0 and
MAX 26.6.0 are pinned, with Python 3.13 for their wheels.

Three toolchain problems were found during implementation:

- Calling the standard `atan2` from a Metal kernel emitted an unresolved `atan2f`
  and failed compilation. Spawn headings are now baked into the static map. The
  runtime needs no inverse trigonometry or platform-specific workaround.
- An asynchronous CPU closure dispatch outlived borrowed pointers and crashed.
  CPU execution now uses synchronous `parallelize`, with CPU GEMM completing
  before the next kernel. CPU-only builds exclude GPU compilation entirely.
- Mojo 1.1 `input()` leaks a duplicated stdin descriptor on each call. The
  viewer reuses one buffered input stream and closes it on exit. A 600-frame
  native protocol test covers this failure alongside wall impacts and finishes.

An early isolated research environment could import Torch MPS buffers into MAX
through DLPack, but the `CustomOpLibrary` device helper rejected `mps`. The final
runtime uses native Mojo buffers, avoiding this interop dependency altogether.

## Evidence and remaining portability work

CPU/Metal tests match vehicle dynamics and sensing on flat and ramp maps. An
independent Torch oracle checks all 9,029 policy/value derivatives, with stricter
CPU tolerance and a bounded tolerance for fast GPU matmul. Checkpoint tests move
weights and Adam state between devices and batch sizes. Ramp training learned a
policy that completed 256 cars × 1,000 deterministic steps with zero failures.
The WebGL viewer was checked in Chrome for policy playback, keyboard input,
pause, single-step, reset, and rendering.

There are no Metal/CUDA/HIP-specific application kernels. Mojo currently supports
Apple silicon Macs, Linux, and Windows through WSL; consult its requirements for
GPU/driver coverage. NVIDIA, AMD, and WSL execution still need hardware
validation. Source portability is established by shared APIs; measured behavior
and speed here are Apple CPU/Metal results.
