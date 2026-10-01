# warporacer3d

Fast racing on **flat roads and ramps**, with simulation and PPO written in Mojo.
The same kernels run on CPU, Apple Metal, NVIDIA CUDA, or AMD HIP. Python prepares
static maps and hosts a small browser viewer; it is outside the training loop.

The car is a seven-state dynamic bicycle. Road geometry supplies height, pitch,
and roll; tire forces, steering response, and gravity supply the motion. Road
edges and walls end an episode. There is no suspension, jumping, or contact solver.
This deliberately matches the flat-road/ramp scope.

## Try it

Use Python 3.13 and the pinned Mojo/MAX toolchain:

```sh
uv sync
uv run python app.py
```

The viewer builds the runtime, opens a local browser, and starts the ramp map.
Drive with **W/S** and **A/D** or the arrow keys. Space pauses; R resets. Select a
car, change camera, inspect lidar and trails, or advance one step. The panel shows
speed, grade, height, steering, reward, progress, resets, and finishes. Cars run
independently, so they can overlap. The viewer uses WebGL without external JS libraries.

```sh
# Flat circuit; force CPU execution (also avoids compiling GPU kernels).
uv run python app.py maps/3d/flat.yaml --device cpu --cars 1

# Watch a trained policy. Switch to keyboard driving whenever you like.
uv run python app.py maps/3d/ramp.yaml --checkpoint build/ramp.wrppo --cars 16
```

## Train, evaluate, benchmark

```sh
uv run python prepare.py maps/3d/ramp.yaml build/ramp.wrmap
uv run mojo build main.mojo -o build/racer

# 256 cars, 300 PPO iterations, checkpoint saved every 100 iterations and at exit.
build/racer train build/ramp.wrmap gpu 256 300 build/ramp.wrppo
build/racer eval build/ramp.wrmap gpu 256 1000 build/ramp.wrppo
build/racer benchmark build/ramp.wrmap gpu 1024 1000

# Continue training, optionally changing backend and batch size.
build/racer train build/ramp.wrmap cpu 32 100 build/ramp.wrppo build/ramp.wrppo
```

CLI arguments are `MODE MAP [DEVICE] [CARS] [STEPS_OR_ITERATIONS] [CHECKPOINT] [RESUME]`.
`DEVICE` is `auto` (the default), `cpu`, or `gpu`; Mojo selects the installed GPU
backend. A training iteration collects 32 steps per car, then runs four PPO epochs.
Evaluation uses deterministic actions and reports reward, failures, and finishes.
Checkpoints include weights, Adam moments, optimizer step, iteration, and RNG seed;
resuming starts fresh driving episodes. Old Torch checkpoints are incompatible.

For a CPU-only binary, including on Macs without the Metal compiler:

```sh
uv run mojo build main.mojo -D CPU_ONLY=true -o build/racer-cpu
build/racer-cpu benchmark build/ramp.wrmap cpu 32 1000
```

CPU is usually best for one car. The GPU becomes useful with larger batches.
Default sensing is 64 planar beams over 270°, with a 10 m range. Compile with
`-D BEAMS=108 -D RANGE=20` for the original `warporacer` sensor count and range.
Use the same sensing configuration to train, evaluate, and load checkpoints.
The browser viewer uses the default configuration.

## Maps

`prepare.py` reads the existing YAML/mesh/ROS-image loaders and bakes a 2.5 cm
height/gradient/clearance grid plus a uniformly sampled route into `.wrmap`.
Triangle geometry remains available to the viewer. Road edges, holes, and thin
walls are conservatively rasterized; the chassis uses swept clearance queries.
Obstacle projections block the entire road column. Overlapping road levels are
rejected, and ramps must be below about 45°. Existing overpass/jump maps require
the reference simulator.

## Performance and portability

The local Apple M5 Pro runs both simulation and the entire learner on Metal.
Measured comparisons, methodology, and differences from the original 2D racer
and Newton implementation are in [BENCHMARKS.md](BENCHMARKS.md).

The source shares one CPU/GPU implementation; only dispatch and MAX matmul select
a backend. Apple CPU/Metal are tested here. CUDA, HIP, and Windows/WSL still need
hardware validation. Mojo currently supports Apple silicon Macs, compatible Linux
hosts, and Windows through WSL, rather than native Windows. Consult the current
[Mojo requirements](https://mojolang.org/docs/requirements/) for supported GPUs
and drivers.

This Mac's Metal toolchain is installed and works. On a fresh Mac, install Xcode
and, if needed, run `xcodebuild -downloadComponent MetalToolchain`.
[GPU_INVESTIGATION.md](GPU_INVESTIGATION.md) records the platform choices and
compiler issues encountered.

## Read the code / verify it

Start with [racer/vehicle.mojo](racer/vehicle.mojo), then
[racer/terrain.mojo](racer/terrain.mojo) and [racer/env.mojo](racer/env.mojo).
[ARCHITECTURE.md](ARCHITECTURE.md) explains the equations, buffer layout, PPO, and
scope. There is no general physics framework or autodiff framework in the new runtime.

```sh
uv run python -m unittest discover -s tests -p 'test_p*.py'
uv run python prepare.py maps/3d/flat.yaml build/flat.wrmap
uv run python prepare.py maps/3d/ramp.yaml build/ramp.wrmap
uv run mojo -I . tests/test_mojo.mojo build/flat.wrmap build/ramp.wrmap

# Optional independent autograd check: Torch is a test/reference dependency.
uv run --extra reference mojo -I . tests/gradient.mojo cpu build/flat.wrmap build/cpu-gradient.bin
uv run --extra reference mojo -I . tests/gradient.mojo gpu build/flat.wrmap build/gpu-gradient.bin
uv run --extra reference python tests/check_gradient.py build/cpu-gradient.bin build/gpu-gradient.bin

# Comparison with the sibling original racer, using its own virtual environment.
uv run python compare.py --original ../warporacer
uv run python compare.py --original ../warporacer --cars 256 --train-iterations 10 --output build/training-comparison.json
```

The previous Newton/Warp/Torch runtime remains available via
`uv run --extra reference python main.py`. Its code and existing tests are retained;
its [usage](docs/reference/README.md) and
[architecture](docs/reference/ARCHITECTURE.md) are archived separately.
