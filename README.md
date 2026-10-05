# warporacer3d

F1TENTH-style racing simulation with terrain (ramps, banks, overpasses), a
planar lidar, PPO training and a browser viewer, in Mojo on MAX. Kernels are
closures run by MAX's `elementwise`, matrix products are MAX's `matmul`, sums
are MAX's `sum`, so the same code runs on the CPU and on Metal, CUDA or HIP.
Tracks are compiled in Python with numpy, scipy and scikit-image.

```sh
uv sync
uv run mojo build main.mojo -o build/racer
uv run build/racer train ramp gpu 1024 300 build/agent.npy
uv run build/racer eval ramp gpu 1024 1000 build/agent.npy
uv run build/racer benchmark ramp gpu 4096 1000
uv run build/racer serve ramp auto 16 8765 build/agent.npy   # open http://127.0.0.1:8765
```

Run the binary through `uv run` from the project directory: it imports
`racer/track.py` and `racer/serve.py` and reads `viewer.html`.

```
racer train     TRACK [auto|gpu|cpu] [cars] [iterations] [checkpoint.npy] [resume.npy]
racer eval      TRACK [auto|gpu|cpu] [cars] [steps] [checkpoint.npy]
racer benchmark TRACK [auto|gpu|cpu] [cars] [steps]
racer serve     TRACK [auto|gpu|cpu] [cars] [port] [checkpoint.npy|-]
```

`TRACK` is `flat`, `ramp`, `bank`, `bridge`, or a map YAML. Checkpoints are
`.npy` files holding the weights; resuming restarts Adam. Viewer keys: W/S
throttle, A/D steer, Space pause, R reset. `-D CPU_ONLY=true` builds without a
GPU toolchain; `-D BEAMS=1081` uses every beam of the lidar.

Until the Apple GPU fixes in MAX's two-phase reduction and `transpose_b` gemv
ship in a nightly, build against a checkout of the `apple-gpu-fixes` branch of
github.com/modular/modular:

```sh
set M ~/git/modular   # checkout on apple-gpu-fixes
set L .venv/lib/python3.13/site-packages/modular/lib/mojo
env MODULAR_MOJO_MAX_IMPORT_PATH=$M/max/mojo uv run mojo build -I $M/max/kernels/src -I $L main.mojo -o build/racer
```

## Tracks

A track is a closed route of evenly spaced waypoints plus grids over the same
xy extent: per layer, a signed clearance to the nearest wall, the road height,
and the nearest waypoint of every cell. One layer is enough unless the route
crosses over itself; then the deck and the ground under it are separate
layers, and the car (and every lidar beam) uses the layer whose height is
nearest its own z. The built-in tracks are ribbons around a circle (`ramp`
adds hills, `bank` tilts the road) and a figure eight with an overpass
(`bridge`). A YAML map is either a ROS occupancy image, as produced by SLAM,
with optional ramp boxes laid on the floor, or a route with a height profile:

```yaml
image: track.png          # or .pgm; free pixels >= 230; the route is the skeleton loop
resolution: 0.05
origin: [-10.7, -20.6, 0]
ramps:                    # optional: boxes with a sloped top
  - {center: [2.0, 1.5], heading: 0.0, length: 1.6, width: 1.0, height: 0.25}
```

```yaml
route: loop.txt           # closed loop, one `x y z` per line
half_width: 1.3
bank: 0.0                 # rise per metre outward
resolution: 0.025
```

## Tests

```sh
uv run mojo run -I . tests/test_mojo.mojo        # add `cpu` to skip the GPU
```

Dynamics, a synthetic corridor (lidar ranges, crash, respawn, progress), the
bridge layers, CPU/GPU parity, GAE, and the analytic PPO gradient against
central finite differences.

## Layout

| file | what it does |
|---|---|
| `main.mojo` | CLI: train, eval, benchmark, serve |
| `racer/track.py` | tracks: ribbons, occupancy images (EDT, skeleton centerline), layers, one float32 array |
| `racer/device.mojo` | `Device`: buffers, matrix views, `run` (MAX elementwise), `gemm` (MAX matmul), `sum` (MAX reduction) |
| `racer/sim.mojo` | `Map` lookups, kinematic bicycle on a slope with a friction circle, sphere-traced lidar, `Sim` |
| `racer/ppo.mojo` | `Policy`: tanh MLP as matmuls with epilogues; `Trainer`: rollouts, GAE, clipped PPO, Adam, checkpoints |
| `racer/serve.py` | `http.server` bridge for the viewer |
| `viewer.html` | three.js viewer of the track layers, cars and lidar |

Activations are feature-major (one column per sample) with a constant-1 row,
so biases are weight columns and every forward, backward and weight-gradient
product is a plain `matmul` (with `transpose_b` for the gradients).
