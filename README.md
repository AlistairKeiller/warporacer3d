# warporacer3d

Terrain-following racing simulation with a planar lidar, PPO training, a map
compiler, and a browser viewer, in Mojo on MAX. Matrix products are MAX's
`matmul`, kernels are closures run by MAX's `elementwise`, so the same code
runs on the CPU and on Metal, CUDA, or HIP. Python is used only through
interop for the viewer's HTTP server and PNG inflation (standard library).

```sh
uv sync
uv run mojo build main.mojo -o build/racer
uv run build/racer prepare ramp build/ramp.wrmap
uv run build/racer train build/ramp.wrmap gpu 1024 300 build/agent.wrppo
uv run build/racer eval build/ramp.wrmap gpu 1024 1000 build/agent.wrppo
uv run build/racer benchmark build/ramp.wrmap gpu 4096 1000
uv run build/racer serve build/ramp.wrmap auto 16 8765 build/agent.wrppo   # open http://127.0.0.1:8765
```

Run the binary through `uv run` (it needs the venv's Python) from the project
directory (it reads `viewer.html` and `racer/serve.py`).

```
racer prepare   flat|ramp|bank|MAP.yaml OUT.wrmap [resolution=0.025]
racer train     MAP.wrmap [auto|gpu|cpu] [cars] [iterations] [checkpoint] [resume]
racer eval      MAP.wrmap [auto|gpu|cpu] [cars] [steps] [checkpoint]
racer benchmark MAP.wrmap [auto|gpu|cpu] [cars] [steps]
racer serve     MAP.wrmap [auto|gpu|cpu] [cars] [port] [checkpoint|-]
```

Viewer keys: W/S throttle, A/D steer, Space pause, R reset. `-D CPU_ONLY=true`
builds without a GPU toolchain; `-D BEAMS=1081` uses every beam of the lidar.

## Maps

Every road boundary edge is extruded into a 0.5 m wall, so the lidar always
sees the track and leaving the road is a crash. Built-in `flat`, `ramp`, and
`bank` are ribbons around a circle. The chassis attitude (height, pitch, roll)
comes from the road height under the four wheels, so a faceted mesh does not
pitch the car at every facet. A YAML map is either a ROS occupancy image
(the route is the longest skeleton loop of the free space) or OBJ meshes plus
a text route:

```yaml
image: track.png          # or .pgm; free pixels >= 230
resolution: 0.05
origin: [-10.7, -20.6, 0]
```

```yaml
mesh: road.obj            # comma-separated list; shared edges cancel, the rest get walls
obstacles: cones.obj      # optional, comma-separated; never driven on
route: route.txt          # lines: x y z [half_width]
closed: false
```

## Tests

```sh
uv run mojo -I . tests/test_mojo.mojo        # add `cpu` to skip the GPU
```

Dynamics, map continuity (no per-cell height jumps), smooth wheel-based
attitude on the ramp, a thin wall seen by the lidar with crash/finish/reset,
CPU/GPU parity, GAE boundaries, and the analytic PPO gradient against central
finite differences.

## Layout

| file | what it does |
|---|---|
| `main.mojo` | CLI: prepare, train, eval, benchmark, serve |
| `racer/device.mojo` | `Device`: buffers, matrix views, `run` (MAX elementwise), `gemm` (MAX matmul), `sum` |
| `racer/map.mojo` | `.wrmap` header, bilinear surface / route projection used by kernels, shared vector helpers |
| `racer/vehicle.mojo`, `lidar.mojo` | bicycle dynamics; 270-degree planar lidar: one thread per beam marches the clearance field, then walks coarse-cell triangle lists |
| `racer/simulation.mojo` | `Sim`: spawn, physics with rewards and resets, sensing, action sampling |
| `racer/network.mojo` | `Policy`: tanh MLP forward and backward as matmuls with epilogues |
| `racer/ppo.mojo` | `Trainer`: rollouts, GAE, clipped PPO minibatches, Adam, checkpoints |
| `racer/compile.mojo` | map compiler: ribbons, OBJ + route, occupancy images, walls, grids (parallel distance transform), cell lists |
| `racer/serve.py` | `http.server` bridge for the viewer |
| `viewer.html` | WebGL viewer; decodes the raw `.wrmap` and binary state snapshots |

Activations are feature-major (one column per sample) with a constant-1 row,
so biases are weight columns and every forward, backward, and weight-gradient
product is a plain `matmul` (with `transpose_b` for the gradients).
