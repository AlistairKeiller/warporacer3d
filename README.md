# warporacer3d

Terrain-following racing simulation, lidar, PPO training, map compiler, and a
browser viewer, all in Mojo. One binary, no Python on any path; `uv` only
installs the Mojo toolchain.

```sh
uv sync
uv run mojo build main.mojo -o build/racer
build/racer prepare ramp build/ramp.wrmap
build/racer train build/ramp.wrmap gpu 1024 300 build/agent.wrppo
build/racer eval build/ramp.wrmap gpu 1024 1000 build/agent.wrppo
build/racer benchmark build/ramp.wrmap gpu 4096 1000
build/racer serve build/ramp.wrmap auto 16 8765 build/agent.wrppo   # then open http://127.0.0.1:8765
```

```
racer prepare   flat|ramp|bank|MAP.yaml OUT.wrmap [resolution=0.025]
racer train     MAP.wrmap [auto|gpu|cpu] [cars] [iterations] [checkpoint] [resume]
racer eval      MAP.wrmap [auto|gpu|cpu] [cars] [steps] [checkpoint]
racer benchmark MAP.wrmap [auto|gpu|cpu] [cars] [steps]
racer serve     MAP.wrmap [auto|gpu|cpu] [cars] [port] [checkpoint|-]
```

Viewer keys: W/S throttle, A/D steer, Space pause, R reset. The device is
Metal, CUDA, or HIP when `auto`; `cpu` runs the same kernels as parallel SIMD
loops. `-D CPU_ONLY=true` builds without any GPU toolchain.

## Maps

Built-in `flat`, `ramp`, and `bank` are ribbons around a circle. A YAML map is
either a ROS occupancy image or meshes plus a route:

```yaml
image: track.png          # or .pgm; free pixels >= 230, walls on every edge
resolution: 0.05
origin: [-10.7, -20.6, 0]
```

```yaml
parts:                    # or just `mesh: road.obj`
  - mesh: road.obj
  - mesh: cones.obj
    obstacle: true
route: route.txt          # lines: x y z [half_width]
closed: false
```

Road meshes must be edge-matched (shared edges cancel; the remaining edges are
road boundaries). Slopes up to 45 degrees; no bridges.

```sh
build/racer prepare ../warporacer/maps/my_map.yaml build/my_map.wrmap 0.05
```

## Tests

```sh
build/racer prepare flat build/flat.wrmap
uv run mojo -I . tests/test_mojo.mojo build/flat.wrmap build/ramp.wrmap        # add `cpu` to skip the GPU
```

The suite covers vehicle dynamics, CPU/GPU parity of dynamics, sensing and
gradients, GAE boundaries, checkpoint round trips, the analytic PPO gradient
against central finite differences, map geometry, and a thin wall seen by the
lidar.

## Layout

| file | what it does |
|---|---|
| `main.mojo` | CLI: prepare, train, eval, benchmark, serve |
| `racer/layout.mojo` | arena layout (one float32 buffer), network sizes, shuffle permutation |
| `racer/device.mojo` | CPU/GPU dispatch of thin kernels, RNG hash, float I/O |
| `racer/vehicle.mojo`, `terrain.mojo`, `lidar.mojo` | bicycle dynamics, clearance field and route projection, BVH ray casts |
| `racer/simulation.mojo` | spawn, one environment step, lidar sensing; block-per-car policy kernel |
| `racer/network.mojo` | tanh MLP forward/backward, 64x64 tiled GPU matmul, split-K weight gradients |
| `racer/ppo.mojo` | rollout, GAE, clipped PPO loss, Adam |
| `racer/checkpoint.mojo` | `.wrppo` checkpoints (model, Adam moments, step, seed) |
| `racer/track.mojo`, `compile.mojo`, `image.mojo` | routes, meshes, OBJ/YAML/PGM/PNG loading, distance field, BVH, spawn validation |
| `racer/http.mojo` | blocking HTTP/1.0 server on libc sockets for the viewer |
| `viewer.html` | WebGL viewer; decodes the raw `.wrmap` and binary state snapshots |

Per step the GPU runs three launches (policy MLP as one 64-thread block per
car, dynamics per car, lidar per ray); training updates use ten launches per
minibatch. Observations are padded with a constant-1 lane so biases are
weight rows and every matmul width is a multiple of 16.
