# Measured performance

Apple M5 Pro: 18 CPU cores, 20 GPU cores, 48 GB; macOS 27.0, Xcode 27.0.
Mojo 1.1.0 / MAX 26.6.0. Measurements taken 2026-10-01. Native Metal kernels and
GPU matmul ran on this machine. These are Apple results, not CUDA/HIP projections.

## Compared with original `warporacer`

The original racer runs Warp simulation on CPU and exposes Torch MPS tensors on
Mac, copying actions/observations across that boundary through its public `step`
API. Mojo keeps the entire simulation/learner on its selected device.

Both use a 6 m-radius, 2.6 m-wide flat circuit and a 2.5 cm grid. The original has
108 beams over 270°, with a 20 m range; the matched Mojo build uses those settings.
Original geometry is an analytic raster annulus; Mojo uses the demo triangle
ribbon and conservative rasterization. Shapes, sensor origins, dynamics, collision
footprints, rewards, and trajectories differ. This compares complete implementations,
not identical physics or an isolated language speedup.

Median of three synchronized 1,000-step timings per configuration. Warmup, map
preparation, compilation, construction, and rendering are excluded. Actions are
held at steering rate 0.1 and throttle 0.7. Both reset terminal cars and recompute
sensing. Original stepping includes the CPU/MPS bridge; Mojo consumes resident
actions, as it does during training.

| Cars | Original CPU + MPS | Mojo CPU, matched lidar | Mojo Metal, matched lidar | Metal / original |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.790 ms | 0.012 ms | 0.075 ms | 10.5× |
| 32 | 1.808 ms | 0.151 ms | 0.084 ms | 21.5× |
| 128 | 4.948 ms | 0.417 ms | 0.088 ms | 56.0× |
| 1,024 | 30.768 ms | 1.881 ms | 0.209 ms | 147.5× |

At 1,024 cars, matched-sensor Metal stepping reaches **4.91 million transitions/s**,
about **147×** the original on this Mac. CPU reaches 544,000 transitions/s at the
same sensor settings. For one car, CPU has lower latency than GPU dispatch.

## Default sensing: 64 beams / 10 m

| Cars | Mojo CPU | Mojo Metal | Metal transitions/s |
| ---: | ---: | ---: | ---: |
| 1 | 0.006 ms | 0.074 ms | 13,454 |
| 32 | 0.106 ms | 0.084 ms | 382,003 |
| 128 | 0.407 ms | 0.083 ms | 1,550,388 |
| 1,024 | 1.167 ms | 0.157 ms | 6,529,863 |

Default Metal stepping reaches **6.53 million transitions/s** at 1,024 cars.
This sensor setting is deliberately smaller than the original, so use the matched
table when assessing the platform change.

## Full PPO training

256 cars on the flat circuit; median of three ten-iteration timings. Original
PPO is warmed up for two iterations. Mojo kernels are built ahead of time; each
run includes logging and its final checkpoint save. These are each implementation’s
actual training defaults, rather than equal-sized network FLOP benchmarks.

| Runtime | Sensor beams / range | Training transitions/s |
| --- | --- | ---: |
| Original Warp + Torch MPS | 108 / 20 m | 21,641 |
| Mojo CPU | 64 / 10 m | 139,932 |
| Mojo Metal | 64 / 10 m | 226,771 |
| Mojo CPU | 108 / 20 m | 127,538 |
| Mojo Metal | 108 / 20 m | 224,092 |

Matched-sensor Mojo Metal training is about **10.4×** faster here. This includes
a smaller shared 64-wide network, four epochs, and 32-step rollouts. Original PPO
uses separate 256-wide actor/critic networks, up to five epochs, 24-step rollouts,
running normalization, and KL-based learning-rate/epoch adjustment. The original
uses MPS fp16 autocast; Mojo uses MAX fast GPU matmul. Simpler learning machinery
and a different model contribute to the result.

A longer ramp run (256 cars, 300 iterations) sustained **228,758 training
transitions/s**. Its reward/step rose from approximately zero to 0.358. A separate
256-car, 1,000-step deterministic ramp evaluation measured reward/step 0.339 with
**zero road/wall failures**. This demonstrates learning on the demo map, not
generalization to every imported track.

## Compared with Newton `warporacer3d`

The retained Newton implementation uses eleven rigid bodies/car, suspension and
wheel contacts, 16 substeps, 12 XPBD iterations, and 324 physical 3D lidar rays.
Its synchronized ramp benchmark uses 30 steps after five warmup steps, with
graph replay enabled. This full-contact model solves a broader problem.

| Cars | Newton CPU step | Transitions/s |
| ---: | ---: | ---: |
| 1 | 2.07 ms | 483 |
| 32 | 38.38 ms | 834 |
| 128 | 151.69 ms | 844 |

Removing contact solving and simplifying sensing account for much of the gap.
The Mojo model prescribes road contact and ends an episode at an edge or wall;
it cannot replace Newton for jumping, suspension response, landings, or bridges.

## Reproduce

From the repository root:

```sh
uv run python compare.py --original ../warporacer --steps 1000 --repeats 3
uv run python compare.py --original ../warporacer --cars 256 --train-iterations 10 --repeats 3 --output build/training-comparison.json
uv run --extra reference python benchmark.py --device cpu --num-envs 1 32 128 --steps 30 --warmup 5 --output build/newton-comparison.json
```

The original sibling repository needs its own working `.venv`. `compare.py`
uses that interpreter in a separate process to avoid package-name collisions.
Raw measurements are archived in [m5-pro.json](docs/benchmarks/m5-pro.json).
CUDA, HIP, and WSL still require hardware validation; their speed is not measured here.
