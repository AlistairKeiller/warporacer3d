# Mojo racer architecture

The runtime owns only the flat-road/ramp racing problem. It consists of shared
scalar vehicle/terrain kernels and a fixed small PPO network. Python imports maps
once; Mojo owns all simulation and learning state thereafter.

| File | Responsibility |
| --- | --- |
| `prepare.py` | Load meshes/images, bake conservative clearance, height, road gradients, route, and spawns |
| `racer/core.mojo` | Own two device buffers, CPU/GPU dispatch, seeded random numbers, binary reads |
| `racer/terrain.mojo` | Validate assets, query support, march lidar, project onto a local route window |
| `racer/vehicle.mojo` | Seven-state bicycle and its surface-relative frame |
| `racer/env.mojo` | Four substeps, swept chassis checks, reward/termination/reset, observations |
| `racer/network.mojo` | Arena offsets, initialization, MAX matmul, fixed network forward/backward |
| `racer/ppo.mojo` | Rollout, GAE, minibatches, clipped losses, gradient clipping, Adam |
| `racer/checkpoint.mojo` | Versioned model/optimizer checkpoint, atomic replacement |
| `racer/evaluate.mojo` | Deterministic driving and aggregate metrics |
| `racer/interactive.mojo` | Local viewer controls and host snapshots |
| `main.mojo` | Benchmark/train/eval/serve entry point |
| `app.py`, `ui/viewer.html` | Local HTTP host and dependency-free WebGL viewer |

## Vehicle and coordinates

World coordinates are XYZ, with Z up. The dynamic state is `(x, y, heading, u, v,
yaw_rate, steering_angle)`. Heading is the azimuth of the forward direction;
`u`, `v`, and yaw rate are expressed in the road-relative frame. The car weighs
3 kg, has a 0.3302 m wheelbase, and a 0.58 × 0.38 m chassis.

At each substep the map supplies height and road gradients `(sx, sy)`. Normalize
`(cos(heading), sin(heading), sx*cos(heading)+sy*sin(heading))` for the forward
axis, and `(-sx, -sy, 1)` for the road normal. Their cross product gives the left
axis. Gravity contributes `-g*forward.z` and `-g*left.z` to the two velocity
components; available normal load is `mass*g*normal.z`. This supplies uphill,
downhill, and cross-slope effects. The browser derives exactly the same pose.

Steering actions command angle rate, bounded to ±3.2 rad/s, with an angle limit
of ±0.4189 rad. Throttle commands signed motor force, reduced by a simple speed
curve. Linear tire slip forces use front/rear contact velocities, regularize the
speed denominator at 0.5 m/s, and saturate at the friction limit. The rear tire
shares its friction circle between drive and lateral force. Drag dissipates
forward velocity. Grip and motor strength vary by ±15% on each episode.

The stiff lateral/yaw pair is solved as a 2×2 implicit linear update before force
saturation. Four 1/240 s substeps make one 1/60 s action step. Reverse uses the
same equations, avoiding divisions by signed or zero speed. Convert local yaw
rate back to azimuth rate using the surface frame. When the road grade changes,
project the old tangent velocity into the new tangent plane. A sharp transition
can dissipate energy; this is prescribed road following, without flight or
suspension dynamics.

## Geometry and sensing

The offline compiler rasterizes upward-facing road triangles and their planar
height/gradients. Projected obstacle faces occupy every cell they intersect,
including zero-width vertical walls. The union of road polygons supplies actual
boundaries: internal tessellation edges disappear, while even sub-cell road
holes remain blocked. Overlapping heights at one XY position are rejected.

The clearance field is a Euclidean distance transform to occupied cell centres,
minus half a cell diagonal. Runtime lookup subtracts the query's displacement
from the selected cell centre as well. This is a conservative lower bound, so
sphere tracing cannot step over an occupied wall cell. Lidar has a finite march
budget and may shorten grazing rays rather than overshoot a barrier. Height uses
the stored plane, so flat and ramp interiors are not stair-stepped.

Three circles centred at longitudinal offsets `-0.15, 0, +0.15` m, each of radius
0.24 m, enclose the rectangular chassis. Sample them before and after every
substep, inflating by half the maximum centre motion plus 1 mm. This catches thin
walls and road edges conservatively, including turning. It can reject a near
miss; there is no bounce or sliding contact response.

The route is resampled uniformly to at most 0.15 m spacing. Search ±8 segments
around the previous segment, instead of jumping to a globally nearest segment
at a crossing. Project progress in 3D arc length, handle closed-route wrapping,
and bound reward progress by actual distance travelled. Open routes end near
the final station. Route width also supplies an off-course check.

Observations have eight scaled values (steering, forward/lateral velocity, yaw
rate, longitudinal/lateral gravity direction, normal Z, clearance), followed by
64 default planar lidar beams. These rays query XY road boundaries/obstacle
projections; they are not physical horizontal 3D rays. Range is 10 m and FOV 270°.
Height and orientation are derived from terrain, never integrated independently.

## Ownership and execution

A `Device` owns the terrain buffer and one Float32 arena. The arena begins with
state `[N,14]`, observations `[N,OBS]`, actions `[N,2]`, and reward/terminal reason
`[N,2]`. The remaining training regions contain weights, gradients, two Adam
moments, rollout storage, activations/deltas/transposes, and reduction scratch.
`Memory` calculates every region in one place. No hot-path allocation stores
vehicle or rollout state outside these buffers.

Each GPU thread integrates one car, including termination and deterministic
reset. Lidar runs one thread per ray. CPU dispatch calls the identical functions
through synchronous `parallelize`; no asynchronous closure retains borrowed
pointers. Only matrix multiplication has a separate CPU/GPU library call. MAX
owns portable GEMM optimization, so the project has no Metal/CUDA/HIP kernels.

A normal training step never transfers cars, observations, actions, rollouts,
weights, or gradients to the host. Eight statistics are read once per iteration;
checkpoints read the parameter region periodically. The browser deliberately
uses a separate snapshot path and does not participate in training.

## Learning and episodes

The default network is `72 → 64 tanh → 64 tanh → (two Gaussian means, value)` plus
two learned log standard deviations: 9,029 parameters. Explicit derivatives
cover this fixed network and clipped PPO losses; there is no general autodiff
system. MAX's fast GPU matmul may use reduced product precision; the gradient
oracle checks both individual error and relative norm. CPU uses stricter FP32
checks.

Collect 32 steps/car. Compute GAE with gamma 0.99 and lambda 0.95, normalize
advantages across the rollout, then train four epochs with four minibatches.
Two modular shears provide a bijective shuffle for arbitrary batch sizes without
a shuffle buffer. PPO ratio/value clips are 0.2, value loss weight is 0.5,
gradient norm limit is 0.5, and Adam uses learning rate 3e-4 and epsilon 1e-5.
Observations have fixed physical scaling; there are no running RMS buffers or
adaptive learning-rate machinery. Gaussian samples are stored before actuator
clamping, so their likelihoods stay consistent.

Reward is ten times progress in metres, minus a small steering penalty. A road
or wall failure subtracts two; an open-route finish adds two. Reasons are
`0=running, 1=collision/unsupported/off-course, 2=time limit, 3=finish`.
Episodes are explicitly finite at 3,000 steps, so every terminal reason stops
GAE bootstrapping. A rollout boundary still bootstraps normally. Finished cars
reset in the same kernel; their terminal reward/reason accompanies the next
spawn's observation. Viewer episode counters/trails expose these resets.

Checkpoints store version, network dimensions, lidar range, iteration, optimizer step, full
32-bit seed, weights, gradients, and Adam moments. Write a temporary file, close
it, then atomically rename it over the destination. They can move between batch
sizes and backends; resume starts fresh simulation episodes. Sensor dimensions
and range must match. Assets and checkpoints are validated before entering pointer kernels.

## Deliberate limits and validation

This model stays attached to the road at a crest. It does not simulate wheels,
vertical acceleration, jumps, suspension, collision response, deformable terrain,
overpasses, or car-to-car collisions. Use the retained Newton implementation for
those tasks. Tire constants are a compact racing approximation, not a calibrated
replacement for CommonRoad's passenger-car MB model.

Behavior checks cover rest, drive/coast/reverse, slope gravity, orthonormal
frames, shuffling, CPU/Metal parity on flat/ramp maps, pure sensing, selective
reset/time limits, GAE episode boundaries, and cross-device checkpoints. Geometry
checks cover bridges, invalid assets, out-of-road routes, and sub-cell walls.
An independent Torch autograd oracle checks every PPO/network derivative.
Longer training and deterministic evaluation establish that the model learns
rather than merely running quickly. The previous Python tests still pass.
