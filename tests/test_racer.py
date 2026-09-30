"""Behavior checks for CPU debug mode and CUDA (RACER_DEVICE=cuda:0)."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from warporacer.agent import Agent
from warporacer.ppo import PPO, gae
from warporacer.sim import Env, RouteData, SimConfig, project_route, route_delta
from warporacer.track import MeshPart, Route, Track, demo_track, ribbon

DEVICE = os.environ.get("RACER_DEVICE") or None
SMALL = {"lidar_beams": 3, "lidar_elevations": (0,)}


def straight(slope=0, width=20):
    route = Route(
        [[-50, 0, -50 * slope], [50, 0, 50 * slope]], half_width=width, closed=False
    )
    return Track([ribbon(route)], route)


def environment(track=None, n=1, **config):
    return Env(
        track or straight(), n, device=DEVICE, config=SimConfig(**(SMALL | config))
    )


def advance(env, steps, actions=(0, 0)):
    action = (
        torch.tensor(actions, device=env.torch_device).float().expand(env.num_envs, 2)
    )
    terminals = []
    for _ in range(steps):
        env.step(action)
        terminals.extend(env.reason[env.done.bool()].cpu().tolist())
    return terminals


def place(env, position, rotation=(0, 0, 0, 1), speed=0):
    """Place the entire car coherently for isolated collision/sensor tests."""
    batch = env.batches[0]
    rest = batch.rest.numpy()
    rot = Rotation.from_quat(rotation)
    pose = np.column_stack(
        [
            rot.apply(rest[:, :3]) + position,
            (rot * Rotation.from_quat(rest[:, 3:])).as_quat(),
        ]
    )
    velocity = np.zeros((len(pose), 6), np.float32)
    velocity[:, :3] = rot.apply([speed, 0, 0])
    for state in (batch.state, batch.other):
        state.body_q.assign(pose)
        state.body_qd.assign(velocity)
    batch.episodes.position.assign([position])
    # Find the corresponding arc position once; normal stepping uses a local window.
    route = env.tracks[0].route
    t = np.clip(
        ((position - route.points[: len(route.delta)]) * route.delta).sum(1)
        / route.lengths**2,
        0,
        1,
    )
    j = np.argmin(
        np.linalg.norm(
            position - (route.points[: len(t)] + t[:, None] * route.delta), axis=1
        )
    )
    batch.episodes.progress.assign([route.distance[j] + t[j] * route.lengths[j]])
    batch.episodes.crashed.zero_()
    env.observe()


@wp.kernel
def projections(
    route: RouteData,
    points: wp.array[wp.vec3],
    previous: wp.array[float],
    result: wp.array[wp.vec4],
):
    i = wp.tid()
    result[i] = project_route(points[i], previous[i], 0.8, route)


@wp.kernel
def progress_deltas(
    route: RouteData, pairs: wp.array[wp.vec2], result: wp.array[float]
):
    i = wp.tid()
    result[i] = route_delta(pairs[i][0], pairs[i][1], route)


class PhysicsTests(unittest.TestCase):
    def test_rest_and_timestep_convergence(self):
        heights = []
        for substeps in (24, 48):
            env = environment(substeps=substeps, iterations=32)
            self.assertEqual(advance(env, 90), [])
            heights.append(float(env.poses_t[0, 2]))
            self.assertLess(float(env.obs[0, 1:7].abs().max()), 0.03)
            self.assertEqual(env.obs[0, 10:14].tolist(), [1] * 4)
        self.assertGreater(heights[0], 0.1)
        self.assertLess(abs(heights[0] - heights[1]), 0.004)

    def test_drive_reverse_and_world_isolation(self):
        env = environment(n=2)
        before = env.poses_t.clone()
        actions = torch.tensor([[0, 0.6], [0, 0]], device=env.torch_device)
        for _ in range(120):
            env.step(actions)
            self.assertFalse(bool(env.done.any()))
        self.assertGreater(float(env.poses_t[0, 0] - before[0, 0]), 2)
        self.assertLess(float((env.poses_t[1, :2] - before[1, :2]).abs().max()), 0.03)
        self.assertLess(
            abs(float(env.obs[0, 0])), 0.03
        )  # many complete wheel revolutions
        reverse = environment()
        self.assertEqual(advance(reverse, 90, (0, -0.5)), [])
        self.assertLess(float(reverse.poses_t[0, 0]), -1)
        self.assertLess(float(reverse.rew[0]), 0)

    def test_steering_turns_and_respects_limits(self):
        env = environment()
        self.assertEqual(advance(env, 90, (0.15, 0.25)), [])
        self.assertGreater(float(env.poses_t[0, 1]), 0.1)
        self.assertGreater(float(env.obs[0, 0]), 0.2)
        self.assertLess(abs(float(env.obs[0, 0])), env.car.steer_max + 0.02)
        self.assertGreater(abs(float(env.poses_t[0, 5])), 0.05)

    def test_motor_speed_and_unpowered_coasting(self):
        env = environment()
        self.assertEqual(advance(env, 150, (0, 0.8)), [])
        speed = float(env.obs[0, 1])
        self.assertGreater(speed, 2)
        self.assertLess(speed, env.car.speed_limit + 0.2)
        self.assertEqual(advance(env, 90), [])
        self.assertLess(float(env.obs[0, 1]), speed - 0.2)

    def test_incline_gravity_and_banking(self):
        env = environment(straight(0.15))
        start = env.poses_t.clone()
        self.assertGreater(abs(float(env.obs[0, 7])), 0.1)
        self.assertEqual(advance(env, 100), [])
        self.assertLess(float(env.poses_t[0, 0]), float(start[0, 0]) - 0.1)
        self.assertLess(float(env.poses_t[0, 2]), float(start[0, 2]))
        bank = environment(demo_track("bank"))
        self.assertGreater(abs(float(bank.obs[0, 8])), 0.15)
        self.assertEqual(advance(bank, 60), [])

    def test_airborne_and_landing(self):
        env = environment()
        place(env, [0, 0, 1])
        self.assertEqual(env.obs[0, 10:14].tolist(), [0] * 4)
        self.assertEqual(advance(env, 10), [])
        self.assertLess(float(env.poses_t[0, 2]), 1)
        self.assertEqual(advance(env, 80), [])
        self.assertEqual(env.obs[0, 10:14].tolist(), [1] * 4)
        self.assertLess(float(env.poses_t[0, 2]), 0.2)

    def test_wall_at_driving_speed(self):
        import trimesh

        road = straight()
        box = trimesh.creation.box(extents=(0.1, 10, 1))
        box.apply_translation([1, 0, 0.5])
        road.parts.append(MeshPart(box.vertices, box.faces, obstacle=True))
        env = environment(road)
        place(env, [0, 0, env.car.ride_height], speed=5)
        reasons = advance(env, 20)
        self.assertIn(1, reasons)
        self.assertTrue(bool(torch.isfinite(env.obs).all()))

    def test_ramp_gap_takeoff_and_landing(self):
        env = environment(demo_track("jump"))
        place(env, [4, 0, env.car.ride_height])
        action = torch.tensor([[0.0, 0.8]], device=env.torch_device)
        airborne, landed = False, False
        for _ in range(220):
            env.step(action)
            self.assertFalse(bool(env.done.any()))
            contacts = int(env.obs[0, 10:14].sum())
            x = float(env.poses_t[0, 0])
            airborne |= x > 10 and contacts == 0
            landed |= airborne and x > 11.2 and contacts == 4
        self.assertTrue(airborne)
        self.assertTrue(landed)


class EnvironmentTests(unittest.TestCase):
    def test_observation_and_selective_full_car_reset(self):
        env = environment(n=2)
        advance(env, 8, (0.2, 0.3))
        batch = env.batches[0]
        pose = batch.state.body_q.numpy().copy()
        clocks = batch.episodes.steps.numpy().copy()
        resets = batch.episodes.resets.numpy().copy()
        obs = env.obs.clone()
        env.observe()
        np.testing.assert_array_equal(batch.state.body_q.numpy(), pose)
        np.testing.assert_array_equal(batch.episodes.steps.numpy(), clocks)
        np.testing.assert_array_equal(batch.episodes.resets.numpy(), resets)
        torch.testing.assert_close(env.obs, obs, rtol=0, atol=0)
        bodies = batch.bodies.numpy()
        other_q = batch.state.body_q.numpy()[bodies[1]].copy()
        other_v = batch.state.body_qd.numpy()[bodies[1]].copy()
        other_motor = float(batch.episodes.motor.numpy()[1])
        env.reset([1, 0])
        np.testing.assert_array_equal(batch.state.body_q.numpy()[bodies[1]], other_q)
        np.testing.assert_array_equal(batch.state.body_qd.numpy()[bodies[1]], other_v)
        np.testing.assert_array_equal(batch.state.body_qd.numpy()[bodies[0]], 0)
        np.testing.assert_array_equal(batch.other.body_qd.numpy()[bodies[0]], 0)
        self.assertEqual(float(batch.episodes.motor.numpy()[1]), other_motor)
        self.assertEqual(batch.episodes.steps.numpy().tolist(), [0, clocks[1]])
        self.assertEqual(
            batch.episodes.resets.numpy().tolist(), [resets[0] + 1, resets[1]]
        )
        self.assertEqual(float(batch.episodes.target.numpy()[0]), 0)
        np.testing.assert_array_equal(
            batch.control.joint_target_q.numpy()[batch.steering_target.numpy()[0]], 0
        )
        np.testing.assert_array_equal(
            batch.control.joint_f.numpy()[batch.drive_dof.numpy()[0]], 0
        )

    def test_seeded_resets(self):
        first, second = (
            environment(demo_track("ramp"), n=2),
            environment(demo_track("ramp"), n=2),
        )
        for mask in ([1, 0], [0, 1], [1, 1]):
            first.reset(mask)
            second.reset(mask)
            torch.testing.assert_close(first.obs, second.obs, rtol=0, atol=0)
            torch.testing.assert_close(first.poses_t, second.poses_t, rtol=0, atol=0)
            np.testing.assert_array_equal(
                first.batches[0].episodes.motor.numpy(),
                second.batches[0].episodes.motor.numpy(),
            )

    def test_rotation_preserves_output_buffers_and_multi_map(self):
        env = environment(n=2)
        pointers = [t.data_ptr() for t in (env.obs, env.rew, env.done, env.poses_t)]
        env.rotate([demo_track("ramp"), demo_track("overpass")])
        self.assertEqual(
            pointers, [t.data_ptr() for t in (env.obs, env.rew, env.done, env.poses_t)]
        )
        self.assertEqual(len(env.batches), 2)
        self.assertEqual(advance(env, 3), [])
        self.assertTrue(bool(torch.isfinite(env.obs).all()))
        if env.device.is_cuda:
            for batch in env.batches:
                self.assertEqual(batch.state.body_q.device, env.device)
                self.assertEqual(batch.contacts.rigid_contact_count.device, env.device)
            self.assertTrue(env.obs.is_cuda)

    def test_timeout_and_open_route_finish(self):
        env = environment(max_steps=2)
        advance(env, 1)
        self.assertEqual(advance(env, 1), [4])
        self.assertEqual(int(env.batches[0].episodes.steps.numpy()[0]), 0)
        place(env, [49.9, 0, env.car.ride_height])
        self.assertEqual(advance(env, 1), [5])

    def test_prolonged_rollover(self):
        env = environment(rollover_steps=3)
        place(env, [0, 0, 1], Rotation.from_euler("x", 180, degrees=True).as_quat())
        self.assertEqual(advance(env, 3), [3])
        self.assertLess(float(env.obs[0, 9]), -0.99)  # respawned upright

    def test_overpass_lidar_uses_full_pose_and_correct_level(self):
        env = environment(
            demo_track("overpass"),
            lidar_beams=1,
            lidar_fov=0,
            lidar_elevations=(-90, 0, 90),
            lidar_mount=(0, 0, 0.06),
        )
        h = env.car.ride_height
        place(env, [0, 0, h])
        self.assertAlmostEqual(float(env.obs[0, 14]), h + 0.06, delta=0.02)
        self.assertAlmostEqual(float(env.obs[0, 16]), 2 - 0.15 - h - 0.06, delta=0.04)
        place(env, [0, 0, 2 + h])
        self.assertAlmostEqual(float(env.obs[0, 14]), h + 0.06, delta=0.02)
        self.assertEqual(float(env.obs[0, 16]), env.config.lidar_range)
        # Roll the sensor 180 degrees: its down ray now points up into the bridge.
        place(env, [0, 0, h], Rotation.from_euler("x", 180, degrees=True).as_quat())
        self.assertAlmostEqual(float(env.obs[0, 14]), 2 - 0.15 - h + 0.06, delta=0.04)

    def test_crossing_continuity_and_lap_wrapping(self):
        track = demo_track("overpass")
        env = environment(track)
        batch = env.batches[0]
        # Same XY, different heights and distant arc positions.
        points = wp.array([[0, 0, 2], [0, 0, 0]], dtype=wp.vec3, device=env.device)
        previous = wp.array(
            [0.1, track.route.distance[128]], dtype=float, device=env.device
        )
        result = wp.empty(2, dtype=wp.vec4, device=env.device)
        wp.launch(
            projections,
            2,
            inputs=[batch.route, points, previous, result],
            device=env.device,
        )
        self.assertAlmostEqual(float(result.numpy()[0, 0]), 0, delta=0.01)
        self.assertAlmostEqual(
            float(result.numpy()[1, 0]), float(track.route.distance[128]), delta=0.01
        )
        # Now put both crossings on the same level; route adjacency still selects the branch.
        route = Route(track.route.points * np.array([1, 1, 0]))
        batch.route.points.assign(route.points)
        batch.route.delta.assign(route.delta)
        batch.route.lengths.assign(route.lengths)
        batch.route.distance.assign(route.distance)
        batch.route.total = route.total
        points.assign([[0, 0, 0], [0, 0, 0]])
        previous.assign([0.1, route.distance[128]])
        wp.launch(
            projections,
            2,
            inputs=[batch.route, points, previous, result],
            device=env.device,
        )
        self.assertAlmostEqual(
            float(result.numpy()[1, 0]), float(route.distance[128]), delta=0.01
        )
        pairs = wp.array(
            [[0.1, route.total - 0.1], [route.total - 0.1, 0.1]],
            dtype=wp.vec2,
            device=env.device,
        )
        deltas = wp.empty(2, dtype=float, device=env.device)
        wp.launch(
            progress_deltas, 2, inputs=[batch.route, pairs, deltas], device=env.device
        )
        np.testing.assert_allclose(deltas.numpy(), [0.2, -0.2], atol=1e-4)


class MapAndLearningTests(unittest.TestCase):
    def test_mesh_indices_cannot_wrap_into_valid_triangles(self):
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        for dtype in (np.int64, np.uint64):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                MeshPart(vertices, np.array([[2**32, 1, 2]], dtype=dtype))

    def test_mesh_import_bakes_scene_transforms(self):
        import trimesh

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route = straight().route
            np.savez(
                root / "route.npz", points=route.points, half_width=route.half_width
            )
            mesh = ribbon(route)
            np.savez(
                root / "road.npz", vertices=mesh.vertices, triangles=mesh.triangles
            )
            (root / "map.yaml").write_text(
                "mesh: road.npz\nroute: route.npz\nclosed: false\n"
            )
            loaded = Track.load(root / "map.yaml")
            np.testing.assert_array_equal(loaded.parts[0].vertices, mesh.vertices)
            scene = trimesh.Scene()
            transform = np.eye(4)
            transform[2, 3] = 3
            scene.add_geometry(
                trimesh.Trimesh(mesh.vertices, mesh.triangles), transform=transform
            )
            scene.export(root / "road.glb")
            (root / "map.yaml").write_text(
                "mesh: road.glb\nroute: route.npz\nclosed: false\n"
            )
            self.assertAlmostEqual(float(Track.load(root / "map.yaml").bounds[1, 2]), 3)

    def test_legacy_image_and_all_demo_maps(self):
        root = Path(__file__).resolve().parents[1]
        legacy = Track.load(root / "maps/my_map.yaml")
        self.assertEqual(len(legacy.parts), 2)
        self.assertTrue(legacy.parts[1].obstacle)
        self.assertTrue(legacy.route.closed)
        env = environment(legacy)
        self.assertEqual(advance(env, 2), [])
        for path in sorted((root / "maps/3d").glob("*.yaml")):
            env.rotate(Track.load(path))
            self.assertTrue(bool(torch.isfinite(env.obs).all()))
            self.assertEqual(advance(env, 2), [])
            if path.stem == "jump":
                batch = env.batches[0]
                centers = batch.track.route.points[:-1] + 0.5 * batch.track.route.delta
                self.assertFalse(
                    np.any(
                        (centers[batch.route.spawns.numpy(), 0] > 10)
                        & (centers[batch.route.spawns.numpy(), 0] < 10.6)
                    )
                )

    def test_gae_stops_at_terminal(self):
        rewards = torch.tensor([[1.0], [2.0], [3.0]])
        dones = torch.tensor([[0.0], [1.0], [0.0]])
        values = torch.tensor([[0.5], [0.7], [100.0], [101.0]])
        result = gae(rewards, dones, values, gamma=1, lam=1)
        torch.testing.assert_close(result, torch.tensor([[2.5], [1.3], [4.0]]))

    def test_ppo_update(self):
        env = environment(n=2, max_steps=3)
        agent = Agent(env.obs_dim, env.act_dim, hidden=32).to(env.torch_device)
        before = torch.cat([p.detach().flatten().clone() for p in agent.parameters()])
        ppo = PPO(env, agent, rollouts=4, epochs=1, minibatches=2)
        log = ppo.iterate()
        after = torch.cat([p.detach().flatten() for p in agent.parameters()])
        self.assertTrue(all(np.isfinite(v) for v in log.values()))
        self.assertFalse(torch.equal(before, after))
        self.assertEqual(ppo.global_step, 8)
        self.assertIn("ep_return", log)


@unittest.skipUnless(torch.cuda.is_available(), "Requires an NVIDIA CUDA device")
class CudaTests(unittest.TestCase):
    def test_switching_torch_streams_orders_inputs_and_outputs(self):
        device = DEVICE if DEVICE and DEVICE.startswith("cuda") else "cuda:0"
        env = Env(straight(), 1, device=device, config=SimConfig(**SMALL))
        self.assertTrue(env.stream.is_blocking)
        env.step(torch.zeros((1, 2), device=env.torch_device))  # compile before delays
        streams = [torch.cuda.Stream(device=env.torch_device) for _ in range(2)]
        snapshots = []
        target = 0.0
        for k, steering in enumerate([0.4, -0.2, 0.1, 0.0] * 2):
            with torch.cuda.stream(streams[k % 2]):
                actions = torch.empty((1, 2), device=env.torch_device)
                # Keep the producer busy so an omitted dependency reads stale actions.
                torch.cuda._sleep(2_000_000)
                actions[0, 0], actions[0, 1] = steering, 0.2
                env.step(actions)
                env.observe()
                target += steering * env.car.steer_rate * env.config.dt
                snapshots.append(
                    (
                        wp.to_torch(env.batches[0].episodes.target).clone(),
                        target,
                        env.obs.clone(),
                    )
                )
        for stream in streams:
            stream.synchronize()
        for actual, expected, obs in snapshots:
            self.assertAlmostEqual(float(actual[0]), expected, delta=1e-6)
            self.assertTrue(bool(torch.isfinite(obs).all()))
        with torch.cuda.stream(streams[0]):
            env.reset(torch.ones(1, device=env.torch_device, dtype=torch.int32))
            cleared = wp.to_torch(env.batches[0].episodes.target).clone()
        streams[0].synchronize()
        self.assertEqual(float(cleared[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
