"""Racer: compile maps, train and evaluate PPO drivers, and serve the viewer."""
from std.sys import argv
from std.time import perf_counter, perf_counter_ns
from std.math import isfinite
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, Unit
from std.python import Python, PythonObject
from std.python.numpy import copy_to_numpy_array
from racer.device import Device, Buffer, GPU_AVAILABLE, mat, read_floats, write_floats
from racer.compile import load_track, compile
from racer.simulation import Sim, STATE, IN, OUT, CRASH, FINISH
from racer.network import Policy, Activations
from racer.ppo import Trainer, ROLLOUT, checkpoint
from racer.lidar import BEAMS, RANGE, MOUNT_FORWARD, MOUNT_HEIGHT

comptime USAGE = """racer prepare   flat|ramp|bank|MAP.yaml OUT.wrmap [resolution=0.025]
racer train     MAP.wrmap [auto|gpu|cpu] [cars] [iterations] [checkpoint] [resume]
racer eval      MAP.wrmap [auto|gpu|cpu] [cars] [steps] [checkpoint]
racer benchmark MAP.wrmap [auto|gpu|cpu] [cars] [steps]
racer serve     MAP.wrmap [auto|gpu|cpu] [cars] [port] [checkpoint|-]"""


def arg(i: Int, default: String) -> String:
    """Positional argument `i`, or `default` when it was not given."""
    return String(argv()[i]) if len(argv()) > i else default


def as_bytes(values: List[Float32]) raises -> PythonObject:
    return copy_to_numpy_array(Span(values)).tobytes()


struct Runner:
    """A device, `n` cars on a map, and a policy driving them."""

    var device: Device
    var sim: Sim
    var policy: Policy
    var acts: Activations
    var scratch: Buffer  # [2, n]: log probabilities and values nobody reads

    def __init__(out self, gpu: Bool, map_data: List[Float32], n: Int) raises:
        self.device = Device(gpu)
        print("Device:", self.device.ctx.name())
        self.sim = Sim(self.device, map_data, n, 42)
        self.policy = Policy(self.device, 42)
        self.acts = Activations(self.device, n)
        self.scratch = self.device.alloc(2 * n)
        self.sim.spawn_all(self.device)
        self.sim.sense(self.device, mat[IN](self.sim.obs, n))

    def step(mut self, deterministic: Bool, t: Int) raises:
        """One environment step for every car, acting with the policy."""
        var n = self.sim.n
        self.policy.forward(self.device, mat[IN](self.sim.obs, n), self.acts)
        self.sim.sample(
            self.device,
            mat[OUT](self.acts.o, n),
            self.policy.log_std(),
            mat[2](self.sim.actions, n),
            mat[1](self.scratch, n),
            mat[1](self.scratch, n, n),
            deterministic,
            t,
        )
        self.physics()

    def physics(mut self) raises:
        """Advance every car with the actions already in `sim.actions`."""
        var n = self.sim.n
        self.sim.physics(
            self.device,
            mat[2](self.sim.actions, n),
            mat[1](self.sim.reward, n),
            mat[1](self.sim.done, n),
        )
        self.sim.sense(self.device, mat[IN](self.sim.obs, n))

    def train(mut self, iterations: Int, path: String, resume: String) raises:
        var n = self.sim.n
        var trainer = Trainer(self.device, n)
        var first = 0
        if resume:
            var saved = checkpoint(self.device, self.policy, resume)
            trainer.load(self.device, saved)
            first = Int(saved[3])
        var start = perf_counter()
        for iteration in range(first, first + iterations):
            var reward = trainer.rollout(self.device, self.sim, self.policy, iteration)
            var kl = trainer.update(self.device, self.policy, iteration)
            if not isfinite(kl):
                raise Error("training diverged (nonfinite KL)")
            print("iteration", iteration + 1, "reward/step", reward, "KL", kl)
            if (iteration + 1) % 100 == 0:
                trainer.save(self.device, self.policy, path, iteration + 1)
        trainer.save(self.device, self.policy, path, first + iterations)
        var rate = Float64(n * ROLLOUT * iterations) / (perf_counter() - start)
        print("Saved", path, ";", rate, "training transitions/s")

    def evaluate(mut self, steps: Int) raises:
        """Drive deterministically for `steps` and report episode outcomes."""
        var n = self.sim.n
        var reward: Float64 = 0
        var crashes = 0
        var finishes = 0
        for t in range(steps):
            self.step(True, t)
            for r in self.device.read(self.sim.reward):
                reward += Float64(r)
            for d in self.device.read(self.sim.done):
                crashes += Int(d == CRASH)
                finishes += Int(d == FINISH)
        print(
            "reward/step", reward / Float64(n * steps),
            "crashes/car", Float64(crashes) / Float64(n),
            "finishes/car", Float64(finishes) / Float64(n),
        )

    def benchmark(mut self, steps: Int) raises:
        """Time `steps` policy steps of every car after a short warm-up."""

        def timed(mut b: Bencher) raises {ref}:
            """Queue every step, then wait once: the GPU timing is of the stream."""
            var start = perf_counter_ns()
            for t in range(b.num_iters):
                self.step(False, t)
            self.device.ctx.synchronize()
            b.elapsed = Int(perf_counter_ns() - start)

        for t in range(10):
            self.step(False, t)
        self.device.ctx.synchronize()
        var bench = Bench(BenchConfig(num_warmup_iters=0))
        bench.bench_function(timed, BenchId("step"), fixed_iterations=steps)
        print(bench)
        var ms = bench.info_vec[0].result.mean(Unit.ms)
        print(self.sim.n, "cars;", ms, "ms/step;", Float64(self.sim.n) * 1000 / ms, "transitions/s")

    def serve(
        mut self, map_data: List[Float32], name: String, port: Int, trained: Bool
    ) raises:
        """Drive the viewer: Python serves HTTP, each POST /step runs one step
        and answers with every car's state, observations, actions, reward and
        done flag in one feature-major snapshot."""
        var n = self.sim.n
        var info = Python.dict(
            name=name, cars=n, policy=trained, state=STATE, obs=IN, beams=BEAMS,
            range=Float64(RANGE), mount=Python.list(Float64(MOUNT_FORWARD), 0, Float64(MOUNT_HEIGHT)),
        )
        Python.add_to_path(".")
        var bridge = Python.import_module("racer.serve").Bridge(
            port, open("viewer.html", "r").read(), info, as_bytes(map_data)
        )
        print("http://127.0.0.1:" + String(port) + " (Ctrl+C to stop)", flush=True)
        var snapshot = List[Float32](length=(STATE + IN + 4) * n, fill=0)
        var tick = 0
        while True:
            # Request: steering% throttle% reset policy.
            var words = String(bridge.wait()).split()
            if Int(words[2]) > 0:
                self.sim.spawn_all(self.device)
                self.sim.sense(self.device, mat[IN](self.sim.obs, n))
            tick += 1
            if Int(words[3]) > 0 and trained:
                self.step(True, tick)
            else:
                var actions = List[Float32](length=n, fill=Float32(Int(words[0])) / 100)
                actions.extend(List[Float32](length=n, fill=Float32(Int(words[1])) / 100))
                self.device.write(self.sim.actions, actions)
                self.physics()
            var offset = 0
            for buffer in [self.sim.state, self.sim.obs, self.sim.actions, self.sim.reward, self.sim.done]:
                self.device.ctx.enqueue_copy(snapshot.unsafe_ptr().unsafe_offset(offset), buffer)
                offset += len(buffer)
            self.device.ctx.synchronize()
            bridge.reply(as_bytes(snapshot))


def main() raises:
    var mode = arg(1, "")
    if len(argv()) < 3 or mode not in ["prepare", "train", "eval", "benchmark", "serve"]:
        print(USAGE)
        return
    if mode == "prepare":
        var map = compile(load_track(arg(2, "")), Float64(arg(4, "0.025")))
        write_floats(arg(3, ""), map)
        print(
            arg(3, ""), ":", Int(map[2]), "x", Int(map[3]), "cells,", Int(map[8]), "segments,",
            Int(map[14]), "triangles,", Float64(len(map) * 4) / 1e6, "MB",
        )
        return
    var map_data = read_floats(arg(2, ""))
    var choice = arg(3, "auto")
    var count = Int(arg(5, "8765" if mode == "serve" else "1000"))
    var path = arg(6, "agent.wrppo")
    var runner = Runner(GPU_AVAILABLE if choice == "auto" else choice == "gpu", map_data, Int(arg(4, "1024")))
    if mode == "train":
        runner.train(count, path, arg(7, ""))
        return
    var trained = mode == "eval" or (mode == "serve" and path != "-")
    if trained:
        _ = checkpoint(runner.device, runner.policy, path)
    if mode == "serve":
        runner.serve(map_data, arg(2, ""), count, trained)
    elif mode == "benchmark":
        runner.benchmark(count)
    else:
        runner.evaluate(count)
