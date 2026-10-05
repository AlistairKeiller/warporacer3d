"""Racer: train and evaluate PPO drivers on a track, benchmark, or serve the viewer."""
from std.sys import argv
from std.time import perf_counter
from std.math import isfinite
from std.python import Python, PythonObject
from std.python.numpy import copy_to_numpy_array, from_numpy_array
from racer.device import Device, Buffer, GPU_AVAILABLE, mat
from racer.sim import Sim, STATE, IN, OUT, BEAMS, RANGE, MOUNT, CRASH
from racer.ppo import Policy, Activations, Trainer, ROLLOUT, save, load

comptime USAGE = """racer train     TRACK [auto|gpu|cpu] [cars] [iterations] [checkpoint.npy] [resume.npy]
racer eval      TRACK [auto|gpu|cpu] [cars] [steps] [checkpoint.npy]
racer benchmark TRACK [auto|gpu|cpu] [cars] [steps]
racer serve     TRACK [auto|gpu|cpu] [cars] [port] [checkpoint.npy|-]
TRACK is flat, ramp, bank, bridge, or a map YAML (see racer/track.py)."""


def arg(i: Int, default: String) -> String:
    return String(argv()[i]) if len(argv()) > i else default


struct Runner:
    """A device, `n` cars on a track, and a policy driving them."""

    var device: Device
    var sim: Sim
    var policy: Policy
    var acts: Activations
    var scratch: Buffer  # [2, n]: log probabilities and values, unused outside training

    def __init__(out self, gpu: Bool, track: Span[Float32, _], n: Int) raises:
        self.device = Device(gpu)
        print("Device:", self.device.ctx.name())
        self.sim = Sim(self.device, track, n, 42)
        self.policy = Policy(self.device, 42)
        self.acts = Activations(self.device, n)
        self.scratch = self.device.alloc(2 * n)
        self.sim.reset(self.device)

    def step(mut self, deterministic: Bool, t: Int) raises:
        """One environment step for every car, acting with the policy."""
        var n = self.sim.n
        self.policy.forward(self.device, mat[IN](self.sim.obs, n), self.acts)
        self.sim.sample(
            self.device, mat[OUT](self.acts.o, n), self.policy.log_std(), mat[2](self.sim.actions, n),
            mat[1](self.scratch, n), mat[1](self.scratch, n, n), deterministic, t,
        )
        self.physics()

    def physics(mut self) raises:
        var n = self.sim.n
        self.sim.step(self.device, mat[2](self.sim.actions, n), mat[1](self.sim.reward, n), mat[1](self.sim.done, n), mat[IN](self.sim.obs, n))

    def train(mut self, iterations: Int, path: String, resume: String) raises:
        var trainer = Trainer(self.device, self.sim.n)
        var first = load(self.device, self.policy, resume) if resume else 0
        var start = perf_counter()
        for iteration in range(first, first + iterations):
            var reward = trainer.rollout(self.device, self.sim, self.policy, self.acts, iteration)
            var kl = trainer.update(self.device, self.policy)
            if not isfinite(kl):
                raise Error("training diverged (nonfinite KL)")
            print("iteration", iteration + 1, "reward/step", reward, "KL", kl)
            if (iteration + 1) % 100 == 0:
                save(self.device, self.policy, path, iteration + 1)
        save(self.device, self.policy, path, first + iterations)
        var rate = Float64(self.sim.n * ROLLOUT * iterations) / (perf_counter() - start)
        print("Saved", path, ";", rate, "training transitions/s")

    def evaluate(mut self, steps: Int) raises:
        var reward: Float64 = 0
        var crashes = 0
        for t in range(steps):
            self.step(True, t)
            for r in self.device.read(self.sim.reward):
                reward += Float64(r)
            for d in self.device.read(self.sim.done):
                crashes += Int(d == CRASH)
        var n = Float64(self.sim.n)
        print("reward/step", reward / (n * Float64(steps)), "crashes/car", Float64(crashes) / n)

    def benchmark(mut self, steps: Int) raises:
        """Time `steps` policy steps of every car after a short warm-up."""
        for t in range(10):
            self.step(False, t)
        self.device.ctx.synchronize()
        var start = perf_counter()
        for t in range(steps):
            self.step(False, t)
        self.device.ctx.synchronize()
        var ms = (perf_counter() - start) * 1000 / Float64(steps)
        print(self.sim.n, "cars;", ms, "ms/step;", Float64(self.sim.n) * 1000 / ms, "transitions/s")

    def serve(mut self, track: PythonObject, name: String, port: Int, trained: Bool) raises:
        """Drive the viewer: Python serves HTTP, each POST /step runs one step
        and answers with a snapshot of every buffer."""
        var n = self.sim.n
        var info = Python.dict(
            name=name, cars=n, policy=trained, state=STATE, obs=IN, beams=BEAMS, range=Float64(RANGE),
            mount=Python.list(Float64(MOUNT[0]), 0, Float64(MOUNT[2])),
        )
        var bridge = Python.import_module("racer.serve").Bridge(port, open("viewer.html", "r").read(), info, track.tobytes())
        print("http://127.0.0.1:" + String(port) + " (Ctrl+C to stop)", flush=True)
        var snapshot = List[Float32](length=(STATE + IN + 4) * n, fill=0)
        var tick = 0
        while True:
            # Request: steering% throttle% reset policy.
            var words = String(bridge.requests.get()).split()
            if Int(words[2]) > 0:
                self.sim.reset(self.device)
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
            bridge.replies.put(copy_to_numpy_array(Span(snapshot)).tobytes())


def main() raises:
    var mode = arg(1, "")
    if len(argv()) < 3 or mode not in ["train", "eval", "benchmark", "serve"]:
        print(USAGE)
        return
    Python.add_to_path(".")
    var track = Python.import_module("racer.track").load(arg(2, ""))
    var choice = arg(3, "auto")
    var count = Int(arg(5, "8765" if mode == "serve" else "1000"))
    var path = arg(6, "agent.npy")
    var runner = Runner(GPU_AVAILABLE if choice == "auto" else choice == "gpu", from_numpy_array[DType.float32](track), Int(arg(4, "1024")))
    if mode == "train":
        runner.train(count, path, arg(7, ""))
        return
    var trained = mode == "eval" or (mode == "serve" and path != "-")
    if trained:
        _ = load(runner.device, runner.policy, path)
    if mode == "serve":
        runner.serve(track, arg(2, ""), count, trained)
    elif mode == "benchmark":
        runner.benchmark(count)
    else:
        runner.evaluate(count)
