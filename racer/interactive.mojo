"""Line-based local viewer protocol. Training never enters this host I/O path."""
from std.io.io import _fdopen
from std.sys import stdin
from .core import Device, Params, Ptr
from .env import spawn, step, observe, env_size, action_offset, reward_offset, OBS, BEAMS
from .network import forward
from .ppo import input as policy_input, sample


def controls(i: Int, data: Ptr, map: Ptr, p: Params):
    var n = Int(p.envs)
    var a = action_offset(n) + i * 2
    data[unsafe_offset=a] = Float32(p.offset) / 100
    data[unsafe_offset=a + 1] = Float32(p.index) / 100
    if p.flag > 0:
        spawn(i, data, map, p)
    var r = reward_offset(n) + i * 2
    data[unsafe_offset=r] = 0
    data[unsafe_offset=r + 1] = 0


def serve(mut device: Device, n: Int, seed: UInt32) raises:
    print('{"ready":true}', flush=True)
    var snapshot = device.data.create_sub_buffer[DType.float32](0, env_size(n))
    # Mojo 1.1 input() leaks a duplicated descriptor per call. Reuse one
    # buffered stream, whose context closes the descriptor on exit.
    with _fdopen["r"](stdin) as stream:
        while True:
            var command = stream.readline()
            if command == "quit":
                return
            var words = command.split()
            if len(words) != 4:
                raise Error("viewer command requires steering, throttle, reset, policy")
            device.run[controls](
                n,
                Params(
                    Int32(n), seed, Int32(Int(words[0])), Int32(Int(words[1])), Int32(Int(words[2]))
                ),
            )
            if Int(words[2]) > 0:
                device.run[observe](n * BEAMS, Params(Int32(n), seed))
            if Int(words[3]) > 0:
                device.run[policy_input](n * OBS, Params(Int32(n), seed, -1))
                forward(device, n, n, False)
                device.run[sample](n, Params(Int32(n), seed, 0, 0, 1))
            device.run[step](n, Params(Int32(n), seed))
            device.run[observe](n * BEAMS, Params(Int32(n), seed))
            var json = String("[")
            with snapshot.map_to_host() as host:
                for i in range(env_size(n)):
                    if i > 0:
                        json += ","
                    json += String(host[i])
            json += "]"
            print(json, flush=True)
