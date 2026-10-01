"""Export one analytic PPO/backprop gradient for an independent Torch oracle."""
from std.sys import argv
from std.math import sin
from racer.device import Device, Params, Ptr, read_floats
from racer.network import (
    memory,
    initialize,
    forward,
    backward,
    WEIGHTS,
    ROLLOUT,
)
from racer.simulation import OBS
from racer.ppo import gather, loss, std_gradient


def fixture(i: Int, data: Ptr, map: Ptr, p: Params):
    var mem = memory(Int(p.envs))
    for j in range(OBS):
        data[unsafe_offset=mem.obs + i * OBS + j] = 0.2 * sin(
            Float32(i * OBS + j) * 0.12
        )
    data[unsafe_offset=mem.actions + i * 2] = -0.4 + 0.1 * Float32(i % 9)
    data[unsafe_offset=mem.actions + i * 2 + 1] = -0.2 + 0.05 * Float32(i % 11)
    data[unsafe_offset=mem.values + i] = 0.1 * sin(Float32(i))
    data[unsafe_offset=mem.logp + i] = -0.5 - 0.3 * Float32(i % 8)
    data[unsafe_offset=mem.advantage + i] = sin(Float32(i) * 0.7)
    data[unsafe_offset=mem.target + i] = sin(Float32(i) * 0.31)
    if i == 0:
        data[unsafe_offset=mem.stats] = 0
        data[unsafe_offset=mem.stats + 1] = 1


def main() raises:
    var args = argv()
    var n = 2
    var d = Device(
        String(args[1]) == "gpu", memory(n).size, read_floats(String(args[2]))
    )
    var p = Params(Int32(n), 42)
    d.run[initialize](WEIGHTS, p)
    d.run[fixture](n * ROLLOUT, p)
    d.run[gather](16 * OBS, p)
    forward(d, n, 16, True)
    d.run[loss](16, p)
    d.run[std_gradient](3, p)
    backward(d, n, 16)
    d.ctx.synchronize()
    var file = open(String(args[3]), "w")
    var dimensions = List[Float32]()
    dimensions.append(Float32(OBS))
    dimensions.append(Float32(WEIGHTS))
    file.write_all(
        Span(
            unsafe_ptr=dimensions.unsafe_ptr().unsafe_bitcast[UInt8](), length=8
        )
    )
    with d.data.map_to_host() as host:
        file.write_all(
            Span(
                unsafe_ptr=host.unsafe_ptr().unsafe_bitcast[UInt8](),
                length=memory(n).size * 4,
            )
        )
    print("Exported", args[1], "gradient")
