"""Versioned float32 checkpoints: model, Adam moments, step, and RNG seed."""
from std.math import isfinite, floor
from std.ffi import external_call
from .device import Device, read_floats, write_floats
from .layout import memory, WEIGHTS, OBS, H
from .lidar import RANGE

comptime MAGIC = 271828
comptime VERSION = 3
comptime HEADER = 9


def save(
    mut device: Device,
    path: String,
    n: Int,
    iteration: Int,
    optimizer_step: Int,
    seed: UInt32,
) raises:
    var values = List[Float32](length=HEADER, fill=0)
    values[0], values[1], values[2], values[3] = (
        MAGIC,
        VERSION,
        Float32(OBS),
        Float32(H),
    )
    values[4], values[5] = Float32(iteration), Float32(optimizer_step)
    values[6], values[7] = Float32(seed >> 16), Float32(seed & 65535)
    values[8] = RANGE
    values.extend(device.read(memory(n).weights, WEIGHTS * 4))
    var temporary = path + ".tmp"
    var destination = path
    write_floats(temporary, values)
    # C rename is atomic on every supported host (macOS, Linux, and WSL).
    if (
        external_call["rename", Int32](
            temporary.as_c_string_span(), destination.as_c_string_span()
        )
        != 0
    ):
        raise Error("could not replace checkpoint: " + path)


def load(
    mut device: Device, path: String, n: Int
) raises -> Tuple[Int, Int, UInt32]:
    var values = read_floats(path)
    if len(values) != HEADER + WEIGHTS * 4:
        raise Error("checkpoint has an invalid size")
    if (
        values[0] != MAGIC
        or values[1] != VERSION
        or values[2] != Float32(OBS)
        or values[3] != Float32(H)
        or values[8] != RANGE
    ):
        raise Error(
            "checkpoint format, network dimensions, or lidar range do not match"
        )
    for value in values:
        if not isfinite(value):
            raise Error("checkpoint contains nonfinite values")
    for i in range(4, 8):
        var limit: Float32 = 16000000 if i < 6 else 65535
        if values[i] < 0 or values[i] > limit or values[i] != floor(values[i]):
            raise Error("invalid checkpoint counter or seed")
    for i in range(HEADER + WEIGHTS * 3, len(values)):
        if values[i] < 0:
            raise Error("invalid Adam second moment")
    var params = List[Float32](capacity=WEIGHTS * 4)
    for i in range(HEADER, len(values)):
        params.append(values[i])
    device.write(memory(n).weights, params)
    return (
        Int(values[4]),
        Int(values[5]),
        (UInt32(values[6]) << 16) | UInt32(values[7]),
    )
