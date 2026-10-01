"""Versioned float32 checkpoints: model, Adam moments, step, and RNG seed."""
from std.math import isfinite, floor
from std.ffi import external_call
from .core import Device, read_floats
from .network import memory, WEIGHTS, HIDDEN
from .env import OBS
from .terrain import RANGE


def save(
    mut device: Device, path: String, n: Int, iteration: Int, optimizer_step: Int, seed: UInt32
) raises:
    device.ctx.synchronize()
    var header = List[Float32](length=9, fill=0)
    header[0], header[1], header[2], header[3] = 271828, 2, Float32(OBS), Float32(HIDDEN)
    header[4], header[5] = Float32(iteration), Float32(optimizer_step)
    header[6], header[7] = Float32(seed >> 16), Float32(seed & 65535)
    header[8] = RANGE
    var temporary = path + ".tmp"
    var file = open(temporary, "w")
    file.write_all(Span(unsafe_ptr=header.unsafe_ptr().unsafe_bitcast[UInt8](), length=36))
    var params = device.data.create_sub_buffer[DType.float32](memory(n).weights, WEIGHTS * 4)
    with params.map_to_host() as host:
        file.write_all(
            Span(unsafe_ptr=host.unsafe_ptr().unsafe_bitcast[UInt8](), length=WEIGHTS * 16)
        )
    file.close()
    var destination = path
    # C rename is atomic on every supported host (macOS, Linux, and WSL).
    if (
        external_call["rename", Int32](temporary.as_c_string_span(), destination.as_c_string_span())
        != 0
    ):
        raise Error("could not replace checkpoint: " + path)


def load(mut device: Device, path: String, n: Int) raises -> Tuple[Int, Int, UInt32]:
    var values = read_floats(path)
    if len(values) != 9 + WEIGHTS * 4:
        raise Error("checkpoint has an invalid size")
    if (
        values[0] != 271828
        or values[1] != 2
        or values[2] != Float32(OBS)
        or values[3] != Float32(HIDDEN)
        or values[8] != RANGE
    ):
        raise Error("checkpoint format, network dimensions, or lidar range do not match")
    for value in values:
        if not isfinite(value):
            raise Error("checkpoint contains nonfinite values")
    for i in range(4, 8):
        var limit: Float32 = 16000000 if i < 6 else 65535
        if values[i] < 0 or values[i] > limit or values[i] != floor(values[i]):
            raise Error("invalid checkpoint counter or seed")
    for i in range(9 + WEIGHTS * 3, len(values)):
        if values[i] < 0:
            raise Error("invalid Adam second moment")
    var params = device.data.create_sub_buffer[DType.float32](memory(n).weights, WEIGHTS * 4)
    device.ctx.enqueue_copy(params, values.unsafe_ptr().unsafe_offset(9))
    device.ctx.synchronize()
    return (Int(values[4]), Int(values[5]), (UInt32(values[6]) << 16) | UInt32(values[7]))
