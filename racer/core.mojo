"""One CPU/GPU dispatch path; all kernels share the same scalar equations."""
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu import global_idx
from max.algorithm import parallelize
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import sqrt, log, cos
from std.sys import has_accelerator, get_defined_bool

comptime GPU_AVAILABLE = has_accelerator() and not get_defined_bool["CPU_ONLY", False]()

comptime Ptr = Pointer[Float32, MutAnyOrigin]


struct Params(DevicePassable, TrivialRegisterPassable):
    comptime device_type = Self

    def _to_device_type(self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]):
        target.unsafe_bitcast[Self]().write(self)

    @staticmethod
    def get_type_name() -> String:
        return "Params"

    var envs: Int32
    var seed: UInt32
    var offset: Int32
    var index: Int32
    var flag: Int32
    var scale: Float32

    def __init__(
        out self,
        envs: Int32,
        seed: UInt32,
        offset: Int32 = 0,
        index: Int32 = 0,
        flag: Int32 = 0,
        scale: Float32 = 0,
    ):
        self.envs, self.seed, self.offset, self.index, self.flag, self.scale = (
            envs,
            seed,
            offset,
            index,
            flag,
            scale,
        )


def gpu_kernel[
    body: def(Int, Ptr, Ptr, Params) thin -> None
](data: Ptr, terrain: Ptr, p: Params, count: Int32):
    var i = global_idx.x
    if i < Int(count):
        body(i, data, terrain, p)


struct Device(Movable):
    var ctx: DeviceContext
    var gpu: Bool
    var data: DeviceBuffer[DType.float32]
    var terrain: DeviceBuffer[DType.float32]

    def __init__(out self, gpu: Bool, size: Int, map_data: List[Float32]) raises:
        self.gpu = gpu
        comptime if GPU_AVAILABLE:
            self.ctx = DeviceContext() if gpu else DeviceContext(api="cpu")
        else:
            if gpu:
                raise Error("GPU support is unavailable in this build; select cpu")
            self.ctx = DeviceContext(api="cpu")
        self.data = self.ctx.enqueue_create_buffer[DType.float32](size)
        self.terrain = self.ctx.enqueue_create_buffer[DType.float32](len(map_data))
        self.data.enqueue_fill(0)
        self.ctx.enqueue_copy(self.terrain, map_data.unsafe_ptr())
        self.ctx.synchronize()

    def run[body: def(Int, Ptr, Ptr, Params) thin -> None](mut self, count: Int, p: Params) raises:
        comptime if GPU_AVAILABLE:
            if self.gpu:
                self.ctx.enqueue_function[gpu_kernel[body]](
                    self.data,
                    self.terrain,
                    p,
                    Int32(count),
                    grid_dim=(count + 127) // 128,
                    block_dim=128,
                )
                return
        var data = self.data.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var terrain = self.terrain.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

        def cpu_body(i: Int) {data, terrain, p}:
            body(i, data, terrain, p)

        parallelize(cpu_body, count, min(16, max(1, count // 256)), self.ctx)


@inline(.always)
def clamp(x: Float32, lo: Float32, hi: Float32) -> Float32:
    return min(max(x, lo), hi)


@inline(.always)
def uniform(seed: UInt32) -> Float32:
    var x = seed
    x ^= x >> 16
    x *= 0x7FEB352D
    x ^= x >> 15
    x *= 0x846CA68B
    x ^= x >> 16
    # 23 bits keep the half-unit offset representable: strictly 0 < u < 1.
    # A 24-bit midpoint can round to 1 and index past the spawn table.
    return (Float32(x >> 9) + 0.5) / 8388608.0


@inline(.always)
def normal(seed: UInt32) -> Float32:
    return sqrt(-2.0 * log(uniform(seed))) * cos(6.283185307 * uniform(seed + 0x9E3779B9))


def read_floats(path: String) raises -> List[Float32]:
    var file = open(path, "r")
    var raw = file.read_bytes()
    if len(raw) % 4 != 0:
        raise Error("truncated float32 file: " + path)
    var result = List[Float32](capacity=len(raw) // 4)
    var ptr = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(len(raw) // 4):
        result.append(ptr[unsafe_offset=i])
    return result^
