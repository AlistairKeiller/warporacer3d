"""One device abstraction for the CPU and every GPU MAX supports.

Buffers are flat float32 device memory; `mat` views them as row-major
matrices with a static row count. Kernels are closures run per element by
MAX's `elementwise`; matrix products go through MAX's `matmul`.
"""
from std.sys import has_accelerator, get_defined_bool
from layout import TileTensor, Coord, Idx, row_major, ComptimeInt
from layout.tile_layout import RowMajorLayout
from linalg.matmul import matmul
from linalg.utils import elementwise_compute_lambda_type
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, DeviceBuffer

comptime GPU_AVAILABLE = has_accelerator() and not get_defined_bool[
    "CPU_ONLY", False
]()
comptime Buffer = DeviceBuffer[DType.float32]
comptime Mat[rows: Int] = TileTensor[
    DType.float32, RowMajorLayout[ComptimeInt[rows], Int], MutAnyOrigin
]
comptime Kernel = ImplicitlyCopyable & RegisterPassable & def[
    width: Int, alignment: Int = 1
](Coord) -> None
comptime Epilogue = Optional[elementwise_compute_lambda_type]


def mat[rows: Int](mut buffer: Buffer, cols: Int, offset: Int = 0) -> Mat[rows]:
    """A `rows` x `cols` row-major view starting `offset` floats into `buffer`."""
    return TileTensor(
        ptr=buffer.unsafe_ptr()
        .unsafe_origin_cast[MutAnyOrigin]()
        .unsafe_offset(offset),
        layout=row_major(Coord(Idx[rows], cols)),
    )


struct Device(Movable):
    var ctx: DeviceContext
    var gpu: Bool

    def __init__(out self, gpu: Bool) raises:
        comptime if GPU_AVAILABLE:
            self.ctx = DeviceContext() if gpu else DeviceContext(api="cpu")
        else:
            if gpu:
                raise Error("GPU support is unavailable in this build; use cpu")
            self.ctx = DeviceContext(api="cpu")
        self.gpu = gpu

    def alloc(self, count: Int) raises -> Buffer:
        """A zeroed buffer. The fill is waited for: on the CPU, kernels run on
        the calling thread and would otherwise race the queued fill."""
        var buffer = self.ctx.enqueue_create_buffer[DType.float32](count)
        buffer.enqueue_fill(0)
        self.ctx.synchronize()
        return buffer^

    def run[F: Kernel, //](self, kernel: F, shape: Coord) raises:
        """Call `kernel(coord)` once per coordinate: GPU threads or CPU tasks."""
        comptime if GPU_AVAILABLE:
            if self.gpu:
                elementwise[1, target="gpu"](kernel, shape, self.ctx)
                return
        elementwise[1, target="cpu"](kernel, shape, self.ctx)

    def gemm[
        transpose_b: Bool = False, epilogue: Epilogue = None
    ](
        self,
        c: TileTensor[mut=True, address_space=.GENERIC, ...],
        a: TileTensor[address_space=.GENERIC, ...],
        b: TileTensor[address_space=.GENERIC, ...],
    ) raises:
        """c = a @ b (or a @ b^T) with an optional element-wise epilogue."""
        comptime if GPU_AVAILABLE:
            if self.gpu:
                matmul[
                    transpose_b=transpose_b,
                    elementwise_compute_lambda_fn=epilogue,
                    target="gpu",
                ](c, a, b, Optional(self.ctx))
                return
        matmul[
            transpose_b=transpose_b,
            elementwise_compute_lambda_fn=epilogue,
            target="cpu",
        ](c, a, b, Optional(self.ctx))

    def read(self, buffer: Buffer) raises -> List[Float32]:
        var result = List[Float32](length=len(buffer), fill=0)
        self.ctx.enqueue_copy(result.unsafe_ptr(), buffer)
        self.ctx.synchronize()
        return result^

    def write(self, buffer: Buffer, values: List[Float32]) raises:
        self.ctx.enqueue_copy(buffer, values.unsafe_ptr())
        self.ctx.synchronize()

    def upload(self, values: List[Float32]) raises -> Buffer:
        var buffer = self.ctx.enqueue_create_buffer[DType.float32](len(values))
        self.write(buffer, values)
        return buffer^


def read_floats(path: String) raises -> List[Float32]:
    var raw = open(path, "r").read_bytes()
    var result = List[Float32](length=len(raw) // 4, fill=0)
    for i in range(len(result)):
        result[i] = raw.unsafe_ptr().unsafe_bitcast[Float32]()[unsafe_offset=i]
    return result^


def write_floats(path: String, values: List[Float32]) raises:
    var file = open(path, "w")
    file.write_all(
        Span(
            unsafe_ptr=values.unsafe_ptr().unsafe_bitcast[UInt8](),
            length=len(values) * 4,
        )
    )
    file.close()
