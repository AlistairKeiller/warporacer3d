"""One device abstraction for the CPU and every GPU MAX supports.

Buffers are flat float32 device memory; `mat` views them as row-major matrices
with a static row count. Kernels are closures called once per index by MAX's
`elementwise`, matrix products are MAX's `matmul`, row sums are MAX's `sum`.
"""
from std.sys import has_accelerator, get_defined_bool
from std.utils import IndexList
from std.python import Python
from std.python.numpy import copy_to_numpy_array, from_numpy_array
from layout import TileTensor, Coord, Idx, row_major, ComptimeInt
from layout.tile_layout import RowMajorLayout
from linalg.matmul import matmul
from linalg.utils import elementwise_compute_lambda_type
from max.algorithm import elementwise, parallelize
from max.algorithm.reduction import sum as reduce_sum
from max.gpu.host import DeviceContext, DeviceBuffer

comptime GPU_AVAILABLE = has_accelerator() and not get_defined_bool["CPU_ONLY", False]()
comptime Buffer = DeviceBuffer[DType.float32]
comptime Mat[rows: Int] = TileTensor[
    DType.float32, RowMajorLayout[ComptimeInt[rows], Int], MutAnyOrigin
]
comptime Kernel = ImplicitlyCopyable & RegisterPassable & def(Int) -> None
comptime Kernel2 = ImplicitlyCopyable & RegisterPassable & def(Int, Int) -> None
comptime Epilogue = Optional[elementwise_compute_lambda_type]


def mat[rows: Int](buffer: Buffer, cols: Int, offset: Int = 0) -> Mat[rows]:
    """A `rows` x `cols` row-major view starting `offset` floats into `buffer`."""
    var ptr = buffer.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    return TileTensor(ptr=ptr.unsafe_offset(offset), layout=row_major(Idx[rows], cols))


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

    def alloc(self, count: Int, fill: Float32 = 0) raises -> Buffer:
        """A filled buffer. The fill is waited for: on the CPU, kernels run on
        the calling thread and would otherwise race the queued fill."""
        var buffer = self.ctx.enqueue_create_buffer[DType.float32](count)
        buffer.enqueue_fill(fill)
        self.ctx.synchronize()
        return buffer^

    def run[F: Kernel, //](self, kernel: F, n: Int) raises:
        """Call `kernel(i)` for every i < n: one GPU thread each, or CPU tasks."""

        def wrapped[width: Int, alignment: Int = 1](c: Coord) {var}:
            kernel(Int(c[0].value()))

        comptime if GPU_AVAILABLE:
            if self.gpu:
                return elementwise[1, target="gpu"](wrapped, Coord(n), self.ctx)
        parallelize(kernel, n, Optional(self.ctx))

    def run[F: Kernel2, //](self, kernel: F, rows: Int, cols: Int) raises:
        """Call `kernel(i, j)` over rows x cols; adjacent GPU threads take adjacent j."""

        def wrapped[width: Int, alignment: Int = 1](c: Coord) {var}:
            kernel(Int(c[0].value()), Int(c[1].value()))

        def row(i: Int) {var}:
            for j in range(cols):
                kernel(i, j)

        comptime if GPU_AVAILABLE:
            if self.gpu:
                return elementwise[1, target="gpu"](wrapped, Coord(rows, cols), self.ctx)
        parallelize(row, rows, Optional(self.ctx))

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
                return matmul[
                    transpose_b=transpose_b, elementwise_compute_lambda_fn=epilogue, target="gpu"
                ](c, a, b, Optional(self.ctx))
        matmul[transpose_b=transpose_b, elementwise_compute_lambda_fn=epilogue, target="cpu"](
            c, a, b, Optional(self.ctx)
        )

    def sum[rows: Int](self, x: Mat[rows], into: Mat[1], slot: Int = 0) raises:
        """into[0, slot + r] = the sum of row r of x."""

        @__copy_capture(x)
        @__parameter
        def load[width: Int, rank: Int](idx: IndexList[rank]) -> SIMD[DType.float32, width]:
            return x.load[width=width](Coord(idx[0], idx[1]))

        @__copy_capture(into, slot)
        @__parameter
        def store[width: SIMDLength, rank: Int](idx: IndexList[rank], v: SIMD[DType.float32, width]):
            into[0, slot + idx[0]] = v[0]

        var shape = Coord(rows, Int(x.dim[1]()))
        comptime if GPU_AVAILABLE:
            if self.gpu:
                return reduce_sum[DType.float32, load, store, target="gpu", reduce_dim=1](
                    shape, Optional(self.ctx)
                )
        reduce_sum[DType.float32, load, store, target="cpu", reduce_dim=1](shape, Optional(self.ctx))

    def read(self, buffer: Buffer) raises -> List[Float32]:
        var result = List[Float32](length=len(buffer), fill=0)
        self.ctx.enqueue_copy(result.unsafe_ptr(), buffer)
        self.ctx.synchronize()
        return result^

    def write(self, buffer: Buffer, values: Span[Float32, _]) raises:
        self.ctx.enqueue_copy(buffer, values.unsafe_ptr())
        self.ctx.synchronize()

    def upload(self, values: Span[Float32, _]) raises -> Buffer:
        var buffer = self.ctx.enqueue_create_buffer[DType.float32](len(values))
        self.write(buffer, values)
        return buffer^


def save_floats(path: String, values: Span[Float32, _]) raises:
    Python.import_module("numpy").save(path, copy_to_numpy_array(values))


def load_floats(path: String) raises -> List[Float32]:
    var array = Python.import_module("numpy").load(path).astype("float32")
    return List(from_numpy_array[DType.float32](array))
