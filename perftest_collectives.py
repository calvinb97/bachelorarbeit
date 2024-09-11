from numba import cuda
import nbshmem
from numba.types.containers import UniTuple
from numba.types import int64, float64, boolean

sig = (UniTuple(float64[:], 2), UniTuple(float64[:], 2), int64, int64, UniTuple(boolean[:], 2))
sig2d = (UniTuple(float64[:, :], 2), UniTuple(float64[:, :], 2), int64, int64, UniTuple(boolean[:], 2))
# sig = (UniTuple(float64[:], 2), float64[:], int64, int64, UniTuple(boolean[:], 2))


@cuda.jit(sig)
def allreduce_kernel(ary, res, my_pe, npes, sync):
    nbshmem.allreduce_1d_sum(res, ary, ary[my_pe].shape[0], my_pe, sync)


@cuda.jit(sig2d)
def allreduce_2d_kernel(ary, res, my_pe, npes, sync):
    nbshmem.allreduce_2d_sum(res, ary, ary[my_pe].shape[0], my_pe, sync)


blockdim = (8, 8)

# overload = allreduce_kernel.overloads[sig]
overload = allreduce_2d_kernel.overloads[sig2d]
maxblocks = overload.max_cooperative_grid_blocks(blockdim)
print(f"Max blocks: {maxblocks}")