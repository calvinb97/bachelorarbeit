from numba import cuda
import nbshmem
from numba.types.containers import UniTuple
from numba.types import int64, float64, boolean

sig = (UniTuple(float64[:, :], 2), float64[:, :], int64, int64, UniTuple(boolean[:], 2))


@cuda.jit(sig)
def allreduce_1d_kernel(ary, res, my_pe, npes, sync):
    nbshmem.allreduce_sum(res, ary, ary[my_pe].shape[0], my_pe, sync)


blockdim = (24, 24)

overload = allreduce_1d_kernel.overloads[sig]
maxblocks = overload.max_cooperative_grid_blocks(blockdim)
print(f"Max blocks: {maxblocks}")