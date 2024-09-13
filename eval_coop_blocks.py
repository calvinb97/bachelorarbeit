from numba import cuda
import nbshmem
from numba.types.containers import UniTuple
from numba.types import int64, int32, boolean

sig_allreduce_1d = (UniTuple(int32[:], 2), int64, UniTuple(int32[:], 2), int64, int64, UniTuple(boolean[:], 2))
sig_allreduce_2d = (UniTuple(int32[:, :], 2), int64, UniTuple(int32[:, :], 2), int64, int64, UniTuple(boolean[:], 2))
sig_allreduce_bcast = (UniTuple(int32[:], 2), int64, UniTuple(int32[:], 2), int64, int64, UniTuple(boolean[:], 2))
sig_broadcast_1d = (UniTuple(int32[:], 2), int64, UniTuple(int32[:], 2), int64, int64, UniTuple(boolean[:], 2))
sig_broadcast_2d = (UniTuple(int32[:, :], 2), int64, UniTuple(int32[:, :], 2), int64, int64, UniTuple(boolean[:], 2))
sig_alltoall_1d = (UniTuple(int32[:], 2), int64, UniTuple(int32[:], 2), int64, int64, UniTuple(boolean[:], 2))
sig_alltoall_2d = (UniTuple(int32[:, :], 2), int64, UniTuple(int32[:, :], 2), int64, int64, UniTuple(boolean[:], 2))


@cuda.jit(sig_allreduce_1d)
def allreduce_1d_kernel(ary, nelems, res, my_pe, npes, sync):
    nbshmem.allreduce_1d_sum(res, ary, nelems, my_pe, sync)


@cuda.jit(sig_allreduce_2d)
def allreduce_2d_kernel(ary, nelems, res, my_pe, npes, sync):
    nbshmem.allreduce_2d_sum(res, ary, nelems, my_pe, sync)


@cuda.jit(sig_allreduce_bcast)
def allreduce_bcast_kernel(ary, nelems, res, my_pe, npes, sync):
    nbshmem.allreduce_1d_sum_bcast(res, ary, nelems, my_pe, sync)


@cuda.jit(sig_broadcast_1d)
def broadcast_1d_kernel(ary, nelems, bcast_res, my_pe, npes, sync):
    nbshmem.broadcast_1d(bcast_res, ary, nelems, 0, my_pe, sync)


@cuda.jit(sig_broadcast_2d)
def broadcast_2d_kernel(ary, rows, bcast_res, my_pe, npes, sync):
    nbshmem.broadcast_2d(bcast_res, ary, rows, 0, my_pe, sync)


@cuda.jit(sig_alltoall_1d)
def alltoall_1d_kernel(ary, nelems, alltoall_res, my_pe, npes, sync):
    nbshmem.alltoall_1d(alltoall_res, ary, nelems, my_pe, sync)


@cuda.jit(sig_alltoall_2d)
def alltoall_2d_kernel(ary, rows, alltoall_res, my_pe, npes, sync):
    nbshmem.alltoall_2d(alltoall_res, ary, rows, my_pe, sync)


blockdims = [32, 64, 256, 512, 768, 1024, 2048]


for blockdim in blockdims:
    overload = allreduce_1d_kernel.overloads[sig_allreduce_1d]
    maxblocks = overload.max_cooperative_grid_blocks(blockdim)
    print(f"Allreduce 1D, blockdim={blockdim}, max blocks: {maxblocks}, threads: {blockdim * maxblocks}")

    overload = allreduce_2d_kernel.overloads[sig_allreduce_2d]
    maxblocks = overload.max_cooperative_grid_blocks(blockdim)
    print(f"Allreduce 2D, blockdim={blockdim}, max blocks: {maxblocks}, threads: {blockdim * maxblocks}")

    overload = allreduce_bcast_kernel.overloads[sig_allreduce_bcast]
    maxblocks = overload.max_cooperative_grid_blocks(blockdim)
    print(f"Allreduce Bcast, blockdim={blockdim}, max blocks: {maxblocks}, threads: {blockdim * maxblocks}")

    overload = broadcast_1d_kernel.overloads[sig_broadcast_1d]
    maxblocks = overload.max_cooperative_grid_blocks(blockdim)
    print(f"Broadcast 1D, blockdim={blockdim}, max blocks: {maxblocks}, threads: {blockdim * maxblocks}")

    overload = broadcast_2d_kernel.overloads[sig_broadcast_2d]
    maxblocks = overload.max_cooperative_grid_blocks(blockdim)
    print(f"Broadcast 2D, blockdim={blockdim}, max blocks: {maxblocks}, threads: {blockdim * maxblocks}")

    overload = alltoall_1d_kernel.overloads[sig_alltoall_1d]
    maxblocks = overload.max_cooperative_grid_blocks(blockdim)
    print(f"AlltoAll 1D, blockdim={blockdim}, max blocks: {maxblocks}, threads: {blockdim * maxblocks}")

    overload = alltoall_2d_kernel.overloads[sig_alltoall_2d]
    maxblocks = overload.max_cooperative_grid_blocks(blockdim)
    print(f"AlltoAll 2D, blockdim={blockdim}, max blocks: {maxblocks}, threads: {blockdim * maxblocks}\n")