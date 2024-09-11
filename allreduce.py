from numba import cuda
import numpy as np
from mpi4py import MPI
import nbshmem


# from numba.types.containers import UniTuple
# from numba.types import int64, float64, boolean
# sig = (UniTuple(float64[:], 2), int64, UniTuple(float64[:], 2), int64, int64, UniTuple(boolean[:], 2))

@cuda.jit
def allreduce_1d_kernel(ary, ary_size, res, my_pe, npes, sync):
    nbshmem.allreduce_1d_sum(res, ary, ary_size, my_pe, sync)


@cuda.jit
def allreduce_2d_kernel(ary, ary_size, res, my_pe, npes, sync):
    nbshmem.allreduce_2d_sum(res, ary, ary_size, my_pe, sync)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"Hello from {rank}")

nbshmem.init(comm)

ary_size = 16384 * 100
ary = np.arange(ary_size)
ary_shmem = nbshmem.array(ary)

res_shmem = nbshmem.alloc((ary_size, ), np.int64)

sync_local = np.full(3, False)
sync_shmem = nbshmem.array(sync_local)

res_np = ary + ary
print(f"{res_np=}")

print("ALLREDUCE 1D")
for i in range(3):
    start_time = MPI.Wtime()
    allreduce_1d_kernel[(16, 16), (8, 8)](ary_shmem, ary_size, res_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    print(f"iteration {i}: {time} on process {rank}")
    res = res_shmem[rank].copy_to_host()
    print(f"{res=}")
    assert np.allclose(res_np, res)

print("ALLREDUCE 2D")
ary = ary.reshape(10, ary_size // 10)
ary_shmem = nbshmem.array(ary)
res_shmem = nbshmem.alloc((10, ary_size//10), np.int64)
res_np = res_np.reshape((10, ary_size//10))
for i in range(3):
    start_time = MPI.Wtime()
    allreduce_2d_kernel[(16, 16), (8, 8)](ary_shmem, ary_size, res_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    print(f"iteration {i}: {time} on process {rank}")
    res = res_shmem[rank].copy_to_host()
    print(f"{res=}")
    assert np.allclose(res_np, res)
