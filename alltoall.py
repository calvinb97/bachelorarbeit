from numba import cuda
import numpy as np
from mpi4py import MPI
import nbshmem


@cuda.jit
def alltoall_1d_kernel(ary, alltoall_res, my_pe, npes, sync):
    nbshmem.alltoall_2d(alltoall_res, ary, 3, my_pe, sync)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"Hello from {rank}")

nbshmem.init(comm)

if rank == 0:
    ary = np.arange(18).reshape(6, 3)
else:
    ary = np.arange(18).reshape(6, 3) * 10
dary = cuda.to_device(ary)

alltoall_res = nbshmem.alloc((6, 3), np.int64)

sync_local = np.full(3, False)
sync_shmem = nbshmem.array(sync_local)

alltoall_1d_kernel[1, 10](dary, alltoall_res, rank, 2, sync_shmem)
cuda.synchronize()

if rank == 0:
    print(alltoall_res[0].copy_to_host())
    print(f"process {rank} done")
if rank == 1:
    print(alltoall_res[1].copy_to_host())
    print(f"process {rank} done")