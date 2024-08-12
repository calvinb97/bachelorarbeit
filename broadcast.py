from numba import cuda
import numpy as np
from mpi4py import MPI
import nbshmem


@cuda.jit
def broadcast_2d_kernel(ary, bcast_res, my_pe, npes, sync):
    nbshmem.broadcast_2d(bcast_res, ary, 10, 0, my_pe, sync)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"Hello from {rank}")

nbshmem.init(comm)

if rank == 0:
    ary = np.ones((3, 3), dtype=np.int64) * 5
else:
    ary = np.zeros((3, 3), dtype=np.int64)

dary = cuda.to_device(ary)

bcast_res = nbshmem.alloc((3,3), np.int64)

sync_local = np.full(3, False)
sync_shmem = nbshmem.array(sync_local)

broadcast_2d_kernel[1, 10](dary, bcast_res, rank, 2, sync_shmem)
cuda.synchronize()

if rank == 0:
    print(bcast_res[0].copy_to_host())
    print(f"process {rank} done")
if rank == 1:
    print(bcast_res[1].copy_to_host())
    print(f"process {rank} done")