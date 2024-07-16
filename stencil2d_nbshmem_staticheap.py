from numba import cuda
import numpy as np
from mpi4py import MPI
import nbshmem
import math
from numba.types.containers import UniTuple
from numba.types import int64, float64

import os
rand_value = 30
n = 128

basedir = os.path.dirname(os.path.abspath(__file__))
nbshmem_cu = os.path.join(basedir, 'nbshmem', 'nbshmem.cu')

sig = (UniTuple(float64[:, :], 2), UniTuple(float64[:, :], 2), int64, int64,
       UniTuple(float64[:], 2), int64, int64, int64, int64, UniTuple(int64[::1], 2))


@cuda.jit(sig, link=[nbshmem_cu])
def stencil2d_kernel(a_shmem, anew_shmem, n, iterations, diffnorm, my_pe, npes, y_end, x_end, sync):
    x, y = cuda.grid(2)
    bottom = my_pe + 1 if my_pe < (npes-1) else 0
    top = my_pe - 1 if my_pe > 0 else npes - 1

    for i in range(iterations):
        a = a_shmem[my_pe]
        anew = anew_shmem[my_pe]
        if x > 0 and x < x_end and y > 0 and y < y_end:
            anew[x, y] = 0.25 * (a[x, y-1] + a[x, y+1] + a[(x-1), y] + a[(x+1), y])
            local_diffnorm = (anew[x, y] - a[x, y]) * (anew[x, y] - a[x, y])
            cuda.atomic.add(diffnorm[my_pe], i, local_diffnorm)

            if y == (y_end - 1):
                anew[x, 0] = anew[x, y]
            if y == 1:
                anew[x, y_end] = anew[x, y]

            if x == (x_end - 1):
                anew_bottom = anew_shmem[bottom]
                anew_bottom[0, y] = anew[x, y]
            if x == 1:
                anew_top = anew_shmem[top]
                anew_top[x_end, y] = anew[x, y]

        nbshmem.barrier_all_device(sync, my_pe, npes)

        tmp = a_shmem
        a_shmem = anew_shmem
        anew_shmem = tmp


@cuda.jit
def init_kernel(a, my_pe, y_end, x_end):
    x, y = cuda.grid(2)
    if x >= 0 and x <= x_end and y >= 0 and y <= y_end:
        if y == 0 or y == y_end:
            a[x, y] = 30
        if my_pe == 0 and x == 0:
            a[x, y] = 30
        if my_pe == 1 and x == x_end:
            a[x, y] = 30
        if my_pe == 0 and x == y:
            a[x, y] = 30
        if my_pe == 1 and x + x_end - 1 == y:
            a[x, y] = 30


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"Hello from {rank}")

size = 2
chunk_size = (n-2) // 2

sync_shmem, shmem_heaps = nbshmem.init(comm, static_heap=True)

iterations = 500
x_shmem = nbshmem.alloc_like(np.empty((chunk_size+2, n), dtype=np.float64))
xnew_shmem = nbshmem.alloc_like(np.empty((chunk_size+2, n), dtype=np.float64))
diffnorm_shmem = nbshmem.alloc_like(np.empty(iterations, dtype=np.float64))

threadsperblock = (8, 8)
blockspergrid_x = math.ceil((chunk_size + 2) / threadsperblock[0])
blockspergrid_y = math.ceil(n / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
x_end = chunk_size + 1
y_end = n - 1

init_kernel[blockspergrid, threadsperblock](x_shmem[rank], rank, y_end, x_end)
cuda.synchronize()
nbshmem.barrier_all_host()
start_time = MPI.Wtime()
stencil2d_kernel[blockspergrid, threadsperblock](x_shmem, xnew_shmem, n,
                                                 iterations, diffnorm_shmem,
                                                 rank, size, y_end, x_end,
                                                 sync_shmem)
cuda.synchronize()
end_time = MPI.Wtime()
time = end_time - start_time
if rank == 0:
    diffnorm_0 = diffnorm_shmem[0].copy_to_host()
    diffnorm_1 = diffnorm_shmem[1].copy_to_host()
    diffnorm = diffnorm_0 + diffnorm_1
    norm = np.sqrt(diffnorm[::10])
    print(norm)

# print(f"Process {rank} time: {time}")