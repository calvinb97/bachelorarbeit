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
       UniTuple(float64[:], 2), int64, int64, int64, int64,
       UniTuple(int64[::1], 2), UniTuple(float64[:], 2))


@cuda.jit(sig, link=[nbshmem_cu])
def stencil2d_kernel(a_shmem, anew_shmem, n, iterations, diffnorm, 
                     my_pe, npes, y_end, x_end, sync, red):
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

        nbshmem.barrier_all(sync, my_pe, npes)

        tmp = a_shmem
        a_shmem = anew_shmem
        anew_shmem = tmp

    nbshmem.allreduce_sum(red[my_pe], diffnorm, iterations, my_pe, sync)
    idx_1d = nbshmem.get_1d_threadidx_in_grid()
    if idx_1d < iterations:
        red[my_pe][idx_1d] = math.sqrt(red[my_pe][idx_1d])


def init(n):
    mat = np.zeros(shape=(n, n), dtype=np.float64)
    mat[0] = rand_value
    mat[n-1] = rand_value
    mat[:, 0] = rand_value
    mat[:, n-1] = rand_value
    np.fill_diagonal(mat, rand_value)
    return mat


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"Hello from {rank}")

size = 2
chunk_size = (n-2) // 2

nbshmem.init(comm)
x = init(n)
# local_x = np.zeros(shape = (chunk_size, n))
first_row = rank * chunk_size
local_x = np.copy(x[first_row:(first_row+chunk_size+2), :])
x_shmem = nbshmem.array(local_x)
local_xnew = np.zeros_like(local_x)
xnew_shmem = nbshmem.array(local_xnew)
# sync_local = np.zeros(3, dtype=np.int64)
# sync_shmem = nbshmem.array(sync_local)
sync_local = np.full(3, False)
sync_shmem = nbshmem.array(sync_local)

iterations = 2000
diffnorm = np.zeros(iterations)
diffnorm_shmem = nbshmem.array(diffnorm)
red = np.zeros(iterations)
red_shmem = nbshmem.array(red)

threadsperblock = (8, 8)
blockspergrid_x = math.ceil((chunk_size + 2) / threadsperblock[0])
blockspergrid_y = math.ceil(n / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
x_end = chunk_size + 1
y_end = n - 1

start_time = MPI.Wtime()
stencil2d_kernel[blockspergrid, threadsperblock](x_shmem, xnew_shmem, n,
                                                 iterations, diffnorm_shmem,
                                                 rank, size, y_end, x_end,
                                                 sync_shmem, red_shmem)
cuda.synchronize()
end_time = MPI.Wtime()
time = end_time - start_time
if rank == 0:
    diffnorm_0 = diffnorm_shmem[0].copy_to_host()
    diffnorm_1 = diffnorm_shmem[1].copy_to_host()
    diffnorm = diffnorm_0 + diffnorm_1
    norm = np.sqrt(diffnorm[::50])
    print(norm)
    print(red_shmem[rank].copy_to_host()[::50])

# print(f"Process {rank} time: {time}")


# overload = stencil2d_kernel.overloads[sig]
# maxblocks = overload.max_cooperative_grid_blocks(threadsperblock)
# print(f"Max blocks: {maxblocks}")
