from numba import cuda
import numpy as np
from mpi4py import MPI
import nbshmem
import math

rand_value = 30
n = 128


@cuda.jit
def stencil2d_kernel(a_shmem, anew_shmem, n, iter, diffnorm, my_pe, npes, y_end, x_end):
    x, y = cuda.grid(2)
    a = a_shmem[my_pe]
    anew = anew_shmem[my_pe]
    if x > 0 and x < x_end and y > 0 and y < y_end:
        anew[x, y] = 0.25 * (a[x, y-1] + a[x, y+1] + a[(x-1), y] + a[(x+1), y])
        local_diffnorm = (anew[x, y] - a[x, y]) * (anew[x, y] - a[x, y])
        cuda.atomic.add(diffnorm[my_pe], iter, local_diffnorm)

        if y == (y_end - 1):
            anew[x, 0] = anew[x, y]
        if y == 1:
            anew[x, y_end] = anew[x, y]

        bottom = my_pe + 1 if my_pe < (npes-1) else 0
        top = my_pe - 1 if my_pe > 0 else npes - 1

        if x == (x_end - 1):
            anew_bottom = anew_shmem[bottom]
            anew_bottom[0, y] = anew[x, y]
        if x == 1:
            anew_top = anew_shmem[top]
            anew_top[x_end, y] = anew[x, y]


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

iterations = 2000
diffnorm = np.zeros(iterations)
diffnorm_shmem = nbshmem.array(diffnorm)

threadsperblock = (8, 8)
blockspergrid_x = math.ceil((chunk_size + 2) / threadsperblock[0])
blockspergrid_y = math.ceil(n / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
x_end = chunk_size + 1
y_end = n - 1

for i in range(iterations):
    stencil2d_kernel[blockspergrid, threadsperblock](x_shmem, xnew_shmem, n, i,
                                                     diffnorm_shmem, rank, size,
                                                     y_end, x_end)
    cuda.synchronize()
    nbshmem.barrier_all_host()
    tmp = x_shmem
    x_shmem = xnew_shmem
    xnew_shmem = tmp

if rank == 0:
    diffnorm_0 = diffnorm_shmem[0].copy_to_host()
    diffnorm_1 = diffnorm_shmem[1].copy_to_host()
    diffnorm = diffnorm_0 + diffnorm_1
    norm = np.sqrt(diffnorm[::50])
    print(norm)