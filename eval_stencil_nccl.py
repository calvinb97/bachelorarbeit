from numba import cuda
import numpy as np
from mpi4py import MPI
import math
import cupy
from cupyx.distributed import NCCLBackend

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"Hello from {rank}")

# NCCL setup
cupy.cuda.Device(rank).use()
num_devices = 2
nccl_comm = NCCLBackend(num_devices, rank, use_mpi=True)

init_value = 30


@cuda.jit
def stencil_kernel(a, anew, diffnorm, iteration, x_end, y_end):
    x, y = cuda.grid(2)

    if x > 0 and x < x_end and y > 0 and y < y_end:
        anew[x, y] = 0.25 * (a[x, y-1] + a[x, y+1] + a[(x-1), y] + a[(x+1), y])
        local_diffnorm = (anew[x, y] - a[x, y]) * (anew[x, y] - a[x, y])
        cuda.atomic.add(diffnorm, iteration, local_diffnorm)

        if y == (y_end - 1):
            anew[x, 0] = anew[x, y]
        if y == 1:
            anew[x, y_end] = anew[x, y]


def init(n):
    ary = np.zeros(shape=(n, n), dtype=np.float64)
    ary[0] = init_value
    ary[n-1] = init_value
    ary[:, 0] = init_value
    ary[:, n-1] = init_value
    np.fill_diagonal(ary, init_value)
    return ary


n = 128
iterations = 2000
chunk_size = (n-2) // 2
first_row = rank * chunk_size

threadsperblock = (8, 8)
blockspergrid_x = math.ceil((chunk_size + 2) / threadsperblock[0])
blockspergrid_y = math.ceil(n / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
x_end = chunk_size + 1
y_end = n - 1

x = init(n)

# local arrays on GPU
a = cupy.asarray(x[first_row:(first_row+chunk_size+2), :])
a_new = cupy.zeros(shape=(chunk_size + 2, n))
diffnorm = cupy.zeros(iterations)
diffnorm_reduced = cupy.zeros(iterations)

top = 1 if rank == 0 else 0
bottom = 1 if rank == 0 else 0

if rank == 0:
    print("Starting evaluation of Stencil with NCCL")

norm = None
time_results = []
for time_iter in range(11):
    a = cupy.asarray(x[first_row:(first_row+chunk_size+2), :], blocking=True)
    a_new = cupy.zeros(shape=(chunk_size + 2, n))
    diffnorm = cupy.zeros(iterations)
    comm.Barrier()
    start_time = MPI.Wtime()

    for iter in range(iterations):
        stencil_kernel[blockspergrid, threadsperblock](a, a_new, diffnorm, 0, x_end, y_end)
        cupy.cuda.nccl.groupStart()
        nccl_comm.recv(a_new[(chunk_size + 1), 1:(n-1)], bottom)
        nccl_comm.send(a_new[1, 1:(n-1)], top)
        nccl_comm.recv(a_new[0, 1:(n-1)], top)
        nccl_comm.send(a_new[chunk_size, 1:(n-1)], bottom)
        cupy.cuda.nccl.groupEnd()
        tmp = a
        a = a_new
        a_new = tmp

    end_time = MPI.Wtime()
    time = end_time - start_time
    if time_iter == 0:
        print(f"iteration {time_iter}: {time} on process {rank}")
    else:
        time_results.append(time)

    nccl_comm.all_reduce(diffnorm, diffnorm_reduced, op="sum")
    cupy.cuda.runtime.deviceSynchronize()
    diffnorm_host = cupy.asnumpy(diffnorm_reduced)
    if time_iter == 0:
        norm = diffnorm_host
    else:
        assert np.allclose(norm, diffnorm_host)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Stencil NCCL mean: {np.mean(results)}")
    print(f"Stencil NCCL std: {np.std(results)}")
