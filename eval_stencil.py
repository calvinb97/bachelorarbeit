from numba import cuda
import numpy as np
from mpi4py import MPI
import nbshmem
import math

# ignore performance warning for one-block-launch
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

init_value = 30
n = 128


@cuda.jit
def stencil_barrier_kernel(a_shmem, anew_shmem, n, iterations, diffnorm, 
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


@cuda.jit
def stencil_kernel(a_shmem, anew_shmem, n, iter, diffnorm, my_pe, npes, y_end, x_end):
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
    ary = np.zeros(shape=(n, n), dtype=np.float64)
    ary[0] = init_value
    ary[n-1] = init_value
    ary[:, 0] = init_value
    ary[:, n-1] = init_value
    np.fill_diagonal(ary, init_value)
    return ary


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

# synchronization
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


norm = None

# # # # # # # #
# GPU barrier #
# # # # # # # #

if rank == 0:
    print("Starting evaluation of Stencil with GPU barrier")

time_results = []
for i in range(11):
    comm.Barrier()

    # copy initial values to device
    cuda.to_device(local_x, to=x_shmem[rank])
    cuda.to_device(local_xnew, to=xnew_shmem[rank])
    cuda.to_device(np.zeros(iterations), to=diffnorm_shmem[rank])

    comm.Barrier()
    start_time = MPI.Wtime()

    stencil_barrier_kernel[blockspergrid, threadsperblock](x_shmem, xnew_shmem, n,
                                                           iterations, diffnorm_shmem,
                                                           rank, size, y_end, x_end,
                                                           sync_shmem, red_shmem)

    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)
    if rank == 0:
        diffnorm_0 = diffnorm_shmem[0].copy_to_host()
        diffnorm_1 = diffnorm_shmem[1].copy_to_host()
        diffnorm = diffnorm_0 + diffnorm_1
        iteration_norm = np.sqrt(diffnorm[::50])
        if i == 0:
            norm = iteration_norm
        else:
            assert np.allclose(norm, iteration_norm)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Stencil GPU-Barrier: {np.mean(results)}")
    print(f"Stencil GPU-Barrier: {np.std(results)}")


# # # # # # # # # # #
# on-stream barrier #
# # # # # # # # # # #

if rank == 0:
    print("Starting evaluation of Stencil with on-stream barrier")

time_results = []
for i in range(11):
    comm.Barrier()

    # copy initial values to device
    cuda.to_device(local_x, to=x_shmem[rank])
    cuda.to_device(local_xnew, to=xnew_shmem[rank])
    cuda.to_device(np.zeros(iterations), to=diffnorm_shmem[rank])

    stream = cuda.stream()
    comm.Barrier()
    start_time = MPI.Wtime()

    for iter in range(iterations):
        stencil_kernel[blockspergrid, threadsperblock, stream](x_shmem, xnew_shmem, n, iter, diffnorm_shmem, rank, size, y_end, x_end)
        nbshmem.barrier_all_host_on_stream(sync_shmem, stream, 2)
        tmp = x_shmem
        x_shmem = xnew_shmem
        xnew_shmem = tmp

    end_time = MPI.Wtime()
    stream.synchronize()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)
    if rank == 0:
        diffnorm_0 = diffnorm_shmem[0].copy_to_host()
        diffnorm_1 = diffnorm_shmem[1].copy_to_host()
        diffnorm = diffnorm_0 + diffnorm_1
        iteration_norm = np.sqrt(diffnorm[::50])
        assert np.allclose(norm, iteration_norm)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Stencil On-Stream-Barrier: {np.mean(results)}")
    print(f"Stencil On-Stream-Barrier: {np.std(results)}")


# # # # # # # # 
# MPI barrier #
# # # # # # # # 

if rank == 0:
    print("Starting evaluation of Stencil with MPI barrier")

time_results = []
for i in range(11):
    comm.Barrier()

    # copy initial values to device
    cuda.to_device(local_x, to=x_shmem[rank])
    cuda.to_device(local_xnew, to=xnew_shmem[rank])
    cuda.to_device(np.zeros(iterations), to=diffnorm_shmem[rank])

    comm.Barrier()
    start_time = MPI.Wtime()

    for iter in range(iterations):
        stencil_kernel[blockspergrid, threadsperblock](x_shmem, xnew_shmem, n, iter, diffnorm_shmem, rank, size, y_end, x_end)
        nbshmem.barrier_all_host()
        tmp = x_shmem
        x_shmem = xnew_shmem
        xnew_shmem = tmp

    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)
    if rank == 0:
        diffnorm_0 = diffnorm_shmem[0].copy_to_host()
        diffnorm_1 = diffnorm_shmem[1].copy_to_host()
        diffnorm = diffnorm_0 + diffnorm_1
        iteration_norm = np.sqrt(diffnorm[::50])
        assert np.allclose(norm, iteration_norm)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Stencil MPI-Barrier: {np.mean(results)}")
    print(f"Stencil MPI-Barrier: {np.std(results)}")