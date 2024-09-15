from numba import cuda
import numpy as np
from mpi4py import MPI
import nbshmem

import os

basedir = os.path.dirname(os.path.abspath(__file__))
nbshmem_cu = os.path.join(basedir, 'nbshmem', 'nbshmem.cu')


@cuda.jit
def barrier_kernel(sync_shmem, my_pe):
    for i in range(1000):
        if my_pe == 0:
            cuda.nanosleep(10000000)
        nbshmem.barrier_all(sync_shmem, my_pe, 2)


@cuda.jit(link=[nbshmem_cu])
def barrier_volatile_kernel(sync_shmem, my_pe):
    for i in range(1000):
        if my_pe == 0:
            cuda.nanosleep(10000000)
        nbshmem.barrier_all_volatile(sync_shmem, my_pe, 2)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"Hello from {rank}")

nbshmem.init(comm)

sync_local = np.full(3, False)
sync_shmem = nbshmem.array(sync_local)

print("Barrier")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    barrier_kernel[126, 512](sync_shmem, rank)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Barrier mean: {np.mean(results)}")
    print(f"Barrier std: {np.std(results)}")


print("Barrier volatile")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    barrier_volatile_kernel[126, 512](sync_shmem, rank)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Barrier volatile mean: {np.mean(results)}")
    print(f"Barrier volatile std: {np.std(results)}")

