from mpi4py import MPI
import nbshmem
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print("Perftest STATIC HEAP allocations")

start_time = MPI.Wtime()
nbshmem.init(comm, static_heap=True)
end_time = MPI.Wtime()
time = end_time - start_time
print(f"Init (PE {rank}): {time}")


start_time = MPI.Wtime()
nbshmem.alloc((100000000,), np.int8)
end_time = MPI.Wtime()
time = end_time - start_time
print(f"Alloc 100 MB (PE {rank}): {time}")

start_time = MPI.Wtime()
nbshmem.alloc((1000000000,), np.int8)
end_time = MPI.Wtime()
time = end_time - start_time
print(f"Alloc 1 GB (PE {rank}): {time}")

start_time = MPI.Wtime()
nbshmem.alloc((2000000000,), np.int8)
end_time = MPI.Wtime()
time = end_time - start_time
print(f"Alloc 2 GB (PE {rank}): {time}")

start_time = MPI.Wtime()
nbshmem.alloc((4000000000,), np.int8)
end_time = MPI.Wtime()
time = end_time - start_time
print(f"Alloc 4 GB (PE {rank}): {time}")