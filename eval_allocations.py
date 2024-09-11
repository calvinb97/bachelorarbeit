from mpi4py import MPI
import nbshmem
import numpy as np
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parser = argparse.ArgumentParser(description='Evaluation of allocation performance')
parser.add_argument('heap_type', type=str, help='static or dynamic')
parser.add_argument('op', type=str, help='alloc or array')
parser.add_argument('--copy', action='store_true', help='Copy values to allocation')

args = parser.parse_args()

if rank == 0:
    print(f"{args.heap_type} heap allocations with operation nb.{args.op} and copy={args.copy}")


size_bytes = [100000000, 1000000000, 2000000000, 4000000000]
size_print = ["100 MB", "1 GB", "2 GB", "4 GB"]

if args.heap_type == "static":
    start_time = MPI.Wtime()
    nbshmem.init(comm, static_heap=True)
    end_time = MPI.Wtime()
if args.heap_type == "dynamic":
    start_time = MPI.Wtime()
    nbshmem.init(comm)
    end_time = MPI.Wtime()

time = end_time - start_time
print(f"Init (PE {rank}): {time}")

for bytes, size in zip(size_bytes, size_print):
    alloc_shape = (bytes,)

    if args.op == "array":
        ary = np.ones(alloc_shape, dtype=np.int8)

        if args.copy:
            comm.Barrier()
            start_time = MPI.Wtime()
            nbshmem.array(ary)
            end_time = MPI.Wtime()
        else:
            comm.Barrier()
            start_time = MPI.Wtime()
            nbshmem.array(ary, copy=False)
            end_time = MPI.Wtime()

    if args.op == "alloc":
        comm.Barrier()
        start_time = MPI.Wtime()
        nbshmem.alloc(alloc_shape, np.int8)
        end_time = MPI.Wtime()

    time = end_time - start_time
    print(f"Allocation of size {size} (PE {rank}): {time}")