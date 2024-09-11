from mpi4py import MPI
import nbshmem
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

nbshmem.init(comm, static_heap=True)

alloc1 = nbshmem.alloc((1, ), np.int8)
alloc2 = nbshmem.alloc((5, ), np.int64)
alloc3 = nbshmem.alloc((5, ), np.int64)

if rank == 0:
    adr1 = alloc1[0].__cuda_array_interface__["data"][0]
    print(f"{adr1=}")
    if adr1 % np.dtype(np.int8).itemsize == 0:
        print("aligned")
    else:
        print("not aligned")

    adr2 = alloc2[0].__cuda_array_interface__["data"][0]
    print(f"{adr2=}")
    if adr2 % np.dtype(np.int64).itemsize == 0:
        print("aligned")
    else:
        print("not aligned")

    adr3 = alloc3[0].__cuda_array_interface__["data"][0]
    print(f"{adr3=}")
    if adr3 % np.dtype(np.int64).itemsize == 0:
        print("aligned")
    else:
        print("not aligned")