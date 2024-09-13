from numba import cuda
import numpy as np
from mpi4py import MPI
import nbshmem

import cupy
from cupyx.distributed import NCCLBackend


@cuda.jit
def allreduce_1d_kernel(ary, ary_size, res, my_pe, npes, sync):
    nbshmem.allreduce_1d_sum(res, ary, ary_size, my_pe, sync)


@cuda.jit
def allreduce_1d_bcast_kernel(ary, ary_size, res, my_pe, npes, sync):
    nbshmem.allreduce_1d_sum_bcast(res, ary, ary_size, my_pe, sync)


@cuda.jit
def allreduce_2d_kernel(ary, ary_size, res, my_pe, npes, sync):
    nbshmem.allreduce_2d_sum(res, ary, ary_size, my_pe, sync)


@cuda.jit
def broadcast_1d_kernel(ary, nelems, bcast_res, my_pe, npes, sync):
    nbshmem.broadcast_1d(bcast_res, ary, nelems, 0, my_pe, sync)


@cuda.jit
def alltoall_1d_kernel(ary, nelems, alltoall_res, my_pe, npes, sync):
    nbshmem.alltoall_1d(alltoall_res, ary, nelems, my_pe, sync)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"Hello from {rank}")

nbshmem.init(comm)

# NCCL setup
cupy.cuda.Device(rank).use()
num_devices = 2
nccl_comm = NCCLBackend(num_devices, rank, use_mpi=True)


ary_size = 16384 * 10000

ary = np.arange(ary_size, dtype=np.int32)
ary_shmem = nbshmem.array(ary)
res_shmem = nbshmem.alloc((ary_size, ), np.int32)

# Sync array
sync_local = np.full(3, False)
sync_shmem = nbshmem.array(sync_local)



# # # # # # #
# Reduction #
# # # # # # #

reduction_result = 2 * ary

print("Allreduce for 1 Block with 256 Threads")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    allreduce_1d_kernel[1, 256](ary_shmem, ary_size, res_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    h_dest = res_shmem[rank].copy_to_host()
    assert np.allclose(h_dest, reduction_result)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Block (256) Allreduce mean: {np.mean(results)}")
    print(f"Block (256) Allreduce std: {np.std(results)}")


print("Allreduce for 1 Block with 512 Threads")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    allreduce_1d_kernel[1, 512](ary_shmem, ary_size, res_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    h_dest = res_shmem[rank].copy_to_host()
    assert np.allclose(h_dest, reduction_result)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Block (512) Allreduce mean: {np.mean(results)}")
    print(f"Block (512) Allreduce std: {np.std(results)}")


print("Allreduce for 1600 blocks with 32 Threads (max gridsize)")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    allreduce_1d_kernel[1600, 32](ary_shmem, ary_size, res_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    h_dest = res_shmem[rank].copy_to_host()
    assert np.allclose(h_dest, reduction_result)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Allreduce (max grid) mean: {np.mean(results)}")
    print(f"Allreduce (max grid) std: {np.std(results)}")


print("Allreduce_Bcast for 1600 blocks with 32 Threads")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    allreduce_1d_bcast_kernel[1600, 32](ary_shmem, ary_size, res_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    h_dest = res_shmem[rank].copy_to_host()
    assert np.allclose(h_dest, reduction_result)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Allreduce_Bcast (1600*32) mean: {np.mean(results)}")
    print(f"Allreduce_Bcast (1600*32) std: {np.std(results)}")


print("Allreduce with NCCL")
cupy_ary = cupy.arange(ary_size, dtype=np.int32)
cupy_res = cupy.zeros(ary_size, dtype=np.int32)
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    nccl_comm.all_reduce(cupy_ary, cupy_res, op="sum")
    cupy.cuda.runtime.deviceSynchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    assert np.allclose(cupy_res, reduction_result)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Allreduce (NCCL) mean: {np.mean(results)}")
    print(f"Allreduce (NCCL) std: {np.std(results)}")


# # # # # # #
# Broadcast #
# # # # # # #


print("Broadcast for 1 Block with 512 Threads")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    broadcast_1d_kernel[1, 512](ary_shmem, ary_size, res_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    h_dest = res_shmem[rank].copy_to_host()
    assert np.allclose(h_dest, ary)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Block (512) Broadcast mean: {np.mean(results)}")
    print(f"Block (512) Broadcast std: {np.std(results)}")


print("Broadcast for 1 Block with 1024 Threads")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    broadcast_1d_kernel[1, 1024](ary_shmem, ary_size, res_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    h_dest = res_shmem[rank].copy_to_host()
    assert np.allclose(h_dest, ary)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Block (1024) Broadcast mean: {np.mean(results)}")
    print(f"Block (1024) Broadcast std: {np.std(results)}")


print("Broadcast for 240 blocks with 512 Threads (max gridsize)")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    broadcast_1d_kernel[240, 512](ary_shmem, ary_size, res_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    h_dest = res_shmem[rank].copy_to_host()
    assert np.allclose(h_dest, ary)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Broadcast (max grid) mean: {np.mean(results)}")
    print(f"Broadcast (max grid) std: {np.std(results)}")


print("Broadcast with NCCL")
if rank == 0:
    cupy_ary = cupy.arange(ary_size, dtype=np.int32)
else:
    cupy_ary = cupy.zeros(ary_size, dtype=np.int32)
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    nccl_comm.broadcast(cupy_ary)
    cupy.cuda.runtime.deviceSynchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    assert np.allclose(h_dest, cupy_ary)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Broadcast (NCCL) mean: {np.mean(results)}")
    print(f"Broadcast (NCCL) std: {np.std(results)}")


# # # # # # #
# AlltoAll  #
# # # # # # #

print("AlltoAll for 1 Block with 512 Threads")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    alltoall_1d_kernel[1, 512](ary_shmem, ary_size, res_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    h_dest = res_shmem[rank].copy_to_host()
    assert np.allclose(h_dest, ary)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Block (512) Broadcast mean: {np.mean(results)}")
    print(f"Block (512) Broadcast std: {np.std(results)}")
