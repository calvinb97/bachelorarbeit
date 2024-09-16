from numba import cuda
import numpy as np
from mpi4py import MPI
import nbshmem

import cupy
from cupyx.distributed import NCCLBackend

# ignore performance warning for one-block-launch
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


@cuda.jit
def allreduce_1d_kernel(ary, ary_size, res, my_pe, npes, sync):
    nbshmem.allreduce_1d_sum(res, ary, ary_size, my_pe, sync)


@cuda.jit
def allreduce_1d_bcast_kernel(ary, ary_size, res, my_pe, npes, sync):
    nbshmem.allreduce_1d_sum_bcast(res, ary, ary_size, my_pe, sync)


@cuda.jit
def allreduce_2d_kernel(ary, rows, res, my_pe, npes, sync):
    nbshmem.allreduce_2d_sum(res, ary, rows, my_pe, sync)


@cuda.jit
def broadcast_1d_kernel(ary, nelems, bcast_res, my_pe, npes, sync):
    nbshmem.broadcast_1d(bcast_res, ary, nelems, 0, my_pe, sync)


@cuda.jit
def broadcast_2d_kernel(ary, rows, bcast_res, my_pe, npes, sync):
    nbshmem.broadcast_2d(bcast_res, ary, rows, 0, my_pe, sync)


@cuda.jit
def alltoall_1d_kernel(ary, nelems, alltoall_res, my_pe, npes, sync):
    nbshmem.alltoall_1d(alltoall_res, ary, nelems, my_pe, sync)


@cuda.jit
def alltoall_2d_kernel(ary, rows_per_chunk, alltoall_res, my_pe, npes, sync):
    nbshmem.alltoall_2d(alltoall_res, ary, rows_per_chunk, my_pe, sync)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"Hello from {rank}")

nbshmem.init(comm)

# NCCL setup
cupy.cuda.Device(rank).use()
num_devices = 2
nccl_comm = NCCLBackend(num_devices, rank, use_mpi=True)


ary_size = 128 * 1000000
# ary_size = 16384 * 10000
# ary_size = 20

ary = np.arange(ary_size, dtype=np.int32)
ary_shmem = nbshmem.array(ary)
res_shmem = nbshmem.alloc((ary_size, ), np.int32)

ary = ary.reshape(2, ary_size//2)
ary2d_shmem = nbshmem.array(ary)
res2d_shmem = nbshmem.alloc((2, ary_size//2), np.int32)
ary = ary.reshape(ary_size)

# Sync array
sync_local = np.full(3, False)
sync_shmem = nbshmem.array(sync_local)

if rank == 0:
    print(f"*** Starting evaluation for array of size {ary_size * 4 / 1000000} MB ***")


# # # # # # #
# Reduction #
# # # # # # #

reduction_result = 2 * ary

if rank == 0:
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


if rank == 0:
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

if rank == 0:
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

if rank == 0:
    print("Allreduce 2D for 1600 blocks with 32 Threads (max gridsize)")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    allreduce_2d_kernel[1600, 32](ary2d_shmem, 2, res2d_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    h_dest = res2d_shmem[rank].copy_to_host()
    h_dest = h_dest.reshape(ary_size)
    assert np.allclose(h_dest, reduction_result)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Allreduce 2D (max grid) mean: {np.mean(results)}")
    print(f"Allreduce 2D (max grid) std: {np.std(results)}")

if rank == 0:
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


if rank == 0:
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


if rank == 0:
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


if rank == 0:
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


if rank == 0:
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

if rank == 0:
    print("Broadcast 2D for 240 blocks with 512 Threads (max gridsize)")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    broadcast_2d_kernel[240, 512](ary2d_shmem, 2, res2d_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    h_dest = res2d_shmem[rank].copy_to_host()
    h_dest = h_dest.reshape(ary_size)
    assert np.allclose(h_dest, ary)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Broadcast 2D (max grid) mean: {np.mean(results)}")
    print(f"Broadcast 2D (max grid) std: {np.std(results)}")


if rank == 0:
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

ary_0 = np.arange(ary_size, dtype=np.int32)
ary_1 = np.arange(ary_size, dtype=np.int32) * 10
if rank == 0:
    ary_shmem = nbshmem.array(ary_0)
    ary_0 = ary_0.reshape(2, ary_size//2)
    ary2d_shmem = nbshmem.array(ary_0)
    ary_0 = ary_0.reshape(ary_size)
else:
    ary_shmem = nbshmem.array(ary_1)
    ary_1 = ary_1.reshape(2, ary_size//2)
    ary2d_shmem = nbshmem.array(ary_1)
    ary_1 = ary_1.reshape(ary_size)
alltoall_res = np.empty(ary_size, dtype=np.int32)
chunk_size = ary_size // 2
if rank == 0:
    alltoall_res[:chunk_size] = ary_0[:chunk_size]
    alltoall_res[chunk_size:] = ary_1[:chunk_size]
else:
    alltoall_res[:chunk_size] = ary_0[chunk_size:]
    alltoall_res[chunk_size:] = ary_1[chunk_size:]

if rank == 0:
    print("AlltoAll for 1 Block with 512 Threads")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    alltoall_1d_kernel[1, 512](ary_shmem, chunk_size, res_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    h_dest = res_shmem[rank].copy_to_host()
    assert np.allclose(h_dest, alltoall_res)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Block (512) AlltoAll mean: {np.mean(results)}")
    print(f"Block (512) AlltoAll std: {np.std(results)}")


if rank == 0:
    print("AlltoAll for 1 Block with 1024 Threads")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    alltoall_1d_kernel[1, 1024](ary_shmem, chunk_size, res_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    h_dest = res_shmem[rank].copy_to_host()
    assert np.allclose(h_dest, alltoall_res)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"Block (1024) AlltoAll mean: {np.mean(results)}")
    print(f"Block (1024) AlltoAll std: {np.std(results)}")


if rank == 0:
    print("AlltoAll for 320 blocks with 256 Threads (max gridsize)")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    alltoall_1d_kernel[320, 256](ary_shmem, chunk_size, res_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    h_dest = res_shmem[rank].copy_to_host()
    assert np.allclose(h_dest, alltoall_res)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"AlltoAll (max blocks) mean: {np.mean(results)}")
    print(f"AlltoAll (max blocks) std: {np.std(results)}")

if rank == 0:
    print("AlltoAll 2D for 320 blocks with 256 Threads (max gridsize)")
time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    alltoall_2d_kernel[320, 256](ary2d_shmem, 1, res2d_shmem, rank, 2, sync_shmem)
    cuda.synchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    h_dest = res2d_shmem[rank].copy_to_host()
    h_dest = h_dest.reshape(ary_size)
    assert np.allclose(h_dest, alltoall_res)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"AlltoAll 2D (max blocks) mean: {np.mean(results)}")
    print(f"AlltoAll 2D (max blocks) std: {np.std(results)}")


if rank == 0:
    print("AlltoAll with NCCL")
if rank == 0:
    cupy_ary = cupy.arange(ary_size, dtype=np.int32)
else:
    cupy_ary = cupy.arange(ary_size, dtype=np.int32) * 10

cupy_res = cupy.zeros(ary_size, dtype=np.int32)

time_results = []
for i in range(11):
    comm.Barrier()
    start_time = MPI.Wtime()
    cupy.cuda.nccl.groupStart()
    for pe in range(2):
        from_idx = pe * chunk_size
        to_idx = (pe + 1) * chunk_size
        nccl_comm.send(cupy_ary[from_idx:to_idx], pe)
        nccl_comm.recv(cupy_res[from_idx:to_idx], pe)
    cupy.cuda.nccl.groupEnd()
    cupy.cuda.runtime.deviceSynchronize()
    end_time = MPI.Wtime()
    time = end_time - start_time
    if i == 0:
        print(f"iteration {i}: {time} on process {rank}")
    else:
        time_results.append(time)

    # assert correct result
    # cupy_res = cupy_res.reshape(ary_size)
    assert np.allclose(cupy_res, alltoall_res)

gathered_results = comm.gather(time_results)
if rank == 0:
    results = np.array(gathered_results).reshape(20,)
    print(f"AlltoAll (NCCL) mean: {np.mean(results)}")
    print(f"AlltoAll (NCCL) std: {np.std(results)}")

