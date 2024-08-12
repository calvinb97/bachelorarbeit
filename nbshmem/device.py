from numba import cuda
import cffi
from nbshmem import device_internal as internal

ffi = cffi.FFI()

wait_until_volatile = cuda.declare_device('wait_until_cu_volatile',
                                          'void(CPointer(int64), int64)')
wait_until_threadfence = cuda.declare_device('wait_until_cu_fence',
                                             'void(CPointer(int64), int64)')


@cuda.jit(device=True)
def fence():
    """
    Ensures memory ordering of the calling thread.
    """
    cuda.threadfence_system()


@cuda.jit(device=True)
def wait_until(ary, idx, val):
    """
    Busy-waits until condition is true.
    """
    while ary[idx] != val:
        cuda.threadfence()


@cuda.jit(device=True)
def wait_until_c(ary, val):
    """
    Busy-waits until condition is true.
    Implemented with volatile memory access in C.
    """
    ary_ptr = ffi.from_buffer(ary)
    wait_until_volatile(ary_ptr, val)


@cuda.jit(device=True)
def barrier_all_device(sync_shmem, my_pe, npes):
    """
    Synchronized all GPUs and must be called by every Thread.
    """
    x, y = cuda.grid(2)
    g = cuda.cg.this_grid()
    cuda.threadfence_system()
    g.sync()
    if x == 0 and y == 0:
        for i in range(npes):
            cuda.atomic.add(sync_shmem[i], 0, 1)
        wait_until(sync_shmem[my_pe], 0, npes)
    g.sync()
    if x == 0 and y == 0:
        sync_shmem[my_pe][0] = 0
    

@cuda.jit(device=True)
def barrier_all_atomic(sync_shmem, my_pe, npes):
    x, y = cuda.grid(2)
    g = cuda.cg.this_grid()

    counter = sync_shmem[my_pe][0]

    cuda.threadfence_system()
    g.sync()

    if my_pe == 0:
        if cuda.blockIdx.x == 0 and cuda.blockIdx.y == 0:
            if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
                wait_until(sync_shmem[my_pe], 0, counter + (npes - 1))
            cuda.syncthreads()
            idx_in_block = get_idx_1d_block()
            if idx_in_block > 0 and idx_in_block < npes:
                sync_shmem[idx_in_block][0] += 1
    else:
        if x == 0 and y == 0:
            cuda.atomic.add(sync_shmem[0], 0, 1)
            wait_until(sync_shmem[my_pe], 0, counter + 1)
    g.sync()


@cuda.jit(device=True)
def barrier_all(sync_shmem, my_pe, npes):
    """
    Barrier for all threads on all GPUs.
    sync_shmem array is organized as follows:
    | phase | flag GPU 1 | ... | flag GPU n |
    """
    x, y = cuda.grid(2)
    g = cuda.cg.this_grid()

    # current phase (alternating between barrier calls)
    phase = sync_shmem[my_pe][0]

    # synchronize all threads on local GPU
    cuda.threadfence_system()
    g.sync()

    # root PE waits until barrier flags of all PEs are set,
    # then signals phase swap to all other PEs
    if my_pe == 0:
        if cuda.blockIdx.x == 0 and cuda.blockIdx.y == 0:
            idx_in_block = get_idx_1d_block()
            if idx_in_block > 0 and idx_in_block < npes:
                wait_until(sync_shmem[my_pe], idx_in_block, not phase)
            cuda.syncthreads()
            if idx_in_block < npes:
                sync_shmem[idx_in_block][0] = not phase
    else:
        # one thread signals arrival at barrier to root PE
        # and waits for barrier signal from root PE
        if x == 0 and y == 0:
            sync_shmem[0][my_pe] = not phase
            wait_until(sync_shmem[my_pe], 0, not phase)

    # synchronize all threads on local GPU
    g.sync()


@cuda.jit()
def get_idx_1d_block():
    return cuda.threadIdx.x * cuda.blockDim.y + cuda.threadIdx.y


@cuda.jit(device=True)
def get_1d_threadidx_in_grid():
    """
    Returns the one-dimensional grid-global index of the calling thread.
    """
    block_in_grid = cuda.gridDim.y * cuda.blockIdx.x + cuda.blockIdx.y
    thread_in_block = cuda.blockDim.y * cuda.threadIdx.x + cuda.threadIdx.y
    threads_per_block = cuda.blockDim.x * cuda.blockDim.y
    return block_in_grid * threads_per_block + thread_in_block


@cuda.jit(device=True)
def get_num_threads_in_grid():
    """
    Returns the total number of threads in the grid.
    """
    blocks_per_grid = cuda.gridDim.x * cuda.gridDim.y
    threads_per_block = cuda.blockDim.x * cuda.blockDim.y
    return blocks_per_grid * threads_per_block


@cuda.jit(device=True)
def broadcast_1d(dest, src, elems, root, my_pe, sync):
    if root == my_pe:
        idx = get_1d_threadidx_in_grid()
        numthreads = get_num_threads_in_grid()
        for i in range(idx, elems, numthreads):
            for pe in range(len(dest)):
                dest[pe][i] = src[i]
    barrier_all(sync, my_pe, len(dest))


@cuda.jit(device=True)
def broadcast_2d(dest, src, rows, root, my_pe, sync):
    if root == my_pe:
        idx = get_1d_threadidx_in_grid()
        numthreads = get_num_threads_in_grid()
        for i in range(idx, rows * src.shape[1], numthreads):
            row = i // src.shape[1]
            col = i % src.shape[1]
            for pe in range(len(dest)):
                dest[pe][row][col] = src[row][col]
    barrier_all(sync, my_pe, len(dest))


@cuda.jit(device=True)
def alltoall_1d(dest, src, elems, my_pe, sync):
    idx = get_1d_threadidx_in_grid()
    numthreads = get_num_threads_in_grid()
    for i in range(idx, elems * len(dest), numthreads):
        pe = i // elems
        pe_offset = my_pe * elems
        elem_i = i % elems
        dest[pe][pe_offset + elem_i] = src[i]
    barrier_all(sync, my_pe, len(dest))


@cuda.jit(device=True)
def alltoall_2d(dest, src, rows, my_pe, sync):
    idx = get_1d_threadidx_in_grid()
    numthreads = get_num_threads_in_grid()
    for i in range(idx, rows * len(dest) * src.shape[1], numthreads):
        row = i // src.shape[1]
        col = i % src.shape[1]
        pe = row // rows
        elem_row = row % rows
        row_offset = my_pe * rows
        dest[pe][row_offset + elem_row][col] = src[row][col]
    barrier_all(sync, my_pe, len(dest))


@cuda.jit(device=True)
def allreduce_sum(dest, src, elems, my_pe, sync):
    """
    Performs SUM-Allreduce and must be called by every thread.
    """
    if len(src[my_pe].shape) == 1:
        internal.allreduce_1d(dest, src, elems, my_pe, sync)
    if len(src[my_pe].shape) == 2:
        internal.allreduce_2d(dest, src, elems, my_pe, sync)


