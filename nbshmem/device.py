from numba import cuda
import cffi

ffi = cffi.FFI()

wait_until_bool_cu = cuda.declare_device('wait_until_bool_volatile',
                                         'void(CPointer(boolean), int64, boolean)')


@cuda.jit(device=True)
def fence():
    """
    Ensures sequentially consistent ordering of operations
    """
    cuda.threadfence_system()


@cuda.jit(device=True)
def wait_until_ptxtest1(ary, val):
    while ary[0] != val:
        cuda.threadfence()


@cuda.jit(device=True)
def wait_until_ptxtest2(ary, idx, val):
    while ary[idx] != val:
        cuda.threadfence_block()


@cuda.jit(device=True)
def wait_until_ptxtest3(ary, idx, val):
    while ary[idx] != val:
        continue


@cuda.jit(device=True)
def wait_until(ary, idx, val):
    """
    Busy-waits until condition is true.
    """
    while ary[idx] != val:
        cuda.threadfence()


@cuda.jit(device=True)
def wait_until_bool(ary, idx, val):
    """
    Busy-waits until condition is true.
    Implemented with volatile memory access in C.
    """
    ary_ptr = ffi.from_buffer(ary)
    wait_until_bool_cu(ary_ptr, idx, val)


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


@cuda.jit(device=True)
def barrier_all_volatile(sync_shmem, my_pe, npes):
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
                wait_until_bool(sync_shmem[my_pe], idx_in_block, not phase)
            cuda.syncthreads()
            if idx_in_block < npes:
                sync_shmem[idx_in_block][0] = not phase
    else:
        # one thread signals arrival at barrier to root PE
        # and waits for barrier signal from root PE
        if x == 0 and y == 0:
            sync_shmem[0][my_pe] = not phase
            wait_until_bool(sync_shmem[my_pe], 0, not phase)

    # synchronize all threads on local GPU
    g.sync()


@cuda.jit()
def get_idx_1d_block():
    """
    Returns the one-dimensional index of a thread in its block
    """
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
        src = src[my_pe]
        idx = get_1d_threadidx_in_grid()
        numthreads = get_num_threads_in_grid()
        for i in range(idx, elems, numthreads):
            for pe in range(len(dest)):
                dest[pe][i] = src[i]
    barrier_all(sync, my_pe, len(dest))


@cuda.jit(device=True)
def broadcast_2d(dest, src, rows, root, my_pe, sync):
    if root == my_pe:
        src = src[my_pe]
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
    src = src[my_pe]
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
    src = src[my_pe]
    for i in range(idx, rows * len(dest) * src.shape[1], numthreads):
        row = i // src.shape[1]
        col = i % src.shape[1]
        pe = row // rows
        elem_row = row % rows
        row_offset = my_pe * rows
        dest[pe][row_offset + elem_row][col] = src[row][col]
    barrier_all(sync, my_pe, len(dest))


@cuda.jit(device=True)
def allreduce_1d_sum(dest, src, elems, my_pe, sync):
    dest = dest[my_pe]
    idx = get_1d_threadidx_in_grid()
    numthreads = get_num_threads_in_grid()
    for i in range(idx, elems, numthreads):
        dest[i] = 0
    barrier_all(sync, my_pe, len(src))
    for i in range(idx, elems, numthreads):
        for pe in range(len(src)):
            dest[i] += src[pe][i]
    barrier_all(sync, my_pe, len(src))


@cuda.jit(device=True)
def allreduce_2d_sum(dest, src, rows, my_pe, sync):
    dest = dest[my_pe]
    idx = get_1d_threadidx_in_grid()
    numthreads = get_num_threads_in_grid()
    for i in range(idx, rows * dest.shape[1], numthreads):
        row = i // dest.shape[1]
        col = i % dest.shape[1]
        dest[row][col] = 0
    barrier_all(sync, my_pe, len(src))
    for i in range(idx, rows * dest.shape[1], numthreads):
        row = i // dest.shape[1]
        col = i % dest.shape[1]
        for pe in range(len(src)):
            dest[row][col] += src[pe][row][col]
    barrier_all(sync, my_pe, len(src))


@cuda.jit(device=True)
def allreduce_1d_sum_bcast(dest, src, elems, my_pe, sync):
    g = cuda.cg.this_grid()
    my_dest = dest[my_pe]
    idx = get_1d_threadidx_in_grid()
    numthreads = get_num_threads_in_grid()
    if my_pe == 0:
        for i in range(idx, elems, numthreads):
            my_dest[i] = 0
    barrier_all(sync, my_pe, len(src))
    if my_pe == 0:
        for i in range(idx, elems, numthreads):
            for pe in range(len(src)):
                my_dest[i] += src[pe][i]
        g.sync()
    broadcast_1d(dest, dest, elems, 0, my_pe, sync)


@cuda.jit
def barrier_all_kernel(sync_shmem, my_pe, npes):
    barrier_all(sync_shmem, my_pe, npes)
