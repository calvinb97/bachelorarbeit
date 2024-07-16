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
def wait_until(ary, val):
    """
    Busy-waits until condition is true.
    """
    while ary[0] != val:
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
        wait_until(sync_shmem[my_pe], npes)
    g.sync()
    if x == 0 and y == 0:
        sync_shmem[my_pe][0] = 0


@cuda.jit(device=True)
def barrier_all_device_volatile(sync_shmem, my_pe, npes):
    x, y = cuda.grid(2)
    g = cuda.cg.this_grid()
    cuda.threadfence_system()
    g.sync()
    if x == 0 and y == 0:
        for i in range(npes):
            cuda.atomic.add(sync_shmem[i], 0, 1)
        wait_until_c(sync_shmem[my_pe], npes)
    g.sync()
    if x == 0 and y == 0:
        sync_shmem[my_pe][0] = 0


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
    blocks_per_grid = cuda.blockDim.x * cuda.blockDim.y
    threads_per_block = cuda.blockDim.x * cuda.blockDim.y
    return blocks_per_grid * threads_per_block


@cuda.jit(device=True)
def allreduce_sum(dest, src, elems, my_pe, sync):
    """
    Performs SUM-Allreduce and must be called by every thread.
    """
    if len(src[my_pe].shape) == 1:
        internal.allreduce_1d(dest, src, elems, my_pe, sync)
    if len(src[my_pe].shape) == 2:
        internal.allreduce_2d(dest, src, elems, my_pe, sync)
