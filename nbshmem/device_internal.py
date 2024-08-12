from numba import cuda
from nbshmem import device


@cuda.jit(device=True)
def allreduce_1d(dest, src, elems, my_pe, sync):
    """
    Performs one-dimensional allreduce by ideally
    using one thread per element. If there are more
    elements than threads, a thread may perform the
    reduction on more than one element.
    """
    idx = device.get_1d_threadidx_in_grid()
    numthreads = device.get_num_threads_in_grid()
    for i in range(0, elems, numthreads):
        dest[idx] = 0
    device.barrier_all(sync, my_pe, len(src))
    for i in range(idx, elems, numthreads):
        for j in range(len(src)):
            dest[i] += src[j][idx]


@cuda.jit(device=True)
def allreduce_2d(dest, src, elems, my_pe, sync):
    pass