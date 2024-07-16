from numba import cuda
import numpy as np
from numba.cuda.cudadrv import driver, devicearray
from cuda.cuda import CUdeviceptr


HEAP_SIZE = 1000000


class StaticHeap():
    def __init__(self):
        # Array der Größe HEAP_SIZE Bytes
        self.local_heap_buf = np.zeros(HEAP_SIZE, dtype=np.int8)
        self.shmem_heaps = array(self.local_heap_buf)
        self.offset = 0

    def allocate(self, npary):
        shape, strides, dtype = cuda.prepare_shape_strides_dtype(
            npary.shape, npary.strides, npary.dtype, order='C')

        alloc_size = driver.memory_size_from_info(shape, strides, dtype.itemsize)

        shmem_ary = ()

        for i in range(len(self.shmem_heaps)):
            buf_base = self.shmem_heaps[i]
            buf_base_adr = buf_base.__cuda_array_interface__["data"][0]
            alloc_adr = buf_base_adr + self.offset

            data = driver.MemoryPointer(cuda.current_context(),
                                        CUdeviceptr(alloc_adr),
                                        size=alloc_size, owner=None)
            devary = devicearray.DeviceNDArray(shape=shape, strides=strides,
                                               dtype=dtype, gpu_data=data,
                                               stream=None)
            shmem_ary = shmem_ary + (devary, )

        self.offset += alloc_size
        return shmem_ary


class HostContext():
    def init(self, comm, static_heap):
        self.mpi = comm
        self.use_static_heap = static_heap

    def init_static_heap(self):
        self.heap = StaticHeap()


ctx = HostContext()


def init(mpi_comm, static_heap=False):
    """
    Collective function that sets the device for the PE
    and may initialize a static heap.
    """
    ctx.init(mpi_comm, static_heap)
    rank = mpi_comm.Get_rank()
    cuda.select_device(rank)
    if static_heap:
        ctx.init_static_heap()
        sync = ctx.heap.allocate(np.empty(1, dtype=np.int64))
        return sync


def array(npary):
    """
    Collective function that allocates device memory by
    copying the given numpy array to device. Returns a
    tuple containing Numba DeviceArrays of all peer devices
    sorted by PE.
    """
    # TODO: Für mehr als 2 Peers implementieren

    dary = cuda.to_device(npary)
    handle = dary.get_ipc_handle()
    rank = ctx.mpi.Get_rank()
    peer = (rank+1) % 2
    s_req = ctx.mpi.isend(handle, peer)
    r_req = ctx.mpi.irecv(source=peer)
    s_req.wait()
    res = r_req.wait()
    shmem_ary = ()
    for i in range(2):
        if i == ctx.mpi.Get_rank():
            shmem_ary = shmem_ary + (dary, )
        else:
            shmem_ary = shmem_ary + (res.open(), )
    return shmem_ary


def alloc_like(npary):
    """
    Collective function that allocates device memory based on
    the given numpy array but does not copy the numpy array to
    device. Returns a tuple containing Numba DeviceArrays of all
    peer devices sorted by PE.
    """
    if ctx.use_static_heap:
        return ctx.heap.allocate(npary)
    else:
        # TODO: anpassen
        dary = cuda.to_device(npary)
        handle = dary.get_ipc_handle()
        rank = ctx.mpi.Get_rank()
        peer = (rank+1) % 2
        s_req = ctx.mpi.isend(handle, peer)
        r_req = ctx.mpi.irecv(source=peer)
        s_req.wait()
        res = r_req.wait()
        shmem_ary = ()
        for i in range(2):
            if i == ctx.mpi.Get_rank():
                shmem_ary = shmem_ary + (dary, )
            else:
                shmem_ary = shmem_ary + (res.open(), )
        return shmem_ary


def barrier_all_host():
    """
    Collective function that synchronizes all PEs on the host.
    """
    ctx.mpi.Barrier()