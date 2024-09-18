from numba import cuda
import numpy as np
from numba.cuda.cudadrv import driver
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from cuda.cuda import CUdeviceptr
import nbshmem.device as device

# 8 GB
HEAP_SIZE = 8000000000


class StaticHeap():
    def __init__(self, heap_size, my_pe):
        self.heap_size = heap_size
        self.my_pe = my_pe
        self.local_heap = self._alloc_local_heap()
        self.shmem_heaps = map_peer_arrays(self.local_heap)
        self.offset = 0

    def _alloc_local_heap(self):
        # allocates self.heap_size bytes on the GPU
        return DeviceNDArray(shape=(self.heap_size,), strides=(1,),
                             dtype=np.int8)

    def _shmem_alloc_from_heap(self, shape, strides, dtype):
        alloc_size = driver.memory_size_from_info(shape, strides, dtype.itemsize)
        darys = []
        for i in range(len(self.shmem_heaps)):
            buf_base = self.shmem_heaps[i]
            buf_base_adr = buf_base.__cuda_array_interface__["data"][0]
            alloc_adr = buf_base_adr + self.offset
            alloc_adr, alignment_offset = self._align(alloc_adr, dtype)
            data = driver.MemoryPointer(cuda.current_context(),
                                        CUdeviceptr(alloc_adr),
                                        size=alloc_size, owner=None)
            dary = DeviceNDArray(shape=shape, strides=strides,
                                 dtype=dtype, gpu_data=data)
            darys.append(dary)
        self.offset += (alloc_size + alignment_offset)
        return tuple(darys)

    def _align(self, adr, dtype):
        itemsize = dtype.itemsize
        if adr % itemsize == 0:
            return adr, 0
        alignment_offset = itemsize - (adr % itemsize)
        return adr + alignment_offset, alignment_offset

    def alloc_array(self, npary, copy=True):
        shape, strides, dtype = cuda.prepare_shape_strides_dtype(
            npary.shape, npary.strides, npary.dtype, order='C')

        shmem_ary = self._shmem_alloc_from_heap(shape, strides, dtype)

        if copy:
            cuda.to_device(npary, to=shmem_ary[self.my_pe])
        return shmem_ary

    def alloc(self, shape, nptype):
        dtype = np.dtype(nptype)
        if len(shape) == 1:
            strides = (dtype.itemsize,)
        elif len(shape) == 2:
            strides = (dtype.itemsize * shape[1], dtype.itemsize)
        else:
            print("Allocation only implemented for 1D or 2D shape.")
            return
        return self._shmem_alloc_from_heap(shape, strides, dtype)


class HostContext():
    def init(self, comm, static_heap):
        self.mpi = comm
        self.use_static_heap = static_heap

    def init_static_heap(self):
        self.heap = StaticHeap(HEAP_SIZE, self.mpi.Get_rank())


ctx = HostContext()


def map_peer_arrays(my_dary):
    my_handle = my_dary.get_ipc_handle()
    my_rank = ctx.mpi.Get_rank()

    handle_list = ctx.mpi.allgather(my_handle)
    darys = []
    for i, peer_handle in enumerate(handle_list):
        if i == my_rank:
            darys.append(my_dary)
        else:
            darys.append(peer_handle.open())
    return tuple(darys)


def init(mpi_comm, static_heap=False):
    """
    Collective function that sets the device for the PE
    and initializes a static symmetric heap if
    static_heap=True
    """
    ctx.init(mpi_comm, static_heap)
    rank = mpi_comm.Get_rank()
    cuda.select_device(rank)
    if static_heap:
        ctx.init_static_heap()


def array(npary, copy=True):
    """
    Collective function that allocates device memory like
    the given array. If copy=True, the given array is
    copied to the local device array.

    Returns a tuple containing Numba DeviceNDArrays of all
    peer devices sorted by PE.
    """
    if ctx.use_static_heap:
        darys = ctx.heap.alloc_array(npary, copy)
        ctx.mpi.Barrier()
        return darys
    else:
        if copy:
            my_dary = cuda.to_device(npary)
        else:
            shape, strides, dtype = cuda.prepare_shape_strides_dtype(
                npary.shape, npary.strides, npary.dtype, order='C')
            my_dary = DeviceNDArray(shape=shape, strides=strides,
                                    dtype=dtype, stream=0)
        return map_peer_arrays(my_dary)


def alloc(shape, nptype):
    """
    Collective function that allocates device memory with
    given shape und NumPy type.

    Returns a tuple containing Numba DeviceNDArrays of all
    peer devices sorted by PE.
    """
    if ctx.use_static_heap:
        darys = ctx.heap.alloc(shape, nptype)
        ctx.mpi.Barrier()
        return darys
    else:
        dtype = np.dtype(nptype)
        if len(shape) == 1:
            strides = (dtype.itemsize,)
        elif len(shape) == 2:
            strides = (dtype.itemsize * shape[1], dtype.itemsize)
        else:
            print("Allocation only implemented for 1D or 2D shape.")
            return
        my_dary = DeviceNDArray(shape=shape, strides=strides,
                                dtype=dtype, stream=0)
        return map_peer_arrays(my_dary)


def barrier_all_host():
    """
    Collective function that synchronizes all PEs on the host.
    """
    cuda.synchronize()
    ctx.mpi.Barrier()


def barrier_all_host_on_stream(sync_shmem, stream, npes):
    """
    Collective function that synchronizes all PEs in a GPU kernel on stream

    Params:
    sync_shmem: symmetric allocation of boolean synchronization array
    stream: stream for the barrierer to be put on
    npes: number of PEs
    """
    my_pe = ctx.mpi.Get_rank()
    device.barrier_all_kernel[1, 32, stream](sync_shmem, my_pe, npes)