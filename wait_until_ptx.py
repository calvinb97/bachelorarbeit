from numba import cuda
from numba.types import int64
import nbshmem

sig = (int64[:], int64, int64)

ptx = cuda.compile_ptx_for_current_device(nbshmem.wait_until_ptxtest4, sig)

print(ptx)