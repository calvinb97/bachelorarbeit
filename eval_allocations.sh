export NUMBA_CUDA_USE_NVIDIA_BINDING=1

mpirun -n 2 python eval_allocations.py static alloc
mpirun -n 2 python eval_allocations.py static array
mpirun -n 2 python eval_allocations.py static array --copy

mpirun -n 2 python eval_allocations.py dynamic alloc
mpirun -n 2 python eval_allocations.py dynamic array
mpirun -n 2 python eval_allocations.py dynamic array --copy