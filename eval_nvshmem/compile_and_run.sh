nvcc -rdc=true -ccbin g++ -gencode=$NVCC_GENCODE -I $NVSHMEM_HOME/include nvshmem_collectives.cu -o nvshmem_collectives -L $NVSHMEM_HOME/lib -lnvshmem -lnvidia-ml -lcuda -lcudart -lmpi

mpirun -n 2 nvshmem_collectives