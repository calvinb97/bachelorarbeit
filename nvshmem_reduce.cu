#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nvshmemx.h>
#include <nvshmem.h>


__global__ void InitKernel(int* array) {
    int idx = blockIdx .x * blockDim .x + threadIdx .x;
    if (idx < 10) {
        array[idx] = idx;
    }
}

__global__ void ReductionKernel(int* array, int* result) {
    nvshmemx_int_sum_reduce_block(NVSHMEM_TEAM_WORLD, result, array, 10);
}

int main(int argc, char* argv[]) {
    int rank;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("My Rank is: %d\n", rank);
    
    cudaSetDevice(rank);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int my_pe = nvshmem_my_pe();
    printf("My PE is: %d\n", my_pe);

    int* array = (int *)nvshmem_malloc(10 * sizeof(int));
    int* result = (int *)nvshmem_malloc(10 * sizeof(int));
    int* host_result = (int *)malloc(10 * sizeof(int));

    InitKernel<<<1, 10>>>(array);
    dim3 dimBlock(10);
    dim3 dimGrid(1);
    void *args[] = {&array, &result};
    nvshmemx_collective_launch((const void *)ReductionKernel, dimGrid, dimBlock, args, 0, 0);
    cudaMemcpy(host_result, result, 10 * sizeof(int), cudaMemcpyDeviceToHost);

    if (rank == 0) {
        printf("Ergebnis:\n");
        for (int i = 0; i < 10; i++) {
            int res = host_result[i];
            printf("%d\n", res);
        }        
    }

    nvshmem_finalize();
    MPI_Finalize();
}