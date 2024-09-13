#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nvshmemx.h>
#include <nvshmem.h>

__global__ void ReductionKernel(int* array, int* result, int nelems) {
    if (blockIdx.x == 0) {
        nvshmemx_int_sum_reduce_block(NVSHMEM_TEAM_WORLD, result, array, nelems);
    }
}

__global__ void BroadcastKernel(int* array, int* dest, int nelems) {
    if (blockIdx.x == 0) {
        nvshmemx_int_broadcast_block(NVSHMEM_TEAM_WORLD, dest, array, nelems, 0);
    }
}

__global__ void AlltoAllKernel(int* array, int* dest, int nelems) {
    if (blockIdx.x == 0) {
        nvshmemx_int_alltoall_block(NVSHMEM_TEAM_WORLD, dest, array, nelems/2);
    }
}



int* initHostArray(int size) {
    int* array = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        array[i] = i;
    }
    return array;
}

int* initAlltoAllArray(int size, int rank) {
    int* array = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        if (rank == 0) {
            array[i] = i;
        } else {
            array[i] = 10 * i;
        }
    }
    return array;
}

int* allToAllResult(int size, int rank) {
    int* result = (int *)malloc(size * sizeof(int));
    if (rank == 0) {
        for (int i = 0; i < size/2; i++) {
            result[i] = i;
            result[i + size/2] = 10 * i;
        }
    } else {
        for (int i = 0; i < size/2; i++) {
            result[i] = i + size/2;
            result[i + size/2] = 10 * (i + size/2);
        }
    }
    return result;
}

int* reductionResult(int* array, int size) {
    int* result = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        result[i] = 2 * array[i];
    }
    return result;
}

bool assertEquals(int* array1, int* array2, int size) {
    for (int i = 0; i < size; i++) {
        if (array1[i] != array2[i]) {
            return false;
        }
    }
    return true;
}

void calcMetrics(double* timeResults, int rank, double* meanResult, double* stddevResult) {
    double* gatheredTimes = (double *)malloc(20 * sizeof(double));
    MPI_Gather(timeResults + 1, 10, MPI_DOUBLE, gatheredTimes, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double timeSum = 0;
        for (int i = 0; i < 20; i++) {
            timeSum += gatheredTimes[i]; 
        }
        double mean = timeSum / 20;
    
        double varianceSum = 0;
        for (int i = 0; i < 20; i++) {
            double tmp = gatheredTimes[i] - mean;
            tmp = tmp * tmp;
            varianceSum += tmp;
        }
        double variance = varianceSum/19;
        double stddev = sqrt(variance);

        *meanResult = mean;
        *stddevResult = stddev;
    }
}

void evalMaxCoopBlocks(const void* kernel, void** args) {
    const int blockdims[] = {32, 64, 256, 512, 544, 1024};
    for (int i = 0; i < 6; i++) {
        dim3 dimBlock(blockdims[i]);
        int gridsize = 0;
        nvshmemx_collective_launch_query_gridsize(kernel, dimBlock, args, 0, &gridsize);
        int threads = gridsize * blockdims[i];
        printf("blockdim=%d, maxblocks= %d, threads=%d\n", blockdims[i], gridsize, threads);
    }
}


void evalHostReduction(int* result, int* array, int* hostResult, int* hostCalculatedResult,
    	int ary_size, double* timeResults, int rank) {
    double start_time, end_time, time;
    for (int i = 0; i < 11; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        nvshmem_int_sum_reduce(NVSHMEM_TEAM_WORLD, result, array, ary_size);
        end_time = MPI_Wtime();
        time = end_time - start_time;
        timeResults[i] = time;
        if (i == 0) {
            printf("Iteration %d: %f on process %d\n", i, time, rank);
        }

        // assert correct result
        cudaMemcpy(hostResult, result, ary_size * sizeof(int), cudaMemcpyDeviceToHost);
        bool assertion = assertEquals(hostCalculatedResult, hostResult, ary_size);    
        if (!assertion) {
            printf("Assertion failed.\n");
        }
    }
    double mean = 0;
    double stddev = 0;
    calcMetrics(timeResults, rank, &mean, &stddev);
    if (rank == 0) {
        printf("Host Reduction mean: %f\n", mean);
        printf("Host Reduction stddev: %f\n", stddev);
    }
}

void evalBlockReduction(int* result, int* array, int* hostResult, int* hostCalculatedResult,
    	int ary_size, double* timeResults, int rank) {
    double start_time, end_time, time;
    void *args[] = {&array, &result, &ary_size};

    // using one block with max blockdim of 256 threads
    dim3 dimBlock(256);
    dim3 dimGrid(1);
    
    for (int i = 0; i < 11; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        nvshmemx_collective_launch((const void *)ReductionKernel, dimGrid, dimBlock, args, 0, 0);
        cudaDeviceSynchronize();
        end_time = MPI_Wtime();
        time = end_time - start_time;
        timeResults[i] = time;
        if (i == 0) {
            printf("Iteration %d: %f on process %d\n", i, time, rank);
        }

        // assert correct result
        cudaMemcpy(hostResult, result, ary_size * sizeof(int), cudaMemcpyDeviceToHost);
        bool assertion = assertEquals(hostCalculatedResult, hostResult, ary_size);    
        if (!assertion) {
            printf("Assertion failed.\n");
        }
    }
    double mean = 0;
    double stddev = 0;
    calcMetrics(timeResults, rank, &mean, &stddev);
    if (rank == 0) {
        printf("Block Reduction mean: %f\n", mean);
        printf("Block Reduction stddev: %f\n", stddev);
    }
}

void evalHostBroadcast(int* dest, int* array, int* hostArray, int* hostDestArray, int ary_size, double* timeResults, int rank) {
    double start_time, end_time, time;
    for (int i = 0; i < 11; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        nvshmem_int_broadcast(NVSHMEM_TEAM_WORLD, dest, array, ary_size, 0);
        end_time = MPI_Wtime();
        time = end_time - start_time;
        timeResults[i] = time;
        if (i == 0) {
            printf("Iteration %d: %f on process %d\n", i, time, rank);
        }

        // assert correct result
        cudaMemcpy(hostDestArray, dest, ary_size * sizeof(int), cudaMemcpyDeviceToHost);
        bool assertion = assertEquals(hostArray, hostDestArray, ary_size);    
        if (!assertion) {
            printf("Assertion failed.\n");
        }
    }
    double mean = 0;
    double stddev = 0;
    calcMetrics(timeResults, rank, &mean, &stddev);
    if (rank == 0) {
        printf("Host Broadcast mean: %f\n", mean);
        printf("Host Broadcast stddev: %f\n", stddev);
    }
}

void evalBlockBroadcast(int* dest, int* array, int* hostArray, int* hostDestArray, int ary_size, double* timeResults, int rank) {
    double start_time, end_time, time;
    void *args[] = {&array, &dest, &ary_size};

    // if (rank == 0) {
    //     evalMaxCoopBlocks((const void*) BroadcastKernel, args);
    // }

    
    // using one block with max blockdim of 512 threads
    dim3 dimBlock(512);
    dim3 dimGrid(1);
    
    for (int i = 0; i < 11; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        nvshmemx_collective_launch((const void *)BroadcastKernel, dimGrid, dimBlock, args, 0, 0);
        cudaDeviceSynchronize();
        end_time = MPI_Wtime();
        time = end_time - start_time;
        timeResults[i] = time;
        if (i == 0) {
            printf("Iteration %d: %f on process %d\n", i, time, rank);
        }

        // assert correct result
        cudaMemcpy(hostDestArray, dest, ary_size * sizeof(int), cudaMemcpyDeviceToHost);
        bool assertion = assertEquals(hostArray, hostDestArray, ary_size);    
        if (!assertion) {
            printf("Assertion failed.\n");
        } 
    }
    double mean = 0;
    double stddev = 0;
    calcMetrics(timeResults, rank, &mean, &stddev);
    if (rank == 0) {
        printf("Block Broadcast mean: %f\n", mean);
        printf("Block Broadcast stddev: %f\n", stddev);
    }
}

void evalHostAlltoAll(int* dest, int* array, int* hostCalculatedResult, int* hostDestArray, int ary_size, double* timeResults, int rank) {
    double start_time, end_time, time;
    for (int i = 0; i < 11; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        nvshmem_int_alltoall(NVSHMEM_TEAM_WORLD, dest, array, ary_size/2);
        end_time = MPI_Wtime();
        time = end_time - start_time;
        timeResults[i] = time;
        if (i == 0 || i == 1) {
            printf("Iteration %d: %f on process %d\n", i, time, rank);
        }

        // assert correct result
        cudaMemcpy(hostDestArray, dest, ary_size * sizeof(int), cudaMemcpyDeviceToHost);
        bool assertion = assertEquals(hostCalculatedResult, hostDestArray, ary_size);    
        if (!assertion) {
            printf("Assertion failed.\n");
        }
    }
    double mean = 0;
    double stddev = 0;
    calcMetrics(timeResults, rank, &mean, &stddev);
    if (rank == 0) {
        printf("Host AlltoAll mean: %f\n", mean);
        printf("Host AlltoAll stddev: %f\n", stddev);
    }
}

void evalBlockAlltoAll(int* dest, int* array, int* hostCalculatedResult, int* hostDestArray, int ary_size, double* timeResults, int rank) {
    double start_time, end_time, time;
    void *args[] = {&array, &dest, &ary_size};

    // if (rank == 0) {
    //     evalMaxCoopBlocks((const void*) AlltoAllKernel, args);
    // }

    // using one block with max blockdim of 512 threads
    dim3 dimBlock(512);
    dim3 dimGrid(1);
    
    for (int i = 0; i < 11; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        nvshmemx_collective_launch((const void *)AlltoAllKernel, dimGrid, dimBlock, args, 0, 0);
        cudaDeviceSynchronize();
        end_time = MPI_Wtime();
        time = end_time - start_time;
        timeResults[i] = time;
        if (i == 0) {
            printf("Iteration %d: %f on process %d\n", i, time, rank);
        }

        // assert correct result
        cudaMemcpy(hostDestArray, dest, ary_size * sizeof(int), cudaMemcpyDeviceToHost);
        bool assertion = assertEquals(hostCalculatedResult, hostDestArray, ary_size);    
        if (!assertion) {
            printf("Assertion failed.\n");
        } 
    }
    double mean = 0;
    double stddev = 0;
    calcMetrics(timeResults, rank, &mean, &stddev);
    if (rank == 0) {
        printf("Block AlltoAll mean: %f\n", mean);
        printf("Block AlltoAll stddev: %f\n", stddev);
    }
}

void evalReduction(int ary_size, int rank) {
    // host allocations
    int* hostArray = initHostArray(ary_size);
    int* hostCalculatedResult = reductionResult(hostArray, ary_size);
    int* hostResult = (int *)malloc(ary_size * sizeof(int));
    double* timeResults = (double *)malloc(10 * sizeof(double));
    
    // nvshmem allocations
    int* array = (int *)nvshmem_malloc(ary_size * sizeof(int));
    int* result = (int *)nvshmem_malloc(ary_size * sizeof(int));

    // copy array to GPU
    cudaMemcpy(array, hostArray, ary_size * sizeof(int), cudaMemcpyHostToDevice);
    
    evalHostReduction(result, array, hostResult, hostCalculatedResult, ary_size, timeResults, rank);
    evalBlockReduction(result, array, hostResult, hostCalculatedResult, ary_size, timeResults, rank);
    
    free(hostArray);
    free(hostCalculatedResult);
    free(hostResult);
    free(timeResults);
    nvshmem_free(array);
    nvshmem_free(result);    
}


void evalBroadcast(int ary_size, int rank) {
    // host allocations
    int* hostArray = initHostArray(ary_size);
    int* hostDestArray = (int *)malloc(ary_size * sizeof(int));
    double* timeResults = (double *)malloc(10 * sizeof(double));
    
    // nvshmem allocations
    int* array = (int *)nvshmem_malloc(ary_size * sizeof(int));
    int* dest = (int *)nvshmem_malloc(ary_size * sizeof(int));

    // copy array to GPU
    cudaMemcpy(array, hostArray, ary_size * sizeof(int), cudaMemcpyHostToDevice);
    
    evalHostBroadcast(dest, array, hostArray, hostDestArray, ary_size, timeResults, rank);
    evalBlockBroadcast(dest, array, hostArray, hostDestArray, ary_size, timeResults, rank);
    
    free(hostArray);
    free(hostDestArray);
    free(timeResults);
    nvshmem_free(array);
    nvshmem_free(dest);
}

void evalAlltoAll(int ary_size, int rank) {
    // host allocations
    int* hostArray = initAlltoAllArray(ary_size, rank);
    int* hostCalculatedResult = allToAllResult(ary_size, rank);
    int* hostDestArray = (int *)malloc(ary_size * sizeof(int));
    double* timeResults = (double *)malloc(10 * sizeof(double));
    
    // nvshmem allocations
    int* array = (int *)nvshmem_malloc(ary_size * sizeof(int));
    int* dest = (int *)nvshmem_malloc(ary_size * sizeof(int));

    // copy array to GPU
    cudaMemcpy(array, hostArray, ary_size * sizeof(int), cudaMemcpyHostToDevice);
    
    evalHostAlltoAll(dest, array, hostCalculatedResult, hostDestArray, ary_size, timeResults, rank);
    evalBlockAlltoAll(dest, array, hostCalculatedResult, hostDestArray, ary_size, timeResults, rank);
    
    free(hostArray);
    free(hostCalculatedResult);
    free(hostDestArray);
    free(timeResults);
    nvshmem_free(array);
    nvshmem_free(dest);
}

int main(int argc, char* argv[]) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaSetDevice(rank);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    int my_pe = nvshmem_my_pe();

    int ary_size = 16384 * 10000;
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\nReduction for array size %d bytes:\n", ary_size * sizeof(int));
    }
    
    evalReduction(ary_size, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\nBroadcast for array size %d bytes:\n", ary_size * sizeof(int));
    }
    
    evalBroadcast(ary_size, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\nAlltoAll for array size %d bytes:\n", ary_size * sizeof(int));
    }
    
    evalAlltoAll(ary_size, rank);


    nvshmem_finalize();
    MPI_Finalize();
}