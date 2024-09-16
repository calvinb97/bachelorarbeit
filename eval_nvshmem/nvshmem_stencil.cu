#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <nvshmemx.h>
#include <nvshmem.h>

using namespace cooperative_groups;

__global__ void StencilBarrierKernel(double* a, double* a_new, int n, int iterations, double* diffnorm, int top, int bottom, int nx, int y_end) {

    grid_group g = this_grid();
    
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int iter = 0; iter < iterations; iter++) {

        if (ix > 0 && ix < (nx - 1) && iy > 0 && iy < y_end) {
            double new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                     a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
            a_new[iy * nx + ix] = new_val;
            double residue = new_val - a[iy * nx + ix];
            double local_diffnorm = residue * residue;
            atomicAdd(diffnorm + iter, local_diffnorm);

            if (ix == (nx - 2)) {
                a_new[iy * nx]  = new_val;
            }
            if (ix == 1) {
                a_new[iy * nx + (nx - 1)] = new_val;
            }
            
            if (iy == 1) {
                nvshmem_double_p(a_new + y_end * nx + ix, new_val, top);
            }
            if (iy == (y_end - 1)) {
                nvshmem_double_p(a_new + ix, new_val, bottom);
            }
        }   
        g.sync();
        if (ix == 0 && iy == 0) {
            nvshmem_barrier_all();
        }
        g.sync();
        
        double* tmp = a;
        a = a_new;
        a_new = tmp;
    }
}

double* init_array(int n, int init_value) {
    double* x = (double *)malloc(n * n * sizeof(double));
    for (int i = 0; i < (n-1); i++) {
        x[i * n] = init_value;
        x[i * n + (n-1)] = init_value; 
        for (int j = 1; j < (n-1); j++){
            x[i * n + j] = 0;
            if(i == j) {
                x[i * n + j] = init_value;
            }
        }
    }
    for (int j = 0; j < n; j++) {
        x[j] = init_value;
        x[(n-1) * n + j] = init_value;
    }
    return x;
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

    int n = 128;
    int iterations = 2000;

    int chunk_size = (n - 2) / 2;
    
    double* a = (double *)nvshmem_malloc((chunk_size + 2) * n * sizeof(double));
    double* a_new = (double *)nvshmem_malloc((chunk_size + 2) * n * sizeof(double));
    double* diffnorm = (double *)nvshmem_malloc(iterations * sizeof(double));

    dim3 blockDim(8, 8);
    dim3 gridDim(16, 9);

    int top;
    int bottom;
    if (my_pe == 0) {
        top = 1;
        bottom = 1;
    }
    if (my_pe == 1) {
        top = 0;
        bottom = 0;
    }
    
    int nx = n;
    int y_end = chunk_size + 1;

    double* x = init_array(n, 30);
    int my_chunk_start = my_pe * chunk_size * nx;
    int my_chunk_elems = (chunk_size + 2) * nx;



    double* timeResults = (double *)malloc(10 * sizeof(double));
    double start_time, end_time, time;

    if (rank == 0) {
        printf("Starting NVSHMEM Stencil-Kernel with %d iterations\n", iterations);
    }
    
    for (int i = 0; i < 11; i++) {
        cudaMemcpy(a, x + my_chunk_start, my_chunk_elems * sizeof(double), cudaMemcpyHostToDevice);
        a_new = (double *)nvshmem_malloc((chunk_size + 2) * n * sizeof(double));
        diffnorm = (double *)nvshmem_malloc(iterations * sizeof(double));

        void *args[] = {&a, &a_new, &n, &iterations, &diffnorm, &top, &bottom, &nx, &y_end};
        
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        nvshmemx_collective_launch((const void *)StencilBarrierKernel, gridDim, blockDim, args, 0, 0);
        cudaDeviceSynchronize();
        end_time = MPI_Wtime();
        time = end_time - start_time;
        timeResults[i] = time;
        if (i == 0) {
            printf("Iteration %d: %f on process %d\n", i, time, rank);
        }
    }

    double mean = 0;
    double stddev = 0;
    calcMetrics(timeResults, rank, &mean, &stddev);
    if (rank == 0) {
        printf("Stencil mean: %f\n", mean);
        printf("Stencil std: %f\n", stddev);
    }
    
  
    double* h_norm = (double *)malloc(iterations * sizeof(double));
    cudaMemcpy(h_norm, diffnorm, iterations * sizeof(double), cudaMemcpyDeviceToHost);

    double* reduced_norm = (double *)malloc(iterations * sizeof(double));
    MPI_Reduce(h_norm, reduced_norm, iterations, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // print to assert correct norm results
    // if (rank == 0) {
    //     for (int i = 0; i < iterations; i++) {
    //         if (i % 50 == 0) {
    //             printf("%f \n", sqrt(reduced_norm[i]));
    //         }
    //     }
    // }
    
    nvshmem_finalize();
    MPI_Finalize();
}
