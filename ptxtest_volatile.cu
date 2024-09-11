__device__ int wait_until_cu_volatile(long* array, long val) {
    while(*((volatile long *)array) != val) {
        continue;
    }
    return 0;
}

__global__ void Kernel(long* array, long val) {
    wait_until_cu_volatile(array, val);
}
