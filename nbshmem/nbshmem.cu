extern "C" __device__ int wait_until_cu_volatile(int* return_value, long* array, long val) {
    while(*((volatile long *)array) != val) {
        continue;
    }
    return_value[0] = 1;
    return 0;
}

extern "C" __device__ int wait_until_cu_fence(int* return_value, long* array, long val) {
    while(*array != val) {
        __threadfence();
    }
    return_value[0] = 1;
    return 0;
}
