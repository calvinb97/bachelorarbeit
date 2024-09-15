extern "C" __device__ int wait_until_bool_volatile(int* return_value, volatile bool* array, long idx, bool val) {
    while(array[idx] != val) {
        continue;
    }
    return 0;
}

extern "C" __device__ int wait_until_long_volatile(int* return_value, volatile long* array, long idx, long val) {
    while(array[idx] != val) {
        continue;
    }
    return 0;
}