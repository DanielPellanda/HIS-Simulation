#include "cuda-memory.h"

void cudaAlloc(void** p, size_t size) {
    cudaSafeCall(cudaMalloc(p, size));
}

void cudaCopy(void* dest, void* src, size_t size, cudaMemcpyKind type) {
    cudaSafeCall(cudaMemcpy(dest, src, size, type));
}