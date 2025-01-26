#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <limits.h>
#include "cuda-memory.h"
#include "cuda-math.h"

__host__ __device__ int randint() {
    return rand();
}

__host__ __device__ bool randbool() {
    return randint() % 2;
}

__host__ __device__ double randdouble() {
    return (double)randint() / (double)RAND_MAX;
}

__host__ __device__ unsigned char randbyte() {
    return randint() % UCHAR_MAX;
}

__host__ __device__ double langevin(double velocity, double force, double mass) {
    return (-LAMBDA * velocity + force) / mass * TIME_FACTOR;
}

__host__ __device__ int hammingdist(unsigned char byte1, unsigned char byte2) {
    int i;
    int dist = 0;
    unsigned char xor = byte1 ^ byte2;
    for (i = 0; i < BITS_IN_A_BYTE; i++)
        if (getbit(xor, i) == true) 
            dist++;
    return dist;
}

__host__ __device__ Vector2 vector_zero() {
    Vector2 v = {
        .x = 0.0,
        .y = 0.0
    };
    return v;
}