#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include <stdbool.h>

#define LAMBDA 0.1
#define TIME_FACTOR 0.2
#define PI 3.14159265358

/* A 2-D Vector that can be used to define a point in a two-dimensional space. */
typedef struct
{
    float x;
    float y;
} 
Vector2;

/* Generates either true or false randomly. */
__host__ __device__ bool randbool();

/* Generates a random floating point value between 0 and 1. */
__host__ __device__ double randdouble();

/* Generates a byte of random bits. */
__host__ __device__ unsigned char randbyte();

/* Computes the Langevin equation with the given parameters. */
__host__ __device__ double langevin(double velocity, double force, double mass);

/* Computes the Hamming distance (the number of different bits) of the bytes passed as parameters. */
__host__ __device__ int hammingdist(unsigned char byte1, unsigned char byte2);

/* Returns a 2-D vector with both X and Y equal to 0. */
__host__ __device__ Vector2 vector_zero();

#endif