/****************************************************************************
*    
*    HIS Simulator in C/Cuda C++
*
*    Copyright (C) 2025  Daniel Pellanda
*
*    This program is free software: you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    This program is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with this program.  If not, see <https://www.gnu.org/licenses/>
*
****************************************************************************/

#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include <stdbool.h>

#define LAMBDA 0.1
#define TIME_FACTOR 1.0
#define PI 3.14159265358

/* A 2-D Vector that can be used to define a point in a two-dimensional space. */
typedef struct
{
    float x;
    float y;
} 
Vector2;

/* Generates a random integer from a seed. */
__device__ int device_rand(int* seed);

/* Generates either true or false randomly. */
__device__ bool device_randbool(int* seed);

/* Generates a random floating point value between 0 and 1. */
__device__ double device_randdouble(int* seed);

/* Generates a byte of random bits. */
__device__ unsigned char device_randbyte(int* seed);

/* Generates either true or false randomly. */
bool randbool();

/* Generates a random floating point value between 0 and 1. */
double randdouble();

/* Generates a byte of random bits. */
unsigned char randbyte();

/* Computes the Langevin equation with the given parameters. */
__host__ __device__ double langevin(double velocity, double force, double mass);

/* Computes the Hamming distance (the number of different bits) of the bytes passed as parameters. */
__host__ __device__ int hammingdist(unsigned char byte1, unsigned char byte2);

/* Returns a 2-D vector with both X and Y equal to 0. */
__host__ __device__ Vector2 vector_zero();

#endif