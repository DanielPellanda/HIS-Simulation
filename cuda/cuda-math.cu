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

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <limits.h>
#include "cuda-math.h"
#include "cuda-memory.h"

__device__ int device_rand(int* seed) {
    long x = *seed;
    x ^= (x << 21);
    x ^= (x >> 35);
    x ^= (x << 4);
    if (x < 0)
        x *= -1;
    int finalnum = (int)(x % RAND_MAX);
    *seed = finalnum;
    return finalnum;
}

__device__ bool device_randbool(int* seed) {
    return device_rand(seed) % 2;
}

__device__ double device_randdouble(int* seed) {
    return (double)device_rand(seed) / (double)RAND_MAX;
}

__device__ unsigned char device_randbyte(int* seed) {
    return device_rand(seed) % (UCHAR_MAX+1);
}

bool randbool() {
    return rand() % 2;
}

double randdouble() {
    return (double)rand() / (double)RAND_MAX;
}

unsigned char randbyte() {
    return rand() % (UCHAR_MAX+1);
}

__host__ __device__ double langevin(double velocity, double force, double mass) {
    return (-LAMBDA * velocity + force) / mass * TIME_FACTOR;
}

__host__ __device__ int hammingdist(unsigned char byte1, unsigned char byte2) {
    int dist = 0;
    unsigned char xorbyte = byte1 ^ byte2;
    for (int i = 0; i < BITS_IN_A_BYTE; i++)
        if (getbit(xorbyte, i) == true) 
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