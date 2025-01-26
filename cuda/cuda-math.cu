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