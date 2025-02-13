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
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include "cuda-memory.h"

void cudaAlloc(void** p, size_t size) {
    cudaSafeCall(cudaMalloc(p, size));
}

void cudaCopy(void* dest, void* src, size_t size, cudaMemcpyKind type) {
    cudaSafeCall(cudaMemcpy(dest, src, size, type));
}

void* memalloc(size_t size) {
    void* p = malloc(size);
    assert(p != NULL);
    return p;
}

void memfree(void* p) {
    assert(p != NULL);
    free(p);
}

__host__ __device__ bool getbit(unsigned char byte, int position) {
    if (position < 0 || position > BITS_IN_A_BYTE-1)
        return false;
    unsigned char offset = 1;
    for (int i = 1; i <= position; i++)
        offset *= 2;
    unsigned char bit = byte & offset;
    return bit;
}

__host__ __device__ void setbit(unsigned char* byte, bool value, int position) {
    if (position < 0 || position > BITS_IN_A_BYTE-1)
        return;
    unsigned char offset = 1;
    for (int i = 1; i <= position; i++)
        offset *= 2;
    if (value)
        *byte |= offset;
    else
        *byte &= ~offset;
}