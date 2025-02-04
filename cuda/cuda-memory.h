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

#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

#include <stdio.h>
#include <stdlib.h>

#define BITS_IN_A_BYTE 8

/* Allocates a block of memory of the given size inside the global device memory. */
void cudaAlloc(void** p, size_t size);

/* Copies a block of memory of the given size from the host memory to device memory or viceversa. */
void cudaCopy(void* dest, void* src, size_t size, cudaMemcpyKind type);

/* Allocates a block of memory of the given size. 
   This function will assert if it fails the allocation.
   Returns the pointer to the block of memory just created. */
void* memalloc(size_t size);

/* Frees the block of memory referenced by the pointer passed as parameter.
   This function will assert if the pointer is NULL. */
void memfree(void* p);

/* Returns true if inside the byte the bit at the specified position is equal to 1.
   Returns false otherwise. */
__host__ __device__ bool getbit(unsigned char byte, int position);

/* Sets the bit value of a byte in the specified position. */
__host__ __device__ void setbit(unsigned char* byte, bool value, int position);

/* from https://gist.github.com/ashwin/2652488 */

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );

        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

#endif