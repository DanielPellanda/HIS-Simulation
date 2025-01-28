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

#ifndef MEMORY_H
#define MEMORY_H

#include <stdbool.h>
#include <stddef.h>

#define BITS_IN_A_BYTE 8

/* Allocates a block of memory of the given size. 
   This function will assert if it fails the allocation.
   Returns the pointer to the block of memory just created. */
void* memalloc(size_t size);

/* Frees the block of memory referenced by the pointer passed as parameter.
   This function will assert if the pointer is NULL. */
void memfree(void* p);

#ifdef OPEN_MP
   /* Tries to acquire the lock by setting the "lock" pointer passed as reference.
      If the lock is successful the pointer gets set to "new" and returns his old value.
      If the lock is unsuccessful the pointer remains of the same value and returns it. */
   int trylock(int* lock, int old, int new);
#endif

/* Returns true if inside the byte the bit at the specified position is equal to 1.
   Returns false otherwise. */
bool getbit(unsigned char byte, int position);

/* Sets the bit value of a byte in the specified position. */
void setbit(unsigned char* byte, bool value, int position);

#endif