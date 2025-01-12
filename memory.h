#include <stdbool.h>
#include <stddef.h>

#define EXE_MODE 0
#define BITS_IN_A_BYTE 8

/* Allocates a block of memory of the given size. 
   This function will assert if it fails the allocation.
   Returns the pointer to the block of memory just created. */
void* memalloc(size_t size);

/* Frees the block of memory referenced by the pointer passed as parameter.
   This function will assert if the pointer is NULL. */
void memfree(void* p);

/* Returns true if inside the byte the bit at the specified position is equal to 1.
   Returns false otherwise. */
bool getbit(unsigned char byte, int position);

/* Sets the bit value of a byte in the specified position. */
void setbit(unsigned char* byte, bool value, int position);