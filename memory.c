#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include "memory.h"

void* memalloc(size_t size) {
    void* p = malloc(size);
    assert(p != NULL);
    return p;
}

void memfree(void* p) {
    assert(p != NULL);
    free(p);
}

bool getbit(unsigned char byte, int position) {
    if (position < 0 || position > BITS_IN_A_BYTE-1)
        return false;
    unsigned char offset = 1;
    for (int i = 1; i <= position; i++)
        offset *= 2;
    unsigned char bit = byte & offset;
    return bit;
}

void setbit(unsigned char* byte, bool value, int position) {
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

