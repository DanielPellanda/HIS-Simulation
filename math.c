#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <limits.h>
#include "memory.h"
#include "math.h"

bool randbool() {
    return rand() % 2;
}

double randdouble() {
    return (double)rand() / (double)RAND_MAX;
}

unsigned char randbyte() {
    return rand() % UCHAR_MAX;
}

double langevin(double velocity, double force, double mass) {
    return (-LAMBDA * velocity + force) / mass * TIME_FACTOR;
}

int hammingdist(unsigned char byte1, unsigned char byte2) {
    int i;
    int dist = 0;
    unsigned char xor = byte1 ^ byte2;
    for (i = 0; i < BITS_IN_A_BYTE; i++)
        if (getbit(xor, i) == true) 
            dist++;
    return dist;
}

Vector2 vector_zero() {
    Vector2 v = {
        .x = 0.0,
        .y = 0.0
    };
    return v;
}