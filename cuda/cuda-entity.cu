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
#include <math.h>
#include "cuda-entity.h"
#include "cuda-math.h"
#include "cuda-memory.h"
#include "cuda-simulation.h"

__device__ double affinity_potential(unsigned char receptor1, unsigned char receptor2) {
    int dist = hammingdist(receptor1, receptor2);
    if (dist < AFFINITY_MIN)
        return 0.0;
    return pow(BIND_CHANCE, (dist - BITS_IN_A_BYTE) / (AFFINITY_MIN - BITS_IN_A_BYTE));
}

__device__ bool can_entities_bind(Entity* entity, Entity entity2) {
    for (int i = 0; i < RECEPTOR_SIZE; i++) {
        /* Only one pair of receptors needs to bind. */
        if (device_randdouble(&entity->seed) < affinity_potential(entity->receptor[i], entity2.receptor[i]))
            return true;
    }
    return false;
}

__device__ void hypermutation(Entity* entity) {
    // Binomial distribution
    for (int i = 0; i < RECEPTOR_SIZE; i++) {
        for (int j = 0; j < BITS_IN_A_BYTE; j++) {
            if (device_randdouble(&entity->seed) < MUTATION_CHANCE) {
                bool set = !getbit(entity->receptor[i], j);
                setbit(&entity->receptor[i], set, j);
            }
        }
    }
}

__host__ __device__ Entity create_entity(EntityType type, Vector2 position, int seed) {
    Entity entity;
    entity.type = type;
    entity.position = position;
    entity.velocity = vector_zero();
    entity.has_interacted = 1; // begin interactions on the next time step
    entity.has_moved = 1; // being moving on the next time step
    entity.just_created = 1;
    #ifdef __CUDA_ARCH__
        entity.seed = device_rand(&seed); 
        for (int i = 0; i < RECEPTOR_SIZE; i++)
            entity.receptor[i] = device_randbyte(&entity.seed);
    #else
        entity.seed = rand(); 
        for (int i = 0; i < RECEPTOR_SIZE; i++)
            entity.receptor[i] = randbyte();
    #endif
    switch (type) {
        case B_CELL:
            entity.status = CS_INTERNALIZED;
            break;
        default:
            entity.status = CS_ACTIVE;
            break;
    }
    return entity;
}

__host__ __device__ const char* type_to_string(EntityType type) {
    switch (type) {
        case B_CELL:
            return "B Cell";
        case T_CELL:
            return "T Cell";
        case AG_MOLECOLE:
            return "Antigen";
        case AB_MOLECOLE:
            return "Antibody";
        default:
            return "Invalid";
    }
}