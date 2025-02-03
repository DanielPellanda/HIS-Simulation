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
#include "entity.h"
#include "memory.h"

double affinity_potential(unsigned char receptor1, unsigned char receptor2) {
    int dist = hammingdist(receptor1, receptor2);
    if (dist < AFFINITY_MIN)
        return 0.0;
    return pow(BIND_CHANCE, (dist - BITS_IN_A_BYTE) / (AFFINITY_MIN - BITS_IN_A_BYTE));
}

bool can_entities_bind(Entity* entity, Entity* entity2) {
    for (int i = 0; i < RECEPTOR_SIZE; i++) {
        /* Only one pair of receptors needs to bind. */
        if (randdouble() < affinity_potential(entity->receptor[i], entity2->receptor[i]))
            return true;
    }
    return false;
}

void hypermutation(Entity* entity) {
    #ifdef POISSON_MUTATION
        // Poisson distribution

        /* Choose how many and which bits are going to get changed
           and store the indexes in an array. */
        int num_bits = rand() % (BITS_IN_A_BYTE * RECEPTOR_SIZE); // extract bits to change
        int* positions = (int*)memalloc(num_bits * sizeof(int));
        for (int i = 0; i < num_bits; i++) {
            positions[i] = rand() % (BITS_IN_A_BYTE * RECEPTOR_SIZE); // extract index to change
        }

        /* Invert the value of every bit in the position of each index contained in the array. */
        for (int i = 0; i < num_bits; i++) {
            int pos = positions[i] % BITS_IN_A_BYTE;
            int index = positions[i] / BITS_IN_A_BYTE;
            bool set = !getbit(entity->receptor[index], pos);
            setbit(&entity->receptor[index], set, pos);
        }
        memfree(positions);
    #else
        // Binomial distribution
        for (int i = 0; i < RECEPTOR_SIZE; i++) {
            for (int j = 0; j < BITS_IN_A_BYTE; j++) {
                double random = randdouble();
                if (random < MUTATION_CHANCE) {
                    bool set = !getbit(entity->receptor[i], j);
                    setbit(&entity->receptor[i], set, j);
                }
            }
        }
    #endif
}

Entity* create_entity(EntityType type, Vector2 position) {
    Entity* entity = (Entity*)memalloc(sizeof(Entity));
    entity->type = type;
    entity->position = position;
    entity->velocity = vector_zero();
    entity->has_interacted = true; // begin interactions on the next time step
    entity->to_be_removed = false;
    for (int i = 0; i < RECEPTOR_SIZE; i++)
        entity->receptor[i] = randbyte();
    switch (type) {
        case B_CELL:
            entity->status = CS_INTERNALIZED;
            break;
        default:
            entity->status = CS_ACTIVE;
            break;
    }
    return entity;
}

char* type_to_string(EntityType type) {
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
    return "Invalid";
}