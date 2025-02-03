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

#ifndef CUDA_ENTITY_H
#define CUDA_ENTITY_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "cuda-math.h"

#define POISSON_MUTATION

#define RECEPTOR_SIZE 2
#define MUTATION_CHANCE 0.5
#define AFFINITY_MIN 5
#define BIND_CHANCE 0.5
#define AB_CREATED_PER_CELL 4
#define PROXIMITY_DIST 5

/* The type of an entity. */
typedef enum 
{
    // Cells
    B_CELL, // Lymphocyte B
    T_CELL, // Lymphocyte T

    // Molecules
    AG_MOLECOLE, // Antigen
    AB_MOLECOLE, // Antibody

    MAX_ENTITYTYPE
}
EntityType;

/* Defines the current state of the entity. 
   Only used for cell type entities (B_CELL and T_CELL), 
   molecules are always in the active state. */
typedef enum
{
    CS_ACTIVE,
    CS_INTERNALIZED, // Inactive
    CS_STIMULATED, // Duplicating

    MAX_CELLSTATE 
}
CellState;

/* Represents a Cell or Molecule inside our grid. */
typedef struct
{
    EntityType type;
    CellState status;
    Vector2 velocity;
    Vector2 position;
    unsigned char receptor[RECEPTOR_SIZE];

    bool has_interacted;
    bool to_be_removed;
    double seed;
    int lock;
} 
Entity;

/* Calculates the probability for two receptors to bind. */
__device__ double affinity_potential(unsigned char receptor1, unsigned char receptor2);

__device__ double extract_rand(Entity* entity);

/* Determines whether two different entities can bind
   based on the affinity potential of their receptors. */
__device__ bool can_entities_bind(Entity* entity, Entity* entity2);

/* Mutates the receptor of an entity. */
void hypermutation(Entity* entity);

/* Creates a new entity with the type and position specified as parameters. */
Entity* create_entity(EntityType type, Vector2 position);

const char* type_to_string(EntityType type);

#endif