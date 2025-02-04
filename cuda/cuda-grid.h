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

#ifndef CUDA_GRID_H
#define CUDA_GRID_H

#include "cuda-entity.h"
#include "cuda-math.h"

#define GRID_SIZE 1000

/* A grid containing all entities of the system. */
typedef struct
{
   Entity entities[GRID_SIZE*GRID_SIZE];
   // int size[MAX_ENTITYTYPE];
   // int total_size;
   int seed;
}
Grid;

/* Inserts an entity inside the grid. */
void grid_insert(Grid* grid, Entity entity);

/* Removes an entity in the given position from the grid.
   Returns true if an entity was deleted, false otherwise. */
bool grid_remove(Grid* grid, Vector2 position);

__host__ __device__ void grid_check(Grid* grid);

/* Removes all entities and frees the grid. */
void grid_free(Grid* grid);

/* Initializes and allocates the grid. */
Grid* grid_init();

/* Gets the entity with the given position from the grid. 
   Returns NULL if there are no entities in that position. */
__host__ __device__ Entity grid_get(Grid* grid, Vector2 position);

/* Returns true if the given position is not occupied by any entity.
   Returns false if an entity is occupying that position. */
__host__ __device__ bool grid_is_pos_free(Grid* grid, Vector2 position);



/* Clones the specified entity in the nearest free position.
   If no available position can be found, the entity will not get duplicated. */
__device__ void duplicate_entity(Grid* grid, Entity entity);

/* Generates antibodies in the free positions found around the origin position specified.
   The number of antibodies generated is defined by AB_CREATED_PER_CELL. */
__device__ void generate_antibodies(Grid* grid, Entity cell);

/* Process one movement step of the specified entity inside the grid. */
__device__ void diffuse_entity(Grid* grid, Entity* entity);

// __device__ void diffuse_entity(Grid* grid, Entity* entity, curandState* rng);

/* Check and process the first possibile interaction for the entity passed as parameter. */
__device__ void process_interactions(Grid* grid, Entity* entity);

/* Process interactions for B_CELL type entities. */
__device__ void b_cell_interact(Grid* grid, Entity* bcell);

/* Process interactions for AB_MOLECOLE type entities. */
__device__ void antibody_interact(Grid* grid, Entity* antibody);

/* Checks if the position specified respects the boundaries of the grid. */
__host__ __device__ bool is_pos_valid(Vector2 pos);

/* Checks if the two vectors passed as parameters represent the same exact position. */
__host__ __device__ bool is_matching_pos(Vector2 pos, Vector2 pos2);

/* Corrects the position passed as parameter so that it respects the boundaries of the grid. */
__host__ __device__ void adjust_pos(Vector2* pos);

#endif