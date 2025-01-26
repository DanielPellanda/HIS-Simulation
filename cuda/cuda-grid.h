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

/* Defines an element in a list of entities. */
typedef struct Block
{
   Entity* entity;
   struct Block* next;
}
EntityBlock;

/* A list of entities */
typedef struct 
{
   int size;
   EntityBlock* first;
}
EntityList;

/* A grid containing a list of each type of entity */
typedef struct
{
   int total_size;
   EntityList lists[MAX_ENTITYTYPE];
   Entity* entities[GRID_SIZE][GRID_SIZE];
   int locks[GRID_SIZE][GRID_SIZE];
}
Grid;

/* Gets the entity with the given position from the list. 
   Returns NULL if there are no entities in that position. */
Entity* list_get(EntityList* list, Vector2 position);

/* Inserts an entity inside the list. */
void list_insert(EntityList* list, Entity* entity);

/* Removes an entity in the given position from the list.
   Returns true if an entity was deleted, false otherwise. */
bool list_remove(EntityList* list, Vector2 position);

/* Removes all entities from the list. */
void list_clear(EntityList* list);

/* Removes all entities from the list (stored in device memory). */
void list_clear_device(EntityList* list);

/* Copies into device memory the entity list passed as parameter. */
void list_copy_to_device(EntityList* list, Grid* d_grid, EntityList* d_list);

// void list_copy_to_host(EntityList* d_list, Grid* grid, EntityList* list);

/* Returns a pointer of a copy in device memory of the entity block passed as parameter. */
EntityBlock* block_copy_to_device(EntityBlock* block, Grid* d_grid);

// void block_copy_to_host(Entity* d_block, Grid* grid, Entity* block);

/* Inserts an entity inside the grid. */
void grid_insert(Grid* grid, Entity* entity);

/* Removes an entity in the given position from the grid.
   Returns true if an entity was deleted, false otherwise. */
bool grid_remove(Grid* grid, Vector2 position);

/* Removes an entity of a specific type in the given position from the grid.
   Returns true if an entity was deleted, false otherwise. */
bool grid_remove_type(Grid* grid, Vector2 position, EntityType type);

/* Removes all entities and frees the grid. */
void grid_free(Grid* grid);

/* Removes all entities and frees the grid (stored in device memory). */
void grid_free_device(Grid* grid);

/* Initializes and allocates the grid. */
Grid* grid_init();

/* Returns a pointer of a copy in device memory of the grid passed as parameter. */
Grid* grid_copy_to_device(Grid* grid);

// void grid_copy_to_host(Grid* grid, Grid* d_grid);

/* Gets the entity with the given position from the grid. 
   Returns NULL if there are no entities in that position. */
__host__ __device__ Entity* grid_get(Grid* grid, Vector2 position);

/* Gets the entity of a specific type with the given position from the grid. 
   Returns NULL if there are no entities in that position. */
Entity* grid_get_type(Grid* grid, Vector2 position, EntityType type);

/* Returns true if the given position is not occupied by any entity.
   Returns false if an entity is occupying that position. */
__host__ __device__ bool grid_is_pos_free(Grid* grid, Vector2 position);

/* Sets a device pointer with the address of an entity inside a grid stored in device memory. */
__global__ void set_device_pointer_with_entity(Entity** p, Grid* grid, Vector2 position, EntityType type);





/* Returns an array of entities of a specific type close to the specified position.
   The referenced integer parameter gets set to the length of the array generated. */
Entity** look_for_nearby_entities(Grid* grid, Vector2 position, EntityType type, int* count);

Vector2* find_all_free_nearby_pos(Grid* grid, Vector2 reference, int* count);

/* Gets the nearest free available position within PROXIMITY_DIST distance away
   from the origin position passed as parameter.
   Returns NULL if no free position can be found within range. */
Vector2* find_free_pos_nearby(Grid* grid, Vector2 reference);

/* Returns an array of all available positions within PROXIMITY_DIST distance away
   from the origin position passed as parameter. 
   The referenced integer parameter gets set to the length of the array generated. */
Vector2* find_n_free_nearby_pos(Grid* grid, Vector2 reference, int n, int* count);

/* Clones the specified entity in the nearest free position.
   If no available position can be found, the entity will not get duplicated. */
void duplicate_entity(Grid* grid, Entity* entity);

/* Generates antibodies in the free positions found around the origin position specified.
   The number of antibodies generated is defined by AB_CREATED_PER_CELL. */
void generate_antibodies(Grid* grid, Vector2 origin);

/* Process one movement step of the specified entity inside the grid. */
__device__ void diffuse_entity(Grid* grid, Entity* entity, int blocksize, curandState* rng)

/* Check and process the first possibile interaction for the entity passed as parameter. */
__host__ __device__ void scan_interactions(Grid* grid, Entity* entity);

/* Process interactions for B_CELL type entities. */
__host__ __device__ void b_cell_interact(Grid* grid, Entity* bcell);

/* Looks for nearby compatible entities for the B_CELL passed as parameter. */
__device__ void b_cell_look_for_entity(Grid* grid, Entity* cell);

/* Process interactions for T_CELL type entities. */
__device__ void t_cell_interact(Grid* grid, Entity* tcell);

/* Process interactions for AB_MOLECOLE type entities. */
__device__ void antibody_interact(Grid* grid, Entity* antibody);

/* Checks if the position specified respects the boundaries of the grid. */
__host__ __device__ bool is_pos_valid(Vector2 pos);

/* Checks if the two vectors passed as parameters represent the same exact position. */
__host__ __device__ bool is_matching_pos(Vector2 pos, Vector2 pos2);

/* Corrects the position passed as parameter so that it respects the boundaries of the grid. */
__host__ __device__ void adjust_pos(Vector2* pos);

#endif