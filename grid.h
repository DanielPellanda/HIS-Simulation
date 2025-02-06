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

#ifndef GRID_H
#define GRID_H

#include "entity.h"

#define DEFAULT_GRID_SIZE 500

extern int GRID_SIZE;

/* A grid containing all entities of the system. */
typedef struct
{
   Entity** entities;
   int total_size;
   int size[MAX_ENTITYTYPE];
}
Grid;

/* Inserts an entity inside the grid. */
void grid_insert(Grid* grid, Entity entity);

/* Removes an entity in the given position from the grid.
   Returns true if an entity was deleted, false otherwise. */
bool grid_remove(Grid* grid, Vector2 position);

/* Removes all entities and frees the grid. */
void grid_free(Grid* grid);

/* Initializes and allocates the grid. */
Grid* grid_init();

/* Gets the entity with the given position from the grid. 
   Returns NULL if there are no entities in that position. */
Entity grid_get(Grid* grid, Vector2 position);

/* Gets an array with all the entities of the grid. */
Entity** grid_get_all(Grid* grid);

/* Returns true if the given position is not occupied by any entity.
   Returns false if an entity is occupying that position. */
bool grid_is_pos_free(Grid* grid, Vector2 position);




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
void duplicate_entity(Grid* grid, Entity entity);

/* Generates antibodies in the free positions found around the origin position specified.
   The number of antibodies generated is defined by AB_CREATED_PER_CELL. */
void generate_antibodies(Grid* grid, Entity cell);

/* Process one movement step of the specified entity inside the grid. */
void diffuse_entity(Grid* grid, Entity* entity);

/* Check and process the first possibile interaction for the entity passed as parameter. */
void process_interactions(Grid* grid, Entity* entity);

/* Process interactions for B_CELL type entities. */
void b_cell_interact(Grid* grid, Entity* bcell);

/* Process interactions for T_CELL type entities. */
// void t_cell_interact(Grid* grid, Entity* tcell);

/* Process interactions for AB_MOLECOLE type entities. */
void antibody_interact(Grid* grid, Entity* antibody);

/* Checks if the position specified respects the boundaries of the grid. */
bool is_pos_valid(Vector2 pos);

/* Checks if the two vectors passed as parameters represent the same exact position. */
bool is_matching_pos(Vector2 pos, Vector2 pos2);

/* Corrects the position passed as parameter so that it respects the boundaries of the grid. */
void adjust_pos(Vector2* pos);

#endif