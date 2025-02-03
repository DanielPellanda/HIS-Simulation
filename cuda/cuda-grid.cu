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
#include <assert.h>
#include <math.h>
#include "cuda-grid.h"
#include "cuda-memory.h"
#include "cuda-simulation.h"

Entity* list_get(EntityList* list, Vector2 position) {
    EntityBlock** current = &list->first;
    while (*current != NULL) {
        if (is_matching_pos((*current)->entity->position, position)) 
            return (*current)->entity;
        current = &(*current)->next;
    }
    return NULL;
}

void list_insert(EntityList* list, Entity* entity) {
    EntityBlock* last = NULL;
    EntityBlock** current = &list->first;
    while (*current != NULL) {
        last = *current;
        current = &(*current)->next;
    }

    EntityBlock* new_block = (EntityBlock*)memalloc(sizeof(EntityBlock));
    new_block->entity = entity;
    new_block->next = NULL;
    if (list->first == NULL) {
        assert(last == NULL);
        list->first = new_block;
    }
    else {
        assert(last != NULL);
        last->next = new_block;
    }
    list->size++;
}

bool list_remove(EntityList* list, Vector2 position) {
    EntityBlock** current = &list->first;
    while (*current != NULL) {
        if (is_matching_pos((*current)->entity->position, position)) {
            EntityBlock* temp = *current;
            *current = (*current)->next;
            memfree(temp->entity);
            memfree(temp);
            list->size--;
            return true;
        }
        current = &(*current)->next;
    }
    return false;
}

void list_clear(EntityList* list) {
    EntityBlock* current = list->first;
    while (current != NULL) {
        EntityBlock* next = current->next;
        memfree(current->entity);
        memfree(current);
        current = next;
    }
    list->first = NULL;
    list->size = 0;
}

void list_clear_device(EntityList* list) {
    int count = 0;
    EntityBlock* current = list->first;
    while (current != NULL) {
        //printf("%d Address: %p\n", count, current);
        EntityBlock* block = (EntityBlock*)memalloc(sizeof(EntityBlock));
        cudaCopy(block, current, sizeof(EntityBlock), cudaMemcpyDeviceToHost);
        EntityBlock* next = block->next;
        cudaFree(block->entity);
        cudaFree(current);
        memfree(block);
        current = next;
        count++;
    }
    list->first = NULL;
    list->size = 0;
}

void list_copy_to_device(EntityList* list, Grid* h_grid, EntityList* h_list) {
    if (list->size == 0) {
        h_list->first = NULL;
        return;
    }
    assert(list->first != NULL);
    h_list->first = block_copy_to_device(list->first, h_grid);
}

EntityBlock* block_copy_to_device(EntityBlock* block, Grid* h_grid) {
    EntityBlock* h_block = (EntityBlock*)memalloc(sizeof(EntityBlock));
    if (block->next != NULL) {
        h_block->next = block_copy_to_device(block->next, h_grid);
    }
    else {
        h_block->next = NULL;
    }
    Entity* entity = block->entity;
    assert(entity != NULL);
    Entity* d_entity;
    cudaAlloc((void**)&d_entity, sizeof(Entity));
    cudaCopy(d_entity, entity, sizeof(Entity), cudaMemcpyHostToDevice);
    h_block->entity = d_entity;
    h_grid->entities[(int)round(entity->position.x)][(int)round(entity->position.y)] = d_entity;
    EntityBlock* d_block;
    cudaAlloc((void**)&d_block, sizeof(EntityBlock));
    cudaCopy(d_block, h_block, sizeof(EntityBlock), cudaMemcpyHostToDevice);
    assert(block != d_block);
    memfree(h_block);
    return d_block;
}

void grid_insert(Grid* grid, Entity* entity) {
    EntityType type = entity->type;
    list_insert(&grid->lists[type], entity);
    assert(grid->entities[(int)round(entity->position.x)][(int)round(entity->position.y)] == NULL);
    grid->entities[(int)round(entity->position.x)][(int)round(entity->position.y)] = entity;
    grid->total_size++;
}

bool grid_remove(Grid* grid, Vector2 position) {
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        if (grid_remove_type(grid, position, (EntityType)i))
            return true;
    }
    return false;
}

bool grid_remove_type(Grid* grid, Vector2 position, EntityType type) {
    if (list_remove(&grid->lists[type], position)) {
        assert(grid->entities[(int)round(position.x)][(int)round(position.y)] != NULL);
        grid->entities[(int)round(position.x)][(int)round(position.y)] = NULL;
        grid->total_size--;
        return true;
    }
    return false;
}

Grid* grid_init() {
    Grid* grid = (Grid*)memalloc(sizeof(Grid));
    grid->total_size = 0;
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        grid->lists[i].size = 0;
    }
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            grid->entities[i][j] = NULL;
            //grid->locks[i][j] = 0;
        }
    }
    //grid->warp_lock = 0;
    return grid;
}

Grid* grid_copy_to_device(Grid* grid) {
    Grid* h_grid = (Grid*)memalloc(sizeof(Grid));
    h_grid->total_size = grid->total_size;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            h_grid->entities[i][j] = NULL;
            //h_grid->locks[i][j] = 0;
        }
    }
    //h_grid->warp_lock = 0;
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        h_grid->lists[i].size = grid->lists[i].size;
        list_copy_to_device(&grid->lists[i], h_grid, &h_grid->lists[i]);
    }
    Grid* d_grid;
    cudaAlloc((void**)&d_grid, sizeof(Grid));
    cudaCopy(d_grid, h_grid, sizeof(Grid), cudaMemcpyHostToDevice);
    memfree(h_grid);
    return d_grid;
}

__host__ __device__ void grid_check(Grid* grid) {
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        EntityBlock* current = grid->lists[i].first;
        while (current != NULL) {
            printf("Address check: %p\n", current);
            current = current->next;
        }
    }
}

void grid_free(Grid* grid) {
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        list_clear(&grid->lists[i]);
    }
    memfree(grid);
}

void grid_free_device(Grid* grid) {
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        EntityList* list = (EntityList*)memalloc(sizeof(EntityList));
        cudaCopy(list, &grid->lists[i], sizeof(EntityList), cudaMemcpyDeviceToHost);
        list_clear_device(list);
        memfree(list);
    }
    cudaFree(grid);
}

__host__ __device__ Entity* grid_get(Grid* grid, Vector2 position) {
    if (!is_pos_valid(position))
        return NULL;

    return grid->entities[(int)round(position.x)][(int)round(position.y)];
}

__host__ __device__ Entity* grid_get_type(Grid* grid, Vector2 position, EntityType type) {
    if (!is_pos_valid(position))
        return NULL;

    Entity* entity = grid_get(grid, position);
    if (entity != NULL) {
        if (entity->type == type) {
            return entity;
        }
    }
    return NULL;
}

__host__ __device__ bool grid_is_pos_free(Grid* grid, Vector2 position) {
    if (!is_pos_valid(position))
        return false;

    return grid_get(grid, position) == NULL;
}

// __device__ void grid_insert_device(Grid* grid, EntityBlock* block) {
//     EntityType type = block->entity->type;
//     EntityList* list = &grid->lists[type];
//     EntityBlock* last = NULL;
//     EntityBlock** current = &list->first;
//     while (*current != NULL) {
//         last = *current;
//         current = &(*current)->next;
//     }
//     block->next = NULL;
//     if (list->first == NULL) {
//         list->first = block;
//     }
//     else {
//         last->next = block;
//     }
//     list->size++;
//     grid->entities[(int)round(block->entity->position.x)][(int)round(block->entity->position.y)] = block->entity;
//     grid->total_size++;
// }

// __global__ void kernel_grid_insert(Grid* grid, EntityBlock* block) {
//     grid_insert_device(grid, block);
// }







Entity** look_for_nearby_entities(Grid* grid, Vector2 reference, EntityType type, int* count) {
    *count = 0;
    Entity** array = (Entity**)memalloc(grid->lists[type].size * sizeof(Entity*));
    for (int i = -PROXIMITY_DIST; i <= PROXIMITY_DIST; i++) {
        for (int j = -PROXIMITY_DIST; j <= PROXIMITY_DIST; j++) {
            if (i == 0 && j == 0)
                continue;
            Vector2 position = {
                reference.x + i,
                reference.y + j
            };

            Entity* entity = grid_get_type(grid, position, type);
            if (entity != NULL) {
                if (!entity->has_interacted && !entity->to_be_removed) {
                    array[*count] = entity;
                    (*count)++;
                }
            }
        }
    }
    return array;
}

Vector2* find_all_free_nearby_pos(Grid* grid, Vector2 reference, int* count) {
    *count = 0;
    Vector2* array = (Vector2*)memalloc((2*PROXIMITY_DIST+1) * (2*PROXIMITY_DIST+1) * sizeof(Vector2));
    for (int i = -PROXIMITY_DIST; i <= PROXIMITY_DIST; i++) {
        for (int j = -PROXIMITY_DIST; j <= PROXIMITY_DIST; j++) {
            if (i == 0 && j == 0)
                continue;
            Vector2 position = {
                reference.x + i,
                reference.y + j
            };

            if (grid_is_pos_free(grid, position)) {
                array[*count] = position;
                (*count)++;
            }
        }
    }
    return array;
}

Vector2* find_free_pos_nearby(Grid* grid, Vector2 reference) {
    int n = 0;
    Vector2* position = NULL;
    Vector2* free_positions = find_all_free_nearby_pos(grid, reference, &n);
    if (n > 0) {
        position = (Vector2*)memalloc(sizeof(Vector2));
        *position = free_positions[rand() % n];
    }
    memfree(free_positions);
    return position;
}

Vector2* find_n_free_nearby_pos(Grid* grid, Vector2 reference, int max, int* count) {
    int n = 0;
    *count = 0;
    Vector2* free_positions = find_all_free_nearby_pos(grid, reference, &n);
    if (n > 0) {
        Vector2* array = (Vector2*)memalloc(n * sizeof(Vector2));
        for (int i = 0; i < max; i++) {
            if (n < 1)
                break;
            int seed = rand() % n;
            array[*count] = free_positions[seed];
            free_positions[seed] = free_positions[n-1];
            (*count)++;
            n--;
        }
        return array;
    }
    memfree(free_positions);
    return NULL;
}

void duplicate_entity(Grid* grid, Entity* entity) {
    Vector2* position = find_free_pos_nearby(grid, entity->position);
    if (position != NULL) {
        Entity* new_entity = create_entity(entity->type, *position);
        for (int i = 0; i < RECEPTOR_SIZE; i++)
            new_entity->receptor[i] = entity->receptor[i];
        new_entity->velocity = entity->velocity;
        new_entity->status = entity->status;
        new_entity->has_interacted = true;
        grid_insert(grid, new_entity);
    }
}

void generate_antibodies(Grid* grid, Vector2 origin) {
    int count = 0;
    Vector2* positions = find_n_free_nearby_pos(grid, origin, AB_CREATED_PER_CELL, &count);
    if (positions != NULL) {
        for (int i = 0; i < count; i++) {
            grid_insert(grid, create_entity(AB_MOLECOLE, positions[i]));
        }
        memfree(positions);
    }
}

void diffuse_entity(Grid* grid, Entity* entity) {
    double mass = 0.0;
    switch (entity->type) {
        case B_CELL:
        case T_CELL:
            mass = 0.2;
            break;
        case AG_MOLECOLE:
        case AB_MOLECOLE:
            mass = 0.1;
            break;
        default:
            break;
    }

    /* Box Muller */
    double r1 = randdouble();
    double r2 = randdouble();
    double random_x = sqrt(-2 * log(r1)) * cos(2 * PI * r2);
    double random_y = sqrt(-2 * log(r1)) * sin(2 * PI * r2);

    /* Langevin equation */
    entity->velocity.x += langevin(entity->velocity.x, random_x, mass);
    entity->velocity.y += langevin(entity->velocity.y, random_y, mass);

    Vector2 new_position = entity->position;
    new_position.x += entity->velocity.x * TIME_FACTOR;
    new_position.y += entity->velocity.y * TIME_FACTOR;
    adjust_pos(&new_position); // correct the position

    if (is_matching_pos(new_position, entity->position)) {
        entity->position = new_position;
        return;
    }

    if (!grid_is_pos_free(grid, new_position)) {
        /* If the position is not free, try to look for a nearby one. */
        Vector2* pos = find_free_pos_nearby(grid, new_position);
        if (pos == NULL) /* If no free positions can be found, the entity remains stationary. */
            return; 
        new_position = *pos;
        memfree(pos);
    }
    grid->entities[(int)round(entity->position.x)][(int)round(entity->position.y)] = NULL;
    grid->entities[(int)round(new_position.x)][(int)round(new_position.y)] = entity;
    entity->position = new_position;
}

// __device__ void diffuse_entity(Grid* grid, Entity* entity, curandState* rng) {
//     __shared__ int locked;
//     double mass = 0.0;
//     switch (entity->type) {
//         case B_CELL:
//         case T_CELL:
//             mass = 0.2;
//             break;
//         case AG_MOLECOLE:
//         case AB_MOLECOLE:
//             mass = 0.1;
//             break;
//         default:
//             break;
//     }

//     /* Box Muller */
//     double r1 = curand_uniform(&(rng[getthreadindex()]));
//     double r2 = curand_uniform(&(rng[getthreadindex()]));
//     double random_x = sqrt(-2 * log(r1)) * cos(2 * PI * r2);
//     double random_y = sqrt(-2 * log(r1)) * sin(2 * PI * r2);

//     /* Langevin equation */
//     entity->velocity.x += langevin(entity->velocity.x, random_x, mass);
//     entity->velocity.y += langevin(entity->velocity.y, random_y, mass);

//     Vector2 start_position = {
//         .x = entity->position.x,
//         .y = entity->position.y,
//     };
//     Vector2 new_position = {
//         .x = entity->position.x,
//         .y = entity->position.y,
//     };
//     new_position.x += entity->velocity.x * TIME_FACTOR;
//     new_position.y += entity->velocity.y * TIME_FACTOR;
//     adjust_pos(&new_position); // correct the position

//     if (threadIdx.x == 0) {
//         locked = 1;
//     }

//     __syncthreads();

//     if (threadIdx.x == 0) {
//         while (atomicCAS(&grid->warp_lock, 0, 1) != 0);
//         locked = 0;
//     }
//     else {
//         while (locked == 1);
//     }

//     if (is_matching_pos(new_position, start_position)) {
//         entity->position.x = new_position.x;
//         entity->position.y = new_position.y;
//     }
//     else {
// 		while (atomicCAS(&grid->locks[(int)round(new_position.x)][(int)round(new_position.y)], 0, 1) != 0) {
// 			printf("Thread %d - Wait for lock in this position: X=%d, Y=%d, Lock: %d\n", getthreadindex(), (int)round(new_position.x), (int)round(new_position.y), grid->locks[(int)round(new_position.x)][(int)round(new_position.y)]);
// 		}
// 		printf("Thread %d - Acquired lock in this position: X=%d, Y=%d\n", getthreadindex(), (int)round(new_position.x), (int)round(new_position.y));
	
// 		if (!grid_is_pos_free(grid, new_position)) {
// 			Vector2 origin = {
// 				.x = new_position.x,
// 				.y = new_position.y,
// 			};
// 			grid->locks[(int)round(origin.x)][(int)round(origin.y)] = 0;
// 			printf("Thread %d - Released lock in this position: X=%d, Y=%d\n", getthreadindex(), (int)round(origin.x), (int)round(origin.y));
			
// 			/* If the position is not free, try to look for a nearby one. */
// 			int num_found = 0;
// 			for (int i = -PROXIMITY_DIST; i <= PROXIMITY_DIST; i++) {
// 				for (int j = -PROXIMITY_DIST; j <= PROXIMITY_DIST; j++) {
// 					if (i == 0 && j == 0)
// 						continue;
// 					Vector2 position = {
// 						origin.x + i,
// 						origin.y + j
// 					};
	
// 					if (!is_pos_valid(position))
// 						continue;
	
// 					if (grid_is_pos_free(grid, position)) {
// 						num_found++;
// 					}
// 				}
// 			}
// 			/* If no free positions can be found, the entity remains stationary. */
// 			if (num_found == 0)
// 				return;
			
// 			bool found = false;
// 			double r3 = curand_uniform(&(rng[getthreadindex()]));
// 			int rand = (int)(r3 * 10) % num_found;
// 			num_found = 0;
	
// 			for (int i = -PROXIMITY_DIST; i <= PROXIMITY_DIST; i++) {
// 				if (found)
// 					break;
// 				for (int j = -PROXIMITY_DIST; j <= PROXIMITY_DIST; j++) {
// 					if (i == 0 && j == 0)
// 						continue;
// 					Vector2 position = {
// 						origin.x + i,
// 						origin.y + j
// 					};
					
// 					if (!is_pos_valid(position))
// 						continue;
	
// 					while (atomicCAS(&grid->locks[(int)round(position.x)][(int)round(position.y)], 0, 1) != 0) {
// 						printf("Thread %d - Wait for lock in this position while searching: X=%d, Y=%d, Lock: %d\n", getthreadindex(), (int)round(position.x), (int)round(position.y), grid->locks[(int)round(new_position.x)][(int)round(new_position.y)]);
// 					}
// 					printf("Thread %d - Acquired lock in this position while searching: X=%d, Y=%d\n", getthreadindex(), (int)round(position.x), (int)round(position.y));
	
// 					if (grid_is_pos_free(grid, position)) {
// 						if (rand == num_found) {
// 							found = true;
// 							new_position = position;
// 							printf("Thread %d - Found position: X=%d, Y=%d\n", getthreadindex(), (int)round(position.x), (int)round(position.y));
// 							break;
// 						}
// 						else {
// 							grid->locks[(int)round(position.x)][(int)round(position.y)] = 0;
// 							printf("Thread %d - Released lock in this position while searching: X=%d, Y=%d\n", getthreadindex(), (int)round(position.x), (int)round(position.y));
// 						}
// 						num_found++;
// 					}
// 					else {
// 						grid->locks[(int)round(position.x)][(int)round(position.y)] = 0;
// 						printf("Thread %d - Released lock in this position while searching: X=%d, Y=%d\n", getthreadindex(), (int)round(position.x), (int)round(position.y));
// 					}
// 				}
// 			}
	
// 			if (!found) {
// 				printf("Thread %d - Position taken by another thread\n", getthreadindex());
// 				return;
// 			}
// 		}
//         printf("Thread %d - %p Old position: X=%d, Y=%d\n", getthreadindex(), entity, (int)round(start_position.x), (int)round(start_position.y));
// 		printf("Thread %d - %p Assignment of new position: X=%d, Y=%d\n", getthreadindex(), entity, (int)round(new_position.x), (int)round(new_position.y));
//         grid->entities[(int)round(start_position.x)][(int)round(start_position.y)] = NULL;
// 		grid->entities[(int)round(new_position.x)][(int)round(new_position.y)] = entity;
// 		entity->position.x = new_position.x;
// 		entity->position.y = new_position.y;
// 		__threadfence();
// 		grid->locks[(int)round(new_position.x)][(int)round(new_position.y)] = 0;
// 		printf("Thread %d - %p Released lock in this position after assignment: X=%d, Y=%d\n", getthreadindex(), entity, (int)round(new_position.x), (int)round(new_position.y));
//     }

//     __syncthreads();

//     if (threadIdx.x == 0) {
//         grid->warp_lock = 0;
//     }
// }

__device__ void process_interactions(Grid* grid, Entity* entity) {
    if (entity->has_interacted)
        return;

    switch (entity->type) {
        case B_CELL:
            b_cell_interact(grid, entity);
            break;
        case T_CELL:
            t_cell_interact(grid, entity);
            break;
        case AG_MOLECOLE:
            // antigens do nothing as for now
            break;
        case AB_MOLECOLE:
            antibody_interact(grid, entity);
            break;
        default:
            break;
    }
}

__device__ void b_cell_interact(Grid* grid, Entity* cell) {
    EntityType type;
    switch (cell->status)
    {
        case CS_ACTIVE:
            type = T_CELL;
            break;
        case CS_INTERNALIZED:
            type = AG_MOLECOLE;
            break;
        default:
            return;
    }

    Vector2 reference = cell->position;
    for (int i = -PROXIMITY_DIST; i <= PROXIMITY_DIST; i++) {
        for (int j = -PROXIMITY_DIST; j <= PROXIMITY_DIST; j++) {
            if (i == 0 && j == 0)
                continue;
            Vector2 position = {
                reference.x + i,
                reference.y + j
            };

            Entity* entity = grid_get_type(grid, position, type);
            if (entity != NULL) {
                if (cell->status == CS_ACTIVE) { 
                    // found a T cell
                    if (atomicCAS(&entity->lock, 0, 1) == 0) {
                        if (!entity->has_interacted && !entity->to_be_removed) {
                            entity->has_interacted = true;
                            cell->has_interacted = true;
                            cell->status = CS_STIMULATED;
                            __threadfence();
                            entity->lock = 0;
                            return;
                        }
                        entity->lock = 0;
                    }
                }
                if (cell->status == CS_INTERNALIZED && can_entities_bind(cell, entity)) { 
                    // found a compatible antigen
                    if (atomicCAS(&entity->lock, 0, 1) == 0) { 
                        if (!entity->has_interacted && !entity->to_be_removed) {
                            entity->has_interacted = true;
                            cell->has_interacted = true;
                            cell->status = CS_ACTIVE;
                            __threadfence();
                            entity->lock = 0;
                            return;
                        }
                        entity->lock = 0;
                    }
                }
            }
        }
    }
}

__device__ void t_cell_interact(Grid* grid, Entity* tcell) {
    switch (tcell->status) {
        case CS_ACTIVE:
            // wait for a b cell to interact with me
            break;
        default:
            break;
    }
}

__device__ void antibody_interact(Grid* grid, Entity* antibody) {
    EntityType type = AG_MOLECOLE;
    Vector2 reference = antibody->position;
    for (int i = -PROXIMITY_DIST; i <= PROXIMITY_DIST; i++) {
        for (int j = -PROXIMITY_DIST; j <= PROXIMITY_DIST; j++) {
            if (i == 0 && j == 0)
                continue;
            Vector2 position = {
                reference.x + i,
                reference.y + j
            };

            Entity* entity = grid_get_type(grid, position, type);
            if (entity != NULL) {
                if (can_entities_bind(antibody, entity)) {
                    // found a compatible antigen
                    if (atomicCAS(&entity->lock, 0, 1) == 0) {
                        if (!entity->has_interacted && !entity->to_be_removed) { 
                            antibody->has_interacted = true;
                            entity->has_interacted = true;
                            entity->to_be_removed = true;
                            __threadfence();
                            entity->lock = 0;
                            return;
                        }
                        entity->lock = 0;
                    }
                }
            }
        }
    }
}

__host__ __device__ bool is_pos_valid(Vector2 pos) {
    return 
    pos.x >= 0.0f && pos.x <= GRID_SIZE-1.0f &&
    pos.y >= 0.0f && pos.y <= GRID_SIZE-1.0f;
}

__host__ __device__ bool is_matching_pos(Vector2 pos, Vector2 pos2) {
    return
    round(pos.x) == round(pos2.x) &&
    round(pos.y) == round(pos2.y);
}

__host__ __device__ void adjust_pos(Vector2* pos) {
    if (pos->x < 0.0f)
        pos->x = 0.0f;
    if (pos->y < 0.0f)
        pos->y = 0.0f;
    if (pos->x > GRID_SIZE-1.0f)
        pos->x = GRID_SIZE-1.0f;
    if (pos->y > GRID_SIZE-1.0f)
        pos->y = GRID_SIZE-1.0f;
}