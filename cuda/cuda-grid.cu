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

void grid_insert(Grid* grid, Entity entity) {
    assert(grid_is_pos_free(grid, entity.position));
    grid->entities[(int)round(entity.position.x)*GRID_SIZE+(int)round(entity.position.y)] = entity;
    // grid->size[entity.type]++;
    // grid->total_size++;
}

bool grid_remove(Grid* grid, Vector2 position) {
    if (!is_pos_valid(position))
        return false;

    Entity entity = grid_get(grid, position);
    if (entity.type != NONE) {
        // grid->size[entity.type]--;
        // grid->total_size--;
        grid->entities[(int)round(position.x)*GRID_SIZE+(int)round(position.y)].type = NONE;
        return true;
    }
    return false;
}

Grid* grid_init() {
    Grid* grid = (Grid*)memalloc(sizeof(Grid));
    // grid->total_size = 0;
    // for (int i = 0; i < MAX_ENTITYTYPE; i++) {
    //     grid->size[i] = 0;
    // }
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            grid->entities[i*GRID_SIZE+j].type = NONE;
        }
    }
    return grid;
}

void grid_free(Grid* grid) {
    memfree(grid);
}

__host__ __device__ Entity grid_get(Grid* grid, Vector2 position) {
    return grid->entities[(int)round(position.x)*GRID_SIZE+(int)round(position.y)];
}

__host__ __device__ bool grid_is_pos_free(Grid* grid, Vector2 position) {
    if (!is_pos_valid(position))
        return false;

    Entity entity = grid_get(grid, position);
    return entity.type == NONE;
}





__device__ void duplicate_entity(Grid* grid, Entity entity) {
    /* Get the number of the free positions. */
    int num_free = 0;
    Vector2 reference = entity.position;
    Vector2 free_positions[(2*PROXIMITY_DIST+1) * (2*PROXIMITY_DIST+1)];
    for (int i = -PROXIMITY_DIST; i <= PROXIMITY_DIST; i++) {
        for (int j = -PROXIMITY_DIST; j <= PROXIMITY_DIST; j++) {
            if (i == 0 && j == 0)
                continue;
            Vector2 position = {
                reference.x + i,
                reference.y + j
            };

            if (!is_pos_valid(position))
                continue;

            if (grid_is_pos_free(grid, position)) {
                free_positions[num_free] = position;
                num_free++;
            }
        }
    }

    /* There are no free positions to duplicate this entity, let's bail. */
    if (num_free < 1)
        return;

    /* Extract a random index to decide which position will be chosen. */
    int randindex = device_rand(&grid->seed) % num_free;
    Vector2 randpos = free_positions[randindex];
    while (atomicCAS((int*)&grid->entities[(int)round(randpos.x)*GRID_SIZE+(int)round(randpos.y)].type, NONE, entity.type) != NONE) {
        if (num_free < 1)
            return;
        free_positions[randindex] = free_positions[num_free-1];
        num_free--;
        randindex = device_rand(&grid->seed) % num_free;
        randpos = free_positions[randindex];
    }

    Entity new_entity = create_entity(entity.type, randpos, device_rand(&grid->seed));
    for (int i = 0; i < RECEPTOR_SIZE; i++)
        new_entity.receptor[i] = entity.receptor[i];
    new_entity.velocity = entity.velocity;
    new_entity.status = CS_INTERNALIZED;
    grid->entities[(int)round(randpos.x)*GRID_SIZE+(int)round(randpos.y)] = new_entity;
}

__device__ void generate_antibodies(Grid* grid, Entity cell) {
    /* Get the number of the free positions. */
    int num_free = 0;
    Vector2 reference = cell.position;
    Vector2 free_positions[(2*PROXIMITY_DIST+1) * (2*PROXIMITY_DIST+1)];
    for (int i = -PROXIMITY_DIST; i <= PROXIMITY_DIST; i++) {
        for (int j = -PROXIMITY_DIST; j <= PROXIMITY_DIST; j++) {
            if (i == 0 && j == 0)
                continue;
            Vector2 position = {
                reference.x + i,
                reference.y + j
            };

            if (!is_pos_valid(position))
                continue;

            if (grid_is_pos_free(grid, position)) {
                free_positions[num_free] = position;
                num_free++;
            }
        }
    }

    /* There are no free positions to generate some new antibodies, let's bail. */
    if (num_free < 1)
        return;

    /* Extract random indexes to decide where to generate the antibodies. */
    for (int i = 0; i < AB_CREATED_PER_CELL; i++) {
        if (num_free < 1)
            return;

        int randindex = device_rand(&grid->seed) % num_free;
        Vector2 randpos = free_positions[randindex];
        while (atomicCAS((int*)&grid->entities[(int)round(randpos.x)*GRID_SIZE+(int)round(randpos.y)].type, NONE, AB_MOLECOLE) != NONE) {
            if (num_free < 1)
                return;
            free_positions[randindex] = free_positions[num_free-1];
            num_free--;
            randindex = device_rand(&grid->seed) % num_free;
            randpos = free_positions[randindex];
        }
        free_positions[randindex] = free_positions[num_free-1];
        num_free--;

        Entity entity = create_entity(AB_MOLECOLE, randpos, device_rand(&grid->seed));
        for (int i = 0; i < RECEPTOR_SIZE; i++)
            entity.receptor[i] = cell.receptor[i];
        grid->entities[(int)round(randpos.x)*GRID_SIZE+(int)round(randpos.y)] = entity;
    }
}

__device__ void diffuse_entity(Grid* grid, Entity* entity) {
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
    double r1 = device_randdouble(&grid->seed);
    double r2 = device_randdouble(&grid->seed);
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
    }
    else {
        Vector2 reference = new_position;
        if (atomicCAS((int*)&grid->entities[(int)round(new_position.x)*GRID_SIZE+(int)round(new_position.y)].type, NONE, entity->type) != NONE) {
            /* If the position is not free, try looking for a nearby one. */
            /* Get the number of the free positions. */
            int num_free = 0;
            Vector2 free_positions[(2*PROXIMITY_DIST+1) * (2*PROXIMITY_DIST+1)];
            for (int i = -PROXIMITY_DIST; i <= PROXIMITY_DIST; i++) {
                for (int j = -PROXIMITY_DIST; j <= PROXIMITY_DIST; j++) {
                    if (i == 0 && j == 0)
                        continue;
                    Vector2 position = {
                        reference.x + i,
                        reference.y + j
                    };

                    if (!is_pos_valid(position))
                        continue;

                    if (grid_is_pos_free(grid, position)) {
                        free_positions[num_free] = position;
                        num_free++;
                    }
                }
            }

            /* If there are no free positions, the entity remains stationary. */
            if (num_free < 1)
                return;

            /* Extract a random index to decide which position will be chosen. */
            int randindex = device_rand(&grid->seed) % num_free;
            Vector2 randpos = free_positions[randindex];
            while (atomicCAS((int*)&grid->entities[(int)round(randpos.x)*GRID_SIZE+(int)round(randpos.y)].type, NONE, entity->type) != NONE) {
                if (num_free < 1)
                    return;
                free_positions[randindex] = free_positions[num_free-1];
                num_free--;
                randindex = device_rand(&grid->seed) % num_free;
                randpos = free_positions[randindex];
            }
            new_position = randpos;
        }
        Entity new_entity = create_entity(entity->type, new_position, entity->seed);
        for (int i = 0; i < RECEPTOR_SIZE; i++)
            new_entity.receptor[i] = entity->receptor[i];
        new_entity.velocity = entity->velocity;
        new_entity.status = entity->status;

        grid->entities[(int)round(entity->position.x)*GRID_SIZE+(int)round(entity->position.y)].type = NONE;
        grid->entities[(int)round(new_position.x)*GRID_SIZE+(int)round(new_position.y)] = new_entity;
    }
}

__device__ void process_interactions(Grid* grid, Entity* entity) {
    switch (entity->type) {
        case B_CELL:
            if (atomicCAS(&entity->has_interacted, 0, 1) == 0) {
                b_cell_interact(grid, entity);
            }
            break;
        case T_CELL:
            // t cells do nothing
            break;
        case AG_MOLECOLE:
            // antigens do nothing
            break;
        case AB_MOLECOLE:
            if (atomicCAS(&entity->has_interacted, 0, 1) == 0) {
                antibody_interact(grid, entity);
            }
            break;
        default:
            break;
    }
}

__device__ void b_cell_interact(Grid* grid, Entity* cell) {
    EntityType type;
    switch (cell->status) {
        case CS_ACTIVE:
            type = T_CELL;
            break;
        case CS_INTERNALIZED:
            type = AG_MOLECOLE;
            break;
        case CS_STIMULATED:
            cell->status = CS_INTERNALIZED;
            hypermutation(cell);
            duplicate_entity(grid, *cell);
            generate_antibodies(grid, *cell);
            return;
        default:
            return;
    }

    /* Look for nearby entities and bind with a compatible one. */
    Vector2 reference = cell->position;
    for (int i = -PROXIMITY_DIST; i <= PROXIMITY_DIST; i++) {
        for (int j = -PROXIMITY_DIST; j <= PROXIMITY_DIST; j++) {
            if (i == 0 && j == 0)
                continue;
            Vector2 position = {
                reference.x + i,
                reference.y + j
            };

            if (!is_pos_valid(position))
                continue;

            Entity* entity = &grid->entities[(int)round(position.x)*GRID_SIZE+(int)round(position.y)];
            if (entity->type == type) {
                if (cell->status == CS_ACTIVE) { 
                    // found a T cell
                    if (atomicCAS(&entity->has_interacted, 0, 1) == 0) {
                        cell->status = CS_STIMULATED;
                        return;
                    }
                }
                if (cell->status == CS_INTERNALIZED && can_entities_bind(cell, *entity)) { 
                    // found a compatible antigen
                    if (atomicCAS(&entity->has_interacted, 0, 1) == 0) { 
                        cell->status = CS_ACTIVE;
                        return;
                    }
                }
            }
        }
    }
}

__device__ void antibody_interact(Grid* grid, Entity* antibody) {
    /* Look for nearby antigens and eliminate one. */
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

            if (!is_pos_valid(position))
                continue;

            Entity* entity = &grid->entities[(int)round(position.x)*GRID_SIZE+(int)round(position.y)];
            if (entity->type == type) {
                if (can_entities_bind(antibody, *entity)) {
                    // found a compatible antigen
                    if (atomicCAS(&entity->has_interacted, 0, 1) == 0) {
                        grid->entities[(int)round(position.x)*GRID_SIZE+(int)round(position.y)].type = NONE;
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