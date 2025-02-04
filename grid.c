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
#include "grid.h"
#include "memory.h"

int GRID_SIZE = DEFAULT_GRID_SIZE;

void grid_insert(Grid* grid, Entity entity) {
    assert(grid_is_pos_free(grid, entity.position));
    grid->entities[(int)round(entity.position.x)][(int)round(entity.position.y)] = entity;
    grid->size[entity.type]++;
    grid->total_size++;
}

bool grid_remove(Grid* grid, Vector2 position) {
    if (!is_pos_valid(position))
        return false;

    Entity entity = grid_get(grid, position);
    if (entity.type != NONE) {
        grid->size[entity.type]--;
        grid->total_size--;
        grid->entities[(int)round(position.x)][(int)round(position.y)].type = NONE;
        return true;
    }
    return false;
}

Grid* grid_init() {
    Grid* grid = (Grid*)memalloc(sizeof(Grid));
    grid->total_size = 0;
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        grid->size[i] = 0;
    }
    grid->entities = (Entity**)memalloc(GRID_SIZE * sizeof(Entity*));
    for (int i = 0; i < GRID_SIZE; i++) {
        grid->entities[i] = (Entity*)memalloc(GRID_SIZE * sizeof(Entity));
        for (int j = 0; j < GRID_SIZE; j++) {
            grid->entities[i][j].type = NONE;
        }
    }
    return grid;
}

void grid_free(Grid* grid) {
    for (int i = 0; i < GRID_SIZE; i++) {
        memfree(grid->entities[i]);
    }
    memfree(grid->entities);
    memfree(grid);
}

Entity grid_get(Grid* grid, Vector2 position) {
    return grid->entities[(int)round(position.x)][(int)round(position.y)];
}

Entity** grid_get_all(Grid* grid) {
    Entity** array = (Entity**)memalloc(grid->total_size * sizeof(Entity*));
    int k = 0;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (grid->entities[i][j].type != NONE) {
                grid->entities[i][j].has_interacted = 0;
                array[k] = &grid->entities[i][j];
                k++;
            }
        }
    }
    return array;
}

bool grid_is_pos_free(Grid* grid, Vector2 position) {
    if (!is_pos_valid(position))
        return false;

    Entity entity = grid_get(grid, position);
    return entity.type == NONE;
}






Entity** look_for_nearby_entities(Grid* grid, Vector2 reference, EntityType type, int* count) {
    *count = 0;
    Entity** array = (Entity**)memalloc(grid->size[type] * sizeof(Entity*));
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

            Entity entity = grid_get(grid, position);
            if (entity.type == type) {
                if (!entity.has_interacted) {
                    array[*count] = &grid->entities[(int)round(position.x)][(int)round(position.y)];
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

            if (!is_pos_valid(position))
                continue;

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
        int index = rand() % n;
        position = (Vector2*)memalloc(sizeof(Vector2));
        *position = free_positions[index];
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

void duplicate_entity(Grid* grid, Entity entity) {
    Vector2* position = find_free_pos_nearby(grid, entity.position);
    if (position != NULL) {
        Entity new = create_entity(entity.type, *position);
        for (int i = 0; i < RECEPTOR_SIZE; i++)
            new.receptor[i] = entity.receptor[i];
        new.velocity = entity.velocity;
        new.status = entity.status;
        new.has_interacted = 1;
        grid_insert(grid, new);
        memfree(position);
    }
}

void generate_antibodies(Grid* grid, Entity cell) {
    int count = 0;
    Vector2* positions = find_n_free_nearby_pos(grid, cell.position, AB_CREATED_PER_CELL, &count);
    if (positions != NULL) {
        for (int i = 0; i < count; i++) {
            Entity entity = create_entity(AB_MOLECOLE, positions[i]);
            for (int i = 0; i < RECEPTOR_SIZE; i++)
                entity.receptor[i] = cell.receptor[i];
            grid_insert(grid, entity);
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
    }
    else {

        if (!grid_is_pos_free(grid, new_position)) {
            /* If the position is not free, try to look for a nearby one. */
            Vector2* pos = find_free_pos_nearby(grid, new_position);
            if (pos == NULL) /* If no free positions can be found, the entity remains stationary. */
                return; 
            new_position = *pos;
            memfree(pos);
        }

        assert(entity == &grid->entities[(int)round(entity->position.x)][(int)round(entity->position.y)]);
        assert(grid->entities[(int)round(new_position.x)][(int)round(new_position.y)].type == NONE);
        Entity new = create_entity(entity->type, new_position);
        for (int i = 0; i < RECEPTOR_SIZE; i++)
            new.receptor[i] = entity->receptor[i];
        new.velocity = entity->velocity;
        new.status = entity->status;
        grid->entities[(int)round(entity->position.x)][(int)round(entity->position.y)].type = NONE;
        grid->entities[(int)round(new_position.x)][(int)round(new_position.y)] = new;
    }
}

void process_interactions(Grid* grid, Entity* entity) {
    if (entity->has_interacted)
        return;

    switch (entity->type) {
        case B_CELL:
            b_cell_interact(grid, entity);
            entity->has_interacted = 1;
            break;
        case T_CELL:
            // t cells do nothing 
            break;
        case AB_MOLECOLE:
            antibody_interact(grid, entity);
            entity->has_interacted = 1;
            break;
        case AG_MOLECOLE:
            // antigens do nothing 
            break;
        default:
            break;
    }
}

void b_cell_interact(Grid* grid, Entity* bcell) {
    int count = 0;
    Entity** entities;
    switch (bcell->status) {
        case CS_ACTIVE:
            entities = look_for_nearby_entities(grid, bcell->position, T_CELL, &count);
            if (entities != NULL) {
                for (int i = 0; i < count; i++) {
                    if (entities[i]->status == CS_ACTIVE) {
                        if (!entities[i]->has_interacted) {
                            bcell->status = CS_STIMULATED;
                            entities[i]->has_interacted = 1;
                            break;
                        }
                    }
                }
                memfree(entities);
            }
            break;
        case CS_INTERNALIZED:
            entities = look_for_nearby_entities(grid, bcell->position, AG_MOLECOLE, &count);
            if (entities != NULL) {
                for (int i = 0; i < count; i++) {
                    if (can_entities_bind(*bcell, *entities[i])) {
                        if (!entities[i]->has_interacted) {
                            bcell->status = CS_ACTIVE;
                            entities[i]->has_interacted = 1;
                            break;
                        }
                    }
                }
                memfree(entities);
            }
            break;
        case CS_STIMULATED:
            bcell->status = CS_INTERNALIZED;
            hypermutation(bcell);
            duplicate_entity(grid, *bcell);
            generate_antibodies(grid, *bcell);
            break;
        default:
            break;
    }
}

// void t_cell_interact(Grid* grid, Entity* tcell) {
//     switch (tcell->status) {
//         // case CS_STIMULATED:
//         //     tcell->status = CS_ACTIVE;
//         //     tcell->has_interacted = 1;
//         //     duplicate_entity(grid, *tcell);
//         //     break;
//         case CS_ACTIVE:
//             // wait for a b cell to interact with me
//             break;
//         default:
//             break;
//     }
// }

void antibody_interact(Grid* grid, Entity* antibody) {
    int count = 0;
    Entity** antigens = look_for_nearby_entities(grid, antibody->position, AG_MOLECOLE, &count);
    if (antigens != NULL) {
        for (int i = 0; i < count; i++) {
            if (can_entities_bind(*antibody, *antigens[i])) {
                if (!antigens[i]->has_interacted) {
                    antigens[i]->has_interacted = 1;
                    grid_remove(grid, antigens[i]->position);
                    assert(antigens[i]->type == NONE);
                    assert(antibody->type == AB_MOLECOLE);
                    break;
                }
            }
        }
        memfree(antigens);
    }
}

bool is_pos_valid(Vector2 pos) {
    return 
    pos.x >= 0.0f && pos.x <= GRID_SIZE-1.0f &&
    pos.y >= 0.0f && pos.y <= GRID_SIZE-1.0f;
}

bool is_matching_pos(Vector2 pos, Vector2 pos2) {
    return
    round(pos.x) == round(pos2.x) &&
    round(pos.y) == round(pos2.y);
}

void adjust_pos(Vector2* pos) {
    if (pos->x < 0.0f)
        pos->x = 0.0f;
    if (pos->y < 0.0f)
        pos->y = 0.0f;
    if (pos->x > GRID_SIZE-1.0f)
        pos->x = GRID_SIZE-1.0f;
    if (pos->y > GRID_SIZE-1.0f)
        pos->y = GRID_SIZE-1.0f;
}