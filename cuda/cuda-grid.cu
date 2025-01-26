#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include "cuda-grid.h"
#include "cuda-memory.h"

extern "C" {
    #include "../memory.h"
}

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

    EntityBlock* new_entity = (EntityBlock*)memalloc(sizeof(EntityBlock));
    new_entity->entity = entity;
    new_entity->next = NULL;
    if (list->first == NULL) {
        assert(last == NULL);
        list->first = new_entity;
    }
    else {
        assert(last != NULL);
        last->next = new_entity;
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
    EntityBlock* current = list->first;
    while (current != NULL) {
        EntityBlock* next = current->next;
        cudaFree(current->entity);
        cudaFree(current);
        current = next;
    }
    list->first = NULL;
    list->size = 0;
}

void list_copy_to_device(EntityList* list, Grid* d_grid, EntityList* d_list) {
    if (list->first == NULL) {
        d_list->first = NULL;
        d_list->size = 0;
        return;
    }
    d_list->first = block_copy_to_device(list->first, d_grid);
}

// void list_copy_to_host(EntityList* d_list, Grid* grid, EntityList* list) {
//     if (d_list->first == NULL) {
//         list->first = NULL;
//         list->size = 0;
//         return;
//     }
//     block_copy_to_host(d_list->first, grid, list->first);
// }

EntityBlock* block_copy_to_device(EntityBlock* block, Grid* d_grid) {
    EntityBlock* d_block;
    cudaAlloc((void**)&d_block, sizeof(EntityBlock));
    cudaCopy(d_block, block, sizeof(EntityBlock), cudaMemcpyHostToDevice);
    cudaAlloc((void**)&d_block->entity, sizeof(Entity));
    cudaCopy(d_block->entity, block->entity, sizeof(Entity), cudaMemcpyHostToDevice);
    Entity* address;
    cudaCopy(address, &d_block->entity, sizeof(Entity*), cudaMemcpyDeviceToHost);
    cudaCopy(&d_grid->entities[(int)round(d_block->entity->position.x)][(int)round(d_block->entity->position.y)], address, sizeof(Entity*), cudaMemcpyHostToDevice);
    d_grid->entities[(int)round(d_block->entity->position.x)][(int)round(d_block->entity->position.y)] = d_block->entity;
    if (block->next != NULL) { // recoursive call
        d_block->next = block_copy_to_device(block->next);
    }
    return d_block;
}

// void block_copy_to_host(Entity* d_block, Grid* grid, Entity* block) {
//     cudaCopy(block, d_block, sizeof(EntityBlock), cudaMemcpyDeviceToHost);
//     cudaCopy(block->entity, d_block->entity, sizeof(Entity), cudaMemcpyDeviceToHost);
//     grid->entities[(int)round(block->entity->position.x)][(int)round(block->entity->position.y)] = block->entity;
//     if (d_block->next != NULL) { // recoursive call
//         block_copy_to_host(d_block->next, grid, block->next);
//     }
// }

void grid_insert(Grid* grid, Entity* entity) {
    EntityType type = entity->type;
    list_insert(&grid->lists[type], entity);
    assert(grid->entities[(int)round(entity->position.x)][(int)round(entity->position.y)] == NULL);
    grid->entities[(int)round(entity->position.x)][(int)round(entity->position.y)] = entity;
    grid->total_size++;
}

bool grid_remove(Grid* grid, Vector2 position) {
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        if (grid_remove_type(grid, position, i))
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
            grid->locks[i][j] = 0;
        }
    }
    return grid;
}

Grid* grid_copy_to_device(Grid* grid) {
    Grid* d_grid;
    cudaAlloc((void**)&d_grid, sizeof(Grid));
    cudaCopy(d_grid, grid, sizeof(Grid), cudaMemcpyHostToDevice);
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        list_copy_to_device(grid->lists[i], d_grid, d_grid->lists[i]);
    }
}

// void grid_copy_to_host(Grid* grid, Grid* d_grid) {
//     cudaCopy(grid, d_grid, sizeof(Grid), cudaMemcpyDeviceToHost);
//     for (int i = 0; i < MAX_ENTITYTYPE; i++) {
//         list_copy_to_host(grid->lists[i], d_grid, d_grid->lists[i]);
//     }
// }

void grid_free(Grid* grid) {
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        list_clear(&grid->lists[i]);
    }
    memfree(grid);
}

void grid_free_device(Grid* grid) {
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        list_clear_device(&grid->lists[i]);
    }
    cudaFree(grid);
}

__host__ __device__ Entity* grid_get(Grid* grid, Vector2 position) {
    if (!is_pos_valid(position))
        return NULL;

    return grid->entities[(int)round(position.x)][(int)round(position.y)];
}

Entity* grid_get_type(Grid* grid, Vector2 position, EntityType type) {
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

__global__ void set_device_pointer_with_entity(Entity** p, Grid* grid, Vector2 position, EntityType type) {
    *p = grid_get_type(p, position, type);
}





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
        memfree(position);
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

__device__ void diffuse_entity(Grid* grid, Entity* entity, int index, curandState* rng) {
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
    double r1 = curand_uniform(&(rng[index]));
    double r2 = curand_uniform(&(rng[index]));
    double random_x = sqrt(-2 * log(r1)) * cos(2 * PI * r2);
    double random_y = sqrt(-2 * log(r1)) * sin(2 * PI * r2);

    /* Langevin equation */
    entity->velocity.x += langevin(entity->velocity.x, random_x, mass);
    entity->velocity.y += langevin(entity->velocity.y, random_y, mass);

    Vector2 new_position = entity->position;
    new_position.x += entity->velocity.x * TIME_FACTOR;
    new_position.y += entity->velocity.y * TIME_FACTOR;
    adjust_pos(&new_position); // correct the position

    while (atomicCas(&grid->locks[(int)round(new_position.x)][(int)round(new_position.y)], 0, 1) != 0);

    if (!grid_is_pos_free(grid, new_position)) {
        grid->locks[(int)round(new_position.x)][(int)round(new_position.y)] = 0;

        /* If the position is not free, try to look for a nearby one. */
        int num_found = 0;
        for (int i = -PROXIMITY_DIST; i <= PROXIMITY_DIST; i++) {
            for (int j = -PROXIMITY_DIST; j <= PROXIMITY_DIST; j++) {
                if (i == 0 && j == 0)
                    continue;
                Vector2 position = {
                    reference.x + i,
                    reference.y + j
                };

                while (atomicCas(&grid->locks[(int)round(position.x)][(int)round(position.y)], 0, 1) != 0);

                if (grid_is_pos_free(grid, position)) {
                    int rand = (int)(curand_uniform(&(rng[index])) * 10) % num_found;
                    if (rand == 0) {
                        if (num_found > 0) {
                            grid->locks[(int)round(new_position.x)][(int)round(new_position.y)] = 0;
                        }
                        new_position = position;
                    }
                    else {
                        grid->locks[(int)round(position.x)][(int)round(position.y)] = 0;
                    }
                    num_found++;
                }
            }
        }
        /* If no free positions can be found, the entity remains stationary. */
        if (num_found == 0)
            return;
    }
    grid->entities[(int)round(entity->position.x)][(int)round(entity->position.y)] = NULL;
    grid->entities[(int)round(new_position.x)][(int)round(new_position.y)] = entity;
    entity->position = new_position;
    grid->locks[(int)round(new_position.x)][(int)round(new_position.y)] = 0;
}

__host__ __device__ void scan_interactions(Grid* grid, Entity* entity) {
    if (entity->has_interacted)
        return;

    switch (entity->type) {
        case B_CELL:
            b_cell_interact(grid, entity);
            break;
        case T_CELL:
            t_cell_interact(grid, entity);
            break;
        case AB_MOLECOLE:
            antibody_interact(grid, entity);
            break;
        case AG_MOLECOLE:
            // antigens do nothing as for now
            break;
        default:
            break;
    }
}

__host__ __device__ void b_cell_interact(Grid* grid, Entity* bcell) {
    int count = 0;
    Entity** entities;
    switch (bcell->status) {
        case CS_ACTIVE:
        case CS_INTERNALIZED:
            b_cell_look_for_entity(grid, bcell);
            break;
        case CS_STIMULATED:
            bcell->status = CS_INTERNALIZED;
            bcell->has_interacted = true;
            hypermutation(bcell);
            duplicate_entity(grid, bcell);
            generate_antibodies(grid, bcell->position);
            break;
        default:
            break;
    }
}

__device__ void b_cell_look_for_entity(Grid* grid, Entity* cell) {
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
                    if (!entity->has_interacted && !entity->to_be_removed) {
                        /* Critical section! Needs to run only by one thread at the time for entity. */
                        if (atomicCas(&entity->lock, 0, 1) == 0) {
                            entity->has_interacted = true;
                            cell->has_interacted = true;
                            cell->status = CS_STIMULATED;
                            entity->lock = 0;
                            return;
                        }
                    }
                }
                if (cell->status == CS_INTERNALIZED && can_entities_bind(cell, entity, true)) { 
                    // found a compatible antigen
                    if (!entity->has_interacted && !entity->to_be_removed) {
                        /* Critical section! Needs to run only by one thread at the time for entity. */
                        if (atomicCas(&entity->lock, 0, 1) == 0) { 
                            entity->has_interacted = true;
                            cell->has_interacted = true;
                            cell->status = CS_ACTIVE;
                            entity->lock = 0;
                            return;
                        }
                    }
                }
            }
        }
    }
}

__device__ void t_cell_interact(Grid* grid, Entity* tcell) {
    switch (tcell->status) {
        case CS_STIMULATED:
            tcell->status = CS_ACTIVE;
            tcell->has_interacted = true;
            duplicate_entity(grid, tcell);
            break;
        case CS_ACTIVE:
            // wait for a b cell to interact with me
            break;
        default:
            break;
    }
}

__device__ void antibody_interact(Grid* grid, Entity* antibody) {
    EntityType type = AG_MOLECOLE;
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
                if (can_entities_bind(antibody, entity, true)) {
                    // found a compatible antigen
                    if (!entity->has_interacted && !entity->to_be_removed) { 
                        /* Critical section! Needs to run only by one thread at the time for entity. */
                        if (atomicCas(&entity->lock, 0, 1) == 0) {
                            antibody->has_interacted = true;
                            entity->has_interacted = true;
                            entity->to_be_removed = true;
                            entity->lock = 0;
                            return;
                        }
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