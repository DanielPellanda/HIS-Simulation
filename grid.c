#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include "grid.h"
#include "memory.h"

int GRID_SIZE = DEFAULT_GRID_SIZE;

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

    EntityBlock* new = (EntityBlock*)memalloc(sizeof(EntityBlock));
    new->entity = entity;
    new->next = NULL;
    if (list->first == NULL) {
        assert(last == NULL);
        list->first = new;
    }
    else {
        assert(last != NULL);
        last->next = new;
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

void list_clear(EntityList* list)  {
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

void grid_insert(Grid* grid, Entity* entity) {
    EntityType type = entity->type;
    list_insert(&grid->lists[type], entity);
    #ifdef FAST_GRID_SEARCH
        assert(grid->entities[(int)round(entity->position.x)][(int)round(entity->position.y)] == NULL);
        grid->entities[(int)round(entity->position.x)][(int)round(entity->position.y)] = entity;
    #endif
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
        #ifdef FAST_GRID_SEARCH
            assert(grid->entities[(int)round(position.x)][(int)round(position.y)] != NULL);
            grid->entities[(int)round(position.x)][(int)round(position.y)] = NULL;
        #endif
        //assert(grid_get_type(grid, position, type) == NULL);
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
    #ifdef FAST_GRID_SEARCH
        grid->entities = (Entity***)memalloc(GRID_SIZE * sizeof(Entity**));
        for (int i = 0; i < GRID_SIZE; i++) {
            grid->entities[i] = (Entity**)memalloc(GRID_SIZE * sizeof(Entity*));
            for (int j = 0; j < GRID_SIZE; j++) {
                grid->entities[i][j] = NULL;
            }
        }
    #endif
    return grid;
}

void grid_free(Grid* grid) {
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        list_clear(&grid->lists[i]);
    }
    #ifdef FAST_GRID_SEARCH
        for (int i = 0; i < GRID_SIZE; i++) {
            memfree(grid->entities[i]);
        }
        memfree(grid->entities);
    #endif
    memfree(grid);
}

Entity* grid_get(Grid* grid, Vector2 position) {
    if (!is_pos_valid(position))
        return NULL;

    #ifdef FAST_GRID_SEARCH
        return grid->entities[(int)round(position.x)][(int)round(position.y)];
    #else
        for (int i = 0; i < MAX_ENTITYTYPE; i++) {
            Entity* e = grid_get_type(grid, position, i);
            if (e != NULL)
                return e;
        }
        return NULL;
    #endif
}

Entity* grid_get_type(Grid* grid, Vector2 position, EntityType type) {
    if (!is_pos_valid(position))
        return NULL;

    #ifdef FAST_GRID_SEARCH
        Entity* entity = grid_get(grid, position);
        if (entity != NULL) {
            if (entity->type == type) {
                return entity;
            }
        }
        return NULL;
    #else
        return list_get(&grid->lists[type], position);
    #endif
}

Entity** grid_get_all(Grid* grid) {
    Entity** array = (Entity**)memalloc(grid->total_size * sizeof(Entity*));
    int count = 0;
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        EntityBlock** current = &grid->lists[i].first;
        while (*current != NULL) {
            assert((*current)->entity != NULL);
            (*current)->entity->has_interacted = false;
            array[count] = (*current)->entity;
            current = &(*current)->next;
            count++;
        }
    }
    assert(count == grid->total_size);
    return array;
}

bool grid_is_pos_free(Grid* grid, Vector2 position) {
    if (!is_pos_valid(position))
        return false;

    return grid_get(grid, position) == NULL;
}




Entity** look_for_nearby_entities(Grid* grid, Vector2 reference, EntityType type, int* count) {
    *count = 0;
    Entity** array = (Entity**)memalloc(grid->lists[type].size * sizeof(Entity*));

    #ifdef FAST_GRID_SEARCH
        if (grid->lists[type].size > (2*PROXIMITY_DIST+1)*(2*PROXIMITY_DIST+1)) {
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
    #endif

    EntityBlock** current = &grid->lists[type].first;
    while (*current != NULL) {
        if (
        /* Skip entities that have already interacted or are about to be removed. */
        !(*current)->entity->has_interacted && !(*current)->entity->to_be_removed && 
        abs(reference.x - (*current)->entity->position.x) <= PROXIMITY_DIST && 
        abs(reference.y - (*current)->entity->position.y) <= PROXIMITY_DIST) {
            array[*count] = (*current)->entity;
            (*count)++;
        }
        current = &(*current)->next;
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
        Entity* new = create_entity(entity->type, *position);
        for (int i = 0; i < RECEPTOR_SIZE; i++)
            new->receptor[i] = entity->receptor[i];
        new->velocity = entity->velocity;
        new->status = entity->status;
        new->has_interacted = true;
        memfree(position);
        grid_insert(grid, new);
    }
}

void generate_antibodies(Grid* grid, Vector2 origin) {
    int count = 0;
    Vector2* positions = find_n_free_nearby_pos(grid, origin, AB_CREATED_PER_CELL, &count);
    if (positions != NULL) {
        for (int i = 0; i < count; i++) {
            //assert(grid_is_pos_free(grid, positions[i]));
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

    if (!grid_is_pos_free(grid, new_position)) {
        /* If the position is not free, try to look for a nearby one. */
        Vector2* pos = find_free_pos_nearby(grid, new_position);
        if (pos == NULL) /* If no free positions can be found, the entity remains stationary. */
            return; 
        new_position = *pos;
        memfree(pos);
    }

    #ifdef FAST_GRID_SEARCH
        grid->entities[(int)round(entity->position.x)][(int)round(entity->position.y)] = NULL;
        grid->entities[(int)round(new_position.x)][(int)round(new_position.y)] = entity;
    #endif

    entity->position = new_position;
}

void scan_interactions(Grid* grid, Entity* entity) {
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

void b_cell_interact(Grid* grid, Entity* bcell) {
    int count = 0;
    Entity** entities;
    switch (bcell->status) {
        case CS_ACTIVE:
            entities = look_for_nearby_entities(grid, bcell->position, T_CELL, &count);
            if (entities != NULL) {
                for (int i = 0; i < count; i++) {
                    if (entities[i]->status == CS_ACTIVE) {
                        bcell->status = CS_STIMULATED;
                        bcell->has_interacted = true;
                        entities[i]->has_interacted = true;
                        break;
                    }
                }
                memfree(entities);
            }
            break;
        case CS_INTERNALIZED:
            entities = look_for_nearby_entities(grid, bcell->position, AG_MOLECOLE, &count);
            if (entities != NULL) {
                for (int i = 0; i < count; i++) {
                    if (can_entities_bind(bcell, entities[i], true)) {
                        bcell->status = CS_ACTIVE;
                        bcell->has_interacted = true;
                        entities[i]->has_interacted = true;
                        break;
                    }
                }
                memfree(entities);
            }
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

void t_cell_interact(Grid* grid, Entity* tcell) {
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

void antibody_interact(Grid* grid, Entity* antibody) {
    int count = 0;
    Entity** antigens = look_for_nearby_entities(grid, antibody->position, AG_MOLECOLE, &count);
    if (antigens != NULL) {
        for (int i = 0; i < count; i++) {
            if (can_entities_bind(antibody, antigens[i], true)) {
                antibody->has_interacted = true;
                antigens[i]->has_interacted = true;
                antigens[i]->to_be_removed = true;
                // grid_remove_type(grid, antigens[i]->position, antigens[i]->type);
                break;
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