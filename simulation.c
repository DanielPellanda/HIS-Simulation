#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include "simulation.h"
#include "memory.h"

void time_step(Grid* grid) {
    /* Generate the process order of each entity. */
    Entity** entity_list = generate_order(grid);
    if (entity_list == NULL)
        return;

    int size = grid->total_size;
    /* Check for interactions first. */
    for (int i = 0; i < size; i++) {
        if (entity_list[i] == NULL)
            continue;
        if (entity_list[i]->to_be_removed)
            continue;
        scan_interactions(grid, entity_list[i]);
    }
    /* Process their movement. */
    for (int i = 0; i < size; i++) {
        if (entity_list[i] == NULL)
            continue;
        if (entity_list[i]->to_be_removed) {
            grid_remove_type(grid, entity_list[i]->position, entity_list[i]->type);
            continue;
        }
        diffuse_entity(grid, entity_list[i]);
    }
    memfree(entity_list);
}

Entity** generate_order(Grid* grid) {
    Entity** array = grid_get_all(grid);
    if (array == NULL)
        return NULL;

    if (grid->total_size > 0) {
        /* Shuffle the array. */
        for (int i = 0; i < grid->total_size-1; i++) {
            assert(array[i] != NULL);
            int j = i + rand() / (RAND_MAX / (grid->total_size - i) + 1);
            Entity* tmp = array[j];
            array[j] = array[i];
            array[i] = tmp;
        }
    }
    return array;
}