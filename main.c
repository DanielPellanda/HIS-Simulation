#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "memory.h"
#include "simulation.h"

Grid* generate_grid() {
    Vector2* positions = (Vector2*)memalloc(GRID_HEIGHT * GRID_WIDTH * sizeof(Vector2));

    int n = 0;
    for (int i = 0; i < GRID_WIDTH; i++) {
        for (int j = 0; j < GRID_HEIGHT; j++) {
            Vector2 p = {
                .x = i,
                .y = j
            };
            positions[n] = p;
            n++;
        }
    }

    Grid* grid = grid_init();
    for (int i = 0; i < B_CELL_NUM; i++) {
        if (n < 1)
            break;
        int index = rand() % n;
        Vector2 p = positions[index];
        positions[index] = positions[n - 1];
        positions[n - 1] = p;
        n--;

        grid_insert(grid, create_entity(B_CELL, p));
    }

    for (int i = 0; i < T_CELL_NUM; i++) {
        if (n < 1)
            break;
        int index = rand() % n;
        Vector2 p = positions[index];
        positions[index] = positions[n - 1];
        positions[n - 1] = p;
        n--;

        grid_insert(grid, create_entity(T_CELL, p));
    }

    for (int i = 0; i < AG_MOLECULE_NUM; i++) {
        if (n < 1)
            break;
        int index = rand() % n;
        Vector2 p = positions[index];
        positions[index] = positions[n - 1];
        positions[n - 1] = p;
        n--;

        grid_insert(grid, create_entity(AG_MOLECOLE, p));
    }
    memfree(positions);
    return grid;
}

void print_grid(Grid* grid) {
    for (int i = 0; i < GRID_WIDTH; i++) {
        printf("[ ");
        for (int j = 0; j < GRID_HEIGHT; j++) {
            Vector2 position = {
                .x = i,
                .y = j
            };

            Entity* entity = grid_get(grid, position);
            if (entity != NULL) {
                switch (entity->type) {
                    case B_CELL:
                        printf("B ");
                        continue;
                    case T_CELL:
                        printf("T ");
                        continue;
                    case AG_MOLECOLE:
                        printf("* ");
                        continue;
                    case AB_MOLECOLE:
                        printf("$ ");
                        continue;
                    default:
                        continue;
                }
            }
            printf("- ");
        }
        printf("]\n");
    }
    printf("\n");
}

void print_element_count(Grid* grid) {
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        printf("%s elements: %d\n", type_to_string(i), grid->lists[i].size);
    }
    printf("\n");
}

void print_element_pos(Grid* grid) {
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        EntityBlock** current = &grid->lists[i].first;
        while (*current != NULL) {
            EntityBlock** next = &(*current)->next;
            printf("Type: %s - Position: [X = %f; Y = %f]\n", type_to_string(i), (*current)->entity->position.x, (*current)->entity->position.y);
            current = next;
        }
    }
    printf("\n");
}

void check_grid(Grid* grid) {
    int count = 0;
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        int list_count = 0;
        EntityBlock** current = &grid->lists[i].first;
        while (*current != NULL) {
            assert((*current)->entity != NULL);
            assert((*current)->entity->type == i);
            assert(is_pos_valid((*current)->entity->position));

            Entity* occupant = grid_get(grid, (*current)->entity->position);
            assert(occupant != NULL);
            if (occupant != (*current)->entity) {
                printf("Occupant Entity => Type: %s - Position: [X = %f; Y = %f]\n", type_to_string(i), (*current)->entity->position.x, (*current)->entity->position.y);
                printf("Occupant Entity => Type: %s - Position: [X = %f; Y = %f]\n", type_to_string(occupant->type), occupant->position.x, occupant->position.y);
            }
            assert(occupant == (*current)->entity);

            current = &(*current)->next;
            count++;
            list_count++;
        }
        assert(grid->lists[i].size == list_count);
    }
    assert(grid->total_size == count);
}

void debug_grid(Grid* grid, int step) {
    #ifdef DEBUG
        print_grid(grid);
        print_element_count(grid);
        #ifdef DEBUG_POSITIONS
            print_element_pos(grid);
        #endif
        printf("Time Step %d\n\n", step + 1);
    #endif

    #ifdef ASSERT
        check_grid(grid);
    #endif
} 

int main(int argc, char *argv[]) {

    srand(time(NULL));

    struct timeval start, end;
    gettimeofday(&start, NULL);

    Grid* grid = generate_grid();
    debug_grid(grid, -1);

    for (int i = 0; i < TIMESTEPS; i++) {
        time_step(grid);
        debug_grid(grid, i);
        #ifdef TERMINATE_ON_ZERO_AG
            if (grid->lists[AG_MOLECOLE].size == 0)
                break;
        #endif
    }

    gettimeofday(&end, NULL);

    int elapsed = (int)((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000);
    printf("Computed %d timesteps in a %dx%d grid. Elapsed time: %d ms\n", TIMESTEPS, GRID_WIDTH, GRID_HEIGHT, elapsed);

    grid_free(grid);
    return 0;
}
