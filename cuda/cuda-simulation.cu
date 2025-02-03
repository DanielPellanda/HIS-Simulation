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
#include <math.h>
#include <assert.h>
#include "cuda-simulation.h"
#include "cuda-memory.h"
#include "cuda-grid.h"
#include "cuda-math.h"
#include "cuda-memory.h"
#include "lib/pbPlots.hpp"
#include "lib/supportLib.hpp"

using namespace std;

int TIMESTEPS = DEFAULT_TIMESTEPS;
int B_CELL_NUM = DEFAULT_B_CELLS;
int T_CELL_NUM = DEFAULT_T_CELLS;
int AG_MOLECULE_NUM = DEFAULT_AG_MOLECULES;

void time_step(Grid* grid) {
    int count = 0;
    int cpu_count = 0;
    int gpu_count = 0;
    int size = grid->total_size;

    Entity** gpu_entities;                                               // entities with interactions done by the GPU
    Entity** entity_list = (Entity**)memalloc(size * sizeof(Entity*));   // all entities
    int* cpu_indexes = (int*)memalloc(size * sizeof(int));               // indexes from entity_list for all cpu_entities
    cudaAlloc((void**)&gpu_entities, size * sizeof(Entity*));

    Grid* d_grid = grid_copy_to_device(grid);

    /* Gather all entities from the grid. */
    kernel_gather_entities<<<1, 1>>>(d_grid, gpu_entities);
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        EntityBlock** current = &grid->lists[i].first;
        while (*current != NULL) {
            Entity* entity = (*current)->entity;
            assert(entity != NULL);
            entity->has_interacted = false;
            entity_list[count] = entity;
            if (!entity->to_be_removed) {
                switch (entity->type) {
                    case B_CELL:
                        if (entity->status == CS_STIMULATED) { // duplication handled with the CPU
                            cpu_indexes[cpu_count] = count;
                            cpu_count++;
                            break;
                        }
                    case T_CELL:
                    case AG_MOLECOLE:
                    case AB_MOLECOLE: // the rest is handled by the GPU
                        gpu_count++;
                        break;
                    default:
                        break;
                }
            }
            current = &(*current)->next;
            count++;
        }
    }
    cudaCheckError(); // synchronize gpu threads
    assert(count == grid->total_size);

    /* Interactions of duplicating B Cells must be done by the CPU. 
       The rest are handled by the GPU. */
    kernel_process_interactions<<<(gpu_count + BLKDIM-1)/BLKDIM, BLKDIM>>>(d_grid, gpu_entities, gpu_count);
    for (int i = 0; i < cpu_count; i++) {
        Entity* bcell = entity_list[cpu_indexes[i]];
        if (bcell == NULL)
            continue;

        bcell->status = CS_INTERNALIZED;
        bcell->has_interacted = true;
        hypermutation(bcell);
        duplicate_entity(grid, bcell);
        generate_antibodies(grid, bcell->position);
    }
    cudaCheckError(); // synchronize gpu with cpu

    int cpu_i = 0;
    int gpu_i = 0;
    for (int i = 0; i < size; i++) {
        if (entity_list[i] == NULL)
            continue;
        if (cpu_indexes[cpu_i] != i) {
            Entity** gpu_pointer = (Entity**)memalloc(sizeof(Entity*));
            cudaCopy(gpu_pointer, &gpu_entities[gpu_i], sizeof(Entity*), cudaMemcpyDeviceToHost);
            cudaCopy(entity_list[i], *gpu_pointer, sizeof(Entity), cudaMemcpyDeviceToHost);
            memfree(gpu_pointer);
            gpu_i++;
        }
        else {
            cpu_i++;
        }
        if (entity_list[i]->to_be_removed) {
            grid_remove_type(grid, entity_list[i]->position, entity_list[i]->type);
            continue;
        }
        entity_list[i]->seed = randdouble();
        /* Process the entity movement. */
        diffuse_entity(grid, entity_list[i]);
    }
    memfree(entity_list);
    memfree(cpu_indexes);
    grid_free_device(d_grid);
    assert(cpu_i == cpu_count);
    assert(gpu_i == gpu_count);
}

__device__ int getthreadindex() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void kernel_gather_entities(Grid* grid, Entity** gpu_entities) {
    int gpu_count = 0;
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        EntityBlock** current = &grid->lists[i].first;
        while (*current != NULL) {
            Entity* entity = (*current)->entity;
            entity->has_interacted = false;
            if (!entity->to_be_removed) {
                switch (entity->type) {
                    case B_CELL:
                        if (entity->status == CS_STIMULATED) // duplication handled with the CPU
                            break;
                    case T_CELL:
                    case AG_MOLECOLE:
                    case AB_MOLECOLE: // the rest is handled by the GPU
                        gpu_entities[gpu_count] = entity;
                        gpu_count++;
                        break;
                    default:
                        break;
                }
            }
            current = &(*current)->next;
        }
    }
}

__global__ void kernel_process_interactions(Grid* grid, Entity** array, int size) {
    int threadidx = getthreadindex();
    if (threadidx >= size)
        return;
    if (array[threadidx] == NULL)
        return;
    if (array[threadidx]->to_be_removed)
        return;

    process_interactions(grid, array[threadidx]);
}


// __global__ void kernel_diffuse_entity(Grid* grid, Entity** array, int size, curandState* rng) {
//     int threadidx = getthreadindex();
//     if (threadidx >= size)
//         return;
//     if (array[threadidx] == NULL)
//         return;
//     if (array[threadidx]->to_be_removed)
//         return;
        
//     diffuse_entity(grid, array[threadidx], rng);
// }

Grid* generate_grid() {
    int n = 0;
    Vector2* positions = (Vector2*)memalloc(GRID_SIZE * GRID_SIZE * sizeof(Vector2));

    /* Gather all free positions. */
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            Vector2 p = {
                .x = (float)i,
                .y = (float)j
            };
            positions[n] = p;
            n++;
        }
    }

    Grid* grid = grid_init();

    /* Populate the grid with B cells. */
    populate_grid(grid, B_CELL, B_CELL_NUM, positions, &n);

    /* Populate the grid with T cells. */
    populate_grid(grid, T_CELL, T_CELL_NUM, positions, &n);

    /* Populate the grid with Antigens. */
    populate_grid(grid, AG_MOLECOLE, AG_MOLECULE_NUM, positions, &n);

    memfree(positions);
    return grid;
}

void reinsert_antigens(Grid* grid) {
    int n = 0;
    Vector2* positions = (Vector2*)memalloc(GRID_SIZE * GRID_SIZE * sizeof(Vector2));

    /* Gather all free positions. */
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            Vector2 p = {
                .x = (float)i,
                .y = (float)j
            };

            if (grid_is_pos_free(grid, p)) {
                positions[n] = p;
                n++;
            }
        }
    }

    /* Repopulate the grid with Antigens. */
    populate_grid(grid, AG_MOLECOLE, AG_MOLECULE_NUM, positions, &n);

    memfree(positions);
}

void populate_grid(Grid* grid, EntityType type, int n, Vector2* positions, int* length) {
    for (int i = 0; i < n; i++) {
        if (*length < 1) /* If there are no free positions, stop. */
            break;

        /* Extract a random index from the array of free positions. */
        int index = rand() % *length;
        /* Swap the chosen element at index with the element in the last position
           and decrease the array size. */
        Vector2 p = positions[index];
        positions[index] = positions[*length - 1];
        (*length)--;

        grid_insert(grid, create_entity(type, p));
    }
}

void plot_graph(Grid* grid, char* name) {
    ScatterPlotSettings* settings = GetDefaultScatterPlotSettings();
	settings->width = 600;
	settings->height = 400;
	settings->autoBoundaries = false;
	settings->autoPadding = true;
    settings->title = toVector(L"Humoral Response");
    settings->xAxisAuto = true;
    settings->xLabel = toVector(L"X");
    settings->yAxisAuto = true;
    settings->yLabel = toVector(L"Y");
    settings->xMax = GRID_SIZE - 1.0;
    settings->yMax = GRID_SIZE - 1.0;
    settings->xMin = 0.0;
    settings->yMin = 0.0;
    settings->showGrid = false;

    double** xs = (double**)memalloc(grid->total_size*sizeof(double*));
    double** ys = (double**)memalloc(grid->total_size*sizeof(double*));
    vector<double>** xsv = (vector<double>**)memalloc(grid->total_size*sizeof(vector<double>*));
    vector<double>** ysv = (vector<double>**)memalloc(grid->total_size*sizeof(vector<double>*));
	ScatterPlotSeries** series = (ScatterPlotSeries**)memalloc(grid->total_size*sizeof(ScatterPlotSeries*));
    
    int index = 0;
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        if (grid->lists[i].size == 0)
            continue;

        int count = 0;
        size_t size = grid->lists[i].size * sizeof(double);

        xs[index] = (double*)memalloc(size);
        ys[index] = (double*)memalloc(size);

        EntityBlock** current = &grid->lists[i].first;
        while (*current != NULL) {
            xs[index][count] = (*current)->entity->position.x;
            ys[index][count] = (*current)->entity->position.y;
            current = &(*current)->next;
            count++;
        }
        assert(count == grid->lists[i].size);

        xsv[index] = new vector<double>(xs[index], xs[index]+sizeof(xs[index])*grid->lists[i].size/sizeof(double));
        ysv[index] = new vector<double>(ys[index], ys[index]+sizeof(ys[index])*grid->lists[i].size/sizeof(double));

        series[index] = GetDefaultScatterPlotSeriesSettings();
        series[index]->xs = xsv[index];
	    series[index]->ys = ysv[index];
        series[index]->linearInterpolation = false;
        series[index]->pointType = toVector(L"dots");
        switch ((EntityType)i) {
            case B_CELL:
                series[index]->color = CreateRGBColor(0.0, 1.0, 0.0);     // green
                break;
            case T_CELL:
                series[index]->color = CreateRGBColor(0.0, 1.0, 1.0);     // cyan
                break;
            case AG_MOLECOLE:
                series[index]->color = CreateRGBColor(1.0, 0.0, 0.0);     // red
                break;
            case AB_MOLECOLE:
                series[index]->color = CreateRGBColor(0.0, 0.0, 1.0);     // blue
                break;
            default:
                break;
        }
        settings->scatterPlotSeries->push_back(series[index]);
        index++;
    }

    StringReference* error = CreateStringReferenceLengthValue(0, L' ');
    RGBABitmapImageReference* canvas = CreateRGBABitmapImageReference();
    bool success = DrawScatterPlotFromSettings(canvas, settings, error);

    if (!success) {
        printf("Graph Error: %ls\n", error->string);
        memfree(error);
        return;
    }
    memfree(error);
    
	vector<double>* pngdata = ConvertToPNG(canvas->image);
	WriteToFile(pngdata, name);
	DeleteImage(canvas->image);

    for (int i = 0; i < index; i++) {
        memfree(xs[i]);
        memfree(ys[i]);
        memfree(xsv[i]);
        memfree(ysv[i]);
        memfree(series[i]);
    }
    memfree(series);
    memfree(xs);
    memfree(ys);
    memfree(xsv);
    memfree(ysv);
    memfree(pngdata);
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
            if (occupant == NULL) {
                printf("Missing entity => %p Type: %s - Position: [X=%d, Y=%d]\n", (*current)->entity, type_to_string((EntityType)i), (int)round((*current)->entity->position.x), (int)round((*current)->entity->position.y));
                exit(0);
            }
            //assert(occupant != NULL);
            if (occupant != (*current)->entity) {
                printf("Occupant Entity => %p Type: %s - Position: [X=%d, Y=%d]\n", (*current)->entity, type_to_string((EntityType)i), (int)round((*current)->entity->position.x), (int)round((*current)->entity->position.y));
                printf("Occupant Entity => %p Type: %s - Position: [X=%d, Y=%d]\n", occupant, type_to_string(occupant->type), (int)round(occupant->position.x), (int)round(occupant->position.y));
                exit(0);
            }
            //assert(occupant == (*current)->entity);

            current = &(*current)->next;
            count++;
            list_count++;
        }
        assert(grid->lists[i].size == list_count);
    }
    assert(grid->total_size == count);
}

void print_grid(Grid* grid) {
    for (int i = 0; i < GRID_SIZE; i++) {
        printf("[ ");
        for (int j = 0; j < GRID_SIZE; j++) {
            Vector2 position = {
                .x = (float)i,
                .y = (float)j
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
        printf("%s elements: %d\n", type_to_string((EntityType)i), grid->lists[i].size);
    }
    printf("\n");
}

void print_element_pos(Grid* grid) {
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        EntityBlock** current = &grid->lists[i].first;
        while (*current != NULL) {
            EntityBlock** next = &(*current)->next;
            printf("Type: %s - Position: [X = %f; Y = %f]\n", type_to_string((EntityType)i), (*current)->entity->position.x, (*current)->entity->position.y);
            current = next;
        }
    }
    printf("\n");
}