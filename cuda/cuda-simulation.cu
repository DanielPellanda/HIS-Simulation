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
#include "cuda-simulation.h"
#include "cuda-memory.h"
#include "cuda-grid.h"
#include "cuda-math.h"

extern "C" {
    #include "../lib/pbPlots.h"
    #include "../lib/supportLib.h"
    #include "../memory.h"
}

int TIMESTEPS = DEFAULT_TIMESTEPS;
int B_CELL_NUM = DEFAULT_B_CELLS;
int T_CELL_NUM = DEFAULT_T_CELLS;
int AG_MOLECULE_NUM = DEFAULT_AG_MOLECULES;

void time_step(Grid* grid) {
    int count = 0;
    int gpu_count = 0;
    int cpu_count = 0;
    int size = grid->total_size;

    /* Initialize seed for random number generator. */
    curandState* rng;
    cudaAlloc((void**)&rng, sizeof(curandState));
    kernel_init_rng<<<(size + BLKDIM-1)/BLKDIM>>>(rng);
    Grid* d_grid = grid_copy_to_device(grid);

    Entity** gpu_entities;                                               // entities with interactions done by the GPU
    Entity** entity_list = (Entity**)memalloc(size * sizeof(Entity*));   // all entities
    Entity** cpu_entities = (Entity**)memalloc(size * sizeof(Entity*));  // entities with interactions done by the CPU
    int* gpu_indexes = (int*)memalloc(size * sizeof(int));               // indexes from entity_list for all gpu_entities
    cudaAlloc((void**)&gpu_entities, size * sizeof(Entity*));

    /* Gather all entities from the grid. */
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        EntityBlock** current = &grid->lists[i].first;
        while (*current != NULL) {
            Entity* entity = (*current)->entity;
            assert(entity != NULL);
            entity->has_interacted = false;
            array[count] = entity;
            if (!entity->to_be_removed) {
                switch (entity->type) {
                    case B_CELL:
                        if (entity->status == CS_STIMULATED) { // duplication handled with the CPU
                            cpu_entities[cpu_count] = entity;
                            cpu_count++;
                            break;
                        }
                    case T_CELL:
                    case AG_MOLECOLE:
                    case AB_MOLECOLE: // the rest is handled by the GPU
                        // fill the device grid with pointers referencing device memory
                        set_device_pointer_with_entity<<<1, 1>>>(&gpu_entities[gpu_count], d_grid, entity->position, entity->type);
                        gpu_indexes[gpu_count] = count;
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
    cudaCheckError(); // synchronize gpu with cpu
    assert(count == grid->total_size);

    /* Interactions of duplicating B Cells must be done by the CPU. 
       The rest are handled by the GPU. */
    kernel_scan_interactions<<<(gpu_count + BLKDIM-1)/BLKDIM>>>(d_grid, gpu_entities, gpu_count, BLKDIM);
    for (int i = 0; i < cpu_count; i++) {
        if (cpu_entities[i] == NULL)
            continue;
        if (cpu_entities[i]->to_be_removed)
            continue;
        scan_interactions(grid, cpu_entities[i]);
    }
    cudaCheckError(); // synchronize gpu with cpu

    /* Copy all entity edits done by in device memory back into host memory. */
    for (int i = 0; i < gpu_count; i++) {
        if (entity_list[gpu_indexes[i]] == NULL)
            continue;
        if (entity_list[gpu_indexes[i]]->to_be_removed)
            continue;
        cudaCopy(entity_list[gpu_indexes[i]], gpu_entities[i], sizeof(Entity), cudaMemcpyDeviceToHost);
    }
    memfree(cpu_entities);
    memfree(gpu_indexes);
    cudaFree(gpu_entities);
    grid_free_device(d_grid);

    /* Copy the new grid into device memory. */
    d_grid = grid_copy_to_device(grid);

    /* Create a clone of entity_list into device memory and fill it with device memory addresses. */
    Entity** d_entity_list;
    cudaAlloc((void**)&d_entity_list, size * sizeof(Entity*));
    for (int i = 0; i < size; i++) {
        set_device_pointer_with_entity<<<1, 1>>>(&d_entity_list[i], d_grid, entity_list[i]->position, entity_list[i]->type);
    }
    cudaCheckError(); // synchronize gpu threads

    /* Process the entity movement. */
    kernel_diffuse_entity<<<(size + BLKDIM-1)/BLKDIM>>>(d_grid, d_entity_list, size, BLKDIM, rng);
    cudaCheckError(); // synchronize gpu with cpu

    /* Remove all entities marked to be deleted and copy in host memory all entities modified by the GPU. */
    for (int i = 0; i < size; i++) {
        if (entity_list[i] == NULL)
            continue;
        if (entity_list[i]->to_be_removed) {
            grid_remove_type(grid, entity_list[i]->position, entity_list[i]->type);
            continue;
        }
        cudaCopy(entity_list[i], d_entity_list[i], sizeof(Entity), cudaMemcpyDeviceToHost);
    }
    
    memfree(entity_list);
    cudaFree(d_entity_list);
    cudaFree(rng);
    grid_free_device(d_grid);
}

__global__ void kernel_init_rng(curandState* rng) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(clock(), 1, 1, &rng[index]);
}

__global__ void kernel_scan_interactions(Grid* grid, Entity** array, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;
    if (array[index] == NULL)
        return;
    if (array[index]->to_be_removed)
        return;
    
    scan_interactions(grid, array[index]);
}

__global__ void kernel_diffuse_entity(Grid* grid, Entity** array, int size, curandState* rng) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;
    if (array[index] == NULL)
        return;
    if (array[index]->to_be_removed)
        return;
    
    diffuse_entity(grid, array[index], index, rng);
}

Grid* generate_grid() {
    int n = 0;
    Vector2* positions = (Vector2*)memalloc(GRID_SIZE * GRID_SIZE * sizeof(Vector2));

    /* Gather all free positions. */
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            Vector2 p = {
                .x = i,
                .y = j
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
                .x = i,
                .y = j
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
    settings->title = L"Humoral Response";
    settings->titleLength = wcslen(settings->title);
    settings->xAxisAuto = true;
    settings->xLabel = L"X";
    settings->xLabelLength = wcslen(settings->xLabel);
    settings->yAxisAuto = true;
    settings->yLabel = L"Y";
    settings->yLabelLength = wcslen(settings->yLabel);
    settings->xMax = GRID_SIZE - 1.0;
    settings->yMax = GRID_SIZE - 1.0;
    settings->xMin = 0.0;
    settings->yMin = 0.0;
    settings->showGrid = false;

    double** xs = (double**)memalloc(grid->total_size*sizeof(double*));
    double** ys = (double**)memalloc(grid->total_size*sizeof(double*));
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

        series[index] = GetDefaultScatterPlotSeriesSettings();
        series[index]->xs = xs[index];
        series[index]->xsLength = grid->lists[i].size;
	    series[index]->ys = ys[index];
	    series[index]->ysLength = grid->lists[i].size;
        series[index]->linearInterpolation = false;
        series[index]->pointType = L"dots";
        series[index]->pointTypeLength = wcslen(series[index]->pointType);
        switch (i) {
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
        index++;
    }
    settings->scatterPlotSeries = series;
	settings->scatterPlotSeriesLength = index;

    StringReference error = {
        .stringLength = 0
    };
    RGBABitmapImageReference* canvas = CreateRGBABitmapImageReference();
    bool success = DrawScatterPlotFromSettings(canvas, settings, &error);

    if (!success) {
        if (error.stringLength > 0) {
            printf("Graph Error: %ls\n", error.string);
        }
        return;
    }
    
    size_t length;
	double *pngdata = ConvertToPNG(&length, canvas->image);
	WriteToFile(pngdata, length, name);
	DeleteImage(canvas->image);

    for (int i = 0; i < index; i++) {
        memfree(xs[i]);
        memfree(ys[i]);
    }
    memfree(series);
    memfree(xs);
    memfree(ys);
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

void print_grid(Grid* grid) {
    for (int i = 0; i < GRID_SIZE; i++) {
        printf("[ ");
        for (int j = 0; j < GRID_SIZE; j++) {
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