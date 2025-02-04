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
    kernel_process_interactions<<<(GRID_SIZE*GRID_SIZE+BLKDIM-1)/BLKDIM, BLKDIM>>>(grid);
    cudaCheckError();

    kernel_diffuse_entity<<<(GRID_SIZE*GRID_SIZE+BLKDIM-1)/BLKDIM, BLKDIM>>>(grid);
    cudaCheckError();
}

__device__ int getthreadindex() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void kernel_process_interactions(Grid* grid) {
    int threadidx = getthreadindex();
    if (threadidx >= GRID_SIZE * GRID_SIZE)
        return;
    if (grid->entities[threadidx].type == NONE)
        return;
    if (grid->entities[threadidx].just_created)
        return;
    
    grid->entities[threadidx].seed = device_rand(grid->seed);
    process_interactions(grid, &grid->entities[threadidx]);
    grid->entities[threadidx].has_moved = 0;
}


__global__ void kernel_diffuse_entity(Grid* grid) {
    int threadidx = getthreadindex();
    if (threadidx >= GRID_SIZE * GRID_SIZE)
        return;
    if (grid->entities[threadidx].type == NONE)
        return;
    if (grid->entities[threadidx].just_created) {
        grid->entities[threadidx].just_created = 0;
        grid->entities[threadidx].has_interacted = 0;
        return;
    }
    
    grid->entities[threadidx].seed = device_rand(grid->seed);
    if (atomicCAS(&grid->entities[threadidx].has_moved, 0, 1) == 0) {
        diffuse_entity(grid, &grid->entities[threadidx]);
    }
    grid->entities[threadidx].has_interacted = 0;
}

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

        Entity entity = create_entity(type, p, rand());
        entity.has_interacted = 0;
        entity.has_moved = 0;
        entity.just_created = 0;
        grid_insert(grid, entity);
    }
}

void plot_graph(Grid* grid, char* name, int timestep) {
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

    int total_size = 0;
    int size[MAX_ENTITYTYPE];
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        size[i] = 0;
    }

    for (int j = 0; j < GRID_SIZE; j++) {
        for (int k = 0; k < GRID_SIZE; k++) {
            if (grid->entities[j*GRID_SIZE+k].type != NONE) {
                size[grid->entities[j*GRID_SIZE+k].type]++;
                total_size++;
            }
        }
    }
    printf("Timestep %d: B-Cells=%d - T-Cells=%d - Antigens=%d - Antibodies=%d\n", 
        timestep, size[B_CELL], size[T_CELL], size[AG_MOLECOLE], size[AB_MOLECOLE]);

    double** xs = (double**)memalloc(total_size*sizeof(double*));
    double** ys = (double**)memalloc(total_size*sizeof(double*));
    vector<double>** xsv = (vector<double>**)memalloc(total_size*sizeof(vector<double>*));
    vector<double>** ysv = (vector<double>**)memalloc(total_size*sizeof(vector<double>*));
	ScatterPlotSeries** series = (ScatterPlotSeries**)memalloc(total_size*sizeof(ScatterPlotSeries*));
    
    int index = 0;
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        if (size[i] == 0)
            continue;

        xs[index] = (double*)memalloc(size[i] * sizeof(double));
        ys[index] = (double*)memalloc(size[i] * sizeof(double));

        int count = 0;
        for (int j = 0; j < GRID_SIZE; j++) {
            for (int k = 0; k < GRID_SIZE; k++) {
                if (grid->entities[j*GRID_SIZE+k].type == i) {
                    xs[index][count] = grid->entities[j*GRID_SIZE+k].position.x;
                    ys[index][count] = grid->entities[j*GRID_SIZE+k].position.y;
                    count++;
                }
            }
        }
        assert(count == size[i]);

        xsv[index] = new vector<double>(xs[index], xs[index]+sizeof(xs[index])*count/sizeof(double));
        ysv[index] = new vector<double>(ys[index], ys[index]+sizeof(ys[index])*count/sizeof(double));

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
        cerr << "Graph Error: ";
		for(int i = 0; i < error->string->size(); i++){
			wcerr << error->string->at(i);
		}
		cerr << endl;
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
        // print_grid(grid);
        // print_element_count(grid);
        printf("Time Step %d\n\n", step + 1);
    #endif
    #ifdef ASSERT
        check_grid(grid);
    #endif
}

void check_grid(Grid* grid) {
    // int count = 0;
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        // int list_count = 0;
        for (int j = 0; j < GRID_SIZE; j++) {
            for (int k = 0; k < GRID_SIZE; k++) {
                Entity entity = grid->entities[j*GRID_SIZE+k];
                if (entity.type == i) {
                    assert((int)round(entity.position.x) == j && (int)round(entity.position.y) == k);
                    // count++;
                    // list_count++;
                }
            }
        }
        // assert(grid->size[i] == list_count);
    }
    // assert(grid->total_size == count);
}

void print_grid(Grid* grid) {
    for (int i = 0; i < GRID_SIZE; i++) {
        printf("[ ");
        for (int j = 0; j < GRID_SIZE; j++) {
            Vector2 position = {
                .x = (float)i,
                .y = (float)j
            };

            Entity entity = grid_get(grid, position);
            if (entity.type != NONE) {
                switch (entity.type) {
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

// void print_element_count(Grid* grid) {
//     for (int i = 0; i < MAX_ENTITYTYPE; i++) {
//         printf("%s elements: %d\n", type_to_string(i), grid->size[i]);
//     }
//     printf("\n");
// }