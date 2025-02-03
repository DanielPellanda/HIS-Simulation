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
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "memory.h"
#include "simulation.h"

void read_parameters(int argc, char *argv[]) {
    if (argc > 1) {
        int steps = atoi(argv[1]);
        if (steps > 0) {
            TIMESTEPS = steps;
        }
    }
    if (argc > 2) {
        int size = atoi(argv[2]);
        if (size > 0) {
            GRID_SIZE = size;
        }
    }
    if (argc > 3) {
        int antigens = atoi(argv[3]);
        if (antigens >= 0) {
            AG_MOLECULE_NUM = antigens;
        }
    }
    if (argc > 4) {
        int b_cells = atoi(argv[4]);
        if (b_cells >= 0) {
            B_CELL_NUM = b_cells;
        }
    }
    if (argc > 5) {
        int t_cells = atoi(argv[5]);
        if (t_cells >= 0) {
            T_CELL_NUM = t_cells;
        }
    }
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    read_parameters(argc, argv);

    char string[64];
    struct timeval start, end;
    gettimeofday(&start, NULL);

    Grid* grid = generate_grid();
    sprintf(string, "his-0-timestep.png");
    plot_graph(grid, string);

    for (int i = 0; i < TIMESTEPS; i++) {
        time_step(grid);
        debug_grid(grid, i);
        #ifdef REINSERT_AG
            if (i % (TIMESTEPS / 2) == 0 && i > 0) {
                reinsert_antigens(grid);
            }
        #endif
        if (i % (TIMESTEPS / 4) == 0) {
            sprintf(string, "his-%d-timestep.png", i);
            plot_graph(grid, string);
        }
        printf("Timestep %d: B-Cells=%d - T-Cells=%d - Antigens=%d - Antibodies=%d\n", 
            i, grid->lists[B_CELL].size, grid->lists[T_CELL].size, grid->lists[AG_MOLECOLE].size, grid->lists[AB_MOLECOLE].size);
        #ifdef TERMINATE_ON_ZERO_AG
            if (grid->lists[AG_MOLECOLE].size == 0)
                break;
        #endif
    }

    sprintf(string, "his-%d-timestep.png", TIMESTEPS);
    plot_graph(grid, string);
    gettimeofday(&end, NULL);

    printf("Timestep %d: B-Cells=%d - T-Cells=%d - Antigens=%d - Antibodies=%d\n", 
        TIMESTEPS, grid->lists[B_CELL].size, grid->lists[T_CELL].size, grid->lists[AG_MOLECOLE].size, grid->lists[AB_MOLECOLE].size);

    int elapsed = (int)((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000);
    printf("Computed %d timesteps in a %dx%d grid. Elapsed time: %d ms\n", TIMESTEPS, GRID_SIZE, GRID_SIZE, elapsed);

    grid_free(grid);
    return 0;
}
