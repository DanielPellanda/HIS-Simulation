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

#ifndef CUDA_SIM_H
#define CUDA_SIM_H

#include <math.h>
#include "cuda-grid.h"
#include "cuda-entity.h"
#include "cuda-math.h"

/* Print on output a visual representation of the grid
   together with the number of each type of entity after each time step. */ 
//#define DEBUG

/* Print the position of each entity after each time step.*/
//#define DEBUG_POSITIONS

/* Run a check on the validity of the grid after every time step. */
//#define ASSERT

/* Terminate the simulation immediately if no antigens can be found. */
//#define TERMINATE_ON_ZERO_AG

/* Reinsert all antigens after the completion of 50% of the timesteps. */
#define REINSERT_AG

#define BLKDIM 32

#define DEFAULT_TIMESTEPS 20000
#define DEFAULT_B_CELLS 20
#define DEFAULT_T_CELLS 20
#define DEFAULT_AG_MOLECULES 400

extern int TIMESTEPS;
extern int B_CELL_NUM;
extern int T_CELL_NUM;
extern int AG_MOLECULE_NUM;

/* Processes one full time step of the simulation and updates the grid. */
void time_step(Grid* grid);

__device__ int getthreadindex();

__global__ void kernel_gather_entities(Grid* grid, Entity** gpu_entities);

/* Process entity interactions for every GPU thread. */
__global__ void kernel_process_interactions(Grid* grid, Entity** array, int size);

/* Process entity movement for every GPU thread. */
// __global__ void kernel_diffuse_entity(Grid* grid, Entity** array, int size, curandState* rng);

/* Creates the grid and populates it with entities. */
Grid* generate_grid();

/* Repopulates the grid with antigens. */
void reinsert_antigens(Grid* grid);

/* Populates the grid with "n" entities of the type passed as parameter.
   The last two parameters are the array of free positions and its length. */
void populate_grid(Grid* grid, EntityType type, int n, Vector2* positions, int* length);

/* Creates a PNG image graph with all the entities of the grid */
void plot_graph(Grid* grid, char* name);

void debug_grid(Grid* grid, int step);

void check_grid(Grid* grid);

void print_grid(Grid* grid);

void print_element_count(Grid* grid);

void print_element_pos(Grid* grid);

#endif