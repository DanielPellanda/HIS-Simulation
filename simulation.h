#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "grid.h"

/* Run a check on the validity of the grid after every time step. */
//#define ASSERT 

/* Print on output a visual representation of the grid
   together with the number of each type of entity after each time step. */ 
//#define DEBUG

/* Print the position of each entity after each time step.*/
//#define DEBUG_POSITIONS

/* Terminate the simulation immediately if no antigens can be found. */
//#define TERMINATE_ON_ZERO_AG

#define TIMESTEPS 2000
#define B_CELL_NUM 5
#define T_CELL_NUM 5
#define AG_MOLECULE_NUM 500

/* Processes one full time step of the simulation 
   and updates the grid. */
void time_step(Grid* grid);

/* Generates the order of entities that will be processed
   in the time step. */
Entity** generate_order(Grid* grid);