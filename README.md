# HIS Simulation in C/Cuda C++
This program runs a small simulation of the Human Immune System, showing the progress of the humoral response after the reaction with an antigen.

The environment is represented as a grid where each cell can be occupied by an entity (cells or molecoles). Entities inside the grid are able to move from one cell to the other and interact with other entities (binding). 
The way each entity will interact is determined by their type and status, entities may be added or removed in the process. To regolate how many actions each entity can perform at the time, the system uses the concept of timestep: during each timestep, every entity can interact and move inside the grid exactly once.
The simulation ends after executing the specified number of timesteps.

The program periodically plots all entities of the system in a grid (using [pbPlots](https://github.com/InductiveComputerScience/pbPlots)) and renders them into png images that can be found in the same folder of the executable.

This repo comes with two builds of the program: a serial version in C and a parallel version using CUDA.

# Serial build

In the serial build, the program is run exclusively by a single CPU thread and therefore execution times will be much slower than the parallel build.

To compile use the following command (requires gcc):
 ```
 make
 ```
 
To run use this:
```
./his-simulation <number of timesteps> <grid size> <number of antigens> <number of B cells> <number of T cells>
```

# Parallel build

With the parallel build, the program will make use of the GPU to run the heaviest sections, allowing significant improvements in execution times.
Unfortunately it's not possible to specify the grid size for this build.

To compile use this command (requires nvcc):
 ```
 make cuda
 ```
 
To run use this:
```
./cuda-his-simulation <number of timesteps> <number of antigens> <number of B cells> <number of T cells>
```