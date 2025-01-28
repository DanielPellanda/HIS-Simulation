CFLAGS=-std=c99 -Wall -Wpedantic -Werror
FILE=his-simulation

all: 				${FILE}

cuda:				cuda-${FILE}

omp:				omp-${FILE}
					
${FILE}:			main.c simulation.o memory.o grid.o math.o entity.o memory.o pbPlots.o supportLib.o
					gcc ${CFLAGS} main.c simulation.o memory.o grid.o math.o entity.o pbPlots.o supportLib.o -o ${FILE} -lm
					rm -f *.o *.png
					
cuda-${FILE}:		cuda/cuda-main.cu cuda/cuda-memory.cu cuda/cuda-math.cu cuda/cuda-entity.cu cuda/cuda-grid.cu cuda/cuda-simulation.cu memory.o pbPlots.o supportLib.o
					nvcc cuda/cuda-main.cu cuda/cuda-memory.cu cuda/cuda-math.cu cuda/cuda-entity.cu cuda/cuda-grid.cu cuda/cuda-simulation.cu memory.o pbPlots.o supportLib.o -o cuda-${FILE} -lm
					rm -f *.o *.png

omp-${FILE}:		main.c simulation.o memory.o grid.o math.o entity.o memory.o pbPlots.o supportLib.o
					gcc ${CFLAGS} -fopenmp -DOPEN_MP main.c simulation.o memory.o grid.o math.o entity.o pbPlots.o supportLib.o -o omp-${FILE} -lm
					rm -f *.o *.png

simulation.o:		simulation.c simulation.h memory.c grid.c math.c entity.c
					gcc ${CFLAGS} -c simulation.c memory.c grid.c math.c entity.c -lm
					
grid.o:				grid.c grid.h entity.c memory.c math.c
					gcc ${CFLAGS} -c grid.c entity.c memory.c math.c -lm

memory.o:			memory.c memory.h
					gcc ${CFLAGS} -c memory.c -lm
					
entity.o:			entity.c entity.h math.c memory.c
					gcc ${CFLAGS} -c entity.c math.c memory.c -lm
					
math.o:				math.c math.h memory.c
					gcc ${CFLAGS} -c math.c memory.c -lm

pbPlots.o:			lib/pbPlots.c lib/pbPlots.h
					gcc -std=c99 -O3 -march=native -c lib/pbPlots.c
					
supportLib.o:		lib/supportLib.c lib/supportLib.h
					gcc -std=c99 -O3 -march=native -c lib/supportLib.c

.PHONY:				clean cuda omp

clean:
					rm -f ${FILE} cuda-${FILE} *.o *.png