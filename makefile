CFLAGS=-std=c99 -Wall -Wpedantic -Werror
FILE=his-simulation

all: 						${FILE}
		
cuda:						cuda-${FILE}
		
omp:						omp-${FILE}
							
${FILE}:					main.c simulation.c memory.c grid.c math.c entity.c c/pbPlots.o c/supportLib.o
							gcc ${CFLAGS} main.c simulation.c memory.c grid.c math.c entity.c pbPlots.o supportLib.o -o ${FILE} -lm
							rm -f *.o *.png
							
omp-${FILE}:				main.c simulation.c memory.c grid.c math.c entity.c c/pbPlots.o c/supportLib.o
							gcc ${CFLAGS} -fopenmp -DOPEN_MP main.c simulation.c memory.c grid.c math.c entity.c pbPlots.o supportLib.o -o omp-${FILE} -lm
							rm -f *.o *.png							

cuda-${FILE}:				cuda/cuda-main.cu cuda/cuda-simulation.o cuda/cuda-math.o cuda/cuda-memory.o cuda/cuda-entity.o cuda/cuda-grid.o cuda/pbPlots.o cuda/supportLib.o
							nvcc cuda/cuda-main.cu cuda-simulation.o cuda-math.o cuda-memory.o cuda-entity.o cuda-grid.o pbPlots.o supportLib.o -o cuda-${FILE} -lm
							rm -f *.o *.png

cuda/cuda-simulation.o:		cuda/cuda-simulation.cu cuda/cuda-simulation.h
							nvcc --relocatable-device-code=true --compile cuda/cuda-simulation.cu -lm
		
cuda/cuda-math.o:			cuda/cuda-math.cu cuda/cuda-math.h
							nvcc --relocatable-device-code=true --compile cuda/cuda-math.cu -lm
							
cuda/cuda-memory.o:			cuda/cuda-memory.cu cuda/cuda-memory.h
							nvcc --relocatable-device-code=true --compile cuda/cuda-memory.cu -lm
							
cuda/cuda-entity.o:			cuda/cuda-entity.cu cuda/cuda-entity.h
							nvcc --relocatable-device-code=true --compile cuda/cuda-entity.cu -lm
							
cuda/cuda-grid.o:			cuda/cuda-grid.cu cuda/cuda-grid.h
							nvcc --relocatable-device-code=true --compile cuda/cuda-grid.cu -lm
		
c/pbPlots.o:				lib/pbPlots.c lib/pbPlots.h
							gcc -std=c99 -O3 -march=native -c lib/pbPlots.c
							
c/supportLib.o:				lib/supportLib.c lib/supportLib.h
							gcc -std=c99 -O3 -march=native -c lib/supportLib.c
							
cuda/pbPlots.o:				cuda/lib/pbPlots.cpp cuda/lib/pbPlots.hpp
							g++ -std=c++98 -O3 -march=native -c cuda/lib/pbPlots.cpp
							
cuda/supportLib.o:			cuda/lib/supportLib.cpp cuda/lib/supportLib.hpp
							g++ -std=c++98 -O3 -march=native -c cuda/lib/supportLib.cpp
		
.PHONY:						clean cuda omp
		
clean:		
							rm -f ${FILE} cuda-${FILE} omp-${FILE} *.o *.png