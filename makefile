FILE=his-simulation

all: 				${FILE}

${FILE}:			main.c simulation.c memory.c math.c entity.c grid.c pbPlots.o supportLib.o
					gcc -std=c99 -Wall -Wpedantic -Werror main.c simulation.c memory.c math.c entity.c grid.c pbPlots.o supportLib.o -o ${FILE} -lm
					rm -f pbPlots.o supportLib.o

pbPlots.o:			lib/pbPlots.c
					gcc -std=c99 -O3 -march=native -c lib/pbPlots.c
					
supportLib.o:		lib/pbPlots.c
					gcc -std=c99 -O3 -march=native -c lib/supportLib.c

.PHONY:				clean

clean:
					rm -f ${FILE} *.png