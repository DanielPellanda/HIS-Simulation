FILE=his-simulation
CFLAGS=-std=c99 -Wall -Wpedantic -Werror

all: 				${FILE}

${FILE}:			main.c simulation.c memory.c math.c entity.c grid.c
					gcc ${CFLAGS} -o ${FILE} main.c simulation.c memory.c math.c entity.c grid.c -lm

.PHONY:				clean

clean:
					rm -f ${FILE} 