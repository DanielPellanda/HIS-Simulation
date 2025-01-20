#ifndef ENTITY_H
#define ENTITY_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "math.h"

#define POISSON_MUTATION

#define RECEPTOR_SIZE 2
#define MUTATION_CHANCE 0.5
#define AFFINITY_MIN 5
#define BIND_CHANCE 0.5
#define AB_CREATED_PER_CELL 4
#define PROXIMITY_DIST 5

/* The type of an entity. */
typedef enum 
{
    // Cells
    B_CELL, // Lymphocyte B
    T_CELL, // Lymphocyte T

    // Molecules
    AG_MOLECOLE, // Antigen
    AB_MOLECOLE, // Antibody

    MAX_ENTITYTYPE
}
EntityType;

/* Defines the current state of the entity. 
   Only used for cell type entities (B_CELL and T_CELL), 
   molecules are always in the active state. */
typedef enum
{
    CS_ACTIVE,
    CS_INTERNALIZED, // Inactive
    CS_STIMULATED, // Duplicating

    MAX_CELLSTATE 
}
CellState;

/* Represents a Cell or Molecule inside our grid. */
typedef struct
{
    EntityType type;
    CellState status;
    Vector2 velocity;
    Vector2 position;
    unsigned char receptor[RECEPTOR_SIZE];

    bool has_interacted;
    bool to_be_removed;
} 
Entity;

/* Calculates the probability for two receptors to bind. */
double affinity_potential(unsigned char receptor1, unsigned char receptor2);

/* Determines whether two different entities can bind
   based on the affinity potential of their receptors. */
bool can_entities_bind(Entity* entity, Entity* entity2, bool specific);

/* Mutates the receptor of an entity. */
void hypermutation(Entity* entity);

/* Creates a new entity with the type and position specified as parameters. */
Entity* create_entity(EntityType type, Vector2 position);

/* Returns a string representing the entity type specified. */
char* type_to_string(EntityType type);

#endif