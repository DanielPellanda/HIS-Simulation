#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include "lib/pbPlots.h"
#include "lib/supportLib.h"
#include "simulation.h"
#include "memory.h"

void time_step(Grid* grid) {
    /* Generate the process order of each entity. */
    Entity** entity_list = generate_order(grid);
    if (entity_list == NULL)
        return;

    int size = grid->total_size;
    /* Check for interactions first. */
    for (int i = 0; i < size; i++) {
        if (entity_list[i] == NULL)
            continue;
        if (entity_list[i]->to_be_removed)
            continue;
        scan_interactions(grid, entity_list[i]);
    }
    /* Process their movement. */
    for (int i = 0; i < size; i++) {
        if (entity_list[i] == NULL)
            continue;
        if (entity_list[i]->to_be_removed) {
            grid_remove_type(grid, entity_list[i]->position, entity_list[i]->type);
            continue;
        }
        diffuse_entity(grid, entity_list[i]);
    }
    memfree(entity_list);
}

Entity** generate_order(Grid* grid) {
    Entity** array = grid_get_all(grid);
    if (array == NULL)
        return NULL;

    if (grid->total_size > 0) {
        /* Shuffle the array. */
        for (int i = 0; i < grid->total_size-1; i++) {
            assert(array[i] != NULL);
            int j = i + rand() / (RAND_MAX / (grid->total_size - i) + 1);
            Entity* tmp = array[j];
            array[j] = array[i];
            array[i] = tmp;
        }
    }
    return array;
}

void plot_graph(Grid* grid, char* name) {
    ScatterPlotSettings* settings = GetDefaultScatterPlotSettings();
	settings->width = 600;
	settings->height = 400;
	settings->autoBoundaries = false;
	settings->autoPadding = true;
    settings->title = L"Entity Positions";
    settings->titleLength = wcslen(settings->title);
    settings->xAxisAuto = true;
    settings->xLabel = L"X";
    settings->xLabelLength = wcslen(settings->xLabel);
    settings->yAxisAuto = true;
    settings->yLabel = L"Y";
    settings->yLabelLength = wcslen(settings->yLabel);
    settings->xMax = GRID_WIDTH - 1.0;
    settings->yMax = GRID_HEIGHT - 1.0;
    settings->xMin = 0.0;
    settings->yMin = 0.0;
    settings->showGrid = false;

    double** xs = (double**)memalloc(grid->total_size*sizeof(double*));
    double** ys = (double**)memalloc(grid->total_size*sizeof(double*));
	ScatterPlotSeries** series = (ScatterPlotSeries**)memalloc(grid->total_size*sizeof(ScatterPlotSeries*));
    
    int index = 0;
    for (int i = 0; i < MAX_ENTITYTYPE; i++) {
        if (grid->lists[i].size == 0)
            continue;

        int count = 0;
        size_t size = grid->lists[i].size * sizeof(double);

        xs[index] = (double*)memalloc(size);
        ys[index] = (double*)memalloc(size);

        EntityBlock** current = &grid->lists[i].first;
        while (*current != NULL) {
            xs[index][count] = (*current)->entity->position.x;
            ys[index][count] = (*current)->entity->position.y;
            current = &(*current)->next;
            count++;
        }
        assert(count == grid->lists[i].size);

        series[index] = GetDefaultScatterPlotSeriesSettings();
        series[index]->xs = xs[index];
        series[index]->xsLength = grid->lists[i].size;
	    series[index]->ys = ys[index];
	    series[index]->ysLength = grid->lists[i].size;
        series[index]->linearInterpolation = false;
        series[index]->pointType = L"dots";
        series[index]->pointTypeLength = wcslen(series[index]->pointType);
        switch (i) {
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
        index++;
    }
    settings->scatterPlotSeries = series;
	settings->scatterPlotSeriesLength = index;

    StringReference error = {
        .stringLength = 0
    };
    RGBABitmapImageReference* canvas = CreateRGBABitmapImageReference();
    bool success = DrawScatterPlotFromSettings(canvas, settings, &error);

    if (!success) {
        if (error.stringLength > 0) {
            printf("Graph Error: %ls\n", error.string);
        }
        return;
    }
    
    size_t length;
	double *pngdata = ConvertToPNG(&length, canvas->image);
	WriteToFile(pngdata, length, name);
	DeleteImage(canvas->image);

    for (int i = 0; i < index; i++) {
        memfree(xs[i]);
        memfree(ys[i]);
    }
    memfree(series);
    memfree(xs);
    memfree(ys);
}