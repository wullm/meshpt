/*******************************************************************************
 * This file is part of MeshPT.
 * Copyright (c) 2020 Willem Elbers (whe@willemelbers.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#ifndef SPATIAL_CACHE_H
#define SPATIAL_CACHE_H

#include "indices.h"

struct spatial_factor_table {
    double *cache;       /* Array of N^3 * N_cache doubles */
    int *read_outs;      /* Counter for the number of reads per grid */
    int *cache_location; /* Index in the cache (-1 if not present) */
    double boxlen;       /* Spatial grid size (boxlen^3) */
    int N;               /* Spatial grid size (N^3) */
    int N_f;             /* Maximum number of spatial factors  */
    int N_cache;         /* Number of spatial factors kept in cache */
    int counter;         /* Number of grids that have been generated */
    int unique;          /* A unique number to prevent filename clashes */
};

int init_spatial_factor_table(struct spatial_factor_table *t, double boxlen,
                              int N, int N_f, int N_cache, int unique);
int free_spatial_factor_table(struct spatial_factor_table *t);

int store_grid(struct spatial_factor_table *t, double *box, int index);
int fetch_grid(struct spatial_factor_table *t, double *box, int index);
int disk_store_grid(int N, double boxlen, double *box, char *fname);
int disk_fetch_grid(double *box, char *fname);

#endif
