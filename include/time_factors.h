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

#ifndef TIME_FACTORS_H
#define TIME_FACTORS_H

#include "indices.h"

#define CONTINUITY_EQ 0
#define EULER_EQ      1

enum ic_type {
    EdS = 0, /* Einstein-de Sitter */
    RD = 1,  /* Radiation domination */
};

struct time_factor_table {
    struct coeff_table ct; /* Table of coefficient names */
    double *table;         /* Array of N_t * N_f time factors X_{i,j}(t) */
    double *dtable_dt;     /* Array of N_t * N_f time derivatives X'_{i,j}(t) */
    double *ic_factor;     /* Array of N_f initial conditions for the factors */
    double *time_sampling; /* Vector of length N_t times */
    int N_t;               /* Number of time steps */
    int N_f;               /* Maximum number of time factors  */
    double time_i;         /* The initial time */
    double time_f;         /* The final time */
    int index_Om_11;       /* Index of the first element of the Omega matrix */
};

int init_time_factor_table(struct time_factor_table *t, int N_t, int N_f,
                           double time_i, double time_f);
int free_time_factor_table(struct time_factor_table *t);

/* Find index along the time direction (0 <= u <= 1 between indx and indx+1) */
int find_time_index(const struct time_factor_table *tab, double t, int *index,
                    double *u);

double interp_time_factor(const struct time_factor_table *tab, double t,
                          int index);
double interp_time_derivative(const struct time_factor_table *tab, double t,
                              int index);

int integrate_time_factor(struct time_factor_table *tab, char which_equation,
                          int source_index_1, int source_index_2,
                          int dest_index_1, int dest_index_2, int n);
int compute_time_factor_EdS(struct time_factor_table *tab, char which_equation,
                            int source_index_1, int source_index_2,
                            int dest_index_1, int dest_index_2, int n);
int compute_time_factor_RD(struct time_factor_table *tab, char which_equation,
                           int source_index_1, int source_index_2,
                           int dest_index_1, int dest_index_2, int n);
int generate_time_factors_at_n(struct time_factor_table *tab,
                               struct coeff_table *c, int n, enum ic_type);
int compute_all_derivatives(struct time_factor_table *tab);

#endif
