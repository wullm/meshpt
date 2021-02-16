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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>

#include "../include/time_factors.h"

int init_time_factor_table(struct time_factor_table *t, int N_t, int N_f,
                           double time_i, double time_f) {
    /* Allocate the arrays */
    t->N_t = N_t;
    t->N_f = N_f;
    t->table = malloc(N_t * N_f * sizeof(double));
    t->dtable_dt = calloc(N_t * N_f, sizeof(double));
    t->ic_factor = malloc(N_f * sizeof(double));
    t->time_sampling = malloc(N_t * sizeof(double));

    if (t->table == NULL || t->dtable_dt == NULL || t->ic_factor == NULL ||
            t->time_sampling == NULL) {
        printf("Error with allocating time factor table.\n");
        return 1;
    }

    /* The initial and final times */
    t->time_i = time_i;
    t->time_f = time_f;

    /* Calculate the time sampling (log-linear sampling) */
    double delta_t = (t->time_f - t->time_i) / (double)N_t;
    for (int i = 0; i < N_t; i++) {
        t->time_sampling[i] = t->time_i + i * delta_t;
    }

    return 0;
}

int free_time_factor_table(struct time_factor_table *t) {
    free(t->table);
    free(t->dtable_dt);
    free(t->ic_factor);
    free(t->time_sampling);
    return 0;
}

/* Find index along the time direction (0 <= u <= 1 between indx and indx+1) */
int find_time_index(const struct time_factor_table *tab, double t, int *index,
                    double *u) {

    /* Number of bins */
    int N_t = tab->N_t;

    if (tab->time_i > tab->time_f) {
        /* Quickly return if we are in the first or last bin */
        if (-t < -tab->time_sampling[0]) {
            *index = 0;
            *u = 0.f;
            return 0;
        } else if (-t >= -tab->time_sampling[N_t - 1]) {
            *index = N_t - 2;
            *u = 1.f;
            return 0;
        }

        /* Find i such that t[i] <= t */
        for (int i = 1; i < N_t; i++) {
            if (-tab->time_sampling[i] >= -t) {
                *index = i - 1;
                break;
            }
        }

        /* Find the bounding values */
        double left = -tab->time_sampling[*index];
        double right = -tab->time_sampling[*index + 1];

        /* Calculate the ratio (X - X_left) / (X_right - X_left) */
        *u = (-t - left) / (right - left);
    } else {
        /* Quickly return if we are in the first or last bin */
        if (t < tab->time_sampling[0]) {
            *index = 0;
            *u = 0.f;
            return 0;
        } else if (t >= tab->time_sampling[N_t - 1]) {
            *index = N_t - 2;
            *u = 1.f;
            return 0;
        }

        /* Find i such that t[i] <= t */
        for (int i = 1; i < N_t; i++) {
            if (tab->time_sampling[i] >= t) {
                *index = i - 1;
                break;
            }
        }

        /* Find the bounding values */
        double left = tab->time_sampling[*index];
        double right = tab->time_sampling[*index + 1];

        /* Calculate the ratio (X - X_left) / (X_right - X_left) */
        *u = (t - left) / (right - left);
    }

    return 0;
}

double interp_time_factor(const struct time_factor_table *tab, double t,
                          int index) {
    /* Find the time index */
    int time_index = -1;
    double u = 0.;
    find_time_index(tab, t, &time_index, &u);

    /* Linearly interpolate */
    double left = tab->table[index * tab->N_t + time_index];
    double right = tab->table[index * tab->N_t + (time_index + 1)];

    return (1 - u) * left + u * right;
}

double interp_time_derivative(const struct time_factor_table *tab, double t,
                              int index) {
    /* Find the time index */
    int time_index = -1;
    double u = 0.;
    find_time_index(tab, t, &time_index, &u);

    /* Linearly interpolate */
    double left = tab->dtable_dt[index * tab->N_t + time_index];
    double right = tab->dtable_dt[index * tab->N_t + (time_index + 1)];

    return (1 - u) * left + u * right;
}

struct fluid_equation_params {
    const struct time_factor_table *t; /* The time factor table */
    char which_equation; /* Sourced by the continuity or Euler equation? */
    int index_1;         /* Index of the first factor of the source term */
    int index_2;         /* Index of the second factor of the source term */
    int n;               /* The order in perturbation theory */
};

int fluid_equation(double eta, const double y[], double f[], void *pars) {
    struct fluid_equation_params *fep = (struct fluid_equation_params *)pars;
    const struct time_factor_table *t = fep->t;

    /* Interpolate the two factors in the source term */
    double factor_1 = interp_time_factor(t, eta, fep->index_1);
    double factor_2 = interp_time_factor(t, eta, fep->index_2);

    /* Interpolate the Omega matrix */
    double Omega_11 = interp_time_factor(t, eta, t->index_Om_11);
    double Omega_12 = interp_time_factor(t, eta, t->index_Om_11 + 1);
    double Omega_21 = interp_time_factor(t, eta, t->index_Om_11 + 2);
    double Omega_22 = interp_time_factor(t, eta, t->index_Om_11 + 3);

    /* The EdS growth factor at order n */
    double D_n = exp(eta * fep->n);

    /* Add the Omega term */
    f[0] = -(Omega_11 * y[0] + Omega_12 * y[1]);
    f[1] = -(Omega_21 * y[0] + Omega_22 * y[1]);

    /* Add the exponential term n*I*y */
    f[0] -= fep->n * y[0];
    f[1] -= fep->n * y[1];

    /* Add the source term */
    if (fep->which_equation == CONTINUITY_EQ) {
        f[0] += factor_1 * factor_2;
    } else if (fep->which_equation == EULER_EQ) {
        f[1] += factor_1 * factor_2;
    }

    return GSL_SUCCESS;
}

// int fluid_equation_jac(double eta, const double y[], double *dfdy, double
// dfdt[], void *pars) {
//     struct fluid_equation_params *fep = (struct fluid_equation_params *)pars;
//     const struct time_factor_table *t = fep->t;
//
//     gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdy, 2, 2);
//     gsl_matrix * m = &dfdy_mat.matrix;
//
//     dfdt[0] = 0.0;
//     dfdt[1] = 0.0;
//
//     /* Interpolate the two factors in the source term */
//     double factor_1 = interp_time_factor(t, eta, fep->index_1);
//     double factor_2 = interp_time_factor(t, eta, fep->index_2);
//
//     /* Interpolate the Omega matrix */
//     double Omega_11 = interp_time_factor(t, eta, t->index_Om_11);
//     double Omega_12 = interp_time_factor(t, eta, t->index_Om_11 + 1);
//     double Omega_21 = interp_time_factor(t, eta, t->index_Om_11 + 2);
//     double Omega_22 = interp_time_factor(t, eta, t->index_Om_11 + 3);
//
//     /* The EdS growth factor at order n */
//     double D_n = exp(eta * fep->n);
//
//     gsl_matrix_set (m, 0, 0, -Omega_11 - fep->n);
//     gsl_matrix_set (m, 0, 1, -Omega_12);
//     gsl_matrix_set (m, 1, 0, -Omega_21);
//     gsl_matrix_set (m, 1, 1, -Omega_22 - fep->n);
//
//     return GSL_SUCCESS;
// }

int integrate_time_factor(struct time_factor_table *tab, char which_equation,
                          int source_index_1, int source_index_2,
                          int dest_index_1, int dest_index_2, int n) {

    /* Differential equation parameters */
    const int dim = 2;
    struct fluid_equation_params fep = {tab, which_equation, source_index_1,
               source_index_2, n
    };

    /* Initial step size (allow for integrating in both directions) */
    double hstart;
    if (tab->time_i < tab->time_f) {
        hstart = +1e-10;
    } else {
        hstart = -1e-10;
    }

    /* The ODE system to be solved */
    gsl_odeiv2_system sys = {fluid_equation, NULL, dim, &fep};
    gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(
                               &sys, gsl_odeiv2_step_rk8pd, hstart, 1e-10, 1e-10);

    /* Initial condition (assume EdS) */
    double eta_i = tab->time_i;
    double D_n = exp(eta_i * n);
    double y_0_EdS = tab->ic_factor[dest_index_1];
    double y_1_EdS = tab->ic_factor[dest_index_2];
    double y[2] = {y_0_EdS, y_1_EdS};

    /* Solve the equation at each time step */
    double eta = eta_i;
    for (int i = 0; i < tab->N_t; i++) {
        double eta_next =
            (i < tab->N_t - 1) ? tab->time_sampling[i + 1] : tab->time_f;
        double D_n_next = exp(eta_next * n);

        int status = gsl_odeiv2_driver_apply(d, &eta, eta_next, y);
        if (status != GSL_SUCCESS) {
            printf("Error with integrating status %d\n", status);
            break;
        }

        /* Store the result */
        tab->table[dest_index_1 * tab->N_t + i] = y[0];
        tab->table[dest_index_2 * tab->N_t + i] = y[1];
    }

    /* Free the driver */
    gsl_odeiv2_driver_free(d);

    return 0;
}

/* Compute the (constant) time factor in the EdS limit (modified with variable
 * Omega_m) */
int compute_time_factor_EdS(struct time_factor_table *tab, char which_equation,
                            int source_index_1, int source_index_2,
                            int dest_index_1, int dest_index_2, int n) {

    /* Constant pre-factor at order n in the EdS limit */
    double pre_factor = 2.0 / (2 * n * n + n - 3);

    printf("prefactor %f\n", pre_factor);

    /* Fetch the EdS factors for the source terms */
    double factor_1 = tab->ic_factor[source_index_1];
    double factor_2 = tab->ic_factor[source_index_2];
    double source = factor_1 * factor_2;

    /* Store the EdS factors */
    if (which_equation == CONTINUITY_EQ) {
        tab->ic_factor[dest_index_1] = pre_factor * source * (n + 0.5);
        tab->ic_factor[dest_index_2] = pre_factor * source * 1.5;
    } else if (which_equation == EULER_EQ) {
        tab->ic_factor[dest_index_1] = pre_factor * source * 1.0;
        tab->ic_factor[dest_index_2] = pre_factor * source * n;
    }

    return 0;
}

/* Compute the (constant) time factor in the radiation domination (RD) limit */
int compute_time_factor_RD(struct time_factor_table *tab, char which_equation,
                           int source_index_1, int source_index_2,
                           int dest_index_1, int dest_index_2, int n) {

    /* Constant pre-factor at order n in the RD limit */
    double pre_factor = 1.0 / (n * n - n);

    printf("prefactor %f\n", pre_factor);

    /* Fetch the RD factors for the source terms */
    double factor_1 = tab->ic_factor[source_index_1];
    double factor_2 = tab->ic_factor[source_index_2];
    double source = factor_1 * factor_2;

    /* Store the RD factors */
    if (which_equation == CONTINUITY_EQ) {
        tab->ic_factor[dest_index_1] = pre_factor * source * (n - 1);
        tab->ic_factor[dest_index_2] = 0;
    } else if (which_equation == EULER_EQ) {
        tab->ic_factor[dest_index_1] = pre_factor * source * 1.0;
        tab->ic_factor[dest_index_2] = pre_factor * source * n;
    }

    return 0;
}

/* Generate time factors at order n from the lower order time factors */
int generate_time_factors_at_n(struct time_factor_table *tab,
                               struct coeff_table *c, int n, enum ic_type ic) {
    if (n < 2)
        return 0;

    /* Loop over all lower orders */
    int counter = 0;
    for (int l = 1; l <= n - 1; l++) {
        /* Determine the maximum index at this order for 'c' and 'd' */
        int N_c = 1 + find_coeff_max_index(c, 'c', l);
        int N_d = 1 + find_coeff_max_index(c, 'd', n - l);

        /* Loop over all pairs (c_{l,i}, d_{n-l,j}) for the continuity eq. */
        for (int i = 0; i < N_c; i++) {
            for (int j = 0; j < N_d; j++) {
                int source_index_1 = find_coeff_index_require(c, 'c', l, i);
                int source_index_2 = find_coeff_index_require(c, 'd', n - l, j);

                int dest_index_1 = add_coeff(c, 'c', n, counter);
                int dest_index_2 = add_coeff(c, 'd', n, counter);

                /* Compute the initial conditions */
                if (ic == EdS) {
                    compute_time_factor_EdS(tab, CONTINUITY_EQ, source_index_1,
                                            source_index_2, dest_index_1,
                                            dest_index_2, n);
                } else if (ic == RD) {
                    compute_time_factor_RD(tab, CONTINUITY_EQ, source_index_1,
                                           source_index_2, dest_index_1,
                                           dest_index_2, n);
                } else {
                    printf("Initial condition type not suppoted.\n");
                    return 1;
                }

                /* Integrate the time factors */
                integrate_time_factor(tab, CONTINUITY_EQ, source_index_1,
                                      source_index_2, dest_index_1,
                                      dest_index_2, n);
                counter++;
            }
        }

        /* Determine the maximum index at this order for 'c' and 'd' */
        int N_d_1 = 1 + find_coeff_max_index(c, 'd', l);
        int N_d_2 = 1 + find_coeff_max_index(c, 'd', n - l);

        /* Loop over all pairs (d_{l,i}, d_{n-l,j}) for the Euler eq. */
        for (int i = 0; i < N_d_1; i++) {
            for (int j = 0; j < N_d_2; j++) {
                int source_index_1 = find_coeff_index_require(c, 'd', l, i);
                int source_index_2 = find_coeff_index_require(c, 'd', n - l, j);

                int dest_index_1 = add_coeff(c, 'c', n, counter);
                int dest_index_2 = add_coeff(c, 'd', n, counter);

                /* Compute the initial conditions */
                if (ic == EdS) {
                    compute_time_factor_EdS(tab, EULER_EQ, source_index_1,
                                            source_index_2, dest_index_1,
                                            dest_index_2, n);
                } else if (ic == RD) {
                    compute_time_factor_RD(tab, EULER_EQ, source_index_1,
                                           source_index_2, dest_index_1,
                                           dest_index_2, n);
                } else {
                    printf("Initial condition type not suppoted.\n");
                    return 1;
                }

                /* Integrate the time factors */
                integrate_time_factor(tab, EULER_EQ, source_index_1,
                                      source_index_2, dest_index_1,
                                      dest_index_2, n);
                counter++;
            }
        }
    }

    return 0;
}

int compute_all_derivatives(struct time_factor_table *tab) {
    for (int i = 0; i < tab->N_f; i++) {
        for (int j = 1; j < tab->N_t - 1; j++) {
            double fl = tab->table[i * tab->N_t + (j - 1)];
            double fr = tab->table[i * tab->N_t + (j + 1)];
            double tl = tab->time_sampling[j - 1];
            double tr = tab->time_sampling[j + 1];

            tab->dtable_dt[i * tab->N_t + j] = (fr - fl) / (tr - tl);
        }
    }

    return 0;
}
