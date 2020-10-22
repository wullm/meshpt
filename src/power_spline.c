/*******************************************************************************
 * This file is part of Mitos.
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

#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <math.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_spline.h>

#include "../include/power_spline.h"

/* Initialize the power spectrum spline */
int initPowerSpline(struct power_spline *spline, int k_acc_size) {

    /* Pre-compute the square root of the power spectrum */
    spline->sqrtPower = malloc(spline->k_size * sizeof(double));
    for (int i=0; i<spline->k_size; i++) {
        spline->sqrtPower[i] = sqrt(spline->Power[i]);
    }

    /* Allocate the k search table */
    spline->k_acc_table = malloc(k_acc_size * sizeof(double));
    spline->k_acc_table_size = k_acc_size;

    if (spline->k_acc_table == NULL) return 1;

    /* Bounding values for the larger table */
    int k_size = spline->k_size;
    double k_min = spline->k[0];
    double k_max = spline->k[k_size-1];

    /* Make the index table */
    for (int i=0; i<k_acc_size; i++) {
        double u = (double) i/k_acc_size;
        double v = k_min + u * (k_max - k_min);

        /* Find the largest bin such that w > k */
        double maxJ = 0;
        for(int j=0; j<k_size; j++) {
            if (spline->k[j] < v) {
                maxJ = j;
            }
        }
        spline->k_acc_table[i] = maxJ;
    }

    return 0;
}

/* Clean the memory */
int cleanPowerSpline(struct power_spline *spline) {
    free(spline->k_acc_table);
    free(spline->sqrtPower);

    return 0;
}

/* Find index along the k direction (0 <= u <= 1 between index and index+1) */
int powerSplineFindK(const struct power_spline *spline, double k, int *index,
                     double *u) {

    /* Bounding values for the larger table */
    int k_acc_table_size = spline->k_acc_table_size;
    int k_size = spline->k_size;
    double k_min = spline->k[0];
    double k_max = spline->k[k_size-1];

    if (k > k_max) {
      *index = k_size - 2;
      *u = 1.0;
      return 0;
    }

    /* Quickly find a starting index using the indexed seach */
    double v = k;
    double w = (v - k_min) / (k_max - k_min);
    int J = floor(w * k_acc_table_size);
    int start = spline->k_acc_table[J < k_acc_table_size ? J : k_acc_table_size - 1];

    /* Search in the k vector */
    int i;
    for (i = start; i < k_size; i++) {
        if (k >= spline->k[i] && k <= spline->k[i + 1]) break;
    }

    /* We found the index */
    *index = i;

    /* Find the bounding values */
    double left = spline->k[*index];
    double right = spline->k[*index + 1];

    /* Calculate the ratio (X - X_left) / (X_right - X_left) */
    *u = (k - left) / (right - left);

    return 0;
}

/* Linear interpolation of the power spectrum */
double powerSplineInterp(const struct power_spline *spline, int k_index,
                         double u_k) {

    /* Retrieve the bounding values */
    double P1 = spline->Power[k_index];
    double P2 = spline->Power[k_index + 1];

    return (1 - u_k) * P1 + u_k * P2;
}

/* Linear interpolation of the square root of the power spectrum */
double sqrtPowerSplineInterp(const struct power_spline *spline, int k_index,
                             double u_k) {

    /* Retrieve the bounding values */
    double P1 = spline->sqrtPower[k_index];
    double P2 = spline->sqrtPower[k_index + 1];

    return (1 - u_k) * P1 + u_k * P2;
}

// /* Container function for simple bilinear interpolation */
// double perturbSplineInterp0(const struct perturb_spline *spline, double k,
//                             double log_tau, int index_src) {
//
//     /* Indices in the k and tau directions */
//     int k_index = 0, tau_index = 0;
//     /* Spacing (0 <= u <= 1) between subsequent indices in both directions */
//     double u_k, u_tau;
//
//     /* Find the indices and spacings */
//     perturbSplineFindTau(spline, log_tau, &tau_index, &u_tau);
//     perturbSplineFindK(spline, k, &k_index, &u_k);
//
//     /* Do the interpolation */
//     return perturbSplineInterp(spline, k_index, tau_index, u_k, u_tau, index_src);
// }
