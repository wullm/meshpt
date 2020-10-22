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

#ifndef POWER_SPLINE_H
#define POWER_SPLINE_H

#define DEFAULT_K_ACC_TABLE_SIZE 1000

struct power_spline {
    /* The data table to be interpolated */
    const double *k;
    const double *Power;
    const int k_size;

    /* For optimization purposes, store the square root of the power */
    double *sqrtPower;

    /* Search table for interpolation acceleration in the k direction */
    double *k_acc_table;

    /* Size of the k acceleration table */
    int k_acc_table_size;
};

/* Initialize the power spectrum spline */
int initPowerSpline(struct power_spline *spline, int k_acc_size);

/* Clean the memory */
int cleanPowerSpline(struct power_spline *spline);

/* Find index along the k direction (0 <= u <= 1 between index and index+1) */
int powerSplineFindK(const struct power_spline *spline, double k, int *index,
                     double *u);

/* Linear interpolation of the power spectrum */
double powerSplineInterp(const struct power_spline *spline, int k_index,
                         double u_k);

/* Linear interpolation of the power spectrum */
double sqrtPowerSplineInterp(const struct power_spline *spline, int k_index,
                             double u_k);
#endif
