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

#include <math.h>
#include "../include/primordial.h"

/* The bare primordial power spectrum, without transfer functions */
double primordialPower(double k, const struct cosmology *cosmo) {
    if (k == 0) return 0;

    double A_s = cosmo->A_s;
    double n_s = cosmo->n_s;
    double k_pivot = cosmo->k_pivot;

    return A_s * pow(k/k_pivot, n_s);
}
