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

#ifndef SPATIAL_FACTORS_H
#define SPATIAL_FACTORS_H

#include "../include/spatial_cache.h"
#include "../include/time_factors.h"

int generate_spatial_factors_at_n(struct spatial_factor_table *sft,
                                  const struct time_factor_table *tft,
                                  struct coeff_table *spatial_coeffs,
                                  const struct coeff_table *time_coeffs, int n,
                                  double time, double k_cutoff);

int aggregate_factors_at_n(struct spatial_factor_table *sft,
                           const struct time_factor_table *tft,
                           struct coeff_table *spatial_coeffs,
                           const struct coeff_table *time_coeffs, int n,
                           double time, double *density, double *flux);

int generate_spatial_factors_at_n_EdS(struct spatial_factor_table *sft,
                                      const struct time_factor_table *tft,
                                      struct coeff_table *spatial_coeffs,
                                      const struct coeff_table *time_coeffs,
                                      int n, double time, double k_cutoff);
#endif
