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

#ifndef SPATIAL_OPERATIONS_H
#define SPATIAL_OPERATIONS_H

int grid_grad_dot(int N, double boxlen, double *box1, double *box2,
                  double *result);
int grid_product(int N, double boxlen, double *box1, double *box2,
                 double *result);
int grid_symmetric_grad(int N, double boxlen, double *box1, double *box2,
                        double *result);

#endif
