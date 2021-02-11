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

#ifndef INDICES_H
#define INDICES_H

/*
 * Coefficients are of the form X_{i,j} with
 *  X a character
 *  i the first subscript
 *  j the second subscript
 */
struct coefficient {
    char letter;
    int subscript_1;
    int subscript_2;
};

struct coeff_table {
    struct coefficient *table; /* Array of coefficients */
    int count;                 /* Number of initialized coefficients */
    int length;                /* Maximum table length */
};

int init_coeff_table(struct coeff_table *t, int len);
int free_coeff_table(struct coeff_table *t);
int find_coeff_index(const struct coeff_table *t, char ltr, int s1, int s2);
int find_coeff_index_require(const struct coeff_table *t, char ltr, int s1,
                             int s2);
int find_coeff_max_index(const struct coeff_table *t, char ltr, int s1);
int add_coeff(struct coeff_table *t, char ltr, int s1, int s2);
int print_coefficients(struct coeff_table *t);

#endif
