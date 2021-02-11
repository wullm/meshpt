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

#include <stdio.h>
#include <stdlib.h>

#include "../include/indices.h"

int init_coeff_table(struct coeff_table *t, int len) {
    t->count = 0;
    t->length = len;
    t->table = malloc(sizeof(struct coefficient) * len);

    if (t->table == NULL) {
        printf("Error with allocating coefficient table.\n");
        return -1;
    }

    return 0;
}

int free_coeff_table(struct coeff_table *t) {
    free(t->table);
    return 0;
}

/* Search the coefficient table and return the index or -1 if not found */
int find_coeff_index(const struct coeff_table *t, char ltr, int s1, int s2) {
    int index = -1;
    for (int i = 0; i < t->count; i++) {
        struct coefficient *c = &t->table[i];
        if (c->letter == ltr && c->subscript_1 == s1 && c->subscript_2 == s2) {
            index = i;
        }
    }
    return index;
}

/* Among all coefficients with fixed letter and first index, find the max of
 * the second index. (Returns -1 if none is present.) */
int find_coeff_max_index(const struct coeff_table *t, char ltr, int s1) {
    int max = -1;
    for (int i = 0; i < t->count; i++) {
        struct coefficient *c = &t->table[i];
        if (c->letter == ltr && c->subscript_1 == s1 && c->subscript_2 > max) {
            max = c->subscript_2;
        }
    }

    return max;
}

/* Find the index of a coefficient. Throw an error if not found */
int find_coeff_index_require(const struct coeff_table *t, char ltr, int s1,
                             int s2) {
    int index = find_coeff_index(t, ltr, s1, s2);
    if (index < 0)
        printf("Error: missing coefficient.\n");
    return index;
}

int add_coeff(struct coeff_table *t, char ltr, int s1, int s2) {
    /* Make sure that the table is large enough */
    if (t->count >= t->length) {
        printf("Error: coefficient table is full.\n");
        return -1;
    }

    /* Make sure that the coefficient is not duplicated */
    int search = find_coeff_index(t, ltr, s1, s2);

    if (search >= 0) {
        printf("Error: duplicate index.\n");
        return -1;
    }

    /* Add the index to the table */
    t->table[t->count].letter = ltr;
    t->table[t->count].subscript_1 = s1;
    t->table[t->count].subscript_2 = s2;
    t->count++;

    /* Return the index of the coefficient */
    return (t->count - 1);
}

int print_coefficients(struct coeff_table *t) {
    for (int i = 0; i < t->count; i++) {
        printf("%03d: %c_{%d,%d}\n", i, t->table[i].letter,
               t->table[i].subscript_1, t->table[i].subscript_2);
    }
    return 0;
}
