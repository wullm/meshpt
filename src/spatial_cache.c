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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hdf5.h>

#include "../include/spatial_cache.h"

int init_spatial_factor_table(struct spatial_factor_table *t, double boxlen,
                              int N, int N_f, int N_cache, int unique) {

    /* Table dimensions (the grids are N^3 cells with physical size boxlen^3) */
    t->N = N;
    t->N_f = N_f;
    t->N_cache = N_cache;
    t->boxlen = boxlen;
    t->unique = unique;
    t->counter = 0;

    /* Allocate read out counter and cache location arrays */
    t->read_outs = calloc(N_f, sizeof(int));
    t->cache_location = calloc(N_f, sizeof(int));
    if (t->read_outs == NULL || t->cache_location == NULL) {
        printf("Error with allocating cache support arrays.\n");
        return -1;
    }

    /* Default values for the cache location (when not present) */
    for (int i = 0; i < N_f; i++) {
        t->cache_location[i] = -1;
    }

    /* Allocate cache */
    if (N_cache > 0) {
        t->cache = malloc(N * N * N * N_cache * sizeof(double));
        if (t->cache == NULL) {
            printf("Error with allocating cache.\n");
            return -1;
        }
    }

    return 0;
}

int free_spatial_factor_table(struct spatial_factor_table *t) {
    free(t->read_outs);
    free(t->cache_location);
    if (t->N_cache > 0) {
        free(t->cache);
    }

    return 0;
}

int store_grid(struct spatial_factor_table *t, double *box, int index) {
    int loc = t->cache_location[index];
    int N = t->N;
    int grid_size = N * N * N;
    int grid_bytes = grid_size * sizeof(double);

    /* If the grid is already in the cache, overwrite it. */
    if (loc >= 0) {
        memcpy(t->cache + grid_size * loc, box, grid_bytes);
        return 0;
    } else if (t->counter < t->N_cache) {
        /* There is room in the cache for another grid */
        loc = t->counter;
        t->counter++;
        t->cache_location[index] = loc;
        memcpy(t->cache + grid_size * loc, box, grid_bytes);

        return 0;
    } else if (t->N_cache > 0) {
        /* Determine if we should kick out another grid from the cache */
        int reads = t->read_outs[index];
        int min_reads = 0;
        int argmin = 0;
        int found_in_cache = 0;
        for (int i = 0; i < t->counter; i++) {
            if (t->cache_location[i] >= 0 && i != index) {
                found_in_cache++;
                if (t->read_outs[i] < min_reads || found_in_cache == 0) {
                    min_reads = t->read_outs[i];
                    argmin = i;
                }
            }
        }

        assert(found_in_cache == t->N_cache);

        if (reads > min_reads) {
            /* Store the least read grid to disk */
            loc = t->cache_location[argmin];
            char fname[50];
            sprintf(fname, "cache/cache_%03d_%06d.h5", argmin, t->unique);
            disk_store_grid(t->N, t->boxlen, t->cache + loc * grid_size, fname);
            t->cache_location[argmin] = -1; // no longer in cache

            /* Overwrite its place in the cache */
            memcpy(t->cache + loc * grid_size, box, grid_bytes);
            t->cache_location[index] = loc;

            return 0;
        }
    }

    /* Saving to cache has not happened, write to disk */
    char fname[50];
    sprintf(fname, "cache/cache_%03d_%06d.h5", index, t->unique);
    disk_store_grid(t->N, t->boxlen, box, fname);

    return 0;
}

int fetch_grid(struct spatial_factor_table *t, double *box, int index) {
    /* Check if the grid is in cache */
    int loc = t->cache_location[index];
    int N = t->N;
    int grid_size = N * N * N;
    int grid_bytes = grid_size * sizeof(double);

    /* Increment the read counter */
    t->read_outs[index]++;

    if (loc >= 0) {
        memcpy(box, t->cache + grid_size * loc, grid_bytes);
    } else {
        char fname[50];
        sprintf(fname, "cache/cache_%03d_%06d.h5", index, t->unique);
        printf("We laden index %d (%d).\n", index, t->read_outs[index]);
        disk_fetch_grid(box, fname);
    }

    return 0;
}

int disk_store_grid(int N, double boxlen, double *box, char *fname) {

    /* Create the hdf5 file */
    hid_t h_file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /* Create the Header group */
    hid_t h_grp =
        H5Gcreate(h_file, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for BoxSize attribute */
    const hsize_t arank = 1;
    const hsize_t adims[1] = {3}; // 3D space
    hid_t h_aspace = H5Screate_simple(arank, adims, NULL);

    /* Create the BoxSize attribute and write the data */
    hid_t h_attr =
        H5Acreate1(h_grp, "BoxSize", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    double boxsize[3] = {boxlen, boxlen, boxlen};
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, boxsize);

    /* Close the attribute, corresponding dataspace, and the Header group */
    H5Aclose(h_attr);
    H5Sclose(h_aspace);
    H5Gclose(h_grp);

    /* Create the Field group */
    h_grp = H5Gcreate(h_file, "/Field", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for the field */
    const hsize_t frank = 3;
    const hsize_t fdims[3] = {N, N, N}; // 3D space
    hid_t h_fspace = H5Screate_simple(frank, fdims, NULL);

    /* Create the dataset for the field */
    hid_t h_data = H5Dcreate(h_grp, "Field", H5T_NATIVE_DOUBLE, h_fspace,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Write the data */
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_fspace, h_fspace, H5P_DEFAULT, box);

    /* Close the dataset, corresponding dataspace, and the Field group */
    H5Dclose(h_data);
    H5Sclose(h_fspace);
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    return 0;
}

int disk_fetch_grid(double *box, char *fname) {
    /* Open the hdf5 file */
    hid_t h_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Open the Field group */
    hid_t h_grp = H5Gopen(h_file, "Field", H5P_DEFAULT);

    /* Open the Field dataset */
    hid_t h_data = H5Dopen2(h_grp, "Field", H5P_DEFAULT);

    /* Open the dataspace and fetch the grid dimensions */
    hid_t h_space = H5Dget_space(h_data);
    int ndims = H5Sget_simple_extent_ndims(h_space);
    hsize_t *dims = malloc(ndims * sizeof(hsize_t));
    H5Sget_simple_extent_dims(h_space, dims, NULL);

    /* Read out the data */
    hid_t h_err =
        H5Dread(h_data, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, box);
    if (h_err < 0) {
        printf("Error reading hdf5 file '%s' in place.\n", fname);
        return 1;
    }

    /* Close the dataspace and dataset */
    H5Sclose(h_space);
    H5Dclose(h_data);

    /* Close the Field group */
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    /* Free memory */
    free(dims);

    return 0;
}

int coeff_to_fname(struct coefficient c, char *fname) {
    sprintf(fname, "%c_%d_%d.h5", c.letter, c.subscript_1, c.subscript_2);
    return 0;
}
