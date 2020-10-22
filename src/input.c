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

#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <assert.h>
#include <math.h>
#include "../include/input.h"
// #include "../include/fft.h"

int readParams(struct params *pars, const char *fname) {
     pars->Seed = ini_getl("Random", "Seed", 1, fname);

     pars->GridSize = ini_getl("Box", "GridSize", 64, fname);
     pars->BoxLen = ini_getd("Box", "BoxLen", 1.0, fname);

     /* Read strings */
     int len = DEFAULT_STRING_LENGTH;
     pars->OutputDirectory = malloc(len);
     pars->OutputFilename = malloc(len);
     pars->Name = malloc(len);
     pars->PerturbFile = malloc(len);
     ini_gets("Output", "Directory", "./output", pars->OutputDirectory, len, fname);
     ini_gets("Output", "Filename", "particles.hdf5", pars->OutputFilename, len, fname);
     ini_gets("Simulation", "Name", "No Name", pars->Name, len, fname);
     ini_gets("PerturbData", "File", "", pars->PerturbFile, len, fname);

     return 0;
}

int readUnits(struct units *us, const char *fname) {
    /* Internal units */
    us->UnitLengthMetres = ini_getd("Units", "UnitLengthMetres", 1.0, fname);
    us->UnitTimeSeconds = ini_getd("Units", "UnitTimeSeconds", 1.0, fname);
    us->UnitMassKilogram = ini_getd("Units", "UnitMassKilogram", 1.0, fname);
    us->UnitTemperatureKelvin = ini_getd("Units", "UnitTemperatureKelvin", 1.0, fname);
    us->UnitCurrentAmpere = ini_getd("Units", "UnitCurrentAmpere", 1.0, fname);

    /* Get the transfer functions format */
    char format[DEFAULT_STRING_LENGTH];
    ini_gets("TransferFunctions", "Format", "Plain", format, DEFAULT_STRING_LENGTH, fname);

    /* Format of the transfer functions */
    int default_h_exponent;
    int default_k_exponent;
    int default_sign;
    if (strcmp(format, "CLASS") == 0) {
        default_h_exponent = 1;
        default_k_exponent = 0;
        default_sign = -1;
    } else {
        default_h_exponent = 0;
        default_k_exponent = -2;
        default_sign = +1;
    }
    us->TransferUnitLengthMetres = ini_getd("TransferFunctions", "UnitLengthMetres", MPC_METRES, fname);
    us->Transfer_hExponent = ini_getl("TransferFunctions", "hExponent", default_h_exponent, fname);
    us->Transfer_kExponent = ini_getl("TransferFunctions", "kExponent", default_k_exponent, fname);
    us->Transfer_Sign = ini_getl("TransferFunctions", "Sign", default_sign, fname);

    /* Some physical constants */
    us->SpeedOfLight = SPEED_OF_LIGHT_METRES_SECONDS * us->UnitTimeSeconds
                        / us->UnitLengthMetres;
    us->GravityG = GRAVITY_G_SI_UNITS * us->UnitTimeSeconds * us->UnitTimeSeconds
                    / us->UnitLengthMetres / us->UnitLengthMetres / us->UnitLengthMetres
                    * us->UnitMassKilogram; // m^3 / kg / s^2 to internal
    us->hPlanck = PLANCK_CONST_SI_UNITS / us->UnitMassKilogram / us->UnitLengthMetres
                    / us->UnitLengthMetres * us->UnitTimeSeconds; //J*s = kg*m^2/s
    us->kBoltzmann = BOLTZMANN_CONST_SI_UNITS / us->UnitMassKilogram / us->UnitLengthMetres
                    / us->UnitLengthMetres * us->UnitTimeSeconds * us->UnitTimeSeconds
                    * us->UnitTemperatureKelvin; //J/K = kg*m^2/s^2/K
    us->ElectronVolt = ELECTRONVOLT_SI_UNITS / us->UnitMassKilogram / us->UnitLengthMetres
                    / us->UnitLengthMetres * us->UnitTimeSeconds
                    * us->UnitTimeSeconds; // J = kg*m^2/s^2

    return 0;
}

int readCosmology(struct cosmology *cosmo, struct units *us, const char *fname) {
     cosmo->h = ini_getd("Cosmology", "h", 0.70, fname);
     cosmo->n_s = ini_getd("Cosmology", "n_s", 0.97, fname);
     cosmo->A_s = ini_getd("Cosmology", "A_s", 2.215e-9, fname);
     cosmo->k_pivot = ini_getd("Cosmology", "k_pivot", 0.05, fname);
     cosmo->z_ini = ini_getd("Cosmology", "z_ini", 40.0, fname);

     double H0 = 100 * cosmo->h * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
     cosmo->rho_crit = 3 * H0 * H0 / (8 * M_PI * us->GravityG);

     return 0;
}

int cleanParams(struct params *pars) {
    free(pars->OutputDirectory);
    free(pars->OutputFilename);
    free(pars->Name);
    free(pars->PerturbFile);

    return 0;
}

// /* Read 3D box from disk, allocating memory and storing the grid dimensions */
// int readFieldFile(double **box, int *N, double *box_len, const char *fname) {
//     /* Create the hdf5 file */
//     hid_t h_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
//
//     /* Create the Header group */
//     hid_t h_grp = H5Gopen(h_file, "Header", H5P_DEFAULT);
//
//     /* Read the size of the field */
//     hid_t h_attr, h_err;
//     double boxsize[3];
//
//     /* Open and read out the attribute */
//     h_attr = H5Aopen(h_grp, "BoxSize", H5P_DEFAULT);
//     h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &boxsize);
//     if (h_err < 0) {
//         printf("Error reading hdf5 attribute '%s'.\n", "BoxSize");
//         return 1;
//     }
//
//     /* It should be a cube */
//     assert(boxsize[0] == boxsize[1]);
//     assert(boxsize[1] == boxsize[2]);
//     *box_len = boxsize[0];
//
//     /* Close the attribute, and the Header group */
//     H5Aclose(h_attr);
//     H5Gclose(h_grp);
//
//     /* Open the Field group */
//     h_grp = H5Gopen(h_file, "Field", H5P_DEFAULT);
//
//     /* Open the Field dataset */
//     hid_t h_data = H5Dopen2(h_grp, "Field", H5P_DEFAULT);
//
//     /* Open the dataspace and fetch the grid dimensions */
//     hid_t h_space = H5Dget_space(h_data);
//     int ndims = H5Sget_simple_extent_ndims(h_space);
//     hsize_t *dims = malloc(ndims * sizeof(hsize_t));
//     H5Sget_simple_extent_dims(h_space, dims, NULL);
//     int read_N = dims[0];
//
//     /* We should be in 3D */
//     if (ndims != 3) {
//         printf("Number of dimensions %d != 3.\n", ndims);
//         return 2;
//     }
//     /* It should be a cube (but allow for padding in the last dimension) */
//     if (read_N != dims[1] || (read_N != dims[2] && (read_N+2) != dims[2])) {
//         printf("Non-cubic grid size (%lld, %lld, %lld).\n", dims[0], dims[1], dims[2]);
//         return 2;
//     }
//     /* Store the grid size */
//     *N = read_N;
//
//     /* Allocate the array (wuthout padding) */
//     *box = malloc(read_N * read_N * read_N * sizeof(double));
//
//     /* The hyperslab that should be read (needed in case of padding) */
//     const hsize_t space_rank = 3;
//     const hsize_t space_dims[3] = {read_N, read_N, read_N}; //3D space
//
//     /* Offset of the hyperslab */
//     const hsize_t space_offset[3] = {0, 0, 0};
//
//     /* Create memory space for the chunk */
//     hid_t h_memspace = H5Screate_simple(space_rank, space_dims, NULL);
//     H5Sselect_hyperslab(h_space, H5S_SELECT_SET, space_offset, NULL, space_dims, NULL);
//
//     /* Read out the data */
//     h_err = H5Dread(h_data, H5T_NATIVE_DOUBLE, h_memspace, h_space, H5P_DEFAULT, *box);
//     if (h_err < 0) {
//         printf("Error reading hdf5 file '%s'.\n", fname);
//         return 1;
//     }
//
//     /* Close the dataspaces and dataset */
//     H5Sclose(h_memspace);
//     H5Sclose(h_space);
//     H5Dclose(h_data);
//
//     /* Close the Field group */
//     H5Gclose(h_grp);
//
//     /* Close the file */
//     H5Fclose(h_file);
//
//     /* Free memory */
//     free(dims);
//
//     return 0;
// }
//
// /* Read the box without any checks, assuming we have sufficient memory
//  * allocated. */
// int readFieldFileInPlace(double *box, const char *fname) {
//     /* Create the hdf5 file */
//     hid_t h_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
//
//     /* Open the Field group */
//     hid_t h_grp = H5Gopen(h_file, "Field", H5P_DEFAULT);
//
//     /* Open the Field dataset */
//     hid_t h_data = H5Dopen2(h_grp, "Field", H5P_DEFAULT);
//
//     /* Open the dataspace and fetch the grid dimensions */
//     hid_t h_space = H5Dget_space(h_data);
//     int ndims = H5Sget_simple_extent_ndims(h_space);
//     hsize_t *dims = malloc(ndims * sizeof(hsize_t));
//     H5Sget_simple_extent_dims(h_space, dims, NULL);
//
//     /* Read out the data */
//     hid_t h_err = H5Dread(h_data, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, box);
//     if (h_err < 0) {
//         printf("Error reading hdf5 file '%s' in place.\n", fname);
//         return 1;
//     }
//
//     /* Close the dataspace and dataset */
//     H5Sclose(h_space);
//     H5Dclose(h_data);
//
//     /* Close the Field group */
//     H5Gclose(h_grp);
//
//     /* Close the file */
//     H5Fclose(h_file);
//
//     /* Free memory */
//     free(dims);
//
//     return 0;
// }
