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
#include <string.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>

#include "../include/meshpt.h"

double D(double eta) {
    return exp(eta);
}

int run_meshpt(int N, double boxlen, int nk, void *gridv, void *kvecv,
               void *Pvecv, int nz, void *zvecv, void *Dvecv, int N_SPT,
               double z_ini, double z_final) {

    /* The output grid */
    double *grid = (double *) gridv;

    /* Memory block for the output data */
    double *kvec = (double *) kvecv;
    double *Pvec = (double *) Pvecv;
    double *zvec = (double *) zvecv;
    double *Dvec = (double *) Dvecv;

    /* MeshPT structs */
    struct coeff_table time_coefficients;
    struct coeff_table space_coefficients;
    struct time_factor_table time_factors;
    struct spatial_factor_table space_factors;

    /* Initialize power spectrum interpolation sline */
    struct power_spline spline = {kvec, Pvec, nk};
    init_power_spline(&spline, 100);

    /* Initialize growth factor interpolation sline */
    struct power_spline D_spline = {zvec, Dvec, nz};
    init_power_spline(&D_spline, 100);

    /* Table lengths */
    int min_length = 1000;
    int cache_length = 0;
    int timesteps = 100;

    /* Starting and ending times */
    double a_ini = 1./(1+z_ini);
    double a_final = 1./(1+z_final);
    double t_i = log(a_ini);
    double t_f = log(a_final);

    printf("%f %f\n", t_i, t_f);

    /* Store a grid */
    int s = 101;
    rng_state seed = rand_uint64_init(s);

    /* A unique number to prevent filename clashes */
    int unique = (int)(sampleUniform(&seed) * 1e6);

    /* Initialize the tables */
    init_coeff_table(&time_coefficients, min_length);
    init_coeff_table(&space_coefficients, min_length);
    init_time_factor_table(&time_factors, timesteps, min_length, t_i, t_f);
    init_spatial_factor_table(&space_factors, boxlen, N, min_length,
                              cache_length, unique);

    /* Fill in the (2x2) Omega(time) matrix */
    int Omega_11 = add_coeff(&time_coefficients, 'O', 1, 1);
    int Omega_12 = add_coeff(&time_coefficients, 'O', 1, 2);
    int Omega_21 = add_coeff(&time_coefficients, 'O', 2, 1);
    int Omega_22 = add_coeff(&time_coefficients, 'O', 2, 2);
    time_factors.index_Om_11 = Omega_11;

    /* For Einstein-de Sitter, it's just constant */
    for (int i = 0; i < time_factors.N_t; i++) {
        time_factors.table[Omega_11 * time_factors.N_t + i] = 0.;
        time_factors.table[Omega_12 * time_factors.N_t + i] = -1.;
        time_factors.table[Omega_21 * time_factors.N_t + i] = -1.5;
        time_factors.table[Omega_22 * time_factors.N_t + i] = 0.5;
    }

    /* Fill in the first time factor c_{1,0} = d_{1,0} = D(t) */
    int index1 = add_coeff(&time_coefficients, 'c', 1, 0);
    int index2 = add_coeff(&time_coefficients, 'd', 1, 0);

    for (int i = 0; i < time_factors.N_t; i++) {
        double t = time_factors.time_sampling[i];
        int index;
        double u;
        double z = 1./exp(t) - 1;
        power_spline_find_k(&D_spline, z, &index, &u);

        time_factors.table[index1 * time_factors.N_t + i] = power_spline_interp(&D_spline, index, u) / exp(t);
        time_factors.table[index2 * time_factors.N_t + i] = power_spline_interp(&D_spline, index, u) / exp(t);
    }

    /* Set the constant EdS limit coefficients */
    time_factors.EdS_factor[index1] = 1.0;
    time_factors.EdS_factor[index2] = 1.0;

    /* Allocate array for the primordial Gaussian field */
    fftw_complex *fbox = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    double *box = malloc(N * N * N * sizeof(double));

    /* Generate a complex Hermitian Gaussian random field */
    generate_complex_grf(fbox, N, boxlen, &seed);
    enforce_hermiticity(fbox, N, boxlen);

    /* Apply the interpolated power spectrum */
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_sqrt_power_spline, &spline);

    /* Apply a k-cutoff to address UV divergences */
    double k_max = 1.0 * 0.6737;
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_lowpass, &k_max);

    /* Fourier transform the grid */
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);
    fft_execute(c2r);
    fft_normalize_c2r(box, N, boxlen);
    fftw_destroy_plan(c2r);

    /* Free the complex grid */
    fftw_free(fbox);

    /* Fill in the first spatial factor X_{1,0} = Y_{1,0} = linear theory X */
    int index_s1 = add_coeff(&space_coefficients, 'X', 1, 0);
    int index_s2 = add_coeff(&space_coefficients, 'Y', 1, 0);

    store_grid(&space_factors, box, index_s1);
    store_grid(&space_factors, box, index_s2);

    /* Free the real grid */
    fftw_free(box);

    fbox = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    box = malloc(N * N * N * sizeof(double));

    fftw_plan r2c1 =
        fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fftw_plan c2r1 =
        fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);

    /* Generate the higher order time factors */
    for (int n = 0; n < N_SPT; n++) {
      generate_time_factors_at_n(&time_factors, &time_coefficients, n);
    }

    /* Compute the derivatives */
    compute_all_derivatives(&time_factors);

    /* Print a list of all coefficients */
    print_coefficients(&time_coefficients);

    /* Generate the higher order spatial factors */
    double a_begin = log(a_final);
    for (int n = 0; n < N_SPT; n++) {
      generate_spatial_factors_at_n(&space_factors, &time_factors,
                                    &space_coefficients, &time_coefficients, n,
                                    a_begin);
    }

    /* Compute the aggregate fields */
    double *density = malloc(N * N * N * sizeof(double));
    double *flux = malloc(N * N * N * sizeof(double));
    double *load1 = malloc(N * N * N * sizeof(double));
    double *load2 = malloc(N * N * N * sizeof(double));

    memset(density, 0, N * N * N * sizeof(double));
    memset(flux, 0, N * N * N * sizeof(double));

    for (int n = 1; n <= N_SPT; n++) {
        aggregate_factors_at_n(&space_factors, &time_factors,
                               &space_coefficients, &time_coefficients, n,
                               a_begin, density, flux);

        char density_fname[50];
        char flux_fname[50];

        sprintf(density_fname, "density_%d.h5", n);
        sprintf(flux_fname, "flux_%d.h5", n);

        disk_store_grid(N, boxlen, density, density_fname);
        disk_store_grid(N, boxlen, flux, flux_fname);
    }

    /* Copy the full nonlinear density field into the output array */
    memcpy(grid, density, N*N*N*sizeof(double));

    // /* Fourier transform the grid */
    // fft_execute(r2c1);
    // fft_normalize_r2c(fbox, N, boxlen);
    //
    // /* Apply a Gaussian filter */
    // double R_smooth = 10.0 / 0.67;
    // fft_apply_kernel(fbox, fbox, N, boxlen, kernel_gaussian, &R_smooth);
    //
    // /* Inverse Fourier transform the grid */
    // fft_execute(c2r1);
    // fft_normalize_c2r(box, N, boxlen);
    //
    // /* Free the complex grid */
    // fftw_free(fbox);
    //
    // /* Copy the smoothed nonlinear density field into the output array */
    // memcpy(grid, box, N*N*N*sizeof(double));

    /* Free additional gridds */
    free(density);
    free(flux);
    free(load1);
    free(load2);

    /* Retrieve the computed values */
    int n = 3;
    for (int i = 0; i < time_factors.N_t; i++) {
        double t = time_factors.time_sampling[i];

        int new1 = find_coeff_index(&time_coefficients, 'd', n, 1);
        int new2 = find_coeff_index(&time_coefficients, 'd', n, 0);

        double a1 = time_factors.table[new1 * time_factors.N_t + i];
        double a2 = time_factors.table[new2 * time_factors.N_t + i];

        double b1 = time_factors.EdS_factor[new1];
        double b2 = time_factors.EdS_factor[new2];

        printf("%f %f %f %f %f\n", t, a1, a2, b1, b2);
        // printf("%f %f %f %f\n", t, a1, a2, (a1 + a2));
    }

    /* Free MeshPT structs */
    free_coeff_table(&time_coefficients);
    free_coeff_table(&space_coefficients);
    free_time_factor_table(&time_factors);
    free_spatial_factor_table(&space_factors);
    free_power_spline(&spline);
    free_power_spline(&D_spline);

    return 0;
}

int main() {
    printf("Nice try.\n");

    return 0;
}
