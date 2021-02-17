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

int run_meshpt(int N, double boxlen, double Omega_m, int nk, void *gridv,
               void *kvecv, void *Pvecv, int nz, void *zvecv, void *Dvecv,
               void *fvecv, void *Omega21v, void *Omega22v, int N_SPT,
               double z_ini, double z_final) {

    /* The output grid */
    double *grid = (double *)gridv;

    /* Memory block for the input data */
    double *kvec = (double *)kvecv;
    double *Pvec = (double *)Pvecv;
    double *zvec = (double *)zvecv;
    double *Dvec = (double *)Dvecv;
    double *fvec = (double *)fvecv;
    double *Omega21 = (double *)Omega21v;
    double *Omega22 = (double *)Omega22v;

    /* Compute square root of the power spectrun */
    double *sqrtPvec = malloc(nk * sizeof(double));
    for (int i = 0; i < nk; i++) {
        sqrtPvec[i] = sqrt(Pvec[i]);
    }

    /* Compute log of growth factor, which will be used as time variable */
    double *logDvec = malloc(nz * sizeof(double));
    for (int i = 0; i < nz; i++) {
        logDvec[i] = log(Dvec[i]);
    }

    /* MeshPT structs */
    struct coeff_table time_coefficients;
    struct coeff_table space_coefficients;
    struct time_factor_table time_factors;
    struct spatial_factor_table space_factors;

    /* Initialize power spectrum interpolation sline */
    struct strooklat Pspline = {kvec, nk};
    init_strooklat_spline(&Pspline, 100);

    /* Initialize a spline for the time variable */
    struct strooklat spline = {logDvec, nz};
    init_strooklat_spline(&spline, 100);

    /* Table lengths */
    int min_length = 10000;
    int cache_length = 4;
    int timesteps = 100;

    /* Starting and ending times */
    double a_ini = 1. / (1 + z_ini);
    double a_final = 1. / (1 + z_final);
    double t_i = log(a_ini);
    double t_f = log(a_final);

    printf("%f %f\n", t_i, t_f);
    printf("Omega_m: %f\n", Omega_m);

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
        double t = time_factors.time_sampling[i];

        double Omega21_z = strooklat_interp(&spline, Omega21, t);
        double Omega22_z = strooklat_interp(&spline, Omega22, t);

        time_factors.table[Omega_11 * time_factors.N_t + i] = 0.;
        time_factors.table[Omega_12 * time_factors.N_t + i] = -1.;
        time_factors.table[Omega_21 * time_factors.N_t + i] = Omega21_z;
        time_factors.table[Omega_22 * time_factors.N_t + i] = Omega22_z;
        // time_factors.table[Omega_21 * time_factors.N_t + i] = 0;
        // time_factors.table[Omega_22 * time_factors.N_t + i] = -1;

        // printf("%f %f\n", Omega21_z, Omega22_z);
    }

    /* Fill in the first time factor c_{1,0} = d_{1,0} = D(t) */
    int index1 = add_coeff(&time_coefficients, 'c', 1, 0);
    int index2 = add_coeff(&time_coefficients, 'd', 1, 0);

    for (int i = 0; i < time_factors.N_t; i++) {
        time_factors.table[index1 * time_factors.N_t + i] = 1.0;
        time_factors.table[index2 * time_factors.N_t + i] = 1.0;
    }

    /* Set the constant initial condition coefficients */
    time_factors.ic_factor[index1] = 1.0;
    time_factors.ic_factor[index2] = 1.0;

    /* Allocate array for the primordial Gaussian field */
    fftw_complex *fbox = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    double *box = malloc(N * N * N * sizeof(double));

    /* Generate a complex Hermitian Gaussian random field */
    generate_complex_grf(fbox, N, boxlen, &seed);
    enforce_hermiticity(fbox, N, boxlen);

    /* Apply the interpolated power spectrum */
    struct spline_params sp = {&Pspline, sqrtPvec};
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_sqrt_power_spline, &sp);

    int cutoff = 0;

    /* Apply a k-cutoff to address UV divergences */
    if (cutoff) {
        double k_max = 1.0 * 0.6737;
        fft_apply_kernel(fbox, fbox, N, boxlen, kernel_lowpass, &k_max);
    }

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

    fftw_plan r2c1 = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fftw_plan c2r1 = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);

    /* Generate the higher order time factors */
    for (int n = 2; n <= N_SPT; n++) {
        generate_time_factors_at_n(&time_factors, &time_coefficients, n, EdS);
    }

    /* Compute the derivatives */
    compute_all_derivatives(&time_factors);

    /* Print a list of all coefficients */
    print_coefficients(&time_coefficients);

    /* Generate the higher order spatial factors */
    double a_begin = log(a_final);
    for (int n = 2; n <= N_SPT; n++) {
        generate_spatial_factors_at_n(&space_factors, &time_factors,
                                      &space_coefficients, &time_coefficients,
                                      n, a_begin);
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
    memcpy(grid, density, N * N * N * sizeof(double));

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

        int new1 = find_coeff_index(&time_coefficients, 'c', n, 2);
        int new2 = find_coeff_index(&time_coefficients, 'c', n, 3);

        double a1 = time_factors.table[new1 * time_factors.N_t + i];
        double a2 = time_factors.table[new2 * time_factors.N_t + i];

        double b1 = time_factors.ic_factor[new1];
        double b2 = time_factors.ic_factor[new2];

        printf("%f %f %f %f %f\n", t, a1, a2, b1, b2);
        // printf("%f %f %f %f\n", t, a1, a2, (a1 + a2));
    }

    /* Free MeshPT structs */
    free_coeff_table(&time_coefficients);
    free_coeff_table(&space_coefficients);
    free_time_factor_table(&time_factors);
    free_spatial_factor_table(&space_factors);
    free_strooklat_spline(&Pspline);
    free_strooklat_spline(&spline);

    free(sqrtPvec);
    free(logDvec);

    return 0;
}

int main() {
    printf("Nice try.\n");

    return 0;
}

int computeNonlinearGrid(int N, double boxlen, long long inseed, int nk,
                         double factor, void *kvecv, void *power_linv,
                         void *gridv) {

    /* The unit mass of one cell */
    double cell_mass = pow(boxlen / N, 3);

    /* The input data (power spectrum table) */
    double *kvec = (double *)kvecv;
    double *Pvec = (double *)power_linv;

    /* Memory block for the output data */
    double *grid = (double *)gridv;

    /* Compute square root of the power spectrun */
    double *sqrtPvec = malloc(nk * sizeof(double));
    for (int i = 0; i < nk; i++) {
        sqrtPvec[i] = sqrt(Pvec[i]);
    }

    /* Initialize power spectrum interpolation sline */
    struct strooklat Pspline = {kvec, nk};
    init_strooklat_spline(&Pspline, 100);

    /* Seed the random number generator */
    rng_state seed = rand_uint64_init(inseed);

    /* A unique number to prevent filename clashes */
    int unique = (int)(sampleUniform(&seed) * 1e6);

    /* Allocate array for the primordial Gaussian field */
    fftw_complex *fbox = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    fftw_complex *fbox2 = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    double *box = malloc(N * N * N * sizeof(double));
    /* Generate a complex Hermitian Gaussian random field */
    generate_complex_grf(fbox, N, boxlen, &seed);
    enforce_hermiticity(fbox, N, boxlen);

    /* Apply the interpolated power spectrum */
    struct spline_params sp = {&Pspline, sqrtPvec};
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_sqrt_power_spline, &sp);

    /* Apply the smoothing filter */
    double R_smooth = 1.0;
    double k_cutoff = 1.0 / R_smooth;

    /* Fourier transform the grid */
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);
    fft_execute(c2r);
    fft_normalize_c2r(box, N, boxlen);

    // /* Copy the full nonlinear density field into the output array */
    // memcpy(grid, box, N*N*N*sizeof(double));
    //
    // return 0;

    /* Apply spherical collapse transform */
    const double alpha = 1.5;
    for (int i = 0; i < N * N * N; i++) {
        double d = box[i] * factor;
        // if (d < alpha) {
        //     d = -3*pow(1-d/alpha, alpha/3)+3;
        // } else {
        //     d = 3;
        // }
        // grid[i] = d;
        // grid[i] = d / 1; //ZELDOVICH
        grid[i] = d - d * d / 7; // 2LPT-like
    }

    // printf("%f\n", grid[0]);

    /* Fourier transform the grid back */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fft_execute(r2c);
    fft_normalize_r2c(fbox, N, boxlen);

    /* Prepare a second plan */
    fftw_plan c2r_alt =
        fftw_plan_dft_c2r_3d(N, N, N, fbox2, box, FFTW_ESTIMATE);

    /* Free memory */
    free(box);

    /* Destroy structures no longer needed */
    fftw_destroy_plan(c2r_alt);
    fftw_destroy_plan(r2c);
    free(fbox2);

    /* Fourier transform again (note the different real grid) */
    fftw_plan r2c2 = fftw_plan_dft_r2c_3d(N, N, N, grid, fbox, FFTW_ESTIMATE);
    fft_execute(r2c2);
    fft_normalize_r2c(fbox, N, boxlen);

    /* Approximate the potential with the Zel'dovich approximation */
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_inv_poisson, NULL);

    /* Allocate memory for the three displacement grids */
    fftw_complex *f_psi_x = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    fftw_complex *f_psi_y = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    fftw_complex *f_psi_z = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    double *psi_x = malloc(N * N * N * sizeof(double));
    double *psi_y = malloc(N * N * N * sizeof(double));
    double *psi_z = malloc(N * N * N * sizeof(double));

    /* Compute the displacements grids by differentiating the potential */
    fft_apply_kernel(f_psi_x, fbox, N, boxlen, kernel_dx, NULL);
    fft_apply_kernel(f_psi_y, fbox, N, boxlen, kernel_dy, NULL);
    fft_apply_kernel(f_psi_z, fbox, N, boxlen, kernel_dz, NULL);

    /* Fourier transform the potential grids */
    fftw_plan c2r_x =
        fftw_plan_dft_c2r_3d(N, N, N, f_psi_x, psi_x, FFTW_ESTIMATE);
    fftw_plan c2r_y =
        fftw_plan_dft_c2r_3d(N, N, N, f_psi_y, psi_y, FFTW_ESTIMATE);
    fftw_plan c2r_z =
        fftw_plan_dft_c2r_3d(N, N, N, f_psi_z, psi_z, FFTW_ESTIMATE);
    fft_execute(c2r_x);
    fft_execute(c2r_y);
    fft_execute(c2r_z);
    fft_normalize_c2r(psi_x, N, boxlen);
    fft_normalize_c2r(psi_y, N, boxlen);
    fft_normalize_c2r(psi_z, N, boxlen);
    fftw_destroy_plan(c2r_x);
    fftw_destroy_plan(c2r_y);
    fftw_destroy_plan(c2r_z);

    /* Free the complex grids, which are no longer needed */
    free(fbox);
    free(f_psi_x);
    free(f_psi_y);
    free(f_psi_z);

    /* Reset the box array */
    memset(grid, 0, N * N * N * sizeof(double));

    /* Compute the density grid by CIC mass assignment */
    double fac = N / boxlen;
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            for (int z = 0; z < N; z++) {

                double dx = psi_x[row_major(x, y, z, N)];
                double dy = psi_y[row_major(x, y, z, N)];
                double dz = psi_z[row_major(x, y, z, N)];

                double X = x - dx * fac;
                double Y = y - dy * fac;
                double Z = z - dz * fac;

                int iX = (int)floor(X);
                int iY = (int)floor(Y);
                int iZ = (int)floor(Z);

                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        for (int k = -1; k <= 1; k++) {
                            double xx = fabs(X - (iX + i));
                            double yy = fabs(Y - (iY + j));
                            double zz = fabs(Z - (iZ + k));

                            double part_x = xx <= 1 ? 1 - xx : 0;
                            double part_y = yy <= 1 ? 1 - yy : 0;
                            double part_z = zz <= 1 ? 1 - zz : 0;

                            grid[row_major(iX + i, iY + j, iZ + k, N)] +=
                                cell_mass * part_x * part_y * part_z;
                        }
                    }
                }
            }
        }
    }

    free(psi_x);
    free(psi_y);
    free(psi_z);

    /* Clean up the spline */
    free_strooklat_spline(&Pspline);
    free(sqrtPvec);

    return 1;
}
