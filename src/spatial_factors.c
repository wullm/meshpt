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

#include "../include/fft.h"
#include "../include/fft_kernels.h"
#include "../include/spatial_factors.h"
#include "../include/spatial_operations.h"

void matrix_inv_2d(double *in, double *out) {
    double det = in[0] * in[3] - in[1] * in[2];
    if (fabs(det) < 1e-10) {
        printf("Error: encountered (nearly) singular matrix.\n");
    }

    out[0] = in[3] / det;
    out[1] = -in[1] / det;
    out[2] = -in[2] / det;
    out[3] = in[0] / det;
}

/* Generate spatial factors at order n in the EdS limit */
int generate_spatial_factors_at_n_EdS(struct spatial_factor_table *sft,
                                      const struct time_factor_table *tft,
                                      struct coeff_table *spatial_coeffs,
                                      const struct coeff_table *time_coeffs,
                                      int n, double time) {
    if (n < 2)
        return 0;

    const int N = sft->N;
    const double boxlen = sft->boxlen;
    double *input1 = malloc(N * N * N * sizeof(double));
    double *input2 = malloc(N * N * N * sizeof(double));
    double *input3 = malloc(N * N * N * sizeof(double));
    double *result1 = malloc(N * N * N * sizeof(double));
    double *result2 = malloc(N * N * N * sizeof(double));
    double *total = malloc(N * N * N * sizeof(double));

    /* Reset the result grid */
    memset(result1, 0, N * N * N * sizeof(double));
    memset(result2, 0, N * N * N * sizeof(double));

    /* Loop over all lower orders (computing the density grid) */
    for (int l = 1; l <= n - 1; l++) {
        int source_index_1 =
            find_coeff_index_require(spatial_coeffs, 'X', l, 0);
        int source_index_2 =
            find_coeff_index_require(spatial_coeffs, 'Y', n - l, 0);
        int source_index_3 =
            find_coeff_index_require(spatial_coeffs, 'Y', l, 0);

        printf("Doing %d and %d for %d\n", l, (n - l), n);

        /* Fetch the input grids */
        fetch_grid(sft, input1, source_index_1);
        fetch_grid(sft, input2, source_index_2);
        fetch_grid(sft, input3, source_index_3);

        /* Compute the output grids (continuity eq.) */
        grid_grad_dot(N, boxlen, input1, input2, result1);
        grid_product(N, boxlen, input1, input2, result1);

        /* Compute the output grids (Euler eq.) (Note: use input2 twice) */
        grid_symmetric_grad(N, boxlen, input3, input2, result2);
        grid_grad_dot(N, boxlen, input2, input3, result2);
    }

    /* We want to store the grids with the correct prefactor ratio */
    double prefactor_density_1 = 2.0 / ((2 * n + 3) * (n - 1)) * (n + 0.5);
    double prefactor_density_2 = 2.0 / ((2 * n + 3) * (n - 1)) * 1.0;
    double prefactor_flux_1 = 2.0 / ((2 * n + 3) * (n - 1)) * 1.5;
    double prefactor_flux_2 = 2.0 / ((2 * n + 3) * (n - 1)) * n;

    printf("%f %f\n", prefactor_density_1, prefactor_density_2);
    printf("%f %f\n", prefactor_flux_1, prefactor_flux_2);

    /* Apply the prefactors for the density grid */
    #pragma omp for
    for (int k = 0; k < N * N * N; k++) {
        total[k] =
            prefactor_density_1 * result1[k] + prefactor_density_2 * result2[k];
    }

    fftw_complex *fbox = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));

    fftw_plan r2c1 = fftw_plan_dft_r2c_3d(N, N, N, total, fbox, FFTW_ESTIMATE);
    fftw_plan c2r1 = fftw_plan_dft_c2r_3d(N, N, N, fbox, total, FFTW_ESTIMATE);

    /* Fourier transform the incoming grids */
    int cutoff = 0;
    if (cutoff) {
        fft_execute(r2c1);
        fft_normalize_r2c(fbox, N, boxlen);
        double k_max = (4. / 3.) * 0.6737;
        double R_filter = (10 / 0.6737);
        fft_apply_kernel(fbox, fbox, N, boxlen, kernel_lowpass, &k_max);
        fft_execute(c2r1);
        fft_normalize_c2r(total, N, boxlen);
    }

    /* Store the density grid at this order */
    int density_index = add_coeff(spatial_coeffs, 'X', n, 0);
    store_grid(sft, total, density_index);

    // fft_execute(r2c1);
    // fft_normalize_r2c(fbox, N, boxlen);
    // fft_apply_kernel(fbox, fbox, N, boxlen, kernel_gaussian, &R_filter);
    // fft_execute(c2r1);
    // fft_normalize_c2r(total, N, boxlen);

    char blab[50];
    sprintf(blab, "dens_%d.h5", n);
    disk_store_grid(N, boxlen, total, blab);

    /* Apply the prefactors for the density grid */
    #pragma omp for
    for (int k = 0; k < N * N * N; k++) {
        total[k] =
            prefactor_flux_1 * result1[k] + prefactor_flux_2 * result2[k];
    }

    fftw_plan r2c2 = fftw_plan_dft_r2c_3d(N, N, N, total, fbox, FFTW_ESTIMATE);
    fftw_plan c2r2 = fftw_plan_dft_c2r_3d(N, N, N, fbox, total, FFTW_ESTIMATE);

    /* Fourier transform the incoming grids */
    cutoff = 0;
    if (cutoff) {
        fft_execute(r2c2);
        fft_normalize_r2c(fbox, N, boxlen);
        double k_max = (4. / 3.) * 0.6737;
        double R_filter = (10 / 0.6737);
        fft_apply_kernel(fbox, fbox, N, boxlen, kernel_lowpass, &k_max);
        fft_execute(c2r2);
        fft_normalize_c2r(total, N, boxlen);
    }

    /* Store the grid for result 1 */
    int flux_index = add_coeff(spatial_coeffs, 'Y', n, 0);
    store_grid(sft, total, flux_index);

    // fft_execute(r2c2);
    // fft_normalize_r2c(fbox, N, boxlen);
    // fft_apply_kernel(fbox, fbox, N, boxlen, kernel_gaussian, &R_filter);
    // fft_execute(c2r2);
    // fft_normalize_c2r(total, N, boxlen);

    char bleb[50];
    sprintf(bleb, "flens_%d.h5", n);
    disk_store_grid(N, boxlen, total, bleb);

    free(input1);
    free(input2);
    free(input3);
    free(result1);
    free(result2);
    free(total);

    free(fbox);

    return 0;
}

/* Generate spatial factors at order n from the lower order time & spatial
 * factors */
int generate_spatial_factors_at_n(struct spatial_factor_table *sft,
                                  const struct time_factor_table *tft,
                                  struct coeff_table *spatial_coeffs,
                                  const struct coeff_table *time_coeffs, int n,
                                  double time) {
    if (n < 2)
        return 0;

    const int N = sft->N;
    const double boxlen = sft->boxlen;
    double *input1 = malloc(N * N * N * sizeof(double));
    double *input2 = malloc(N * N * N * sizeof(double));
    double *result = malloc(N * N * N * sizeof(double));
    fftw_complex *fbox = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));

    /* Retrieve the Omega matrix */
    int O_11_index = tft->index_Om_11;
    double O_11 = interp_time_factor(tft, time, O_11_index);
    double O_12 = interp_time_factor(tft, time, O_11_index + 1);
    double O_21 = interp_time_factor(tft, time, O_11_index + 2);
    double O_22 = interp_time_factor(tft, time, O_11_index + 3);

    /* Loop over all lower orders */
    int counter = 0;
    for (int l = 1; l <= n - 1; l++) {
        /* Determine the maximum index at this order for 'c' and 'd' */
        int N_X = 1 + find_coeff_max_index(spatial_coeffs, 'X', l);
        int N_Y = 1 + find_coeff_max_index(spatial_coeffs, 'Y', n - l);

        /* Loop over all pairs (X_{l,i}, Y_{n-l,j}) for the continuity eq. */
        for (int i = 0; i < N_X; i++) {
            for (int j = 0; j < N_Y; j++) {
                int source_index_1 =
                    find_coeff_index_require(spatial_coeffs, 'X', l, i);
                int source_index_2 =
                    find_coeff_index_require(spatial_coeffs, 'Y', n - l, j);
                int source_index_time_1 =
                    find_coeff_index_require(time_coeffs, 'c', l, i);
                int source_index_time_2 =
                    find_coeff_index_require(time_coeffs, 'd', n - l, j);

                int dest_index_1 = add_coeff(spatial_coeffs, 'X', n, counter);
                int dest_index_2 = add_coeff(spatial_coeffs, 'Y', n, counter);
                int dest_index_time_1 =
                    find_coeff_index_require(time_coeffs, 'c', n, counter);
                int dest_index_time_2 =
                    find_coeff_index_require(time_coeffs, 'd', n, counter);

                /* Reset the result grid */
                memset(result, 0, N * N * N * sizeof(double));

                /* Retrieve the pre-factors */
                double Dn = exp(n * time);
                double source_prefactor_1 =
                    interp_time_factor(tft, time, source_index_time_1) *
                    exp(l * time);
                double source_prefactor_2 =
                    interp_time_factor(tft, time, source_index_time_2) *
                    exp((n - l) * time);
                double dest_prefactor_1 =
                    interp_time_factor(tft, time, dest_index_time_1) * Dn;
                double dest_prefactor_2 =
                    interp_time_factor(tft, time, dest_index_time_2) * Dn;

                double dest_derivative_1 =
                    interp_time_derivative(tft, time, dest_index_time_1) * Dn;
                double dest_derivative_2 =
                    interp_time_derivative(tft, time, dest_index_time_2) * Dn;

                // printf("%f %f\n", source_prefactor_1, source_prefactor_2);
                // printf("%f %f\n", dest_prefactor_1, dest_prefactor_2);
                // printf("%f %f\n", dest_derivative_1, dest_derivative_2);

                /* Prevent singularities */
                double inv_1 = (dest_prefactor_1 == 0) ? 0 : 1.0 / dest_prefactor_1;
                double inv_2 = (dest_prefactor_2 == 0) ? 0 : 1.0 / dest_prefactor_2;

                /* Fetch the input grids */
                fetch_grid(sft, input1, source_index_1);
                fetch_grid(sft, input2, source_index_2);

                /* Compute the output grid */
                grid_grad_dot(N, boxlen, input1, input2, result);
                grid_product(N, boxlen, input1, input2, result);

                /* Solve for the inverse matrix */
                double mat[] = {
                    O_11 + dest_derivative_1 * inv_1 + n, O_12, O_21, O_22 + dest_derivative_2 * inv_2 + n
                };
                double mat_inv[4];
                matrix_inv_2d(mat, mat_inv);

                /* Fourier transform the grid */
                fftw_plan r2c =
                    fftw_plan_dft_r2c_3d(N, N, N, result, fbox, FFTW_ESTIMATE);
                fft_execute(r2c);
                fft_normalize_r2c(fbox, N, boxlen);
                fftw_destroy_plan(r2c);

                /* Apply k-cutoff */
                int cutoff = 0;
                if (cutoff) {
                    double k_max = (4. / 3.) * 0.6737;
                    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_lowpass, &k_max);
                }

                /* Fourier transform back */
                fftw_plan c2r =
                    fftw_plan_dft_c2r_3d(N, N, N, fbox, result, FFTW_ESTIMATE);
                fft_execute(c2r);
                fft_normalize_c2r(result, N, boxlen);
                fftw_destroy_plan(c2r);

                // printf("\n\n====\n");
                // printf("%f %f\n%f %f\n", mat[0], mat[1], mat[2], mat[3]);
                // printf("%f %f\n%f %f\n", mat_inv[0], mat_inv[1], mat_inv[2],
                //        mat_inv[3]);
                // printf("====\n\n");

                /* We want to store the grids with the correct prefactor ratio
                 */
                double prefactor_1 = mat_inv[0] * source_prefactor_1 *
                                     source_prefactor_2 * inv_1;
                double prefactor_2 = mat_inv[2] * source_prefactor_1 *
                                     source_prefactor_2 * inv_2;

                printf("Prefactors %f %f\n", prefactor_1, prefactor_2);

                /* Apply the prefactor for result 1 */
                #pragma omp for
                for (int k = 0; k < N * N * N; k++) {
                    result[k] *= prefactor_1;
                }

                /* Store the grid for result 1 */
                store_grid(sft, result, dest_index_1);

                /* Undo prefactor 1 and apply the prefactor for 2 */
                #pragma omp for
                for (int k = 0; k < N * N * N; k++) {
                    result[k] *= prefactor_2;
                }

                /* Store the grid for result 1 */
                store_grid(sft, result, dest_index_2);

                counter++;
            }
        }

        /* Determine the maximum index at this order for 'c' and 'd' */
        int N_Y_1 = 1 + find_coeff_max_index(spatial_coeffs, 'Y', l);
        int N_Y_2 = 1 + find_coeff_max_index(spatial_coeffs, 'Y', n - l);

        /* Loop over all pairs (Y_{l,i}, Y_{n-l,j}) for the Euler eq. */
        for (int i = 0; i < N_Y_1; i++) {
            for (int j = 0; j < N_Y_2; j++) {
                int source_index_1 =
                    find_coeff_index_require(spatial_coeffs, 'Y', l, i);
                int source_index_2 =
                    find_coeff_index_require(spatial_coeffs, 'Y', n - l, j);
                int source_index_time_1 =
                    find_coeff_index_require(time_coeffs, 'd', l, i);
                int source_index_time_2 =
                    find_coeff_index_require(time_coeffs, 'd', n - l, j);

                int dest_index_1 = add_coeff(spatial_coeffs, 'X', n, counter);
                int dest_index_2 = add_coeff(spatial_coeffs, 'Y', n, counter);
                int dest_index_time_1 =
                    find_coeff_index_require(time_coeffs, 'c', n, counter);
                int dest_index_time_2 =
                    find_coeff_index_require(time_coeffs, 'd', n, counter);

                /* Reset the result grid */
                memset(result, 0, N * N * N * sizeof(double));

                /* Retrieve the pre-factors */
                double Dn = exp(n * time);
                double source_prefactor_1 =
                    interp_time_factor(tft, time, source_index_time_1) *
                    exp(l * time);
                double source_prefactor_2 =
                    interp_time_factor(tft, time, source_index_time_2) *
                    exp((n - l) * time);
                double dest_prefactor_1 =
                    interp_time_factor(tft, time, dest_index_time_1) * Dn;
                double dest_prefactor_2 =
                    interp_time_factor(tft, time, dest_index_time_2) * Dn;

                double dest_derivative_1 =
                    interp_time_derivative(tft, time, dest_index_time_1) * Dn;
                double dest_derivative_2 =
                    interp_time_derivative(tft, time, dest_index_time_2) * Dn;

                // printf("%f %f\n", source_prefactor_1, source_prefactor_2);
                // printf("%f %f\n", dest_prefactor_1, dest_prefactor_2);
                // printf("%f %f\n", dest_derivative_1, dest_derivative_2);

                /* Prevent singularities */
                double inv_1 = (dest_prefactor_1 == 0) ? 0 : 1.0 / dest_prefactor_1;
                double inv_2 = (dest_prefactor_2 == 0) ? 0 : 1.0 / dest_prefactor_2;

                /* Fetch the input grids */
                fetch_grid(sft, input1, source_index_1);
                fetch_grid(sft, input2, source_index_2);

                /* Compute the output grid */
                grid_symmetric_grad(N, boxlen, input1, input2, result);
                grid_grad_dot(N, boxlen, input1, input2, result);

                /* Solve for the inverse matrix */
                double mat[] = {
                    O_11 + dest_derivative_1 * inv_1 + n, O_12, O_21, O_22 + dest_derivative_2 * inv_2 + n
                };
                double mat_inv[4];
                matrix_inv_2d(mat, mat_inv);

                /* We want to store the grids with the correct prefactor ratio
                 */
                double prefactor_1 = mat_inv[1] * source_prefactor_1 *
                                     source_prefactor_2 * inv_1;
                double prefactor_2 = mat_inv[3] * source_prefactor_1 *
                                     source_prefactor_2 * inv_2;

                printf("Prefactors %f %f\n", prefactor_1, prefactor_2);

                /* Fourier transform the grid */
                fftw_plan r2c =
                    fftw_plan_dft_r2c_3d(N, N, N, result, fbox, FFTW_ESTIMATE);
                fft_execute(r2c);
                fft_normalize_r2c(fbox, N, boxlen);
                fftw_destroy_plan(r2c);

                /* Apply k-cutoff */
                int cutoff = 0;
                if (cutoff) {
                    double k_max = (4. / 3.) * 0.6737;
                    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_lowpass, &k_max);
                }

                /* Fourier transform back */
                fftw_plan c2r =
                    fftw_plan_dft_c2r_3d(N, N, N, fbox, result, FFTW_ESTIMATE);
                fft_execute(c2r);
                fft_normalize_c2r(result, N, boxlen);
                fftw_destroy_plan(c2r);

                /* Apply the prefactor for result 1 */
                #pragma omp for
                for (int k = 0; k < N * N * N; k++) {
                    result[k] *= prefactor_1;
                }

                /* Store the grid for result 1 */
                store_grid(sft, result, dest_index_1);

                /* Undo prefactor 1 and apply the prefactor for 2 */
                #pragma omp for
                for (int k = 0; k < N * N * N; k++) {
                    result[k] *= prefactor_2 / prefactor_1;
                }

                /* Store the grid for result 1 */
                store_grid(sft, result, dest_index_2);

                counter++;
            }
        }
    }

    free(input1);
    free(input2);
    free(result);
    free(fbox);

    return 0;
}

int aggregate_factors_at_n(struct spatial_factor_table *sft,
                           const struct time_factor_table *tft,
                           struct coeff_table *spatial_coeffs,
                           const struct coeff_table *time_coeffs, int n,
                           double time, double *density, double *flux) {

    const int N = sft->N;
    const double boxlen = sft->boxlen;
    double *input = malloc(N * N * N * sizeof(double));

    /* Determine the maximum index at this order for 'X' and 'Y' */
    int N_X = 1 + find_coeff_max_index(spatial_coeffs, 'X', n);
    int N_Y = 1 + find_coeff_max_index(spatial_coeffs, 'Y', n);

    /* Aggregate the density grid */
    for (int i = 0; i < N_X; i++) {
        int space_index = find_coeff_index_require(spatial_coeffs, 'X', n, i);
        int time_index = find_coeff_index_require(time_coeffs, 'c', n, i);

        /* Retrieve the pre-factor */
        double Dn = exp(n * time);
        double prefactor = interp_time_factor(tft, time, time_index) * Dn;
        // prefactor = 1.0;

        printf("prefactor %f %f %f\n", prefactor, Dn, prefactor / Dn);

        /* Fetch the grid */
        fetch_grid(sft, input, space_index);

        /* Add the result with the prefactor */
        // #pragma omp for
        for (int j = 0; j < N * N * N; j++) {
            density[j] += prefactor * input[j];
        }
    }

    /* Aggregate the energy flux grid */
    for (int i = 0; i < N_Y; i++) {
        int space_index = find_coeff_index_require(spatial_coeffs, 'Y', n, i);
        int time_index = find_coeff_index_require(time_coeffs, 'd', n, i);

        /* Retrieve the pre-factor */
        double Dn = exp(n * time);
        double prefactor = interp_time_factor(tft, time, time_index) * Dn;
        // prefactor = 1.0;

        printf("prefactor %f %f %f\n", prefactor, Dn, prefactor / Dn);

        /* Fetch the grid */
        fetch_grid(sft, input, space_index);

        /* Add the result with the prefactor */
        #pragma omp for
        for (int j = 0; j < N * N * N; j++) {
            flux[j] += prefactor * input[j];
        }
    }

    /* Free the intermediate grid */
    free(input);

    return 0;
}
