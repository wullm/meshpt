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

#include "../include/fft.h"
#include "../include/fft_kernels.h"
#include "../include/spatial_operations.h"

int grid_grad_dot(int N, double boxlen, double *box1, double *box2,
                  double *result) {

    /* Allocate complex arrays */
    fftw_complex *fbox1 = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    fftw_complex *fbox2 = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));

    /* Allocate real intermediate arrays */
    double *int1 = malloc(N * N * N * sizeof(double));
    double *int2 = malloc(N * N * N * sizeof(double));

    /* The kernels to be applied for each directoin */
    const kernel_func grad_kerns[] = {kernel_dx, kernel_dy, kernel_dz};
    const kernel_func poisson_kerns[] = {
        kernel_dx_inv_poisson, kernel_dy_inv_poisson, kernel_dz_inv_poisson
    };

    for (int i = 0; i < 3; i++) {
        /* Copy over the inout grids */
        memcpy(int1, box1, N * N * N * sizeof(double));
        memcpy(int2, box2, N * N * N * sizeof(double));

        /* Prepare Fourier transforms */
        fftw_plan r2c1 =
            fftw_plan_dft_r2c_3d(N, N, N, int1, fbox1, FFTW_ESTIMATE);
        fftw_plan r2c2 =
            fftw_plan_dft_r2c_3d(N, N, N, int2, fbox2, FFTW_ESTIMATE);
        fftw_plan c2r1 =
            fftw_plan_dft_c2r_3d(N, N, N, fbox1, int1, FFTW_ESTIMATE);
        fftw_plan c2r2 =
            fftw_plan_dft_c2r_3d(N, N, N, fbox2, int2, FFTW_ESTIMATE);

        /* Fourier transform the incoming grids */
        fft_execute(r2c1);
        fft_execute(r2c2);

        /* Normalize the complex grids */
        fft_normalize_r2c(fbox1, N, boxlen);
        fft_normalize_r2c(fbox2, N, boxlen);

        /* Apply the kernels */
        fft_apply_kernel(fbox1, fbox1, N, boxlen, grad_kerns[i], NULL);
        fft_apply_kernel(fbox2, fbox2, N, boxlen, poisson_kerns[i], NULL);

        // /* Apply a k-cutoff to address UV divergences */
        // double k_max = (4./3.) * 0.6737;
        // fft_apply_kernel(fbox1, fbox1, N, boxlen, kernel_lowpass, &k_max);
        // fft_apply_kernel(fbox2, fbox2, N, boxlen, kernel_lowpass, &k_max);

        /* Fourier transform back */
        fft_execute(c2r1);
        fft_execute(c2r2);

        /* Normalize */
        fft_normalize_c2r(int1, N, boxlen);
        fft_normalize_c2r(int2, N, boxlen);

        /* Destroy Fourier plans */
        fftw_destroy_plan(r2c1);
        fftw_destroy_plan(r2c2);
        fftw_destroy_plan(c2r1);
        fftw_destroy_plan(c2r2);

        /* Add to the result */
        #pragma omp for
        for (int j = 0; j < N * N * N; j++) {
            result[j] += int1[j] * int2[j];
        }
    }

    /* Free the intermediate grids */
    free(int1);
    free(int2);
    free(fbox1);
    free(fbox2);

    return 0;
}

int grid_product(int N, double boxlen, double *box1, double *box2,
                 double *result) {

    #pragma omp for
    for (int j = 0; j < N * N * N; j++) {
        result[j] += box1[j] * box2[j];
    }

    return 0;
}

int grid_symmetric_grad(int N, double boxlen, double *box1, double *box2,
                        double *result) {

    /* Allocate complex arrays */
    fftw_complex *fbox1 = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    fftw_complex *fbox2 = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));

    /* Allocate real intermediate arrays */
    double *int1 = malloc(N * N * N * sizeof(double));
    double *int2 = malloc(N * N * N * sizeof(double));

    /* The kernels to be applied for each directoin */
    const kernel_func operations[] = {
        kernel_dx_dx_inv_poisson, kernel_dy_dy_inv_poisson,
        kernel_dz_dz_inv_poisson, kernel_dx_dy_inv_poisson,
        kernel_dx_dz_inv_poisson, kernel_dy_dz_inv_poisson
    };
    const double multiplicities[] = {1, 1, 1, 2, 2, 2};

    for (int i = 0; i < 6; i++) {
        /* Copy over the inout grids */
        memcpy(int1, box1, N * N * N * sizeof(double));
        memcpy(int2, box2, N * N * N * sizeof(double));

        /* Prepare Fourier transforms */
        fftw_plan r2c1 =
            fftw_plan_dft_r2c_3d(N, N, N, int1, fbox1, FFTW_ESTIMATE);
        fftw_plan r2c2 =
            fftw_plan_dft_r2c_3d(N, N, N, int2, fbox2, FFTW_ESTIMATE);
        fftw_plan c2r1 =
            fftw_plan_dft_c2r_3d(N, N, N, fbox1, int1, FFTW_ESTIMATE);
        fftw_plan c2r2 =
            fftw_plan_dft_c2r_3d(N, N, N, fbox2, int2, FFTW_ESTIMATE);

        /* Fourier transform the incoming grids */
        fft_execute(r2c1);
        fft_execute(r2c2);

        /* Normalize the complex grids */
        fft_normalize_r2c(fbox1, N, boxlen);
        fft_normalize_r2c(fbox2, N, boxlen);

        /* Apply the kernels */
        fft_apply_kernel(fbox1, fbox1, N, boxlen, operations[i], NULL);
        fft_apply_kernel(fbox2, fbox2, N, boxlen, operations[i], NULL);

        // /* Apply a k-cutoff to address UV divergences */
        // double k_max = (4./3.) * 0.6737;
        // fft_apply_kernel(fbox1, fbox1, N, boxlen, kernel_lowpass, &k_max);
        // fft_apply_kernel(fbox2, fbox2, N, boxlen, kernel_lowpass, &k_max);

        /* Fourier transform back */
        fft_execute(c2r1);
        fft_execute(c2r2);

        /* Normalize */
        fft_normalize_c2r(int1, N, boxlen);
        fft_normalize_c2r(int2, N, boxlen);

        /* Destroy Fourier plans */
        fftw_destroy_plan(r2c1);
        fftw_destroy_plan(r2c2);
        fftw_destroy_plan(c2r1);
        fftw_destroy_plan(c2r2);

        /* The multiplicity of the term */
        double m = multiplicities[i];

        /* Add to the result */
        #pragma omp for
        for (int j = 0; j < N * N * N; j++) {
            result[j] += m * int1[j] * int2[j];
        }
    }

    /* Free the intermediate grids */
    free(int1);
    free(int2);
    free(fbox1);
    free(fbox2);

    return 0;
}
