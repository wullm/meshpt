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
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <complex.h>

#include "../include/meshpt.h"
// #include "../include/grf_ngeniclike.h"

double gridNGP(const double *box, int N, double x, double y, double z) {
    /* Integer grid position */
    int iX = (int) floor(x);
    int iY = (int) floor(y);
    int iZ = (int) floor(z);

    return box[row_major(iX, iY, iZ, N)];
}

double gridTSC(const double *box, int N, double x, double y, double z) {
    /* Integer grid position */
    int iX = (int) floor(x);
    int iY = (int) floor(y);
    int iZ = (int) floor(z);

    /* Intepolate the necessary fields with CIC or TSC */
    double lookLength = 1.5;
    int lookLftX = (int) floor((x-iX) - lookLength);
    int lookRgtX = (int) floor((x-iX) + lookLength);
    int lookLftY = (int) floor((y-iY) - lookLength);
    int lookRgtY = (int) floor((y-iY) + lookLength);
    int lookLftZ = (int) floor((z-iZ) - lookLength);
    int lookRgtZ = (int) floor((z-iZ) + lookLength);

    /* Accumulate */
    double sum = 0;
    for (int i=lookLftX; i<=lookRgtX; i++) {
        for (int j=lookLftY; j<=lookRgtY; j++) {
            for (int k=lookLftZ; k<=lookRgtZ; k++) {
                double xx = fabs(x - (iX+i));
                double yy = fabs(y - (iY+j));
                double zz = fabs(z - (iZ+k));

                double part_x = xx < 0.5 ? (0.75-xx*xx)
                                        : (xx < 1.5 ? 0.5*(1.5-xx)*(1.5-xx) : 0);
				double part_y = yy < 0.5 ? (0.75-yy*yy)
                                        : (yy < 1.5 ? 0.5*(1.5-yy)*(1.5-yy) : 0);
				double part_z = zz < 0.5 ? (0.75-zz*zz)
                                        : (zz < 1.5 ? 0.5*(1.5-zz)*(1.5-zz) : 0);

                sum += box[row_major(iX+i, iY+j, iZ+k, N)] * (part_x*part_y*part_z);
            }
        }
    }

    return sum;
}

/* Map an index 0-7 and base vector (x,y,z) to (x',y',z'). */
static inline int x_from_vertex_id(int x_0, int index) {
    return (index==0 || index==3 || index==4 || index==7) ? x_0 : (x_0 + 1);
}

static inline int y_from_vertex_id(int y_0, int index) {
    return (index==0 || index==1 || index==4 || index==5) ? y_0 : (y_0 + 1);
}

static inline int z_from_vertex_id(int z_0, int index) {
    return (index==0 || index==1 || index==2 || index==3) ? z_0 : (z_0 + 1);
}

int gridPowerSpec(int N, double boxlen, int bins, void *gridv, void *kbinsv,
                  void *Pbinsv, void *countbinsv) {

    /* The input grid */
    double *grid = (double *) gridv;

    /* Memory block for the output data */
    double *k_in_bins = (double *) kbinsv;
    double *power_in_bins = (double *) Pbinsv;
    int *obs_in_bins = (int *) countbinsv;

    /* Allocate memory for the Fourier transform */
    fftw_complex *fbox = malloc(N*N*(N/2+1) * sizeof(fftw_complex));

    /* Fourier transform the grid */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, grid, fbox, FFTW_ESTIMATE);
    fft_execute(r2c);
    fft_normalize_r2c(fbox, N, boxlen);
    fftw_destroy_plan(r2c);

    /* Compute the power spectrum */
    calc_cross_powerspec(N, boxlen, fbox, fbox, bins, k_in_bins,
                         power_in_bins, obs_in_bins);

    /* Free memory */
    free(fbox);

    return 1;
}

int computeNonlinearGrid(int N, double boxlen, long long inseed, int nk,
                         double factor, void *kvecv, void *power_linv,
                         void *gridv) {

    /* Timer */
    // struct timeval time_stop, time_start;
    // gettimeofday(&time_start, NULL);

    /* The input data (power spectrum table) */
    double *kvec = (double *) kvecv;
    double *power_lin = (double *) power_linv;

    /* Memory block for the output data */
    double *grid = (double *) gridv;

    /* Initialize power spectrum interpolation sline */
    struct power_spline spline = {kvec, power_lin, nk};
    initPowerSpline(&spline, 100);

    /* Seed the random number generator */
    rng_state seed = rand_uint64_init(inseed);

    /* Allocate array for the primordial Gaussian field */
    fftw_complex *fbox = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    fftw_complex *fbox2 = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    double *box = malloc(N*N*N * sizeof(double));
    /* Generate a complex Hermitian Gaussian random field */
    // header(0, "Generating Primordial Fluctuations");
    generate_complex_grf(fbox, N, boxlen, &seed);
    enforce_hermiticity(fbox, N, boxlen);

    /* Apply the interpolated power spectrum */
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_sqrt_power_spline, &spline);

    /* Fourier transform the grid */
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);
    fft_execute(c2r);
    fft_normalize_c2r(box, N, boxlen);

    /* Apply spherical collapse transform */
    const double alpha = 1.68;
    for (int i=0; i<N*N*N; i++) {
        double d = box[i] * factor;
        if (d < alpha) {
            d = -3*pow(1-d/alpha, alpha/3)+3;
        } else {
            d = 3;
        }
        grid[i] = d;
    }

    /* Fourier transform the grid back */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fft_execute(r2c);
    fft_normalize_r2c(fbox, N, boxlen);

    /* Prepare a second plan */
    fftw_plan c2r_alt = fftw_plan_dft_c2r_3d(N, N, N, fbox2, box, FFTW_ESTIMATE);

    /* For different smoothing levels */
    for (int i=0; i<5; i++) {
        double R_smooth = pow(2,i);

        /* Apply the smoothing filter */
        fft_apply_kernel(fbox2, fbox, N, boxlen, kernel_gaussian, &R_smooth);

        /* Fourier transform back */
        fft_execute(c2r_alt);
        fft_normalize_c2r(box, N, boxlen);

        /* Check whether the grid is below the limit */
        for (int j=0; j<N*N*N; j++) {
            if (box[j] >= alpha) {
                grid[j] = 3;
            }
        }
    }

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
    fftw_complex *f_psi_x = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    fftw_complex *f_psi_y = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    fftw_complex *f_psi_z = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    double *psi_x = malloc(N*N*N * sizeof(double));
    double *psi_y = malloc(N*N*N * sizeof(double));
    double *psi_z = malloc(N*N*N * sizeof(double));

    /* Compute the displacements grids by differentiating the potential */
    fft_apply_kernel(f_psi_x, fbox, N, boxlen, kernel_dx, NULL);
    fft_apply_kernel(f_psi_y, fbox, N, boxlen, kernel_dy, NULL);
    fft_apply_kernel(f_psi_z, fbox, N, boxlen, kernel_dz, NULL);

    /* Fourier transform the potential grids */
    fftw_plan c2r_x = fftw_plan_dft_c2r_3d(N, N, N, f_psi_x, psi_x, FFTW_ESTIMATE);
    fftw_plan c2r_y = fftw_plan_dft_c2r_3d(N, N, N, f_psi_y, psi_y, FFTW_ESTIMATE);
    fftw_plan c2r_z = fftw_plan_dft_c2r_3d(N, N, N, f_psi_z, psi_z, FFTW_ESTIMATE);
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

    // memcpy(grid, psi_x, N*N*N*sizeof(double));

    // /* Fourier transform again */
    // fft_execute(c2r);
    // fft_normalize_c2r(box, N, boxlen);
    // fftw_destroy_plan(c2r);

    /* Reset the box array */
    memset(box, 0, N*N*N*sizeof(double));

    /* Compute the density grid by CIC mass assignment */
    double fac = N/boxlen;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<N; z++) {

                double dx = psi_x[row_major(x, y, z, N)];
                double dy = psi_y[row_major(x, y, z, N)];
                double dz = psi_z[row_major(x, y, z, N)];

                double X = x - dx*fac;
                double Y = y - dy*fac;
                double Z = z - dz*fac;

                int iX = (int) floor(X);
                int iY = (int) floor(Y);
                int iZ = (int) floor(Z);

                for (int i=-1; i<=1; i++) {
        			for (int j=-1; j<=1; j++) {
        				for (int k=-1; k<=1; k++) {
                            double xx = fabs(X - (iX + i));
                            double yy = fabs(Y - (iY + j));
                            double zz = fabs(Z - (iZ + k));

                            double part_x = xx <= 1 ? 1 - xx : 0;
                            double part_y = yy <= 1 ? 1 - yy : 0;
                            double part_z = zz <= 1 ? 1 - zz : 0;

                            box[row_major(iX+i, iY+j, iZ+k, N)] += 1.0 * part_x * part_y * part_z;
                        }
                    }
                }
            }
        }
    }



    /* The decomposition of every cube into 6 tetrahedra is via */
    // const int decomposition[24] = {4,0,3,1, 7,4,3,1, 7,5,4,1, 7,2,5,1,
    //                         7,3,2,1, 7,6,5,2};

    /* Compute the density grid by CIC mass assignment */
    // long double fac = N/boxlen;
    // for (int x=0; x<N; x++) {
    //     for (int y=0; y<N; y++) {
    //         for (int z=0; z<N; z++) {
    //             /* The six tetrahedra in this cube */
    //             for (int d=0; d<6; d++) {
	// 		        double barycentre_a_x = 0, barycentre_b_x = 0;
	// 		        double barycentre_a_y = 0, barycentre_b_y = 0;
	// 		        double barycentre_a_z = 0, barycentre_b_z = 0;
    //
	// 		        //For each of the 4 vertices
	// 		        for (int v=0; v<4; v++) {
	// 		            int vid = decomposition[v + d*4];
    //                     int id = row_major(x_from_vertex_id(x, vid), y_from_vertex_id(y, vid), z_from_vertex_id(z, vid), N);
    //
    //                     double X = x_from_vertex_id(x, vid) - psi_x[id]*fac;
    //                     double Y = y_from_vertex_id(y, vid) - psi_y[id]*fac;
    //                     double Z = z_from_vertex_id(z, vid) - psi_z[id]*fac;
    //
    //                     barycentre_b_x += X * 0.25;
    //                     barycentre_b_y += Y * 0.25;
    //                     barycentre_b_z += Z * 0.25;
    //
	// 		            // barycentre_b_x += 0.25 * cos(2*M_PI*(X / boxlen));
	// 		            // barycentre_b_y += 0.25 * cos(2*M_PI*(Y / boxlen));
	// 		            // barycentre_b_z += 0.25 * cos(2*M_PI*(Z / boxlen));
	// 		            // barycentre_a_x += 0.25 * sin(2*M_PI*(X / boxlen));
	// 		            // barycentre_a_y += 0.25 * sin(2*M_PI*(Y / boxlen));
	// 		            // barycentre_a_z += 0.25 * sin(2*M_PI*(Z / boxlen));
	// 		        }
    //
	// 				// //Determine the correctly wrapped coordinates
	// 				// double a_x = barycentre_a_x;
	// 				// double b_x = barycentre_b_x;
	// 				// double a_y = barycentre_a_y;
	// 				// double b_y = barycentre_b_y;
	// 				// double a_z = barycentre_a_z;
	// 				// double b_z = barycentre_b_z;
    //                 //
	// 				// //Convert back to our original coordinates
	// 				// double X = (atan2(-1*a_x, -1*b_x) + M_PI) * (boxlen / (2*M_PI));
	// 				// double Y = (atan2(-1*a_y, -1*b_y) + M_PI) * (boxlen / (2*M_PI));
	// 				// double Z = (atan2(-1*a_z, -1*b_z) + M_PI) * (boxlen / (2*M_PI));
    //
    //                 double X = barycentre_b_x;
    //                 double Y = barycentre_b_y;
    //                 double Z = barycentre_b_z;
    //
    //                 int iX = (int) floor(X);
    //                 int iY = (int) floor(Y);
    //                 int iZ = (int) floor(Z);
    //
    //                 for (int i=-1; i<=1; i++) {
    //                     for (int j=-1; j<=1; j++) {
    //                         for (int k=-1; k<=1; k++) {
    //                             double xx = fabs(X - (iX + i));
    //                             double yy = fabs(Y - (iY + j));
    //                             double zz = fabs(Z - (iZ + k));
    //
    //                             double part_x = xx <= 1 ? 1 - xx : 0;
    //                             double part_y = yy <= 1 ? 1 - yy : 0;
    //                             double part_z = zz <= 1 ? 1 - zz : 0;
    //
    //                             box[row_major(iX+i, iY+j, iZ+k, N)] += 1.0 * part_x * part_y * part_z / 24;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }


    /* Copy over into the output grid */
    memcpy(grid, box, N*N*N*sizeof(double));

    /* Free memory */
    free(box);

    /* Clean up the spline */
    cleanPowerSpline(&spline);

    // /* Timer */
    // gettimeofday(&time_stop, NULL);
    // long unsigned microsec = (time_stop.tv_sec - time_start.tv_sec) * 1000000
    //                        + time_stop.tv_usec - time_start.tv_usec;
    // message(0, "\nTime elapsed: %.5f s\n", microsec/1e6);

    return 1;
}
