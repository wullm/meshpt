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

/* Fast 3x3 determinant */
static inline double det3(double *M) {
    return M[0] * (M[4] * M[8] - M[5] * M[7])
         - M[1] * (M[3] * M[8] - M[5] * M[6])
         + M[2] * (M[3] * M[7] - M[4] * M[6]);
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
    struct timeval time_stop, time_start;
    gettimeofday(&time_start, NULL);

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
    double *box = malloc(N*N*N * sizeof(double));

    /* Generate a complex Hermitian Gaussian random field */
    header(0, "Generating Primordial Fluctuations");
    generate_complex_grf(fbox, N, boxlen, &seed);
    enforce_hermiticity(fbox, N, boxlen);

    /* Apply the interpolated power spectrum */
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_sqrt_power_spline, &spline);

    /* Fourier transform the grid */
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);
    fft_execute(c2r);
    fft_normalize_c2r(box, N, boxlen);
    // fftw_destroy_plan(c2r);

    /* Apply the overall factor */
    for (int i=0; i<N*N*N; i++) {
        box[i] *= factor;
    }

    /* Apply spherical collapse transform */
    const double alpha = 1.68;
    for (int i=0; i<N*N*N; i++) {
        double d = box[i];
        if (d < alpha) {
            d = -3*pow(1-d/alpha, alpha/3)+3;
        } else {
            d = 3;
        }
        grid[i] = d;
    }

    // /* Fourier transform the grid back */
    // fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    // fft_execute(r2c);
    // fft_normalize_r2c(fbox, N, boxlen);
    //
    // /* For different smoothing levels */
    // for (int i=0; i<5; i++) {
    //     double R_smooth = pow(2,i);
    //
    //     /* Apply the smoothing filter */
    //     fft_apply_kernel(fbox, fbox, N, boxlen, kernel_gaussian, &R_smooth);
    //
    //     /* Fourier transform back */
    //     fft_execute(c2r);
    //     fft_normalize_c2r(box, N, boxlen);
    //
    //     /* Check whether the grid is below the limit */
    //     for (int j=0; j<N*N*N; j++) {
    //         double d = box[j];
    //         if (d >= alpha) {
    //             grid[j] = 3;
    //         }
    //     }
    //
    //     /* Fourier transform back */
    //     fft_execute(r2c);
    //     fft_normalize_r2c(fbox, N, boxlen);
    //
    //     /* Undo the smoothing filter */
    //     fft_apply_kernel(fbox, fbox, N, boxlen, kernel_gaussian_inv, &R_smooth);
    // }

    /* Fourier transform again (note the different real grid) */
    fftw_plan r2c2 = fftw_plan_dft_r2c_3d(N, N, N, grid, fbox, FFTW_ESTIMATE);
    fft_execute(r2c2);
    fft_normalize_r2c(fbox, N, boxlen);

    /* Approximate the potential with the Zel'dovich approximation */
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_inv_poisson, NULL);

    /* Allocate memory for the three displacement grids */
    fftw_complex *f_psi_xx = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    fftw_complex *f_psi_xy = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    fftw_complex *f_psi_xz = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    fftw_complex *f_psi_yy = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    fftw_complex *f_psi_yz = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    fftw_complex *f_psi_zz = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    double *psi_xx = malloc(N*N*N * sizeof(double));
    double *psi_xy = malloc(N*N*N * sizeof(double));
    double *psi_xz = malloc(N*N*N * sizeof(double));
    double *psi_yy = malloc(N*N*N * sizeof(double));
    double *psi_yz = malloc(N*N*N * sizeof(double));
    double *psi_zz = malloc(N*N*N * sizeof(double));

    /* Compute the displacements grids by differentiating the potential */
    fft_apply_kernel(f_psi_xx, fbox, N, boxlen, kernel_dx, NULL);
    fft_apply_kernel(f_psi_xx, f_psi_xx, N, boxlen, kernel_dx, NULL);

    fft_apply_kernel(f_psi_xy, fbox, N, boxlen, kernel_dx, NULL);
    fft_apply_kernel(f_psi_xy, f_psi_xy, N, boxlen, kernel_dy, NULL);

    fft_apply_kernel(f_psi_xz, fbox, N, boxlen, kernel_dx, NULL);
    fft_apply_kernel(f_psi_xz, f_psi_xz, N, boxlen, kernel_dz, NULL);

    fft_apply_kernel(f_psi_yy, fbox, N, boxlen, kernel_dy, NULL);
    fft_apply_kernel(f_psi_yy, f_psi_yy, N, boxlen, kernel_dy, NULL);

    fft_apply_kernel(f_psi_yz, fbox, N, boxlen, kernel_dy, NULL);
    fft_apply_kernel(f_psi_yz, f_psi_yz, N, boxlen, kernel_dz, NULL);

    fft_apply_kernel(f_psi_zz, fbox, N, boxlen, kernel_dz, NULL);
    fft_apply_kernel(f_psi_zz, f_psi_zz, N, boxlen, kernel_dz, NULL);

    /* Fourier transform the potential grids */
    fftw_plan c2r_xx = fftw_plan_dft_c2r_3d(N, N, N, f_psi_xx, psi_xx, FFTW_ESTIMATE);
    fftw_plan c2r_xy = fftw_plan_dft_c2r_3d(N, N, N, f_psi_xy, psi_xy, FFTW_ESTIMATE);
    fftw_plan c2r_xz = fftw_plan_dft_c2r_3d(N, N, N, f_psi_xz, psi_xz, FFTW_ESTIMATE);
    fftw_plan c2r_yy = fftw_plan_dft_c2r_3d(N, N, N, f_psi_yy, psi_yy, FFTW_ESTIMATE);
    fftw_plan c2r_yz = fftw_plan_dft_c2r_3d(N, N, N, f_psi_yz, psi_yz, FFTW_ESTIMATE);
    fftw_plan c2r_zz = fftw_plan_dft_c2r_3d(N, N, N, f_psi_zz, psi_zz, FFTW_ESTIMATE);
    fft_execute(c2r_xx);
    fft_execute(c2r_xy);
    fft_execute(c2r_xz);
    fft_execute(c2r_yy);
    fft_execute(c2r_yz);
    fft_execute(c2r_zz);
    fft_normalize_c2r(psi_xx, N, boxlen);
    fft_normalize_c2r(psi_xy, N, boxlen);
    fft_normalize_c2r(psi_xz, N, boxlen);
    fft_normalize_c2r(psi_yy, N, boxlen);
    fft_normalize_c2r(psi_yz, N, boxlen);
    fft_normalize_c2r(psi_zz, N, boxlen);

    for (int i=0; i<N*N*N; i++) {
        double d_dxx = -psi_xx[i];
        double d_dxy = -psi_xy[i];
        double d_dxz = -psi_xz[i];
        double d_dyy = -psi_yy[i];
        double d_dyz = -psi_yz[i];
        double d_dzz = -psi_zz[i];

        double M[] = {1+d_dxx, d_dxy, d_dxz,
                      d_dxy, 1+d_dyy, d_dyz,
                      d_dxz, d_dyz, 1+d_dzz};

        double det = fabs(det3(M));

        grid[i] = det;
    }

    // fftw_destroy_plan(r2c);
    //
    // /* Approximate the potential with the Zel'dovich approximation */
    // fft_apply_kernel(fbox, fbox, N, boxlen, kernel_inv_poisson, NULL);
    //
    // /* Allocate memory for the three displacement grids */
    // fftw_complex *f_psi_x = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    // fftw_complex *f_psi_y = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    // fftw_complex *f_psi_z = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    // double *psi_x = malloc(N*N*N * sizeof(double));
    // double *psi_y = malloc(N*N*N * sizeof(double));
    // double *psi_z = malloc(N*N*N * sizeof(double));
    //
    // /* Compute the displacements grids by differentiating the potential */
    // fft_apply_kernel(f_psi_x, fbox, N, boxlen, kernel_dx, NULL);
    // fft_apply_kernel(f_psi_y, fbox, N, boxlen, kernel_dy, NULL);
    // fft_apply_kernel(f_psi_z, fbox, N, boxlen, kernel_dz, NULL);
    //
    // /* Fourier transform the potential grids */
    // fftw_plan c2r_x = fftw_plan_dft_c2r_3d(N, N, N, f_psi_x, psi_x, FFTW_ESTIMATE);
    // fftw_plan c2r_y = fftw_plan_dft_c2r_3d(N, N, N, f_psi_y, psi_y, FFTW_ESTIMATE);
    // fftw_plan c2r_z = fftw_plan_dft_c2r_3d(N, N, N, f_psi_z, psi_z, FFTW_ESTIMATE);
    // fft_execute(c2r_x);
    // fft_execute(c2r_y);
    // fft_execute(c2r_z);
    // fft_normalize_c2r(psi_x, N, boxlen);
    // fft_normalize_c2r(psi_y, N, boxlen);
    // fft_normalize_c2r(psi_z, N, boxlen);
    // fftw_destroy_plan(c2r_x);
    // fftw_destroy_plan(c2r_y);
    // fftw_destroy_plan(c2r_z);
    //
    // /* Free the complex grids, which are no longer needed */
    // free(fbox);
    // free(f_psi_x);
    // free(f_psi_y);
    // free(f_psi_z);
    //
    // // /* Fourier transform again */
    // // fft_execute(c2r);
    // // fft_normalize_c2r(box, N, boxlen);
    // // fftw_destroy_plan(c2r);
    //
    // /* Reset the box array */
    // memset(box, 0, N*N*N*sizeof(double));
    //
    // /* Compute the density grid by CIC mass assignment */
    // // long double fac = N/boxlen;
    // // for (int x=0; x<N; x++) {
    // //     for (int y=0; y<N; y++) {
    // //         for (int z=0; z<N; z++) {
    // //             double dx = psi_x[row_major(x,y,z,N)];
    // //             double dy = psi_y[row_major(x,y,z,N)];
    // //             double dz = psi_z[row_major(x,y,z,N)];
    // //
    // //             double X = x - dx*fac;
    // //             double Y = y - dy*fac;
    // //             double Z = z - dz*fac;
    // //
    // //             int iX = (int) floor(X);
    // //             int iY = (int) floor(Y);
    // //             int iZ = (int) floor(Z);
    // //
    // //             for (int i=-1; i<=1; i++) {
    // //     			for (int j=-1; j<=1; j++) {
    // //     				for (int k=-1; k<=1; k++) {
    // //                         double xx = fabs(X - (iX + i));
    // //                         double yy = fabs(Y - (iY + j));
    // //                         double zz = fabs(Z - (iZ + k));
    // //
    // //                         double part_x = xx <= 1 ? 1 - xx : 0;
    // //                         double part_y = yy <= 1 ? 1 - yy : 0;
    // //                         double part_z = zz <= 1 ? 1 - zz : 0;
    // //
    // //                         box[row_major(iX+i, iY+j, iZ+k, N)] += 1.0 * part_x * part_y * part_z;
    // //                     }
    // //                 }
    // //             }
    // //         }
    // //     }
    // // }
    //
    // /* The decomposition of every cube into 6 tetrahedra is via */
    // const int decomposition[24] = {4,0,3,1, 7,4,3,1, 7,5,4,1, 7,2,5,1,
    //                         7,3,2,1, 7,6,5,2};
    //
    // /* Compute the density grid by CIC mass assignment */
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
    // memcpy(grid, box, N*N*N*sizeof(double));

    /* Free memory */
    // free(fbox);
    free(box);

    /* Clean up the spline */
    cleanPowerSpline(&spline);

    /* Timer */
    gettimeofday(&time_stop, NULL);
    long unsigned microsec = (time_stop.tv_sec - time_start.tv_sec) * 1000000
                           + time_stop.tv_usec - time_start.tv_usec;
    message(0, "\nTime elapsed: %.5f s\n", microsec/1e6);

    return 1;
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Read options */
    const char *fname = argv[1];
    header(0, "MeshPT Realization Generator");
    message(0, "The parameter file is '%s'\n", fname);

    /* Timer */
    struct timeval time_stop, time_start;
    gettimeofday(&time_start, NULL);

    /* MeshPT structuress */
    struct params pars;
    struct units us;
    struct cosmology cosmo;
    // struct perturb_data ptdat;
    // struct perturb_spline spline;
    // struct perturb_params ptpars;

    /* Read parameter file for parameters, units, and cosmological values */
    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, &us, fname);


    message(0, "The output directory is '%s'.\n", pars.OutputDirectory);
    message(0, "Creating initial conditions for '%s'.\n", pars.Name);

    /* Read the perturbation data file */
    // readPerturb(&pars, &us, &ptdat);
    // readPerturbParams(&pars, &us, &ptpars);

    /* Do a sanity check */
    // if (fabs(cosmo.h - ptpars.h) / cosmo.h > 1e-5) {
    //     catch_error(1, "ERROR: h from parameter file does not match perturbation file.\n");
    // }

    /* Merge cdm & baryons into one set of transfer functions (replacing cdm) */
    // if (pars.MergeDarkMatterBaryons) {
    //     header(rank, "Merging cdm & baryon transfer functions, replacing cdm.");
    //
    //     /* The indices of the density transfer functions */
    //     int index_cdm = findTitle(ptdat.titles, "d_cdm", ptdat.n_functions);
    //     int index_b = findTitle(ptdat.titles, "d_b", ptdat.n_functions);
    //
    //     /* Find the present-day background densities */
    //     int today_index = ptdat.tau_size - 1; // today corresponds to the last index
    //     double Omega_cdm = ptdat.Omega[ptdat.tau_size * index_cdm + today_index];
    //     double Omega_b = ptdat.Omega[ptdat.tau_size * index_b + today_index];
    //
    //     /* Do a sanity check */
    //     assert(fabs(Omega_b - ptpars.Omega_b) / Omega_b < 1e-5);
    //
    //     /* Use the present-day densities as weights */
    //     double weight_cdm = Omega_cdm / (Omega_cdm + Omega_b);
    //     double weight_b = Omega_b / (Omega_cdm + Omega_b);
    //
    //     message(rank, "Using weights [w_cdm, w_b] = [%f, %f]\n", weight_cdm, weight_b);
    //
    //     /* Merge the density & velocity transfer runctions, replacing cdm */
    //     mergeTransferFunctions(&ptdat, "d_cdm", "d_b", weight_cdm, weight_b);
    //     mergeTransferFunctions(&ptdat, "t_cdm", "t_b", weight_cdm, weight_b);
    //     /* Merge the background densities, replacing cdm */
    //     mergeBackgroundDensities(&ptdat, "d_cdm", "d_b", 1.0, 1.0); //replace with sum
    // }

    /* Initialize the interpolation spline for the perturbation data */
    // initPerturbSpline(&spline, DEFAULT_K_ACC_TABLE_SIZE, &ptdat);

    /* Seed the random number generator */
    rng_state seed = rand_uint64_init(pars.Seed);

    /* Determine the starting conformal time */
    // cosmo.log_tau_ini = perturbLogTauAtRedshift(&spline, cosmo.z_ini);

    // /* Print some useful numbers */
    // if (rank == 0) {
    //     header(rank, "Settings");
    //     printf("Random numbers\t\t [seed] = [%ld]\n", pars.Seed);
    //     printf("Starting time\t\t [z, tau] = [%.2f, %.2f U_T]\n", cosmo.z_ini, exp(cosmo.log_tau_ini));
    //     printf("Primordial power\t [A_s, n_s, k_pivot] = [%.4e, %.4f, %.4f U_L]\n", cosmo.A_s, cosmo.n_s, cosmo.k_pivot);
    //
    //     header(rank, "Requested Particle Types");
    //     for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
    //         /* The current particle type */
    //         struct particle_type *ptype = types + pti;
    //         printf("Particle type '%s' (N^3 = %d^3).\n", ptype->Identifier, ptype->CubeRootNumber);
    //     }
    // }

    /* Create Gaussian random field */
    const int N = pars.GridSize;
    const double boxlen = pars.BoxLen;

    /* Allocate array for the primordial Gaussian field */
    fftw_complex *grf = malloc(N*N*(N/2+1) * sizeof(fftw_complex));
    double *grid = malloc(N*N*N * sizeof(double));

    // /* Allocate distributed memory arrays (one complex & one real) */
    // struct distributed_grid grf;
    // alloc_local_grid(&grf, N, boxlen, MPI_COMM_WORLD);
    //
    /* Generate a complex Hermitian Gaussian random field */
    header(0, "Generating Primordial Fluctuations");
    generate_complex_grf(grf, N, boxlen, &seed);
    enforce_hermiticity(grf, N, boxlen);

    /* Apply the bare power spectrum, without any transfer functions */
    fft_apply_kernel(grf, grf, N, boxlen, kernel_power_no_transfer, &cosmo);

    /* Fourier transform the grid */
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, grf, grid, FFTW_ESTIMATE);
    fft_execute(c2r);
    fft_normalize_c2r(grid, N, boxlen);
    fftw_destroy_plan(c2r);

    /* Apply spherical collapse transform */
    const double alpha = 1.68;
    for (int i=0; i<N*N*N; i++) {
        double d = grid[i];
        if (d > alpha) {
            d = -3*pow(1-d/alpha, alpha/3)+3;
        } else {
            d = 3;
        }
        grid[i] = d;
    }

    /* Fourier transform the grid back */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, grid, grf, FFTW_ESTIMATE);
    fft_execute(r2c);
    fft_normalize_r2c(grf, N, boxlen);
    fftw_destroy_plan(r2c);

    /* Approximate the potential with the Zel'dovich approximation */
    fft_apply_kernel(grf, grf, N, boxlen, kernel_inv_poisson, NULL);

    /* Free memory */
    free(grf);
    free(grid);

    // /* Apply the bare power spectrum, without any transfer functions */
    // fft_apply_kernel_dg(&grf, &grf, kernel_power_no_transfer, &cosmo);
    //
    // /* Execute the Fourier transform and normalize */
    // fft_c2r_dg(&grf);
    //
    // /* Generate a filename */
    // char grf_fname[DEFAULT_STRING_LENGTH];
    // sprintf(grf_fname, "%s/%s%s", pars.OutputDirectory, GRID_NAME_GAUSSIAN, ".hdf5");
    //
    // /* Export the real GRF */
    // int err = writeFieldFile_dg(&grf, grf_fname);
    // catch_error(err, "Error while writing '%s'.\n", fname);
    // message(rank, "Pure Gaussian Random Field exported to '%s'.\n", grf_fname);
    //
    // /* Create smaller (zoomed out) copies of the Gaussian random field */
    // for (int i=0; i<2; i++) {
    //     /* Size of the smaller grid */
    //     int M;
    //
    //     /* Generate a filename */
    //     char small_fname[DEFAULT_STRING_LENGTH];
    //
    //     /* We do this twice, once if the user requests a SmallGridSize and
    //      * another time if the user requests a FireboltGridSize. */
    //     if (i == 0) {
    //         M = pars.SmallGridSize;
    //         sprintf(small_fname, "%s/%s%s", pars.OutputDirectory,  GRID_NAME_GAUSSIAN_SMALL, ".hdf5");
    //     } else {
    //         M = pars.FireboltGridSize;
    //         sprintf(small_fname, "%s/%s%s", pars.OutputDirectory,  GRID_NAME_GAUSSIAN_FIREBOLT, ".hdf5");
    //     }
    //
    //     if (M > 0) {
    //         /* Allocate memory for the smaller grid on each node */
    //         double *grf_small = fftw_alloc_real(M * M * M);
    //
    //         /* Shrink (our local slice of) the larger grf grid */
    //         shrinkGrid_dg(grf_small, &grf, M, N);
    //
    //         /* Add the contributions from all nodes and send it to the root node */
    //         if (rank == 0) {
    //             MPI_Reduce(MPI_IN_PLACE, grf_small, M * M * M, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    //         } else {
    //             MPI_Reduce(grf_small, grf_small, M * M * M, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    //         }
    //
    //         /* Export the assembled smaller copy on the root node */
    //         if (rank == 0) {
    //             writeFieldFile(grf_small, M, boxlen, small_fname);
    //             message(rank, "Smaller copy of the Gaussian Random Field exported to '%s'.\n", small_fname);
    //         }
    //
    //         /* Free the small grid */
    //         fftw_free(grf_small);
    //     }
    // }
    //
    //
    //
    // /* Go back to momentum space */
    // fft_r2c_dg(&grf);
    //
    // /* Retrieve background densities from the perturbations data file */
    // header(rank, "Fetching Background Densities");
    // retrieveDensities(&pars, &cosmo, &types, &ptdat);
    // retrieveMicroMasses(&pars, &cosmo, &types, &ptpars);
    //
    // /* Find the interpolation index along the time dimension */
    // int tau_index; //greatest lower bound bin index
    // double u_tau; //spacing between subsequent bins
    // perturbSplineFindTau(&spline, cosmo.log_tau_ini, &tau_index, &u_tau);
    //
    // /* Allocate a second grid to compute densities */
    // struct distributed_grid grid;
    // alloc_local_grid(&grid, N, boxlen, MPI_COMM_WORLD);
    //
    // /* Allocate a third grid to compute the potential */
    // struct distributed_grid potential;
    // alloc_local_grid(&potential, N, boxlen, MPI_COMM_WORLD);
    //
    // /* Allocate a fourth grid to compute derivatives */
    // struct distributed_grid derivative;
    // alloc_local_grid(&derivative, N, boxlen, MPI_COMM_WORLD);
    //
    // /* Sanity check */
    // assert(grf.local_size == grid.local_size);
    // assert(grf.local_size == derivative.local_size);
    // assert(grf.X0 == grid.X0);
    // assert(grf.X0 == derivative.X0);
    //
    //
    // /* We calculate derivatives using FFT kernels */
    // const kernel_func derivative_kernels[] = {kernel_dx, kernel_dy, kernel_dz};
    // const char *letter[] = {"x_", "y_", "z_"};
    //
    // header(rank, "Computing Perturbation Grids");
    //
    // /* For each particle type, compute displacement & velocity grids */
    // for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
    //     struct particle_type *ptype = types + pti;
    //     const char *Identifier = ptype->Identifier;
    //     const char *density_title = ptype->TransferFunctionDensity;
    //     const char *velocity_title = ptype->TransferFunctionVelocity;
    //
    //     /* Generate filenames for the grid exports */
    //     char density_filename[DEFAULT_STRING_LENGTH];
    //     char potential_filename[DEFAULT_STRING_LENGTH];
    //     char velocity_filename[DEFAULT_STRING_LENGTH];
    //     char velopot_filename[DEFAULT_STRING_LENGTH];
    //     char derivative_filename[DEFAULT_STRING_LENGTH];
    //
    //     generateFieldFilename(&pars, density_filename, Identifier, GRID_NAME_DENSITY, "");
    //     generateFieldFilename(&pars, potential_filename, Identifier, GRID_NAME_POTENTIAL, "");
    //     generateFieldFilename(&pars, velocity_filename, Identifier, GRID_NAME_THETA, "");
    //     generateFieldFilename(&pars, velopot_filename, Identifier, GRID_NAME_THETA_POTENTIAL, "");
    //
    //     /* Generate density field, compute the potential and its derivatives */
    //     if (strcmp("", density_title) != 0) {
    //
    //         message(rank, "Computing density & displacement grids for '%s'.\n", Identifier);
    //
    //         /* Generate density grid by applying the transfer function to the GRF */
    //         err = generatePerturbationGrid(&cosmo, &spline, &grf, &grid, density_title, density_filename);
    //         catch_error(err, "Error while generating '%s'.", density_filename);
    //
    //         /* Fourier transform the density grid */
    //         fft_r2c_dg(&grid);
    //
    //         /* Should we solve the Monge-Ampere equation or approximate with Zel'dovich? */
    //         if (ptype->CyclesOfMongeAmpere > 0) {
    //             /* Solve the Monge Ampere equation */
    //             err = solveMongeAmpere(&potential, &grid, &derivative, ptype->CyclesOfMongeAmpere);
    //         } else {
    //             /* Approximate the potential with the Zel'dovich approximation */
    //             fft_apply_kernel_dg(&potential, &grid, kernel_inv_poisson, NULL);
    //         }
    //
    //         /* We now have the potential grid in momentum space */
    //         assert(potential.momentum_space == 1);
    //
    //         /* Undo the TSC window function for later */
    //         struct Hermite_kern_params Hkp;
    //         Hkp.order = 3; //TSC
    //         Hkp.N = N;
    //         Hkp.boxlen = boxlen;
    //
    //         /* Apply the kernel */
    //         fft_apply_kernel_dg(&potential, &potential, kernel_undo_Hermite_window, &Hkp);
    //
    //         /* Compute three derivatives of the potential grid */
    //         for (int i=0; i<3; i++) {
    //             /* Apply the derivative kernel */
    //             fft_apply_kernel_dg(&derivative, &potential, derivative_kernels[i], NULL);
    //
    //             /* Fourier transform to get the real derivative grid */
    //             fft_c2r_dg(&derivative);
    //
    //             /* Generate the appropriate filename */
    //             generateFieldFilename(&pars, derivative_filename, Identifier, GRID_NAME_DISPLACEMENT, letter[i]);
    //
    //             /* Export the derivative grid */
    //             writeFieldFile_dg(&derivative, derivative_filename);
    //         }
    //
    //         /* Finally, Fourier transform the potential grid to configuration space */
    //         fft_c2r_dg(&potential);
    //
    //         /* Export the potential grid */
    //         writeFieldFile_dg(&potential, potential_filename);
    //     }
    //
    //     /* Generate flux density field, flux potential, and its derivatives */
    //     if (strcmp("", velocity_title) != 0) {
    //
    //         message(rank, "Computing flux density & velocity grids for '%s'.\n", Identifier);
    //
    //         /* Generate flux grid by applying the transfer function to the GRF */
    //         err = generatePerturbationGrid(&cosmo, &spline, &grf, &grid, velocity_title, velocity_filename);
    //         catch_error(err, "Error while generating '%s'.", velocity_filename);
    //
    //         /* Fourier transform the flux density grid */
    //         fft_r2c_dg(&grid);
    //
    //         /* Compute flux potential grid by applying the inverse Poisson kernel */
    //         fft_apply_kernel_dg(&potential, &grid, kernel_inv_poisson, NULL);
    //
    //         /* Undo the TSC window function for later */
    //         struct Hermite_kern_params Hkp;
    //         Hkp.order = 3; //TSC
    //         Hkp.N = N;
    //         Hkp.boxlen = boxlen;
    //
    //         /* Apply the kernel */
    //         fft_apply_kernel_dg(&potential, &potential, kernel_undo_Hermite_window, &Hkp);
    //
    //         /* Compute three derivatives of the flux potential grid */
    //         for (int i=0; i<3; i++) {
    //             /* Apply the derivative kernel */
    //             fft_apply_kernel_dg(&derivative, &potential, derivative_kernels[i], NULL);
    //
    //             /* Fourier transform to get the real derivative grid */
    //             fft_c2r_dg(&derivative);
    //
    //             /* Generate the appropriate filename */
    //             generateFieldFilename(&pars, derivative_filename, Identifier, GRID_NAME_VELOCITY, letter[i]);
    //
    //             /* Export the derivative grid */
    //             writeFieldFile_dg(&derivative, derivative_filename);
    //         }
    //
    //         /* Finally, Fourier transform the flux potential grid to configuration space */
    //         fft_c2r_dg(&potential);
    //
    //         /* Export the flux potential grid */
    //         writeFieldFile_dg(&potential, velopot_filename);
    //     }
    // }
    //
    // /* We are done with the GRF, density, and derivative grids */
    // free_local_grid(&grid);
    // free_local_grid(&potential);
    // free_local_grid(&grf);
    // free_local_grid(&derivative);

    // /* Compute SPT grids */
    // header(rank, "Computing SPT Corrections");
    // err = computePerturbedGrids(&pars, &us, &cosmo, types, GRID_NAME_DENSITY, GRID_NAME_THETA);
    // if (err > 0) exit(1);

    /* Clean up */
    cleanParams(&pars);

    /* Timer */
    gettimeofday(&time_stop, NULL);
    long unsigned microsec = (time_stop.tv_sec - time_start.tv_sec) * 1000000
                           + time_stop.tv_usec - time_start.tv_usec;
    message(0, "\nTime elapsed: %.5f s\n", microsec/1e6);

}
