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
