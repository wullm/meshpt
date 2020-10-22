/*******************************************************************************
 * This file is part of Mitos.
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

#include <hdf5.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

#include "../include/calc_powerspec.h"
#include "../include/fft.h"

static inline double sinc(double x) {
    return (x == 0) ? 1 : sin(x)/x;
}

void calc_cross_powerspec(int N, double boxlen, const fftw_complex *box1,
                          const fftw_complex *box2, int bins, double *k_in_bins,
                          double *power_in_bins, int *obs_in_bins) {

    const double boxvol = boxlen*boxlen*boxlen;
    const double dk = 2*M_PI/boxlen;
    const double max_k = sqrt(3)*dk*N/2;
    const double min_k = dk;

    const double log_max_k = log(max_k);
    const double log_min_k = log(min_k);

    /* Reset the bins */
    for (int i=0; i<bins; i++) {
        k_in_bins[i] = 0;
        power_in_bins[i] = 0;
        obs_in_bins[i] = 0;
    }

    /* Calculate the power spectrum */
    double kx,ky,kz,k;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                /* Calculate the wavevector */
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                if (k==0) continue; //skip the DC mode

                /* Compute the bin */
                const float u = (log(k) - log_min_k) / (log_max_k - log_min_k);
                const int bin = floor((bins - 1) * u);
                const int id = row_major_half(x, y, z, N);

                assert(bin >= 0 && bin < bins);

                /* Compute the power <X,Y> with X,Y complex */
                double a1 = creal(box1[id]), a2 = creal(box2[id]);
                double b1 = cimag(box1[id]), b2 = cimag(box2[id]);
                double Power = a1*a2 + b1*b2;

                /* All except the z=0 and the z=N/2 planes count double */
                int multiplicity = (z==0 || z==N/2) ? 1 : 2;

                /* Add to the tables */
                k_in_bins[bin] += multiplicity * k;
				power_in_bins[bin] += multiplicity * Power;
				obs_in_bins[bin] += multiplicity;
            }
        }
    }

    /* Divide to obtain averages */
	for (int i=0; i<bins; i++) {
		k_in_bins[i] /= obs_in_bins[i];
		power_in_bins[i] /= obs_in_bins[i];
		power_in_bins[i] /= boxvol;
	}
}

void calc_cross_powerspec_2d(int N, double anglesize, const fftw_complex *box1,
                             const fftw_complex *box2, int bins, double *l_in_bins,
                             double *power_in_bins, int *obs_in_bins) {

    const double norm = pow(anglesize*M_PI/180/(N*N),2);
    const double dl = 360.0/anglesize;
    const double max_l = sqrt(3)*dl*N/2;
    const double min_l = dl;

    const double log_max_l = log(max_l);
    const double log_min_l = log(min_l);

    /* Reset the bins */
    for (int i=0; i<bins; i++) {
        l_in_bins[i] = 0;
        power_in_bins[i] = 0;
        obs_in_bins[i] = 0;
    }

    /* Calculate the power spectrum */
    double lx,ly,l;
    for (int x=0; x<N; x++) {
        for (int y=0; y<=N/2; y++) {
            /* Calculate the wavevector */
            lx = (x > N/2) ? (x - N)*dl : x*dl;
            ly = (y > N/2) ? (y - N)*dl : y*dl;
            l = sqrt(lx*lx + ly*ly);

            if (l==0) continue; //skip the DC mode

            /* Compute the bin */
            const float u = (log(l) - log_min_l) / (log_max_l - log_min_l);
            const int bin = floor((bins - 1) * u);
            const int id = x*(N/2+1) + y;

            assert(bin >= 0 && bin < bins);

            /* Compute the power <X,Y> with X,Y complex */
            double a1 = creal(box1[id]), a2 = creal(box2[id]);
            double b1 = cimag(box1[id]), b2 = cimag(box2[id]);
            double Power = a1*a2 + b1*b2;

            /* All except the z=0 and the z=N/2 planes count double */
            int multiplicity = (y==0 || y==N/2) ? 1 : 2;

            /* Add to the tables */
            l_in_bins[bin] += multiplicity * l;
			power_in_bins[bin] += multiplicity * Power;
			obs_in_bins[bin] += multiplicity;
        }
    }

    /* Divide to obtain averages */
	for (int i=0; i<bins; i++) {
		l_in_bins[i] /= obs_in_bins[i];
		power_in_bins[i] /= obs_in_bins[i];
		power_in_bins[i] *= norm;
	}
}
