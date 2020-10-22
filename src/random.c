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

#include <stdlib.h>
#include <math.h>

#include "../include/random.h"
#include "../include/meshpt.h"

/* Generate a uniform variable on the open unit interval */
double sampleUniform(rng_state *state) {
    const uint64_t A = rand_uint64(state);
    const double RM = (double) UINT64_MAX + 1;
    return ((double) A + 0.5) / RM;
}

/* Generate standard normal variable with Box-Mueller */
double sampleNorm(rng_state *state) {
    /* Generate random integers */
    const uint64_t A = rand_uint64(state);
    const uint64_t B = rand_uint64(state);
    const double RMax = (double) UINT64_MAX + 1;

    /* Map the random integers to the open (!) unit interval */
    const double u = ((double) A + 0.5) / RMax;
    const double v = ((double) B + 0.5) / RMax;

    /* Map to two Gaussians (the second is not used - inefficient) */
    const double z0 = sqrt(-2 * log(u)) * cos(2 * M_PI * v);
    //double z1 = sqrt(-2 * log(u)) * sin(2 * M_PI * v);

    return z0;
}

/* Fermi-Dirac function */
double fd_pdf(double x, void *params) {
    /* Unpack the parameters */
    double *pars = (double*) params;
    double T = pars[0];
    double mu = pars[1];

    /* Calculate the unnormalized Fermi-Dirac density function */
    // return (x <= 0) ? 0 : x*x/(exp((x - mu)/T) + 1)*0.25*(exp(-x / T / 4.67) + 3*exp(-x / T / 0.0935));
    // return (x <= 0) ? 0 : x*x/(exp((x - mu)/T) + 1)*(0.01 + exp(-x / T / 0.23375));
    return (x <= 0) ? 0 : x*x/(exp((x - mu)/T) + 1)*(0.6 + exp(-x / T / 1.868) + 3*exp(-x / T / 0.0935))/4.6;
}

/* Bose-Einstein function */
double be_pdf(double x, void *params) {
    /* Unpack the parameters */
    double *pars = (double*) params;
    double T = pars[0];
    double mu = pars[1];

    /* Calculate the unnormalized Fermi-Dirac density function */
    return (x <= 0) ? 0 : x*x/(exp((x - mu)/T) - 1);
}


double numericalCDF(double xl, double xr, int samples, pdf f, void *params) {
    /* Midpoint rule integration */
    double out = 0;
    double delta = (xr - xl)/samples;
    for (int i=0; i<samples; i++) {
        double x = xl + (i + 0.5) * delta;
        out += delta * f(x, params);
    }

    return out;
}


int initSampler(struct sampler *s, pdf f, double xl, double xr, void *params) {
    /* Set-up */
    int samples = NUMERICAL_CDF_SAMPLES;
    int search_table_length = SEARCH_TABLE_LENGTH;

    /* Store the parameters and endpoints */
    s->xl = xl;
    s->xr = xr;
    s->f = f;
    s->params = params;

    /* Normalize the pdf */
    s->norm = 1.0 / numericalCDF(xl, xr, samples, f, params);

    /* Create the intervals, starting with just one */
    s->intervalNum = 1;
    s->intervals = malloc(s->intervalNum * sizeof(struct interval));

    /* Initially, the first interval covers the entire domain */
    s->intervals[0].id = 0;
    s->intervals[0].l = xl;
    s->intervals[0].r = xr;
    s->intervals[0].Fl = 0.0;
    s->intervals[0].Fr = 1.0;

    /* The current interval under consideration */
    int current_interval_id = 0;

    char done = 0;
    while(!done) {
        /* The interval under consideration */
        struct interval *iv = &s->intervals[current_interval_id];

        /* Check if the interval is too big (covers more than 5%) */
        if (iv->Fr - iv->Fl > 0.05) {
            /* Split the interval in half */
            int err = splitInterval(s, current_interval_id);
            if (err > 0) return 1;

        } else if (iv->r >= xr) {
            /* Stop if we are at the end */
            done = 1;
        } else {
            /* Move on to the next interval */
            current_interval_id = iv->nid;
        }
    }

    /* Return to the first interval */
    current_interval_id = 0;

    /* Now calculate Hermite polynomials in intervals and split them up if
     * they are not monotonic or if the error is too big. */
    done = 0;
    while(!done) {
        /* The interval under consideration */
        struct interval *iv = &s->intervals[current_interval_id];

        /* Evaluate the normalized pdf at the endpoints */
        double fl = s->norm * f(iv->l, params);
        double fr = s->norm * f(iv->r, params);

        /* Calculate the cubic Hermite approximation */
        iv->a0 = iv->l;
        iv->a1 = (iv->Fr - iv->Fl)/fl;
        iv->a2 = 3*(iv->r - iv->l) - (iv->Fr - iv->Fl)*(2./fl + 1./fr);
        iv->a3 = 2*(iv->l - iv->r) + (iv->Fr - iv->Fl)*(1./fl + 1./fr);

        /* Evaluate the error at the midpoint */
        double u = 0.5*(iv->Fr + iv->Fl);
        double H = iv->a0 + iv->a1*0.5 + iv->a2*pow(0.5,2) + iv->a3*pow(0.5,3);
        iv->error = fabs(s->norm * numericalCDF(xl, H, samples, f, params) - u);

        /* Monotonicity check */
        double delta = (iv->Fr - iv->Fl)/(iv->r - iv->l);
        char monotonic = (delta <= 3*fl) && (delta <= 3*fr);

        /* If the error is too big or if the polynomial is not monotonic */
        if (iv->error > 1e-6 || !monotonic) {
            /* Split the interval in half */
            int err = splitInterval(s, current_interval_id);
            if (err > 0) return 1;

        } else if (iv->r >= xr) {
            /* Stop if we are at the end */
            done = 1;
        } else {
            /* Move on to the next interval */
            current_interval_id = iv->nid;
        }
    }



    /* Sort the intervals */
    qsort(s->intervals, s->intervalNum, sizeof(struct interval), compareByLeft);

    // for (int i=0; i<s->intervalNum; i++) {
    //     printf("%d %e %e\n", i, s->intervals[i].l, s->intervals[i].Fl);
    // }

    /* Allocate memory for the search table */
    s->I_max = search_table_length;
    s->index = (double*) malloc(s->I_max * sizeof(double));

    /* Generate the index search table */
    for (int i=0; i<s->I_max; i++) {
        double u = (double) i / s->I_max;

        /* Find the largest interval such that u > F(p) */
        double maxJ = 0;
        int int_i = 0;
        for(int j=0; j<s->intervalNum; j++) {
            if (s->intervals[j].Fr < u && s->intervals[j].r > maxJ) {
                maxJ = s->intervals[j].r;
                int_i = j;
            }
        }
        s->index[i] = int_i;
    }

    return 0;
}

/* Split an interval and half and update the links */
int splitInterval(struct sampler *s, int current_interval_id) {
    /* Set-up */
    int samples = NUMERICAL_CDF_SAMPLES;

    /* The current interval that will be halved */
    struct interval *iv = &s->intervals[current_interval_id];

    /* Split the interval in half */
    double m = iv->l + 0.5*(iv->r - iv->l);
    double Fm = s->norm * numericalCDF(s->xl, m, samples, s->f, s->params);

    /* ID of the new interval */
    int id = s->intervalNum;
    s->intervalNum++;

    /* Allocate more memory for the new interval */
    s->intervals = realloc(s->intervals, s->intervalNum * sizeof(struct interval));

    if (s->intervals == NULL) {
        printf("Error reallocating memory for the intervals.\n");
        return 1;
    }

    /* Update the pointer to the current interval */
    iv = &s->intervals[current_interval_id];

    /* Insert the right half as a new interval */
    s->intervals[id].id = id;
    s->intervals[id].l = m;
    s->intervals[id].r = iv->r;
    s->intervals[id].Fl = Fm;
    s->intervals[id].Fr = iv->Fr;
    s->intervals[id].nid = iv->nid; //link to the old interval's right-neighbour

    /* Update the old interval to cover just the left half */
    iv->r = m;
    iv->Fr = Fm;
    iv->nid = id; //link the left-half to the right-half

    return 0;
}

int cleanSampler(struct sampler *s) {
    free(s->intervals);
    free(s->index);

    return 0;
}

double samplerCustom(struct sampler *s, rng_state *state) {
    /* Generate uniform random variate */
    const uint64_t ul = rand_uint64(state);
    const double u = (double) ul / UINT64_MAX;

    /* Use the search table to find a nearby interval */
    const int I_max = s->I_max;
    int If = floor(u * I_max);
    int idx = s->index[If < I_max ? If : I_max-1];

    /* Find the exact interval, i.e. the largest interval such that u > F(p) */
    double maxJ = 0;
    int int_i = idx;
    for(int i = idx; i < s->intervalNum; i++) {
        if (s->intervals[i].Fr < u && s->intervals[i].r > maxJ) {
            maxJ = s->intervals[i].r;
            int_i = i;
        } else {
            break;
        }
    }

    struct interval *iv = &s->intervals[int_i];

    double u_tilde = (u - iv->Fl)/(iv->Fr - iv->Fl);
    double H = iv->a0 + iv->a1*u_tilde + iv->a2*u_tilde*u_tilde + iv->a3*u_tilde*u_tilde*u_tilde;

    return H;
}
