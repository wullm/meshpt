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

#ifndef INPUT_H
#define INPUT_H

#define DEFAULT_STRING_LENGTH 150

#define KM_METRES 1000
#define MPC_METRES 3.085677581282e22

#define SPEED_OF_LIGHT_METRES_SECONDS 2.99792e8
#define GRAVITY_G_SI_UNITS 6.67428e-11 // m^3 / kg / s^2
#define PLANCK_CONST_SI_UNITS 6.62607015e-34 //J s
#define BOLTZMANN_CONST_SI_UNITS 1.380649e-23 //J / K
#define ELECTRONVOLT_SI_UNITS 1.602176634e-19 // J

/* The .ini parser library is minIni */
#include "../parser/minIni.h"
// #include "../include/output.h"

struct params {
    /* Random parameters */
    long int Seed;

    /* Box parameters */
    int GridSize;
    double BoxLen;

    /* Simulation parameters */
    char *Name;
    char *PerturbFile;
    char MergeDarkMatterBaryons;

    /* Output parameters */
    char *OutputDirectory;
    char *OutputFilename;

};

struct units {
    double UnitLengthMetres;
    double UnitTimeSeconds;
    double UnitMassKilogram;
    double UnitTemperatureKelvin;
    double UnitCurrentAmpere;

    /* Units for the transfer function input data */
    double TransferUnitLengthMetres;
    int Transfer_hExponent; //1 for h/Mpc; 0 for 1/Mpc
    int Transfer_kExponent; //0 for CLASS; -2 for CAMB/CMBFAST/Eisenstein-Hu
    int Transfer_Sign; //-1 for CLASS; +1 for CAMB/CMBFAST/Eisenstein-Hu

    /* Physical constants in internal units */
    double SpeedOfLight;
    double GravityG;
    double hPlanck;
    double kBoltzmann;
    double ElectronVolt;
};

struct cosmology {
    double h;
    double n_s;
    double A_s;
    double k_pivot;
    double z_ini;
    double log_tau_ini; //conformal time
    double rho_crit; //not user-specified, but inferred from h
};

int readParams(struct params *parser, const char *fname);
int readUnits(struct units *us, const char *fname);
int readCosmology(struct cosmology *cosmo, struct units *us, const char *fname);

int cleanParams(struct params *parser);

int readFieldFile(double **box, int *N, double *box_len, const char *fname);
int readFieldFileInPlace(double *box, const char *fname);


static inline void generateFieldFilename(const struct params *pars, char *fname,
                                         const char *Identifier, const char *title,
                                         const char *extra) {
    sprintf(fname, "%s/%s_%s%s.%s", pars->OutputDirectory, title, extra,
            Identifier, "hdf5");
}


#endif
