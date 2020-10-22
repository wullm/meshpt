import numpy as np
import ctypes
from matplotlib import pyplot as plt
from classy import Class
import h5py

N = 256
L = 64.0
bins = 50;

k_nyq = np.pi * N / L;
print("0.5 * k_nyq = ", 0.5 * k_nyq);

theRatios = np.zeros(bins);
thek = np.zeros(bins);

M = 1;

for sample in range(M):

    theP1 = np.zeros(bins);
    theP2 = np.zeros(bins);

    for run in range(2):
        seed = 100 + int(sample/2)
        phase_factor = -1 if sample%2 else 1


        A_s = 2.090524e-09
        # A_s *= 1.23
        h = 0.7737
        n_s = 0.954
        k_pivot = 0.05
        oc = 0.1194
        # oc *= 1.1
        ob = 0.02233
        om = oc + ob
        Om = om/h**2
        f = Om**(2/9)
        Ok = 0

        if (run == 1):
            # delta = -0.05
            # dH = (-Om - f/3)*delta
            # h *= (1 + dH)
            # Ok = -(Om + 2*f/3)*delta
            # oc *= (1-2*dH)
            # ob *= (1-2*dH)
            # n_s *= 0.99
            oc *= 0.95
            ob *= 0.95
            # A_s *= 0.99;

        print(seed, phase_factor, A_s)

        # Define your cosmology (what is not specified will be set to CLASS default parameters)
        params = {
            'output': 'mPk,dTk',
            'A_s': A_s,
            'n_s': n_s,
            'k_pivot': k_pivot,
            'h': h,
            'omega_b': ob,
            'omega_cdm': oc,
            'Omega_k': Ok,
            # 'N_ncdm': 1,
            # 'm_ncdm': 0.05,
            # 'deg_ncdm': 1,
            'k_step_sub': 0.015,
            'k_step_super': 0.0001,
            'P_k_max_1/Mpc': 100,
            'N_ur': 3.046,
            'z_pk': 0.3}

        # Create an instance of the CLASS wrapper
        cosmo = Class()

        # Set the parameters to the cosmological code
        cosmo.set(params)

        # Run the whole code. Depending on your output, it will call the
        # CLASS modules more or less fast. For instance, without any
        # output asked, CLASS will only compute background quantities,
        # thus running almost instantaneously.
        # This is equivalent to the beginning of the `main` routine of CLASS,
        # with all the struct_init() methods called.
        cosmo.compute()

        # Store the power spectrum in a vector
        z = 0.3;
        kvec = np.exp(np.log(10) * np.arange(-5, 1, 0.1));
        Pvec = np.zeros(len(kvec));
        for i in range(len(kvec)):
            Pvec[i] = cosmo.pk_lin(kvec[i], z);

        # Transfer = cosmo.get_transfer();
        # kvec = Transfer["k (h/Mpc)"] * h;
        # density = -Transfer["d_tot"] / kvec**2;
        # Pvec = A_s * (kvec/k_pivot)**n_s * density**2

        f=h5py.File("/home/qvgd89/mitos_spt2/mitos/output/density_cdm.hdf5", mode="r")


        #Include the MeshPT library
        lib = ctypes.cdll.LoadLibrary('./meshpt.so')

        nk = len(kvec);


        c_nk = ctypes.c_int(nk);
        cN = ctypes.c_int(N)
        cL = ctypes.c_double(L)
        cPFac = ctypes.c_double(phase_factor)
        cSeed = ctypes.c_longlong(seed)

        # Allocate memory for a grid
        grid = np.zeros(N**3);
        # grid = f["Field/Field"][:,:,:256]

        c_kvec = ctypes.c_void_p(kvec.ctypes.data);
        c_Pvec = ctypes.c_void_p(Pvec.ctypes.data);
        c_grid = ctypes.c_void_p(grid.ctypes.data);

        lib.computeNonlinearGrid(cN, cL, cSeed, c_nk, cPFac, c_kvec, c_Pvec, c_grid);

        grid = grid.reshape(N, N, N);

        # Calculate the power spectrum
        bins = 50
        k_in_bins = np.zeros(bins);
        P_in_bins = np.zeros(bins);
        obs_in_bins = np.zeros(bins).astype(int);

        c_bins = ctypes.c_int(bins);
        c_kbins = ctypes.c_void_p(k_in_bins.ctypes.data);
        c_Pbins = ctypes.c_void_p(P_in_bins.ctypes.data);
        c_obins = ctypes.c_void_p(obs_in_bins.ctypes.data);

        lib.gridPowerSpec(cN, cL, c_bins, c_grid, c_kbins, c_Pbins, c_obins);

        if (run == 0):
            thek = k_in_bins;
            theP1 = P_in_bins;
            link = kvec
            linP1 = Pvec
        else:
            theP2 = P_in_bins;
            linP2 = Pvec

    ratio = theP1 / theP2;
    theRatios += ratio/M;

#Fit polynomial
select = np.where((thek<3.14) * (1- np.isnan(thek)));
degree = 7
poly = np.poly1d(np.polyfit(np.log(thek[select][:-1]), theRatios[select][:-1], deg=degree));
finer_ln_ks = np.arange(np.log(thek[select][0]), np.log(thek[select][-1]), 0.001);
Rs = poly(finer_ln_ks);
ln_k_max = finer_ln_ks[Rs.argmax()];
k_max = np.exp(ln_k_max);
R_max = Rs.max();
half_max = (R_max-1)*0.75+1;

coeffs = poly.coeffs
coeffs[-1] -= half_max
roots_half_max = np.poly1d(coeffs).roots;
real_roots_half_max = roots_half_max[roots_half_max == roots_half_max.real].real;
real_roots_half_max.sort();
left_root_half_max = real_roots_half_max[real_roots_half_max<ln_k_max][-1];
right_root_half_max = real_roots_half_max[real_roots_half_max>ln_k_max][0];

k_left = np.exp(left_root_half_max)
k_right = np.exp(right_root_half_max)
FWHM = (k_right - k_left);

slope_left = poly.deriv()(left_root_half_max);
slope_right = poly.deriv()(right_root_half_max);

print("k_max = ", k_max);
print("R_max = ", R_max);
print("FWHM = ", FWHM)
print("k_left, k_right = ", k_left, k_right)
print("slope_left, slope_right = ", slope_left, slope_right)

linPRatio = linP1/linP2;
plt.semilogx(thek[:-1], theRatios[:-1]);
plt.semilogx(link, linPRatio);
plt.semilogx(np.exp(finer_ln_ks), Rs);
plt.semilogx(np.exp(finer_ln_ks), 0*Rs + half_max);
plt.show();

# #Select bins with enough observations
# select = np.where(obs_in_bins > 5);
# k_in_bins = k_in_bins[select];
# P_in_bins = P_in_bins[select];
# obs_in_bins = obs_in_bins[select];

# plt.loglog(k_in_bins, P_in_bins);
# plt.show();

# for i in range(len(k_in_bins)):
#     if not np.isnan(k_in_bins[i]):
#         print(k_in_bins[i], P_in_bins[i]);

# plt.imshow(grid[0:50].mean(axis=0), cmap="magma");
