from classy import Class
import ctypes
import numpy as np
from matplotlib import pyplot as plt

#Run MeshPT using input tables
PowData = np.loadtxt("PowData.txt")
CosmoData = np.loadtxt("CosmoData.txt")

#Compute the power spectrum
def compute_PS(grid, N, L, z, mask, defilter):
    #Half the grid length rounded down
    Nhalf = int(N/2)

    #Calculate the wavevectors
    dk = 2*np.pi / L
    modes = np.arange(N)
    modes[modes > N/2] -= N
    kx = modes * dk
    ky = modes * dk
    kz = modes[:Nhalf+1] * dk

    #Calculate the N*N*(N/2+1) grid of wavenumbers
    KX,KY,KZ = np.meshgrid(kx,ky,kz)
    K2 = KX**2 + KY**2 + KZ**2
    k_cube = np.sqrt(K2)

    #Make a copy of the grid so as not to destroy the input
    grid_cp = grid * 1.0

    #Fourier transform the grid
    fgrid = np.fft.rfftn(grid_cp)
    fgrid = fgrid * (L*L*L) / (N*N*N)
    Pgrid = np.abs(fgrid)**2

    #The Nyquist frequency
    k_nyq = np.pi * N / L

    #Undo the CIC window function if desired
    if (defilter == True):
        Wx = Wy = Wz = np.ones_like(KX)
        Wx[KX != 0] = np.sin(0.5 * KX[KX != 0] * L / N)/(0.5 * KX[KX != 0] * L / N)
        Wy[KY != 0] = np.sin(0.5 * KY[KY != 0] * L / N)/(0.5 * KY[KY != 0] * L / N)
        Wz[KZ != 0] = np.sin(0.5 * KZ[KZ != 0] * L / N)/(0.5 * KZ[KZ != 0] * L / N)
        W = (Wx * Wy * Wz)**2
        Pgrid /= W

    #Multiplicity of modes (double count all planes but z==0 and z==N/2)
    mult = np.ones_like(fgrid) * 2
    mult[:,:,0] = 1
    mult[:,:,-1] = 1

    #Get rid of modes excluded by the mask
    eps = np.finfo(float).eps
    mult[np.abs(mask) < eps] = 0

    #Compute the bin edges at the given redshift
    delta_k = 2*np.pi / L
    k_min = delta_k
    k_max = np.sqrt(3) * delta_k * N/2
    kvec = np.arange(delta_k, k_max, delta_k) # 1/Mpc
    bin_edges = np.zeros(len(kvec)+1)
    bin_edges[0] = k_min
    bin_edges[-1] = k_max
    bin_edges[1:-1] = 0.5 * (kvec[1:] + kvec[:-1])

    #Compute the power spectrum
    obs = np.histogram(k_cube, bin_edges, weights = mult)[0]
    Pow = np.histogram(k_cube, bin_edges, weights = Pgrid * mult)[0]
    avg_k = np.histogram(k_cube, bin_edges, weights = k_cube * mult)[0]

    #Normalization
    Pow = Pow / obs
    Pow = Pow / (L*L*L)
    avg_k = avg_k / obs

    #Convert to real numbers if Im(x) < eps
    Pow = np.real_if_close(Pow)
    avg_k = np.real_if_close(avg_k)
    obs = np.real_if_close(obs)

    #Convert to "dimensionless" (has dimensions mK^2) power spectrum
    Delta2 = Pow * avg_k**3 / (2 * np.pi)
    B = np.array([avg_k, Delta2, Pow]).T
    C = B[np.isnan(avg_k) == False,:]
    #Convert to real numbers if Im(x) < eps
    C = np.real_if_close(C)

    return(C)

#The MeshPT library
lib = ctypes.cdll.LoadLibrary('./meshpt.so')

#Unpack the data
kvec = PowData[:,0].astype(np.double)
Pvec = PowData[:,1].astype(np.double)
sqrtPvec = np.sqrt(Pvec)
zvec = CosmoData[:,0].astype(np.double)
Dvec = CosmoData[:,1].astype(np.double)
Omega_21 = CosmoData[:,2].astype(np.double)
Omega_22 = CosmoData[:,3].astype(np.double)
logDvec = np.log(Dvec)

#Lengths of the vectors
nk = len(kvec)
nz = len(zvec)

#We use Mpc and Gyr (no h)
speed_of_light = 306.601394 # Mpc/Gyr

#Cosmological parameters
h = 0.67
Ocdm = 0.26
Ob = 0.05

#Grid dimensions
N = 256
#Physical dimensions of the box
L = N * 1.0 / h
#Cutoff scale for the primary grid (0 = no cutoff)
k_cutoff = h

#Initial and final (desired output) redshifts of the SPT calculation
z_i = 9.5e13
z_f = 0

#Desired redshift of the linear theory power spectrum (can be set equal to z_i or something else for rescaled ICs)
z_lin = 0

#Desired order in perturbation theory
N_SPT = 2

#Perform rescaling if necessary
D_0 = np.interp(0, zvec, Dvec)
D_f = np.interp(z_f, zvec, Dvec)
D_i = np.interp(z_i, zvec, Dvec)
D_lin = np.interp(z_lin, zvec, Dvec)
print("D_0: ", D_0)
print("D_f: ", D_f)
print("D_lin: ", D_lin)

#MeshPT needs a z=0 power spectrum, so one can rescale a power spectrum to z=0
#if a different input redshift is desired
if (not z_lin == 0):
    print("Rescaling the linear theory power spectrum")
    Pvec *= (D_0 / D_lin)**2

#Allocate the grid
grid = np.zeros((N,N,N))

#Conver types to ctypes
c_N = ctypes.c_int(N);
c_L = ctypes.c_double(L);
c_nk = ctypes.c_int(nk);
c_nz = ctypes.c_int(nz);
c_N_SPT = ctypes.c_int(N_SPT);
c_D_i = ctypes.c_double(D_i);
c_D_f = ctypes.c_double(D_f);
c_k_cutoff = ctypes.c_double(k_cutoff);
c_grid = ctypes.c_void_p(grid.ctypes.data);
c_kvec = ctypes.c_void_p(kvec.ctypes.data);
c_sqrtPvec = ctypes.c_void_p(sqrtPvec.ctypes.data);
c_logDvec = ctypes.c_void_p(logDvec.ctypes.data);
c_Omega_21 = ctypes.c_void_p(Omega_21.ctypes.data);
c_Omega_22 = ctypes.c_void_p(Omega_22.ctypes.data);

#Run MeshPT
lib.run_meshpt(c_N, c_L, c_grid, c_nk, c_kvec, c_sqrtPvec, c_nz, c_logDvec,
               c_Omega_21, c_Omega_22, c_N_SPT, c_D_i, c_D_f, c_k_cutoff)

#Show a slice of the output density field
plt.imshow(grid[100:120].mean(axis=0), cmap="magma")
plt.show()

# #Allocate a grid for the Lagrangian calculation
# grid_lpt = np.zeros((N,N,N))
# c_grid_lpt = ctypes.c_void_p(grid_lpt.ctypes.data);
#
# seed = 101
# PFac = 1.0
# c_seed = ctypes.c_int(seed);
# c_PFac = ctypes.c_double(PFac);
#
# Pvec_LPT_input = Pvec * (D_f/D_0)**2
# c_Pvec_LPT_input = ctypes.c_void_p(Pvec_LPT_input.ctypes.data);
#
# #Compute the LPT solution
# lib.computeNonlinearGrid(c_N, c_L, c_seed, c_nk, c_PFac, c_kvec,
#                          c_Pvec_LPT_input, c_grid_lpt);
#
# #Normalize the LPT grid
# grid_lpt = (grid_lpt - grid_lpt.mean())/grid_lpt.mean()

# #Compute the power spectra
# Nhalf = int(N/2)+1
# mask = np.ones((N,N,Nhalf))
# # C_lpt = compute_PS(grid_lpt, N, L, z_f, mask, True)
# C = compute_PS(grid, N, L, z_f, mask, False)
# C_lpt = np.zeros_like(C)
# k = C[:,0]
#
# #Cut off high k modes
# C = C[k<1,:]
# C_lpt = C_lpt[k<1,:]
#
# #Retrieve the linear power spectrum at these points
# Plin = np.interp(C[:,0], kvec, Pvec) * (D_f/D_0)**2
#
# #Plot the power spectrum
# plt.loglog(C[:,0], C[:,2], label="SPT")
# plt.loglog(C[:,0], Plin, label="CLASS")
# plt.loglog(C_lpt[:,0], C_lpt[:,2], label="LPT")
# plt.legend()
# plt.show()
#
# print("k P(k)");
# for i in range(len(C)):
#     k = C[i,0]
#     print(k, C[i,2], C_lpt[i,2], Plin[i]);
