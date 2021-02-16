from classy import Class
import ctypes
import numpy as np
from matplotlib import pyplot as plt

#Compute the power spectrum
def compute_PS(grid, N, L, z, mask):
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

    # #Undo window function
    # Wx = Wy = Wz = np.ones_like(KX)
    # Wx[KX != 0] = np.sin(0.5 * KX[KX != 0] * L / N)/(0.5 * KX[KX != 0] * L / N)
    # Wy[KY != 0] = np.sin(0.5 * KY[KY != 0] * L / N)/(0.5 * KY[KY != 0] * L / N)
    # Wz[KZ != 0] = np.sin(0.5 * KZ[KZ != 0] * L / N)/(0.5 * KZ[KZ != 0] * L / N)
    # W = (Wx * Wy * Wz)**2
    # Pgrid /= W

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

#We use Mpc and Gyr (no h)
speed_of_light = 306.601394 # Mpc/Gyr

#Cosmological parameters
h = 0.67
Ocdm = 0.26
Ob = 0.05

#Grid dimensions
N = 256
#Physical dimensions of the box
L = N * 2.0 / h

#Initial and final (desired output) redshifts of the SPT calculation
z_i = 9.5e13
z_f = 40

#Desired redshift of the linear theory power spectrum (can be set equal to z_i or something else for rescaled ICs)
z_lin = 0

#Desired order in perturbation theory
N_SPT = 1

#Do the linear theory calculation with CLASS
params = {'output': 'mPk,dTk',
          'P_k_max_1/Mpc': 10,
          # 'N_ncdm': 1,
          # 'm_ncdm': 0.05,
          # 'deg_ncdm': 3,
          'z_pk': 40,
          'h': h,
          'Omega_cdm': Ocdm,
          'Omega_b': Ob}
cosmo = Class()
cosmo.set(params)
cosmo.compute()

#Derive value of Omega_m density
Omega_m = cosmo.Om_m(0); #redshift zero

#Prepare logarithmically spaced array of wavenumbers
kvec = np.exp(np.log(10) * np.arange(-5, 1, 0.1));
nk = len(kvec)

#Interpolate the linear theory power spectrum
Pvec = np.zeros(nk);
for i in range(nk):
    Pvec[i] = cosmo.pk_lin(kvec[i], z_lin);

#Perform rescaling if necessary
if (not z_f == z_lin):
    print("Rescaling the linear theory power spectrum")
    D_f = cosmo.scale_independent_growth_factor(z_f)
    D_lin = cosmo.scale_independent_growth_factor(z_lin)
    print("D_f: ", D_f)
    print("D_lin: ", D_lin)
    Dratio = D_f / D_lin
    Pvec *= Dratio**2

#Get background quantities from CLASS (reverse the order of the arrays)
background = cosmo.get_background()
bg_t = background['proper time [Gyr]'][::-1]
bg_H = background['H [1/Mpc]'][::-1]
bg_D = background['gr.fac. D'][::-1]
bg_f = background['gr.fac. f'][::-1]
bg_z = background['z'][::-1]
bg_a = 1./(1+bg_z)
bg_rho = background['(.)rho_tot'][::-1]
bg_p = background['(.)p_tot'][::-1]

#Size of arrays of background quantities
nz = 1000
zmax = max(1000, z_f * 1.05, z_i * 1.05)
zmin = 0

#Curvature parameter
K = -cosmo.Omega0_k() * cosmo.Hubble(0)**2

#Compute conformal derivative of the Hubble rate
bg_H_prime = -1.5 * (bg_rho + bg_p) * bg_a + K / bg_a
bg_H_dot = bg_H_prime / bg_a
bg_Hdot_over_H2 = bg_H_dot / bg_H**2

#Normalize the growth factor
z_norm = z_f
D_norm = cosmo.scale_independent_growth_factor(z_norm)
a_norm = 1./(1+z_norm)
bg_D = bg_D/D_norm

#Compute central difference derivative of the logarithmic growth rate
bg_df_dlogD = np.gradient(bg_f) / np.gradient(np.log(bg_D))

#Prepare arrays of background factors
amax = 1./(1+zmax)
amin = 1./(1+zmin)
avec = np.exp(np.linspace(np.log(amin), np.log(amax), nz));
zvec = 1./avec - 1;

#Interpolate known quantities
Dvec = np.interp(zvec, bg_z, bg_D)
fvec = np.interp(zvec, bg_z, bg_f)
Hvec = np.interp(zvec, bg_z, bg_H) * speed_of_light # 1/Gyr
df_dlogD = np.interp(zvec, bg_z, bg_df_dlogD)
Hdot_over_H2 = np.interp(zvec, bg_z, bg_Hdot_over_H2)

#Retrieve matter density paramter
Omvec = np.zeros(nz)

#Interpolate arrays of background factors
for i in range(nz):
    Omvec[i] = cosmo.Om_m(zvec[i])

#Compute the time-dependent entries of the Omega matrix
Omega_21 = -1.5 * Omvec / fvec**2
Omega_22 = (2 + Hdot_over_H2 + df_dlogD)/fvec

#Replace redshifts with transformed growth factors
zvec = 1./Dvec - 1
D_f = np.interp(z_f, bg_z, bg_D)
D_i = np.interp(z_i, bg_z, bg_D)
z_f = 1./D_f - 1
z_i = 1./D_i - 1

#Allocate the grid
grid = np.zeros((N,N,N))

#Conver types to ctypes
c_N = ctypes.c_int(N);
c_L = ctypes.c_double(L);
c_Omega_m = ctypes.c_double(Omega_m);
c_nk = ctypes.c_int(nk);
c_nz = ctypes.c_int(nz);
c_N_SPT = ctypes.c_int(N_SPT);
c_z_i = ctypes.c_double(z_i);
c_z_f = ctypes.c_double(z_f);
c_grid = ctypes.c_void_p(grid.ctypes.data);
c_kvec = ctypes.c_void_p(kvec.ctypes.data);
c_Pvec = ctypes.c_void_p(Pvec.ctypes.data);
c_zvec = ctypes.c_void_p(zvec.ctypes.data);
c_Dvec = ctypes.c_void_p(Dvec.ctypes.data);
c_fvec = ctypes.c_void_p(fvec.ctypes.data);
c_Omega_21 = ctypes.c_void_p(Omega_21.ctypes.data);
c_Omega_22 = ctypes.c_void_p(Omega_22.ctypes.data);

#Run MeshPT
lib.run_meshpt(c_N, c_L, c_Omega_m, c_nk, c_grid, c_kvec, c_Pvec, c_nz, c_zvec, c_Dvec, c_fvec, c_Omega_21, c_Omega_22, c_N_SPT, c_z_i, c_z_f)

#Show a slice of the output density field
plt.imshow(grid[100:120].mean(axis=0), cmap="magma")
plt.show()

grid_lpt = np.zeros((N,N,N))
c_grid_lpt = ctypes.c_void_p(grid_lpt.ctypes.data);

seed = 101
PFac = 1.0
c_seed = ctypes.c_int(seed);
c_PFac = ctypes.c_double(PFac);

#Compute the LPT solution
lib.computeNonlinearGrid(c_N, c_L, c_seed, c_nk, c_PFac, c_kvec, c_Pvec, c_grid_lpt);

#Normalize the grids
# grid = (grid - grid.mean())/grid.mean()
grid_lpt = (grid_lpt - grid_lpt.mean())/grid_lpt.mean()

#Compute the power spectra
Nhalf = int(N/2)+1
mask = np.ones((N,N,Nhalf))
C_lpt = compute_PS(grid_lpt, N, L, z_f, mask)
C = compute_PS(grid, N, L, z_f, mask)
k = C[:,0]

#Cut off high k modes
C = C[k<1,:]
C_lpt = C_lpt[k<1,:]

#Retrieve the linear power spectrum at these points
Plin = np.zeros(len(C[:,0]))
for i in range(len(Plin)):
    k = C[i,0]
    Plin[i] = cosmo.pk_lin(k, z_lin) * Dratio**2

#Plot the power spectrum
plt.loglog(C[:,0], C[:,2], label="SPT")
plt.loglog(C[:,0], Plin, label="CLASS")
plt.loglog(C_lpt[:,0], C_lpt[:,2], label="LPT")
plt.legend()
plt.show()

print("k P(k)");
for i in range(len(C)):
    k = C[i,0]
    print(k, C[i,2], C_lpt[i,2], cosmo.pk_lin(k, z_lin) * Dratio**2);
