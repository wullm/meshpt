from classy import Class
import ctypes
import numpy as np
from matplotlib import pyplot as plt

params = {'output': 'mPk',
          'P_k_max_1/Mpc': 10,
          'N_ncdm': 1,
          'm_ncdm': 0.05,
          'deg_ncdm': 3,
          'z_pk': 40}
cosmo = Class()
cosmo.set(params)
cosmo.compute()

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
    Delta2 = Pow * avg_k**1.5
    B = np.array([avg_k, Delta2, Pow]).T
    C = B[np.isnan(avg_k) == False,:]
    #Convert to real numbers if Im(x) < eps
    C = np.real_if_close(C)

    return(C)

lib = ctypes.cdll.LoadLibrary('./meshpt.so')

N = 128
L = 256.0/0.67
z_f = 0
z_i = 499

N_SPT = 4 # order in perturbation theory

grid = np.zeros((N,N,N))
kvec = np.exp(np.log(10) * np.arange(-5, 1, 0.1));
nk = len(kvec)

Pvec = np.zeros(nk);
for i in range(nk):
    Pvec[i] = cosmo.pk_lin(kvec[i], z_f);

nz = 1000
zmax = 1000
zmin = 0

amax = 1./(1+zmax)
amin = 1./(1+zmin)
avec = np.exp(np.linspace(np.log(amin), np.log(amax), nz));
zvec = 1./avec - 1;
Dvec = np.zeros(nz)

for i in range(nz):
    Dvec[i] = cosmo.scale_independent_growth_factor(zvec[i]);

Dvec = Dvec/Dvec[-1]/(1+zvec[-1])

#Conver types to ctypes
c_N = ctypes.c_int(N);
c_L = ctypes.c_double(L);
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

#Run MeshPT
lib.run_meshpt(c_N, c_L, c_nk, c_grid, c_kvec, c_Pvec, c_nz, c_zvec, c_Dvec,
               c_N_SPT, c_z_i, c_z_f)

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
grid_lpt = (grid_lpt - grid_lpt.mean())/grid_lpt.mean() * .67

#Compute the power spectra
Nhalf = int(N/2)+1
mask = np.ones((N,N,Nhalf))
C_lpt = compute_PS(grid_lpt, N, L, z_f, mask)
C = compute_PS(grid, N, L, z_f, mask)
k = C[:,0]

#Cut off high k modes
C = C[k<1,:]
C_lpt = C_lpt[k<1,:]

#Plot the power spectrum
plt.loglog(C[:,0], C[:,1])
plt.loglog(C_lpt[:,0], C_lpt[:,1])
plt.show()
