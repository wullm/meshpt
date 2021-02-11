from classy import Class
import ctypes
import numpy as np
from matplotlib import pyplot as plt

params = {'output': 'mPk',
          'P_k_max_1/Mpc': 10,
          'z_pk': 40}
cosmo = Class()
cosmo.set(params)
cosmo.compute()


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
