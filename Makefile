#Compiler options
GCC = gcc

#Libraries
INI_PARSER = parser/minIni.o
STD_LIBRARIES = -lm
FFTW_LIBRARIES = -lfftw3 -lfftw3_omp -lfftw3_mpi
HDF5_LIBRARIES = -lhdf5
GSL_LIBRARIES = -lgsl -lgslcblas

GSL_INCLUDES =

HDF5_INCLUDES += -I/usr/lib/x86_64-linux-gnu/hdf5/serial/include
HDF5_LIBRARIES += -L/usr/lib/x86_64-linux-gnu/hdf5/serial -I/usr/include/hdf5/serial

#Putting it together
INCLUDES = $(HDF5_INCLUDES) $(GSL_INCLUDES) $(FIREBOLT_INCLUDES)
LIBRARIES = $(INI_PARSER) $(STD_LIBRARIES) $(FFTW_LIBRARIES) $(HDF5_LIBRARIES) $(GSL_LIBRARIES)
CFLAGS = -Wall -Wshadow=global -fopenmp -march=native -O4 -fPIC
LDFLAGS =

OBJECTS = lib/*.o

all:
	make minIni
	mkdir -p lib
	$(GCC) src/input.c -c -o lib/input.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/random.c -c -o lib/random.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/fft.c -c -o lib/fft.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/grf.c -c -o lib/grf.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/primordial.c -c -o lib/primordial.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/power_spline.c -c -o lib/power_spline.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/calc_powerspec.c -c -o lib/calc_powerspec.o $(INCLUDES) $(CFLAGS)

	$(GCC) src/meshpt_lib.c -o meshpt.so -shared $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(LDFLAGS)

	$(GCC) src/meshpt.c -o meshpt $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(LDFLAGS)

	# $(GCC) src/output.c -c -o lib/output.o $(INCLUDES) $(CFLAGS)
	#
	# $(GCC) src/input_mpi.c -c -o lib/input_mpi.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/output_mpi.c -c -o lib/output_mpi.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/distributed_grid.c -c -o lib/distributed_grid.o $(INCLUDES) $(CFLAGS)
	#
	# $(GCC) src/header.c -c -o lib/header.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/random.c -c -o lib/random.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/fft.c -c -o lib/fft.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/grf.c -c -o lib/grf.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/grf_ngeniclike.c -c -o lib/grf_ngeniclike.o $(INCLUDES) $(CFLAGS)
	#
	# $(GCC) src/particle_types.c -c -o lib/particle_types.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/titles.c -c -o lib/titles.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/particle.c -c -o lib/particle.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/calc_powerspec.c -c -o lib/calc_powerspec.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/primordial.c -c -o lib/primordial.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/generate_grids.c -c -o lib/generate_grids.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/shrink_grids.c -c -o lib/shrink_grids.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/poisson.c -c -o lib/poisson.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/monge_ampere.c -c -o lib/monge_ampere.o $(INCLUDES) $(CFLAGS)
	#
	# $(GCC) src/spt_convolve.c -c -o lib/spt_convolve.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/spt_grid.c -c -o lib/spt_grid.o $(INCLUDES) $(CFLAGS)
	#
	# $(GCC) src/perturb_data.c -c -o lib/perturb_data.o $(INCLUDES) $(CFLAGS)
	# $(GCC) src/perturb_spline.c -c -o lib/perturb_spline.o $(INCLUDES) $(CFLAGS)
	#
	# $(GCC) src/firebolt_interface.c -c -o lib/firebolt_interface.o $(INCLUDES) $(CFLAGS)
	#
	# $(GCC) src/grids_interp.c -c -o lib/grids_interp.o $(INCLUDES) $(CFLAGS)

minIni:
	cd parser && make
