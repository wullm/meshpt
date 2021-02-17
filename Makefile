#Compiler options
GCC = gcc

#Libraries
INI_PARSER = parser/minIni.o
STD_LIBRARIES = -lm
FFTW_LIBRARIES = -lfftw3
HDF5_LIBRARIES = -lhdf5
GSL_LIBRARIES = -lgsl -lgslcblas

GSL_INCLUDES =

HDF5_INCLUDES += -I/usr/lib/x86_64-linux-gnu/hdf5/serial/include
HDF5_LIBRARIES += -L/usr/lib/x86_64-linux-gnu/hdf5/serial -I/usr/include/hdf5/serial

#Putting it together
INCLUDES = $(HDF5_INCLUDES) $(GSL_INCLUDES)
# $(INI_PARSER)
LIBRARIES = $(STD_LIBRARIES) $(FFTW_LIBRARIES) $(HDF5_LIBRARIES) $(GSL_LIBRARIES)
CFLAGS = -Wall -fopenmp -march=native -O4 -Wshadow -fPIC

OBJECTS = lib/*.o

all:
	# make minIni
	mkdir -p lib
	mkdir -p cache
	$(GCC) src/random.c -c -o lib/random.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/grf.c -c -o lib/grf.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/fft.c -c -o lib/fft.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/indices.c -c -o lib/indices.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/time_factors.c -c -o lib/time_factors.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/spatial_factors.c -c -o lib/spatial_factors.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/spatial_cache.c -c -o lib/spatial_cache.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/spatial_operations.c -c -o lib/spatial_operations.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/meshpt.c -o meshpt $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS)

	$(GCC) src/meshpt.c -o meshpt.so -shared $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(LDFLAGS)

	# make analyse_tools

format:
	clang-format-10 -style="{BasedOnStyle: LLVM, IndentWidth: 4, AlignConsecutiveMacros: true, IndentPPDirectives: AfterHash}" -i src/*.c include/*.h
	astyle -i src/*.c include/*.h
