##----------------------------------------
## taken from Makefile.inc...

PLAT = linux
CPP = g++


LAPACK_LIB   = -llapack
BLAS_LIB     = -lblas
ARMA_LIB     = -larmadillo
ARPACK_LIB   = -larpack


CPP_WARNINGS = -Wall -pedantic-errors
CPP_DEBUG    = -g
CPP_OPTIM    = -O2

CXXFLAGS    = -std=c++17 $(CPP_DEBUG) $(CPP_WARNINGS) $(CPP_OPTIM)

ALL_LIBS     = $(LAPACK_LIB) $(BLAS_LIB)  $(ARPACK_LIB)

## --- to use the latest version of armadillo in MPIPKS
INC = -I/usr/local/armadillo/usr/local/include

DEPS = util.hpp model.hpp vec3.hpp

##----------------------------------------
## taken from examples/product/simple/Makefile...


qdyn: qhs.o util.o model.o
	$(CPP) $(INC) $(CXXFLAGS) -o qdyn qhs.o util.o model.o $(ALL_LIBS)

qhs.o: qhs.cpp $(DEPS)
	$(CPP) $(INC) $(CXXFLAGS) -c $<

%.o: %.cpp %.hpp $(DEPS)
	$(CPP) $(INC) $(CXXFLAGS) -c $<
	


.PHONY: clean
clean:
	rm -f *~ *.o core qdyn



