include ../../../../Makefile.def

#==============================================================================
# sparseCuda Makefile
#==============================================================================

OBJS       = CudaGenBcsrLinSOE.o CudaGenBcsrLinSolver.o AmgXLinSolver.o CuDSSLinSolver.o ParameterUtils.o

# Include directories
INCLUDES   += -I$(CORE)/system_of_eqn/linearSOE/sparseCuda

ifdef CUDA
    # CUDA available - use NVCC for sparseCuda files
    CXXFLAGS += -D_CUDA
    
    ifdef AMGX
        # AMGX available
        CXXFLAGS += -D_AMGX
        $(info CUDA and AMGX available)
    endif
    
    ifdef CUDSS
        # CUDSS available
        CXXFLAGS += -D_CUDSS
        $(info CUDA and CUDSS available)
    endif
    
    $(info Compiling sparseCuda implementation with NVCC)
    
    # Force NVCC compilation for sparseCuda files
    CXX = nvcc
    CXXFLAGS += -x c++ --std=c++11
else
    # No CUDA available - use GCC with dummy implementations
    CXXFLAGS += -D_NO_CUDA -D_NO_AMGX -D_NO_CUDSS
    $(info CUDA not available - sparseCuda files will use dummy implementations (compiled with GCC))
endif

all:         $(OBJS)

# Miscellaneous
tidy:	
	@$(RM) $(RMFLAGS) Makefile.bak *~ #*# core

clean: tidy
	@$(RM) $(RMFLAGS) $(OBJS) *.o

spotless: clean
	@$(RM) $(RMFLAGS)

wipe: spotless

# DO NOT DELETE THIS LINE -- make depend depends on it.
