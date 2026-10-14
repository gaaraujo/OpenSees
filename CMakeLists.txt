#==============================================================================
#
#        OpenSees -- Open System For Earthquake Engineering Simulation
#                Pacific Earthquake Engineering Research Center
#
#     (c) Copyright 1999-2021 The Regents of the University of California
#                             All Rights Reserved
# (Copyright and Disclaimer @ http://www.berkeley.edu/OpenSees/copyright.html)
#
#------------------------------------------------------------------------------
#
# Major sections:
# - General Setup
# - External Libreries (Conan or User supplied paths)
# - Libraries
# - Executables
# - Option Application
#
#==============================================================================

cmake_minimum_required(VERSION 3.18)
project(OpenSeesFramework LANGUAGES C CXX Fortran)


set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(OPS_EXTERN_SOURCE_DIR "${PROJECT_SOURCE_DIR}/OTHER/")
set(OPS_EXTERNALS_DIR "${PROJECT_SOURCE_DIR}/OTHER/")
set(OPS_BUNDLED_DIR "${PROJECT_SOURCE_DIR}/OTHER/")
set(OPS_SRC_DIR "${PROJECT_SOURCE_DIR}/SRC/")

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PROJECT_SOURCE_DIR}/cmake/cmake")
include(OpenSeesFunctions)

# include user config
include(${PROJECT_SOURCE_DIR}/Conf.cmake)

#
# for ubuntu this has to be before the Compiler SPECIFIC stuff below .. go figure!
#

find_package(MPI)



#cmake -DCMAKE_RULE_MESSAGES:BOOL=OFF -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON .

#
# Compiler SPECIFIC
#



if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
 MESSAGE("STATUS" "COMPILER: Clang (macOS Configuration)")

 set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0")
 set(BUILD_SHARED_LIBS OFF)

 # 1. Ask gfortran for the absolute path to the static (.a) files
 execute_process(COMMAND gfortran --print-file-name=libgfortran.a OUTPUT_VARIABLE GF_STATIC OUTPUT_STRIP_TRAILING_WHITESPACE)
 execute_process(COMMAND gfortran --print-file-name=libquadmath.a OUTPUT_VARIABLE QM_STATIC OUTPUT_STRIP_TRAILING_WHITESPACE)
 execute_process(COMMAND gfortran --print-file-name=libgcc.a OUTPUT_VARIABLE GCC_STATIC OUTPUT_STRIP_TRAILING_WHITESPACE)

 # 2. CLEAR THE IMPLICIT DYNAMIC FLAGS
 # This prevents CMake from adding '-lgfortran -lquadmath' to the end of the command
 set(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES "")

 # 3. Force the linker to use these specific archive files
 # link_libraries applies to all targets created after this line
 link_libraries("${GF_STATIC}" "${QM_STATIC}" "${GCC_STATIC}")
    
 # Optional: ensure we search for static libs first for other dependencies
 set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.dylib")
 # set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
 set(BUILD_SHARED_LIBS OFF)
 # set(CMAKE_EXE_LINKER_FLAGS "-static")

elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")

 # set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")
 set(BUILD_SHARED_LIBS OFF)
 # set(CMAKE_EXE_LINKER_FLAGS "-static")
 add_compile_options(-fPIC)
 set(CMAKE_POSITION_INDEPENDENT_CODE ON)
 set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
 MESSAGE("COMPILER: GNU")

elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
 MESSAGE("COMPILER: Intel")
 add_compile_options(-fPIC)

elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
 MESSAGE("COMPILER: MSVC")
 set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
 set(BUILD_SHARED_LIBS OFF)
 set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
 # Intel Fortran runtime (libmmt) duplicates UCRT math symbols with /MT (see Win64 sln)
 set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /FORCE:MULTIPLE")
 set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /FORCE:MULTIPLE")
else()
  MESSAGE("COMPILER: UKNOWN COMPILER ${CMAKE_CXX_COMPILER_ID}")
endif()


#
# external packages
#

add_library(OPS_External_packages INTERFACE)
add_library(OPS_OS_Specific_libs INTERFACE)

#
# Conan (manages external dependencies for a typical build)
#

MESSAGE("${CMAKE_BINARY_DIR}")


if(CONAN_EXPORTED)
    message(STATUS "USING CONAN (via Transparent Integration)")
    set(USING_CONAN TRUE)
else()
    message(STATUS "NOT USING CONAN (System Search)")
    include(OpenSeesFunctions) # Your custom logic
endif()

# Conan MSVC packages ship Release import libs only; map other optimized configs.
if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
  set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL Release)
endif()

find_package(ZLIB REQUIRED)
find_package(HDF5 REQUIRED)
find_package(TCL REQUIRED)

if(TARGET tcl::tcl)
	  
    message(STATUS "Found tcl::tcl target")
    include_directories(${TCL_INCLUDE_DIR})
    if (TCL_INCLUDE_DIR)
        find_file(TCL_INIT_FILE init.tcl
    	    PATHS 
            "${TCL_INCLUDE_DIR}/../lib"
            "${TCL_INCLUDE_DIR}/../lib/tcl8.6"
            NO_DEFAULT_PATH
    	)
     endif()

    if(TCL_INIT_FILE)
        message(STATUS "Confirmed TCL_INIT_FILE at: ${TCL_INIT_FILE}")
    else()
        message(WARNING "init.tcl not found automatically. Check your Tcl installation structure.")
    endif()

  else()
  
    # This is where Stampede3 usually lands. 
    # We manually create the lowercase target so the linker is happy.
    if(TCL_FOUND AND NOT TARGET tcl::tcl)
        add_library(tcl::tcl UNKNOWN IMPORTED)
        set_target_properties(tcl::tcl PROPERTIES
            IMPORTED_LOCATION "${TCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TCL_INCLUDE_PATH}"
        )
    endif()

    set(TCL_LIBRARIES tcl::tcl)
    include_directories(${TCL_INCLUDE_PATH})
    
    # Use the actual library path found to find init.tcl
    get_filename_component(TCL_LIB_DIR "${TCL_LIBRARY}" DIRECTORY)
    get_filename_component(TCL_INIT_FILE "${TCL_LIB_DIR}/tcl8.6/init.tcl" ABSOLUTE)

    message(STATUS "2 TCL_INIT_FILE at ${TCL_INIT_FILE}")
endif()

message(STATUS "3 TCL_INIT_FILE at ${TCL_INIT_FILE}")

# Prefer Windows dynamic tcl86t.dll so OpenFresco loadPackage shares the host Tcl
# runtime. On this branch the standard OpenSees/SP/MP targets use it when present.
set(OPS_TCL_TARGET tcl::tcl)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  include(OpenSeesWindowsDynamicTcl)
  ops_configure_windows_dynamic_tcl()
  if(DEFINED OPS_WINDOWS_DYNAMIC_TCL_LIBRARY)
    add_library(ops_windows_dynamic_tcl UNKNOWN IMPORTED)
    set_target_properties(ops_windows_dynamic_tcl PROPERTIES
      IMPORTED_LOCATION "${OPS_WINDOWS_DYNAMIC_TCL_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${OPS_WINDOWS_DYNAMIC_TCL_INCLUDE}"
      INTERFACE_LINK_LIBRARIES "ws2_32;netapi32;userenv"
    )
    set(OPS_TCL_TARGET ops_windows_dynamic_tcl)
    set(TCL_INCLUDE_PATH "${OPS_WINDOWS_DYNAMIC_TCL_INCLUDE}")
    set(TCL_INCLUDE_DIR "${OPS_WINDOWS_DYNAMIC_TCL_INCLUDE}")
    set(TCL_LIBRARY "${OPS_WINDOWS_DYNAMIC_TCL_LIBRARY}")
    set(TCL_LIBRARIES ops_windows_dynamic_tcl)
    if(EXISTS "${OPS_WINDOWS_DYNAMIC_TCL_SCRIPT_DIR}/init.tcl")
      set(TCL_INIT_FILE "${OPS_WINDOWS_DYNAMIC_TCL_SCRIPT_DIR}/init.tcl")
    endif()
    include_directories(${OPS_WINDOWS_DYNAMIC_TCL_INCLUDE})
    message(STATUS "Using Windows dynamic Tcl (tcl86t) for OpenSees/SP/MP")
  endif()
endif()


find_package(Eigen3 REQUIRED)

if(TARGET Eigen3::Eigen)
    # This pulls the include directory directly from the target object
    get_target_property(EIGEN_INC Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(${EIGEN_INC})
endif()
 
#if(Eigen3_FOUND)
#$   message(STATUS "OPS >>> FOUND EIGEN at ${Eigen3_INCLUDE_DIR}")  
#   include_directories(${Eigen3_INCLUDE_DIR})
#   add_compile_definitions(_EIGEN3)
#else()
#   message(STATUS "OPS >>> Could not find Eigen3")
#endif()


#
# add git log tag
#

find_package(Git QUIET)

if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} log -1 --format=%H
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else()
    set(GIT_VERSION "unknown")
endif()

# Pass git hash to C/C++ only; nvcc rsp quoting breaks on -DOPENSEES_VERSION="..."
add_compile_definitions($<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:OPENSEES_VERSION="${GIT_VERSION}">)


#
# define user-selectable options for end target
#

set_property(CACHE OPS_FINAL_TARGET PROPERTY STRINGS
    G3 OpenSees OpenSeesMP OpenSeesSP OpenSeesPy)

#
# OS Configuration:
#  defines: compile flags & user supplied paths and includes if not using conan
#

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  add_compile_definitions(_LINUX _UNIX _TCL85)
  if (NOT_USING_CONAN)
    include(OpenSeesDependenciesUnix)
  endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(OPS_Use_Graphics_Option None)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  add_compile_definitions(_LINUX _UNIX _TCL85)
  if (NOT_USING_CONAN)
    include(OpenSeesDependenciesUnix)
  endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(OPS_OS_Specific_libs INTERFACE wsock32 ws2_32)
    add_compile_definitions(_WIN32 _TCL85)
  if (NOT_USING_CONAN)
    include(OpenSeesDependenciesWin)
  endif()
endif()

#==============================================================================
#                            General Setup
#
#==============================================================================
#----------------------------------------------------------------
# Compilers
#----------------------------------------------------------------

# Fortran
#--------------------------------------
enable_language(Fortran)

# C++
#--------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_FORTRAN_STANDARD 08)

set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> cqls <TARGET> <LINK_FLAGS> <OBJECTS>")
#add_compile_options(-g)

# Warnings
#opensees_add_cxx_flag(
#    GNU  -Wall
#    MSVC /W0
#)

# Floating-point
#opensees_add_cxx_flag(
#    GNU  -ffloat-store
#    MSVC /fp:precise
#)

#----------------------------------------------------------------
# Global Includes
#----------------------------------------------------------------
# include paths to main abstract classes

# NOTE BeamIntegration and MatrixUtil need to be removed from element/forceBEamColumn

include_directories(
  ${OPS_SRC_DIR}
  ${OPS_SRC_DIR}/matrix
  ${OPS_SRC_DIR}/handler
  ${OPS_SRC_DIR}/database
  ${OPS_SRC_DIR}/element
  ${OPS_SRC_DIR}/element/forceBeamColumn
  ${OPS_SRC_DIR}/element/nonlinearBeamColumn/matrixutil
  ${OPS_SRC_DIR}/coordTransformation
  ${OPS_SRC_DIR}/damping
  ${OPS_SRC_DIR}/tagged
  ${OPS_SRC_DIR}/tagged/storage
  ${OPS_SRC_DIR}/recorder
  ${OPS_SRC_DIR}/renderer
  ${OPS_SRC_DIR}/damage
  ${OPS_SRC_DIR}/recorder/response
  ${OPS_SRC_DIR}/material
  ${OPS_SRC_DIR}/material/section
  ${OPS_SRC_DIR}/material/uniaxial
  ${OPS_SRC_DIR}/material/nD
  ${OPS_SRC_DIR}/graph/graph
  ${OPS_SRC_DIR}/graph/numberer
  ${OPS_SRC_DIR}/graph/partitioner
  ${OPS_SRC_DIR}/domain/component
  ${OPS_SRC_DIR}/domain/domain
  ${OPS_SRC_DIR}/domain/subdomain
  ${OPS_SRC_DIR}/domain/load
  ${OPS_SRC_DIR}/domain/loadBalancer
  ${OPS_SRC_DIR}/domain/pattern
  ${OPS_SRC_DIR}/domain/groundMotion
  ${OPS_SRC_DIR}/domain/node
  ${OPS_SRC_DIR}/domain/constraints
  ${OPS_SRC_DIR}/domain/region
  ${OPS_SRC_DIR}/analysis/algorithm
  ${OPS_SRC_DIR}/analysis/dof_grp
  ${OPS_SRC_DIR}/analysis/fe_ele
  ${OPS_SRC_DIR}/analysis/algorithm/equiSolnAlgo
  ${OPS_SRC_DIR}/analysis/algorithm/eigenAlgo
  ${OPS_SRC_DIR}/analysis/algorithm/domainDecompAlgo
  ${OPS_SRC_DIR}/analysis/analysis
  ${OPS_SRC_DIR}/analysis/integrator
  ${OPS_SRC_DIR}/analysis/handler
  ${OPS_SRC_DIR}/analysis/numberer
  ${OPS_SRC_DIR}/analysis/model
  ${OPS_SRC_DIR}/convergenceTest
  ${OPS_SRC_DIR}/modelbuilder
  ${OPS_SRC_DIR}/system_of_eqn
  ${OPS_SRC_DIR}/system_of_eqn/linearSOE
  ${OPS_SRC_DIR}/system_of_eqn/linearSOE/sparseCUDA
  ${OPS_SRC_DIR}/system_of_eqn/eigenSOE
  ${OPS_SRC_DIR}/actor/actor
  ${OPS_SRC_DIR}/actor/channel
  ${OPS_SRC_DIR}/actor/objectBroker
  ${OPS_SRC_DIR}/actor/message
  ${OPS_BUNDLED_DIR}/eigenAPI
)

# TODO: Temporary fix; include all directories with .h files through glob
file(GLOB_RECURSE ops_include_files CONFIGURE_DEPENDS
    ${OPS_SRC_DIR}/*.h
    ${OPS_BUNDLED_DIR}/*.h
)
set(OPS_INCLUDE_DIRS "")
foreach(filepath ${ops_include_files})
    get_filename_component(dir_path ${filepath} PATH)
    set(OPS_INCLUDE_DIRS ${OPS_INCLUDE_DIRS} ${dir_path})
endforeach()
list(REMOVE_DUPLICATES OPS_INCLUDE_DIRS)

# Makefile.incl does not add OTHER/METIS to FE_INCLUDES; bundled headers belong only to the METIS
# library target so <metis.h> resolves to system METIS 5 when installed (partition API).
set(OPS_INCLUDE_DIRS_FILTERED "")
foreach(_d IN LISTS OPS_INCLUDE_DIRS)
  if(NOT _d STREQUAL "${PROJECT_SOURCE_DIR}/OTHER/METIS")
    list(APPEND OPS_INCLUDE_DIRS_FILTERED "${_d}")
  endif()
endforeach()
set(OPS_INCLUDE_DIRS "${OPS_INCLUDE_DIRS_FILTERED}")

# METIS 5 for OPS_partition(). Bundled OTHER/METIS is METIS 4 — do not use it.
# Pass -DMETIS5_DIR=... (same style as MUMPS_DIR) or set env METIS5_DIR.
# Defaults: /usr on Linux; ../metis-5.1.0/install on Windows (see README-Windows-METIS5.md).
set(METIS5_DIR "" CACHE PATH "METIS 5 install prefix (include/metis.h + lib)")
if(NOT METIS5_DIR AND DEFINED ENV{METIS5_DIR} AND NOT "$ENV{METIS5_DIR}" STREQUAL "")
  set(METIS5_DIR "$ENV{METIS5_DIR}")
endif()
if(NOT METIS5_DIR)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(METIS5_DIR "/usr")
  elseif(WIN32)
    set(METIS5_DIR "${PROJECT_SOURCE_DIR}/../metis-5.1.0/install")
  endif()
endif()

set(OPENSEES_HAVE_METIS5_HEADER FALSE)
set(OPENSEES_METIS5_INCLUDE_DIR "")
if(METIS5_DIR AND EXISTS "${METIS5_DIR}/include/metis.h")
  # Ubuntu's metis.h puts METIS_VER_* after a long banner; keep enough of the file.
  file(READ "${METIS5_DIR}/include/metis.h" _opensees_metis_hdr LIMIT 12000)
  if(_opensees_metis_hdr MATCHES "METIS_VER_MAJOR")
    set(OPENSEES_HAVE_METIS5_HEADER TRUE)
    set(OPENSEES_METIS5_INCLUDE_DIR "${METIS5_DIR}/include")
  endif()
endif()

message(
  STATUS
  "METIS 5 (partition / OpenSeesMP): METIS5_DIR=${METIS5_DIR}, HAVE_METIS5_HEADER=${OPENSEES_HAVE_METIS5_HEADER}"
)

include_directories(${OPS_INCLUDE_DIRS})
# end TODO

# ##############################################################
#
# NUMERICAL and OpenSees LIBRARIES
#
# ##############################################################

#
# NUMERICAL LIBRARIES
#

add_subdirectory("${PROJECT_SOURCE_DIR}/OTHER/SuperLU_5.1.1")
set (SUPERLU_LIBRARIES SUPERLU)
add_subdirectory("${PROJECT_SOURCE_DIR}/OTHER/UMFPACK")
set (UMFPACK_LIBRARIES UMFPACK)
add_subdirectory("${PROJECT_SOURCE_DIR}/OTHER/AMD")
set (AMD_LIBRARIES AMD)
add_subdirectory("${PROJECT_SOURCE_DIR}/OTHER/ARPACK")
set (ARPACK_LIBRARIES ARPACK)
add_subdirectory("${PROJECT_SOURCE_DIR}/OTHER/CSPARSE")
set (CSPARSE_LIBRARIES CSPARSE)
add_subdirectory("${PROJECT_SOURCE_DIR}/OTHER/ITPACK")
set (ITPACK_LIBRARIES ITPACK)
add_subdirectory("${PROJECT_SOURCE_DIR}/OTHER/tetgen1.4.3")
set (TETGEN_LIBRARIES tet)
add_subdirectory("${PROJECT_SOURCE_DIR}/OTHER/Triangle")
set (TRIANGLE_LIBRARIES triangle)
# add_subdirectory("${PROJECT_SOURCE_DIR}/OTHER/eigenAPI")
# set (EIGENAPI_LIBRARIES EIGENAPI)


if (NOT DEFINED LAPACK_LIBRARIES)
   find_package(LAPACK)
else()
   set (LAPACK_FOUND TRUE CACHE BOOL "providing own lapack lib")
endif()

if (NOT DEFINED Python_LIBRARIES)
   find_package(Python COMPONENTS Interpreter Development)
else()
   set (PYTHON_FOUND TRUE CACHE BOOL "providing own python lib")
endif()

find_package(MPI)
find_package(MKL)
# Do not find CUDAToolkit here: FindCUDAToolkit must run after enable_language(CUDA)
# or CUDA::cusparse etc. may never be created (configure fails on fresh cache).

#set (LAPACK_FOUND FALSE)
#set (BLAS_FOUND FALSE)

if(LAPACK_FOUND)
  message(STATUS "LAPACK was found.")
  message(STATUS "LAPACK_LINKER_FLAGS = ${LAPACK_LINKER_FLAGS}")
  message(STATUS "LAPACK_LIBRARIES = ${LAPACK_LIBRARIES}" )
else()
  add_subdirectory("${PROJECT_SOURCE_DIR}/OTHER/BLAS")
  add_subdirectory("${PROJECT_SOURCE_DIR}/OTHER/LAPACK")
  set(LAPACK_LIBRARIES LAPACK)
endif()

if(PYTHON_FOUND)
  message("Python_FOUND:${Python_FOUND}")
  #message("Python_VERSION:${Python_VERSION}")
  #message("Python_Development_FOUND:${Python_Development_FOUND}")
  message("Python_LIBRARIES:${Python_LIBRARIES}")
  message("Python_INCLUDES:${Python_INCLUDE_DIRS}")
else()
  message(STATUS "PYTHON NOT FOUND")
endif()


if(MPI_FOUND)
  message(STATUS "MPI was found.")
  message(STATUS "MPI was found .. path added ${MPI_C_INCLUDE_DIRS} ${MPI_FOUND}")
  include_directories(SYSTEM ${MPI_C_INCLUDE_DIRS})
  #add_subdirectory("${PROJECT_SOURCE_DIR}/OTHER/MUMPS_5.4.1")
  #if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
  if (NOT DEFINED MUMPS_DIR)
  #    include(ExternalProject)
  #    ExternalProject_Add(mumps
  #       GIT_REPOSITORY    https://github.com/scivision/mumps.git
  #       GIT_TAG           v5.5.1.5
  #       SOURCE_DIR        "${CMAKE_BINARY_DIR}/mumps-src"
  #       BINARY_DIR        "${CMAKE_BINARY_DIR}"
  #       INSTALL_COMMAND   ""
  #     )
  #
  #    set (MUMPS_LIBRARIES "${CMAKE_BINARY_DIR}/libdmumps.lib;${CMAKE_BINARY_DIR}/libmumps_common.lib;${CMAKE_BINARY_DIR}/libpord.lib")
     set (MUMPS_FLAG -D_NOMUMPS)
     set (MUMPS_LIBRARIES "")
  else()
     set (MUMPS_FLAG -D_MUMPS)
     if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
       get_filename_component(MUMPS_DIR "${MUMPS_DIR}" ABSOLUTE)
       set (MUMPS_LIBRARIES
         "${MUMPS_DIR}/dmumps.lib"
         "${MUMPS_DIR}/mumps_common.lib"
         "${MUMPS_DIR}/pord.lib")
     else()
       set (MUMPS_LIBRARIES "${MUMPS_DIR}/libdmumps.a;${MUMPS_DIR}/libmumps_common.a;${MUMPS_DIR}/libpord.a")
     endif()

  endif()

  add_subdirectory("${PROJECT_SOURCE_DIR}/OTHER/SuperLU_DIST_4.3/SRC")
  add_subdirectory("${PROJECT_SOURCE_DIR}/OTHER/METIS")

  # Bundled METIS target is METIS 4. partition() / system <metis.h> use METIS 5.
  # Link libmetis when METIS 5 headers were detected; otherwise keep bundled METIS.
  set(OPENSEES_METIS_LINK_LIB "METIS")
  if(OPENSEES_HAVE_METIS5_HEADER)
    find_library(
      OPENSEES_METIS5_LIBRARY
      NAMES metis
      HINTS
        "${METIS5_DIR}/lib"
        "${METIS5_DIR}/lib64"
      PATHS
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /usr/lib64
        /usr/lib
      DOC "METIS 5 (libmetis) for partition() and MetisWrapper with system metis.h"
    )
    if(OPENSEES_METIS5_LIBRARY)
      set(OPENSEES_METIS_LINK_LIB "${OPENSEES_METIS5_LIBRARY}")
      message(STATUS "MPI executables: linking METIS 5 at ${OPENSEES_METIS5_LIBRARY}")
    else()
      message(
        WARNING
        "METIS 5 headers were found at ${OPENSEES_METIS5_INCLUDE_DIR} but libmetis was not found. "
        "Falling back to bundled METIS 4; OpenSeesMP partition() will remain a stub. "
        "Install libmetis (e.g. apt install libmetis-dev) or set METIS5_DIR."
      )
      set(OPENSEES_HAVE_METIS5_HEADER FALSE)
    endif()
  else()
    message(
      WARNING
      "METIS 5 headers not found; OpenSeesMP partition() will remain a stub. "
      "Install METIS development files (e.g. libmetis-dev) or set -DMETIS5_DIR=/path."
    )
  endif()
else()
  message(STATUS "MPI was NOT found.")
  set(OPENSEES_METIS_LINK_LIB "METIS")
endif()


if(MKL_FOUND)
  message(STATUS "MKL was found.")
  if (NOT DEFINED SCALAPACK_LIBRARIES)
     set (SCALAPACK_LIBRARIES ${MKL_LIBRARIES})
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # Prefer lib/ (oneAPI 2024+) over lib/intel64. The intel64 tree can still
    # contain sequential ScaLAPACK leftovers without mkl_intel_thread.lib.
    if(EXISTS "${MKL_ROOT}/lib/mkl_intel_thread.lib" AND EXISTS "${MKL_ROOT}/lib/mkl_scalapack_lp64.lib")
      set(MKL_LPATH "${MKL_ROOT}/lib")
    elseif(EXISTS "${MKL_ROOT}/lib/intel64/mkl_scalapack_lp64.lib")
      set(MKL_LPATH "${MKL_ROOT}/lib/intel64")
    elseif(EXISTS "${MKL_ROOT}/lib/mkl_scalapack_lp64.lib")
      set(MKL_LPATH "${MKL_ROOT}/lib")
    else()
      message(FATAL_ERROR "MKL lp64 ScaLAPACK libraries not found under ${MKL_ROOT}")
    endif()

    # Link-time only (do not compile OpenSees C++ with /openmp). WSL OpenSeesMP
    # used mkl_gnu_thread + libgomp the same way; sequential MKL here mixed with
    # OPS_Numerics' mkl_intel_thread. One OpenMP runtime: libiomp5md.
    set(_iomp5md "")
    if(OMP_LIBRARY)
      set(_iomp5md "${OMP_LIBRARY}")
    elseif(DEFINED ENV{CMPLR_ROOT} AND EXISTS "$ENV{CMPLR_ROOT}/lib/libiomp5md.lib")
      set(_iomp5md "$ENV{CMPLR_ROOT}/lib/libiomp5md.lib")
    elseif(DEFINED ENV{ONEAPI_ROOT} AND EXISTS "$ENV{ONEAPI_ROOT}/compiler/latest/lib/libiomp5md.lib")
      set(_iomp5md "$ENV{ONEAPI_ROOT}/compiler/latest/lib/libiomp5md.lib")
    else()
      message(FATAL_ERROR
        "libiomp5md.lib not found. Load Intel oneAPI (setvars.bat) so CMPLR_ROOT is set.")
    endif()
    set (SCALAPACK_LIBRARIES "${MKL_LPATH}/mkl_scalapack_lp64.lib;${MKL_LPATH}/mkl_intel_lp64.lib;${MKL_LPATH}/mkl_intel_thread.lib;${_iomp5md};${MKL_LPATH}/mkl_core.lib;${MKL_LPATH}/mkl_blacs_intelmpi_lp64.lib")
  endif()

else()
  message(STATUS "MKL NOT found .. user to provide -DSCALAPACK_LIBRARIES=")

endif()

# CUDA toolkit selection (before check_language / find_package):
# - If CUDAToolkit_ROOT is set (e.g. -DCUDAToolkit_ROOT=/usr/local/cuda), use that tree only:
#   set CMAKE_CUDA_COMPILER to <root>/bin/nvcc so PATH (e.g. conda) does not win.
# - Otherwise, let CMake discover nvcc via check_language (PATH, defaults, etc.).
if(CUDAToolkit_ROOT)
  if(CMAKE_HOST_WIN32)
    set(_OpenSees_nvcc "${CUDAToolkit_ROOT}/bin/nvcc.exe")
  else()
    set(_OpenSees_nvcc "${CUDAToolkit_ROOT}/bin/nvcc")
  endif()
  if(NOT EXISTS "${_OpenSees_nvcc}")
    message(FATAL_ERROR "CUDAToolkit_ROOT='${CUDAToolkit_ROOT}' but '${_OpenSees_nvcc}' does not exist.")
  endif()
  set(CMAKE_CUDA_COMPILER "${_OpenSees_nvcc}" CACHE FILEPATH "CUDA compiler" FORCE)
  message(STATUS "CUDAToolkit_ROOT set — using CUDA compiler: ${_OpenSees_nvcc}")
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
  if(MSVC AND CMAKE_GENERATOR MATCHES "Visual Studio")
    # CMake 3.31 CUDA try_compile via MSBuild fails (MSB4023 in CUDA 12.9.targets); nvcc is fine.
    set(CMAKE_CUDA_COMPILER_WORKS TRUE)
  endif()
  enable_language(CUDA)
  if(WIN32)
    # Shared cudart avoids static device registration at process start.
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared CACHE STRING
        "Shared CUDA runtime" FORCE)
  endif()
  # IMPORTED targets CUDA::cusparse, etc. — must run after enable_language(CUDA).
  # A prior find_package(CUDAToolkit) before enable_language can skip this pass and
  # leave CUDA:: targets undefined on a fresh configure.
  find_package(CUDAToolkit REQUIRED)
  set(HAVE_CUDA TRUE)
else()
  message(STATUS "CUDA compiler not found")
  set(HAVE_CUDA FALSE)
endif()

if(HAVE_CUDA)
  message(STATUS "CUDAToolkit_FOUND: ${CUDAToolkit_FOUND}")
  message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
  message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
  message(STATUS "CUDAToolkit_BIN_DIR: ${CUDAToolkit_BIN_DIR}")
  message(STATUS "CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")
  include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})

  get_filename_component(CMAKE_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}" NAME)
  if(CMAKE_CXX_COMPILER_NAME MATCHES cl AND NOT CMAKE_CXX_COMPILER_NAME MATCHES clang)
    set(CUDA_RUNTIME_LIBS CUDA::cudart)
    set(CUDA_SPARSE_LIBS CUDA::cusparse CUDA::cusolver)
    set(CUDA_BLAS_LIBS CUDA::cublas)
    set(CUDA_LIBRARIES ${CUDA_RUNTIME_LIBS} ${CUDA_SPARSE_LIBS} ${CUDA_BLAS_LIBS})
  else()
    set(CUDA_RUNTIME_LIBS CUDA::cudart)
    set(CUDA_SPARSE_LIBS CUDA::cusparse CUDA::cusolver)
    set(CUDA_BLAS_LIBS CUDA::cublas)
    set(CUDA_LIBRARIES ${CUDA_RUNTIME_LIBS} ${CUDA_SPARSE_LIBS} ${CUDA_BLAS_LIBS} m pthread rt dl)
  endif()

  set(CUDA_FLAG -D_CUDA)
  if (NOT DEFINED AMGX_NO_MPI_DIR)
      message(STATUS "AMGX directory not defined. Please set AMGX_NO_MPI_DIR to the path of the AMGX library.")
      set(AMGX_FLAG -D_NOAMGX)
      set(AMGX_LIBRARIES "")
      set(AMGX_INCLUDE_DIR "")
      set(HAVE_AMGX FALSE)
  else()
      message(STATUS "Compiling with AMGX from ${AMGX_NO_MPI_DIR} (expected non-MPI build)")

      set(AMGX_FLAG -D_AMGX)
      set(HAVE_AMGX TRUE)

      if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
          set(AMGX_LIBRARIES "${AMGX_NO_MPI_DIR}/amgx.lib")
      else()
          set(AMGX_LIBRARIES "${AMGX_NO_MPI_DIR}/libamgx.a")
      endif()

      if(NOT EXISTS "${AMGX_LIBRARIES}")
          message(FATAL_ERROR
              "AMGX_NO_MPI_DIR='${AMGX_NO_MPI_DIR}' but library not found: ${AMGX_LIBRARIES}")
      endif()

      if(EXISTS "${AMGX_NO_MPI_DIR}/../include/amgx_c.h")
        set(AMGX_INCLUDE_DIR "${AMGX_NO_MPI_DIR}/../include")
      else()
        message(FATAL_ERROR "Could not find amgx_c.h in expected location: ${AMGX_NO_MPI_DIR}/../include")
      endif()
      
      message(STATUS "AMGX support enabled.")
      message(STATUS "Adding include directory ${AMGX_INCLUDE_DIR} and compile definition ${AMGX_FLAG}")
  endif()
  
  # CuDSS configuration using CMake config package
  set(CUDSS_HINTS)
  if(DEFINED CUDSS_DIR AND NOT DEFINED CUDSS_ROOT)
    set(CUDSS_ROOT "${CUDSS_DIR}")
  endif()
  if(DEFINED CUDSS_ROOT)
    list(APPEND CUDSS_HINTS
      "${CUDSS_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR}/cmake/cudss"
      "${CUDSS_ROOT}/cmake/cudss"
    )
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    list(APPEND CUDSS_HINTS
      "C:/Program Files/NVIDIA cuDSS/v0.8/lib/${CUDAToolkit_VERSION_MAJOR}/cmake/cudss"
    )
  else()
    list(APPEND CUDSS_HINTS
      /usr/lib/x86_64-linux-gnu/libcudss/${CUDAToolkit_VERSION_MAJOR}/cmake
      /usr/lib/aarch64-linux-gnu/libcudss/${CUDAToolkit_VERSION_MAJOR}/cmake
    )
  endif()

  unset(cudss_DIR CACHE)
  unset(cudss_INCLUDE_DIR CACHE)
  unset(cudss_LIBRARY_DIR CACHE)
  unset(cudss_BINARY_DIR CACHE)
  find_package(cudss QUIET HINTS ${CUDSS_HINTS})
  if(cudss_FOUND AND cudss_VERSION VERSION_LESS "0.8.0")
    message(STATUS "cuDSS ${cudss_VERSION} is too old; OpenSees-CUDA requires >= 0.8.0")
    set(cudss_FOUND FALSE)
  endif()

  if(cudss_FOUND)
      message(STATUS "cuDSS found via CMake config")
      set(CUDSS_FLAG -D_CUDSS)
      set(HAVE_CUDSS TRUE)
      message(STATUS "cuDSS support enabled.")
      message(STATUS "  cuDSS MT mode: -multiThreadingMode 1 with mtlayer shim on PATH (no exe OpenMP link)")
  else()
      message(STATUS "cuDSS not found (CUDA major ${CUDAToolkit_VERSION_MAJOR}). To enable cuDSS:")
      if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
          message(STATUS "  - Install NVIDIA cuDSS for Windows, or")
          message(STATUS "  - Set CUDSS_ROOT / CUDSS_DIR (runtime), or")
          message(STATUS "  - Set CMAKE_PREFIX_PATH to .../lib/${CUDAToolkit_VERSION_MAJOR}/cmake/cudss")
      else()
          message(STATUS "  - Ensure libcudss-dev is installed, or")
          message(STATUS "  - Set CMAKE_PREFIX_PATH to cuDSS CMake config directory")
      endif()
      set(CUDSS_FLAG -D_NOCUDSS)
      set(HAVE_CUDSS FALSE)
  endif()
else()
  message(STATUS "CUDAToolkit was NOT found.")
  set(CUDA_LIBRARIES "")
  set(CUDA_FLAG -D_NO_CUDA)
endif()

message(STATUS "SCALAPACK_LIBRARIES=${SCALAPACK_LIBRARIES}")

# CUDA object-library layout:
#   OPS_Cuda          - CudaBcsrLinSOE / CudaBcsrLinSolver (shared GPU SOE framework)
#   OPS_Cuda_AmgX     - AmgXLinSolver (optional, when AMGX_NO_MPI_DIR is set)
#   OPS_Cuda_CuDSS    - cuDSS direct solver + CudaExplicitAlpha / CudaExplicitAlpha_TP (optional, when cuDSS is found)
#   OPS_Cuda_Parallel - parallel assembly CUDA SOEs for OpenSeesSP / OpenSeesMP (future)
#
# Executables link OPS_Cuda plus whichever backends were built. OpenSeesSP/MP also link
# OPS_Cuda_Parallel when that target exists.

function(ops_cuda_is_windows_executable target_name out_var)
  if(${target_name} STREQUAL "OpenSees" OR ${target_name} STREQUAL "OpenSeesMP"
      OR ${target_name} STREQUAL "OpenSeesSP" OR ${target_name} STREQUAL "OpenSeesPy")
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(ops_cuda_target_hosts_gpu_runtime target_name out_var)
  set(${out_var} FALSE PARENT_SCOPE)
  if(HAVE_CUDA)
    set(${out_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

macro(ops_cuda_apply_base_toolchain target_name)
  target_compile_definitions(${target_name} PRIVATE ${CUDA_FLAG})
  ops_cuda_target_hosts_gpu_runtime(${target_name} _ops_cuda_gpu_runtime)
  if(_ops_cuda_gpu_runtime)
    target_link_libraries(${target_name} PRIVATE ${CUDA_LIBRARIES})
  endif()
  if(WIN32)
    set_target_properties(${target_name} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  endif()
endmacro()

macro(ops_cuda_apply_amgx_toolchain target_name)
  if(HAVE_AMGX)
    target_include_directories(${target_name} PRIVATE ${AMGX_INCLUDE_DIR})
    target_compile_definitions(${target_name} PRIVATE ${AMGX_FLAG})
    ops_cuda_target_hosts_gpu_runtime(${target_name} _ops_cuda_gpu_runtime)
    if(_ops_cuda_gpu_runtime)
      target_link_libraries(${target_name} PRIVATE ${AMGX_LIBRARIES})
    endif()
  endif()
endmacro()

macro(ops_cuda_apply_cudss_toolchain target_name)
  if(HAVE_CUDSS)
    target_compile_definitions(${target_name} PRIVATE ${CUDSS_FLAG})
    if(cudss_INCLUDE_DIR)
      target_include_directories(${target_name} SYSTEM PRIVATE ${cudss_INCLUDE_DIR})
    endif()
    ops_cuda_target_hosts_gpu_runtime(${target_name} _ops_cuda_gpu_runtime)
    if(_ops_cuda_gpu_runtime)
      target_link_libraries(${target_name} PRIVATE cudss)
    endif()
  endif()
endmacro()

macro(ops_cuda_apply_backend_defs target_name)
  ops_cuda_apply_amgx_toolchain(${target_name})
  ops_cuda_apply_cudss_toolchain(${target_name})
endmacro()

# OpenSeesCommands.cpp (Python) lives in OPS_INTERPRETER inside OpenSeesLIB.
# It must see _CUDA/_CUDSS at compile time; GPU backends link only on OpenSeesPy.
macro(ops_cuda_apply_interpreter_defs target_name)
  target_compile_definitions(${target_name} PRIVATE ${CUDA_FLAG})
  if(HAVE_AMGX)
    target_compile_definitions(${target_name} PRIVATE ${AMGX_FLAG})
    if(AMGX_INCLUDE_DIR)
      target_include_directories(${target_name} PRIVATE ${AMGX_INCLUDE_DIR})
    endif()
  endif()
  if(HAVE_CUDSS)
    target_compile_definitions(${target_name} PRIVATE ${CUDSS_FLAG})
    if(cudss_INCLUDE_DIR)
      target_include_directories(${target_name} SYSTEM PRIVATE ${cudss_INCLUDE_DIR})
    endif()
  endif()
endmacro()

function(ops_cuda_link_backends target_name)
  if(NOT HAVE_CUDA)
    return()
  endif()

  target_link_libraries(${target_name} PRIVATE OPS_Cuda)
  if(HAVE_AMGX)
    target_link_libraries(${target_name} PRIVATE OPS_Cuda_AmgX)
  endif()
  if(HAVE_CUDSS)
    target_link_libraries(${target_name} PRIVATE OPS_Cuda_CuDSS)
  endif()
endfunction()

function(ops_cuda_link_parallel_backend target_name)
  if(TARGET OPS_Cuda_Parallel)
    target_link_libraries(${target_name} PRIVATE OPS_Cuda_Parallel)
  endif()
endfunction()

function(configure_cuda_for_target target_name)
  if(NOT HAVE_CUDA)
    return()
  endif()

  if(${target_name} STREQUAL "OPS_Cuda")
    ops_cuda_apply_base_toolchain(${target_name})
    return()
  endif()

  if(${target_name} STREQUAL "OPS_Cuda_AmgX")
    ops_cuda_apply_base_toolchain(${target_name})
    ops_cuda_apply_amgx_toolchain(${target_name})
    target_link_libraries(${target_name} PRIVATE OPS_Cuda)
    return()
  endif()

  if(${target_name} STREQUAL "OPS_Cuda_CuDSS")
    ops_cuda_apply_base_toolchain(${target_name})
    ops_cuda_apply_cudss_toolchain(${target_name})
    target_link_libraries(${target_name} PRIVATE OPS_Cuda)
    return()
  endif()

  if(${target_name} STREQUAL "OPS_Cuda_Parallel")
    target_link_libraries(${target_name} PRIVATE OPS_Cuda)
    return()
  endif()

  if(${target_name} STREQUAL "OPS_INTERPRETER" OR ${target_name} STREQUAL "OPS_InterpTcl")
    ops_cuda_apply_interpreter_defs(${target_name})
    target_link_libraries(${target_name} PRIVATE OPS_Cuda)
    return()
  endif()

  ops_cuda_is_windows_executable(${target_name} _ops_cuda_exe)
  if(NOT _ops_cuda_exe)
    return()
  endif()

  ops_cuda_apply_base_toolchain(${target_name})
  ops_cuda_apply_backend_defs(${target_name})
  ops_cuda_link_backends(${target_name})

  if(${target_name} STREQUAL "OpenSeesSP" OR ${target_name} STREQUAL "OpenSeesMP")
    ops_cuda_link_parallel_backend(${target_name})
  endif()

  ops_cuda_finalize_executable_link(${target_name})
endfunction()

# Windows: link CUDA/cuDSS runtime DLLs normally; GPU code is in the exe via object libs.
function(ops_cuda_finalize_executable_link target_name)
  if(NOT HAVE_CUDA OR NOT WIN32)
    return()
  endif()

  set_target_properties(${target_name} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  target_link_options(${target_name} PRIVATE "LINKER:/NODEFAULTLIB:cudart_static.lib")

  if(CUDAToolkit_LIBRARY_DIR)
    target_link_directories(${target_name} PRIVATE "${CUDAToolkit_LIBRARY_DIR}")
    target_link_options(${target_name} PRIVATE "LINKER:/LIBPATH:${CUDAToolkit_LIBRARY_DIR}")
  endif()
  if(TARGET CUDA::cudadevrt)
    target_link_libraries(${target_name} PRIVATE CUDA::cudadevrt)
  elseif(CUDAToolkit_LIBRARY_DIR)
    target_link_libraries(${target_name} PRIVATE "${CUDAToolkit_LIBRARY_DIR}/cudadevrt.lib")
  endif()

  if(LAPACK_FOUND AND LAPACK_LIBRARIES)
    target_link_libraries(${target_name} PRIVATE ${LAPACK_LIBRARIES})
  endif()

  target_link_libraries(${target_name} PRIVATE ${CUDA_LIBRARIES})
  if(HAVE_AMGX)
    target_link_libraries(${target_name} PRIVATE ${AMGX_LIBRARIES})
  endif()
  if(HAVE_CUDSS)
    target_link_libraries(${target_name} PRIVATE cudss)
  endif()
endfunction()

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#  message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()



add_library(OPS_Numerics INTERFACE)

target_link_libraries(OPS_Numerics INTERFACE
       ${ARPACK_LIBRARIES}
       ${CSPARSE_LIBRARIES}
       ${SUPERLU_LIBRARIES}
       ${UMFPACK_LIBRARIES}
       ${TETGEN_LIBRARIES}
       ${TRIANGLE_LIBRARIES}
       ${AMD_LIBRARIES}
       ${AMD_LIBRARIES}
       ${LAPACK_LIBRARIES}
       ${EIGENAPI_LIBRARIES}
)

set(TCL_LIBRARIES ${TCL_LIBRARY})


#
 # OpenSees OBJECT LIBRARIES
 #

# _PARALLEL_PROCESSING (OpenSeesSP) and _PARALLEL_INTERPRETERS (OpenSeesMP) are
# mutually exclusive per executable. Do not use a global compile definition here;
# SP object libraries and OpenSeesSP get _PARALLEL_PROCESSING when PARALLEL_PROCESSING=ON.

# _PARALLEL_PROCESSING (OpenSeesSP) and _PARALLEL_INTERPRETERS (OpenSeesMP) are
# mutually exclusive per executable. Do not use a global compile definition here;
# SP object libraries and OpenSeesSP get _PARALLEL_PROCESSING when PARALLEL_PROCESSING=ON.

# _PARALLEL_PROCESSING (OpenSeesSP) and _PARALLEL_INTERPRETERS (OpenSeesMP) are
# mutually exclusive per executable. Do not use a global compile definition here;
# SP object libraries and OpenSeesSP get _PARALLEL_PROCESSING when PARALLEL_PROCESSING=ON.

add_library(OPS_Matrix             OBJECT)
add_library(OPS_Actor              OBJECT)
add_library(OPS_ObjectBroker       OBJECT)
add_library(OPS_Handler            OBJECT)
add_library(OPS_Recorder           OBJECT)
add_library(OPS_Reliability        OBJECT)
add_library(OPS_Tagged             OBJECT)
add_library(OPS_Utilities          OBJECT)
add_library(OPS_ModelBuilder       OBJECT)
add_library(OPS_Domain             OBJECT)
add_library(OPS_SysOfEqn           OBJECT)
add_library(OPS_SysOfEqnPython     OBJECT)
#add_library(OPS_SysOfEqnSP        OBJECT)
add_library(OPS_Analysis           OBJECT)
add_library(OPS_ConvergenceTest    OBJECT)
add_library(OPS_Thermal            OBJECT)
add_library(OPS_Element            OBJECT)
add_library(OPS_ElementFortran     OBJECT)
#add_library(OPS_ElementSP         OBJECT)
add_library(OPS_Material           OBJECT)
add_library(OPS_MaterialFortran    OBJECT)
add_library(OPS_Damage             OBJECT)
add_library(OPS_Database           OBJECT)
add_library(OPS_INTERPRETER        OBJECT)



#
# Optional Extensions
#

add_library(OPS_Paraview           OBJECT EXCLUDE_FROM_ALL)
add_library(OPS_Renderer           OBJECT EXCLUDE_FROM_ALL)
add_library(OPS_Graphics           OBJECT EXCLUDE_FROM_ALL)
add_library(OPS_Graphics_Default   OBJECT EXCLUDE_FROM_ALL)
add_library(OPS_Graphics_GL        OBJECT EXCLUDE_FROM_ALL)
add_library(OPS_PFEM               OBJECT EXCLUDE_FROM_ALL)

#
# Tcl Interpreter Library & special include and link directives
#

add_library(OPS_InterpTcl          STATIC)

target_compile_definitions(OPS_InterpTcl PUBLIC _TCL85)
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conanbuildinfo.cmake")
  target_include_directories(OPS_InterpTcl PUBLIC ${TCL_INCLUDE_PATH})
  target_link_libraries(OPS_InterpTcl PRIVATE ${TCL_LIBRARIES})
endif()

#
# OpenSees G3 Library - a slimmed down OpenSees
#

add_library(G3)

target_link_libraries(G3
    coordTransformation
    damping
    OPS_Matrix
    OPS_Analysis
    OPS_ModelBuilder
    OPS_Domain
    OPS_ConvergenceTest
    OPS_Element
    OPS_Material
    OPS_Recorder
    OPS_Handler
    OPS_SysOfEqn
    OPS_Tagged
    OPS_Utilities
    graph
    OPS_Actor
    OPS_ObjectBroker
    OPS_Numerics
    OPS_INTERPRETER
)

#
# OpenSees Libray
#

add_library(OpenSeesLIB  EXCLUDE_FROM_ALL)

#
# For UMFPACK long int DLONG version -- Gustavo, Michael
#
target_compile_definitions(OpenSeesLIB PUBLIC _UMFPACK_DLONG)

target_link_libraries(OpenSeesLIB
    ${OPS_Extension_List}
    OPS_Actor
    OPS_Analysis
    OPS_ConvergenceTest
    OPS_Damage
    OPS_Database
    OPS_Domain
    graph
    OPS_Element
    OPS_ElementFortran
    OPS_Handler
    OPS_INTERPRETER
    OPS_Material
    OPS_MaterialFortran
    OPS_Material_YieldSurface
    OPS_Matrix
    OPS_ModelBuilder
    OPS_ObjectBroker
    OPS_PFEM
    OPS_Recorder
    coordTransformation
    damping
    OPS_Renderer
    OPS_Section_Repres
    OPS_Section_YieldSurface
    OPS_SysOfEqn
    OPS_Thermal
    OPS_OS_Specific_libs
    OPS_Tagged
    OPS_Utilities
)

# CUDA support as OPS_ object libraries; executables link GPU backends directly.
if(HAVE_CUDA)
  add_library(OPS_Cuda OBJECT
    ${OPS_SRC_DIR}/system_of_eqn/linearSOE/sparseCUDA/CuSparseBackend.cu
    ${OPS_SRC_DIR}/system_of_eqn/linearSOE/sparseCUDA/CudaBcsrLinSOE.cu
    ${OPS_SRC_DIR}/system_of_eqn/linearSOE/sparseCUDA/CudaBcsrLinSolver.cu
    ${OPS_SRC_DIR}/system_of_eqn/linearSOE/sparseCUDA/DistributedCudaBcsrLinSOE.cu
  )
  configure_cuda_for_target(OPS_Cuda)

  if(HAVE_AMGX)
    add_library(OPS_Cuda_AmgX OBJECT
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/sparseCUDA/AmgXLinSolver.cu
    )
    configure_cuda_for_target(OPS_Cuda_AmgX)
  endif()

  if(HAVE_CUDSS)
    add_library(OPS_Cuda_CuDSS OBJECT
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/sparseCUDA/CudaCsrMatrix.cu
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/sparseCUDA/CuDSSBackend.cu
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/sparseCUDA/CudaDirectSparseSolver.cu
      ${OPS_SRC_DIR}/analysis/integrator/CudaExplicitAlpha.cu
      ${OPS_SRC_DIR}/analysis/integrator/CudaExplicitAlpha_TP.cu
    )
    configure_cuda_for_target(OPS_Cuda_CuDSS)
  endif()

  configure_cuda_for_target(OPS_INTERPRETER)
  configure_cuda_for_target(OPS_InterpTcl)
endif()

# ########################################################################
#
# EXECUTABLES
#
# ########################################################################

# Adding EXCLUDE_FROM_ALL prevents these from being made by default.

#
# OpenSees Tcl Interpreter
#

add_executable(OpenSees EXCLUDE_FROM_ALL
  ${OPS_SRC_DIR}/tcl/tclAppInit.cpp
  ${OPS_SRC_DIR}/tcl/tclMain.cpp
  ${OPS_SRC_DIR}/tcl/commands.cpp
  ${OPS_SRC_DIR}/actor/objectBroker/FEM_ObjectBrokerAllClasses.cpp
)

if(TARGET TCL::TCL)
    # --- CONAN 2 LOGIC ---
    message(STATUS "Conan TCL found at: ${tcl_PACKAGE_FOLDER}")
    
    # In Conan 2, the library files are typically in the 'lib' subfolder of the package
    set(CONAN_TCL_LIB_DIR "${tcl_PACKAGE_FOLDER}/lib")

    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/tcl8.6)
    
    add_custom_command(
        TARGET OpenSees POST_BUILD
        COMMENT "Copying init.tcl from Conan TCL to binary directory"
        COMMAND ${CMAKE_COMMAND} -E copy
        "${CONAN_TCL_LIB_DIR}/tcl8.6/init.tcl"
        "${CMAKE_CURRENT_BINARY_DIR}/lib/tcl8.6"
    )
    
    # Note: You don't usually need target_include_directories here 
    # because target_link_libraries(OpenSees PRIVATE TCL::TCL) handles it.

else()
    # --- SYSTEM / NON-CONAN LOGIC ---
    message(STATUS "Using System TCL")
    
    target_include_directories(OpenSees PUBLIC ${TCL_INCLUDE_PATH})

    file(COPY ${TCL_INCLUDE_PATH}/../lib/tcl8.6  # Adjust path based on your system layout
         DESTINATION ${PROJECT_BINARY_DIR}/lib/tcl8.6
         FILES_MATCHING PATTERN "*.tcl")
endif()


target_link_libraries(OpenSees PRIVATE
  OPS_InterpTcl
  coordTransformation
  OpenSeesLIB
  OPS_Reliability
  OPS_ReliabilityTcl
  OPS_Numerics
  OPS_Recorder
  ${CMAKE_DL_LIBS}
  HDF5::HDF5
  ${OPS_TCL_TARGET}
  ZLIB::ZLIB
  Eigen3::Eigen
)

# Link CUDA after MKL/numerics.
configure_cuda_for_target(OpenSees)
if(COMMAND ops_add_windows_dynamic_tcl_runtime)
  ops_add_windows_dynamic_tcl_runtime(OpenSees)
endif()


#
# OpenSeesSP Tcl Parallel Interpreter
#


if(MPI_FOUND)

    add_executable(OpenSeesSP EXCLUDE_FROM_ALL
      ${OPS_SRC_DIR}/tcl/mpiMain.cpp
      ${OPS_SRC_DIR}/tcl/tclMain.cpp
      ${OPS_SRC_DIR}/tcl/commands.cpp
      ${OPS_SRC_DIR}/domain/domain/partitioned/PartitionedDomain.cpp
      ${OPS_SRC_DIR}/domain/domain/partitioned/PartitionedDomainEleIter.cpp
      ${OPS_SRC_DIR}/domain/domain/partitioned/PartitionedDomainSubIter.cpp
      ${OPS_SRC_DIR}/actor/objectBroker/FEM_ObjectBrokerAllClasses.cpp
      ${OPS_SRC_DIR}/actor/machineBroker/MPI_MachineBroker.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/diagonal/MPIDiagonalSOE.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/diagonal/MPIDiagonalSolver.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/mumps/MumpsSOE.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/mumps/MumpsSolver.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/mumps/MumpsParallelSOE.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/mumps/MumpsParallelSolver.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/sparseGEN/DistributedSuperLU.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/sparseGEN/DistributedSparseGenColLinSOE.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/sparseGEN/DistributedSparseGenRowLinSOE.cpp
      ${OPS_SRC_DIR}/domain/subdomain/ActorSubdomain.cpp
      ${OPS_SRC_DIR}/domain/subdomain/ShadowSubdomain.cpp
    )


    target_include_directories(OpenSeesSP PRIVATE ${MUMPS_DIR}/_deps/mumps-src/include ${MPI_CXX_INCLUDE_DIRS})
    
    target_compile_options(OpenSeesSP PRIVATE ${MPI_CXX_COMPILE_FLAGS})

    if (DEFINED OPENMPI)
        target_compile_definitions(OpenSeesSP
        PUBLIC _PARALLEL_PROCESSING ${MUMPS_FLAG} _OPENMPI)
    else()
        target_compile_definitions(OpenSeesSP
        PUBLIC _PARALLEL_PROCESSING ${MUMPS_FLAG})
    endif()

    target_link_libraries(OpenSeesSP PRIVATE
       OPS_InterpTcl
       OpenSeesLIB
       OPS_Reliability
       OPS_ReliabilityTcl
       OPS_Recorder
       ${OPENSEES_METIS_LINK_LIB}
       SUPERLU_DIST
       OPS_Numerics
       ${MUMPS_LIBRARIES}
       ${CMAKE_DL_LIBS}
       HDF5::HDF5
       ${OPS_TCL_TARGET}
       ZLIB::ZLIB
       Eigen3::Eigen
       ${MPI_CXX_LIBRARIES}
       ${SCALAPACK_LIBRARIES}
       ${MPI_Fortran_LIBRARIES}
       ${MPI_CXX_LINK_FLAGS}
    )

    configure_cuda_for_target(OpenSeesSP)
    if(COMMAND ops_add_windows_dynamic_tcl_runtime)
      ops_add_windows_dynamic_tcl_runtime(OpenSeesSP)
    endif()


endif()


#
# OpenSeesMP Tcl MPI Interpreter
#

if(MPI_FOUND)

    add_executable(OpenSeesMP EXCLUDE_FROM_ALL
      ${OPS_SRC_DIR}/tcl/mpiParameterMain.cpp
      ${OPS_SRC_DIR}/tcl/tclMain.cpp
      ${OPS_SRC_DIR}/tcl/commands.cpp
      ${OPS_SRC_DIR}/domain/domain/partitioned/PartitionedDomain.cpp
      ${OPS_SRC_DIR}/domain/domain/partitioned/PartitionedDomainEleIter.cpp
      ${OPS_SRC_DIR}/domain/domain/partitioned/PartitionedDomainSubIter.cpp
      ${OPS_SRC_DIR}/actor/machineBroker/MPI_MachineBroker.cpp
      ${OPS_SRC_DIR}/actor/objectBroker/FEM_ObjectBrokerAllClasses.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/diagonal/MPIDiagonalSOE.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/diagonal/MPIDiagonalSolver.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/mumps/MumpsSOE.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/mumps/MumpsSolver.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/mumps/MumpsParallelSOE.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/mumps/MumpsParallelSolver.cpp
      ${OPS_SRC_DIR}/system_of_eqn/linearSOE/sparseCSR/MumpsCsrBackend.cpp
      ${OPS_SRC_DIR}/domain/subdomain/ActorSubdomain.cpp
      ${OPS_SRC_DIR}/domain/subdomain/ShadowSubdomain.cpp
    )

    # Compile OpenSeesMiscCommands with _PARALLEL_INTERPRETERS so OPS_partition
    # (and other MPI misc commands) get real bodies. The stub archive member in
    # OpenSeesLIB is not pulled when this TU is present on the executable.
    if(OPENSEES_HAVE_METIS5_HEADER)
      target_sources(OpenSeesMP PRIVATE
        ${OPS_SRC_DIR}/interpreter/OpenSeesMiscCommands.cpp
      )
      target_include_directories(OpenSeesMP BEFORE PRIVATE
        "${OPENSEES_METIS5_INCLUDE_DIR}"
      )
    endif()

    target_include_directories(OpenSeesMP PRIVATE ${MUMPS_DIR}/_deps/mumps-src/include ${MPI_CXX_INCLUDE_DIRS})

    target_compile_options(OpenSeesMP PRIVATE ${MPI_CXX_COMPILE_FLAGS})

    if (NOT DEFINED MUMPS_DIR)
        add_dependencies(OpenSeesMP mumps)
    endif()

    if (DEFINED OPENMPI)
        target_compile_definitions(OpenSeesMP
        PUBLIC _PARALLEL_INTERPRETERS ${MUMPS_FLAG} _OPENMPI)
    else()
        target_compile_definitions(OpenSeesMP
        PUBLIC _PARALLEL_INTERPRETERS ${MUMPS_FLAG})
    endif()

    target_link_libraries(OpenSeesMP PRIVATE
       OPS_InterpTcl
       OpenSeesLIB
       OPS_Reliability
       OPS_ReliabilityTcl
       OPS_Recorder
       ${OPENSEES_METIS_LINK_LIB}
       SUPERLU_DIST
       OPS_Numerics
       ${MUMPS_LIBRARIES}
       ${CMAKE_DL_LIBS}
       HDF5::HDF5
       ${OPS_TCL_TARGET}
       ZLIB::ZLIB
       Eigen3::Eigen
       ${MPI_CXX_LIBRARIES}
       ${SCALAPACK_LIBRARIES}
       ${MPI_Fortran_LIBRARIES}
       ${MPI_CXX_LINK_FLAGS}
    )
    configure_cuda_for_target(OpenSeesMP)
    if(COMMAND ops_add_windows_dynamic_tcl_runtime)
      ops_add_windows_dynamic_tcl_runtime(OpenSeesMP)
    endif()

endif()


#
# OpenSeesPy Module
#

add_library(OpenSeesPy SHARED EXCLUDE_FROM_ALL
    ${OPS_SRC_DIR}/interpreter/PythonModule.cpp
    ${OPS_SRC_DIR}/actor/objectBroker/FEM_ObjectBrokerAllClasses.cpp
)

set_target_properties(OpenSeesPy PROPERTIES PREFIX "")

target_include_directories(OpenSeesPy PUBLIC ${Python_INCLUDE_DIRS})

target_link_libraries(OpenSeesPy PRIVATE
   OpenSeesLIB
   OPS_SysOfEqnPython   
   OPS_Reliability
   OPS_Recorder
   OPS_Numerics
   HDF5::HDF5	
   ZLIB::ZLIB	
   Eigen3::Eigen	
   ${Python_LIBRARIES}
)

configure_cuda_for_target(OpenSeesPy)

#
# INSTALL
#

install (TARGETS OpenSees DESTINATION bin)
if (TCL_INIT_FILE)
   install(FILES "${TCL_INIT_FILE}"
       DESTINATION lib/tcl8.6 
   )
endif()
install(PROGRAMS $<TARGET_FILE:OpenSeesPy> 
        DESTINATION bin 
        RENAME opensees.so)


# Set selected executable to be built by default
set_target_properties(${OPS_FINAL_TARGET} PROPERTIES EXCLUDE_FROM_ALL OFF)

# Add sources to targets
add_subdirectory(${OPS_SRC_DIR})




#==============================================================================
#                            Configure targets
#
#==============================================================================

#target_link_libraries(OPS_Numerics     INTERFACE ${OPS_Numlib_List})
#message("Numlibs: ${OPS_Numlib_List}")
#target_link_libraries(OPS_Element      PRIVATE ${OPS_Element_List})
#target_link_libraries(OPS_ObjectBroker PRIVATE ${OPS_Element_List})
#target_link_libraries(OPS_InterpTcl    PRIVATE ${OPS_Element_List})



#==============================================================================
#                            Apply Options
#
#==============================================================================
#----------------------------
# PARALLEL_PROCESSING (OpenSeesSP object libraries + OPS_InterpTcl)
#----------------------------
if(PARALLEL_PROCESSING)
  message(STATUS "PARALLEL_PROCESSING: applying _PARALLEL_PROCESSING to object libraries for OpenSeesSP")

  function(ops_apply_parallel_processing_to_object_libs dir)
    get_property(_ops_targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_ops_tgt IN LISTS _ops_targets)
      if(NOT TARGET "${_ops_tgt}")
        continue()
      endif()
      get_target_property(_ops_type "${_ops_tgt}" TYPE)
      if(_ops_type STREQUAL "OBJECT_LIBRARY")
        target_compile_definitions("${_ops_tgt}" PRIVATE _PARALLEL_PROCESSING)
      endif()
    endforeach()
    get_property(_ops_subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
    foreach(_ops_subdir IN LISTS _ops_subdirs)
      ops_apply_parallel_processing_to_object_libs("${_ops_subdir}")
    endforeach()
  endfunction()

  foreach(_ops_tgt
      OPS_Matrix OPS_Actor OPS_ObjectBroker OPS_Handler OPS_Recorder
      OPS_Reliability OPS_Tagged OPS_Utilities OPS_ModelBuilder OPS_Domain
      OPS_SysOfEqn OPS_Analysis OPS_ConvergenceTest OPS_Thermal OPS_Element
      OPS_ElementFortran OPS_Material OPS_MaterialFortran OPS_Damage
      OPS_Database OPS_INTERPRETER OPS_Paraview OPS_Renderer OPS_Graphics
      OPS_Graphics_Default OPS_Graphics_GL OPS_PFEM OpenSeesLIB OPS_InterpTcl)
    if(TARGET "${_ops_tgt}")
      get_target_property(_ops_type "${_ops_tgt}" TYPE)
      if(_ops_type STREQUAL "OBJECT_LIBRARY" OR _ops_tgt STREQUAL "OPS_InterpTcl")
        target_compile_definitions("${_ops_tgt}" PRIVATE _PARALLEL_PROCESSING)
      endif()
    endif()
  endforeach()

  ops_apply_parallel_processing_to_object_libs("${PROJECT_SOURCE_DIR}/SRC")
endif()

#----------------------------
# FMK
#----------------------------
if(FMK)
    add_compile_definitions(
        _HAVE_Damage2p
        _HAVE_PSUMAT
        _HAVE_PML
        _FILIP_LHNMYS
    )
endif()

#----------------------------
# Extensions
#----------------------------
message("OPS >>> Configuring OpenSees extensions")
foreach(extension IN LISTS OPS_SysOfEqn_List OPS_Element_List OPS_Extension_List)
    string(TOUPPER "${extension}" ext_flag)
    string(REGEX REPLACE "^OPS_" "OPSDEF_" ext_flag "${ext_flag}")
    add_compile_definitions(${ext_flag})
endforeach()
foreach(extension IN LISTS OPS_Exclude_List)
    string(TOUPPER "${extension}" ext_flag)
    string(REGEX REPLACE "^OPS_" "OPS_EXCLUDE_" ext_flag "${ext_flag}")
    message("    Adding macro definition '${ext_flag}'")
    add_compile_definitions(${ext_flag})
endforeach()


# Renderer
#----------------------------
if (${OPS_Use_Graphics_Option} STREQUAL "Base")
  target_link_libraries(${OPS_FINAL_TARGET} OPS_Graphics_Default OPS_Graphics)

elseif (${OPS_Use_Graphics_Option} STREQUAL "OpenGL")
  message("OPS >>> Including OpenGL graphics option")
  set_source_files_properties(
      "${OPS_SRC_DIR}/recorder/AlgorithmIncrements.cpp"
      "${OPS_SRC_DIR}/recorder/FilePlotter.cpp"
      "${OPS_SRC_DIR}/renderer/main.cpp"
      "${OPS_SRC_DIR}/renderer/OpenGlDevice.cpp"
      "${OPS_SRC_DIR}/renderer/OpenGlRenderer.cpp"
      "${OPS_SRC_DIR}/tcl/TclFeViewer.cpp"
      "${OPS_SRC_DIR}/tcl/TclVideoPlayer.cpp"

    PROPERTIES COMPILE_DEFINITIONS _AGL
  )
  target_link_libraries(${OPS_FINAL_TARGET} OPS_Graphics_GL OPS_Graphics)
else()
  add_compile_definitions(_NOGRAPHICS)
endif()


# Reliability
#----------------------------
#if ("OPS_Reliability" IN_LIST OPS_Extension_List)
    add_compile_definitions(_RELIABILITY)
    #target_link_libraries(${OPS_FINAL_TARGET} OPS_Reliability)
#endif()


#----------------------------
# HDF5
#----------------------------

if(HDF5_FOUND)
   include_directories(${HDF5_INCLUDE_DIR})
   set(_hdf5_libs hdf5 hdf5_cpp)
   if (HDF5_VERSION VERSION_GREATER_EQUAL 1.12.0)
         add_compile_definitions(_H5DRM)
         add_compile_definitions(_HDF5)
         message(STATUS "OPS >>> Have HDF5 and VERSION >= 1.12.0")
   endif()
else()
   message(STATUS "OPS >>> Could not find HDF5")
endif()

#----------------------------
# Eigen3
#----------------------------
if(Eigen3_FOUND)
   include_directories(${Eigen3_INCLUDE_DIR})
   add_compile_definitions(_EIGEN3)
else()
   message(STATUS "OPS >>> Could not find Eigen3")
endif()

if (OPS_Use_Dev_Directories)
  add_subdirectory("${PROJECT_SOURCE_DIR}/DEVELOPER/")
endif()
