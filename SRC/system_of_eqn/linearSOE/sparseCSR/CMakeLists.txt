#==============================================================================
# 
#        OpenSees -- Open System For Earthquake Engineering Simulation
#                Pacific Earthquake Engineering Research Center
#
#==============================================================================
# MumpsCsrBackend.cpp is not listed here: it needs MUMPS and MPI headers and is
# compiled into the executables that link MUMPS (see OpenSeesMP in the root CMakeLists).
target_sources(OPS_SysOfEqn
    PRIVATE
        CsrMatrix.cpp
        SparseCsrLinSOE.cpp
        DistributedSparseCsrLinSOE.cpp

    PUBLIC
        CsrDirectBackend.h
        CsrMatrix.h
        SparseCsrLinSOE.h
        DistributedSparseCsrLinSOE.h
)

target_include_directories(OPS_SysOfEqn PUBLIC ${CMAKE_CURRENT_LIST_DIR})
