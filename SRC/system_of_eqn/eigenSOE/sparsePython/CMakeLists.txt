target_sources(OPS_SysOfEqnPython
  PRIVATE
    SparsePythonCompressedEigenSOE.cpp
    SparsePythonCompressedEigenSolver.cpp
    SparsePythonCOOEigenSOE.cpp
    SparsePythonCOOEigenSolver.cpp
    SparsePythonEigenCommon.cpp
    SparsePythonEigenFactory.cpp
    DistributedSparsePythonEigenSOE.cpp
  PUBLIC
    SparsePythonCompressedEigenSOE.h
    SparsePythonCompressedEigenSolver.h
    SparsePythonCOOEigenSOE.h
    SparsePythonCOOEigenSolver.h
    SparsePythonEigenCommon.h
    SparsePythonEigenFactory.h
    DistributedSparsePythonEigenSOE.h
)

find_package(Python3 COMPONENTS Development REQUIRED)
target_link_libraries(OPS_SysOfEqnPython PRIVATE Python3::Python)
