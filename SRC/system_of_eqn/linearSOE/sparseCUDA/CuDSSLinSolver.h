/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
**                                                                    **
** (C) Copyright 1999, The Regents of the University of California    **
** All Rights Reserved.                                               **
**                                                                    **
** Commercial use of this program without express permission of the   **
** University of California, Berkeley, is strictly prohibited.  See   **
** file 'COPYRIGHT'  in main directory for information on usage and   **
** redistribution,  and for a DISCLAIMER OF ALL WARRANTIES.           **
**                                                                    **
** Developed by:                                                      **
**   Frank McKenna (fmckenna@ce.berkeley.edu)                         **
**   Gregory L. Fenves (fenves@ce.berkeley.edu)                       **
**   Filip C. Filippou (filippou@ce.berkeley.edu)                     **
**                                                                    **
** ****************************************************************** */                                                                        
                                                                       
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCUDA/CuDSSLinSolver.h
//
// Written: gaaraujo
// Created: 10/2025
//
// Description: This file contains the class definition for CuDSSLinSolver.
// An CuDSSLinSolver object can be constructed to solve a CudaGenBcsrLinSOE
// object. It obtains the solution by making calls to the
// CuDSS library developed by NVIDIA.
//
// The CuDSS library provides a high-performance GPU-accelerated direct solver
// for sparse linear systems of the form AX = B. The cuDSS functionality allows 
// flexibility in matrix properties and solver configuration, as well as 
// execution parameters like CUDA streams.
//
// This solver will only be available if OpenSees is compiled with CUDSS
// support enabled (i.e., with the compile flag -D_CUDSS). This wrapper 
// is implemented for single-CPU and single-GPU systems only. 
//

#ifndef CuDSSLinSolver_h
#define CuDSSLinSolver_h

// OpenSees includes
#include <CudaGenBcsrLinSolver.h>
#include <OPS_Stream.h>

// C++ includes
#include <string>
#include <vector>

// Matrix type for cuDSS: full, symmetric, or SPD (symmetric positive definite).
// symmetric and spd both use symmetric lower storage in CudaGenBcsrLinSOE.
enum class CuDSSMatrixType { FULL, SYMMETRIC, SPD };

// cuDSS includes
#ifdef _CUDSS
#include <cudss.h>
#endif // _CUDSS

class CuDSSLinSolver : public CudaGenBcsrLinSolver
{
public:
    // Constructor with default parameters
    // useMultiGPU: use cuDSS multi-GPU (MG) mode; deviceIndices: GPU IDs (empty = use all)
    CuDSSLinSolver(
        CudaPrecision precision = CudaPrecision::dDDI,
        bool verbose = false,
        bool hybridMemoryMode = false,
        const std::vector<size_t>& hybridDeviceMemoryLimits = {},
        bool hybridExecuteMode = false,
        bool multiThreadingMode = false,
        const char* threadingLibPath = nullptr,
        CuDSSMatrixType cudssMatType = CuDSSMatrixType::FULL,
        bool useMultiGPU = false,
        const std::vector<int>& deviceIndices = {}
    );
    
    // Destructor
    ~CuDSSLinSolver();
    
    // Solver methods
    int solve(void) override;
    int setSize(void) override;
    int setLinearSOE(CudaGenBcsrLinSOE &theSOE) override;

protected:

private:
    // Verbosity flag
    bool m_verbose;
    
    // Hybrid mode settings
    bool m_hybridMemoryMode;
    std::vector<size_t> m_hybridDeviceMemoryLimits;  // Per-device limit (empty = use heuristic)
    bool m_hybridExecuteMode;
    
    // Multi-threading settings
    bool m_multiThreadingMode;
    std::string m_threadingLibPath;

    // Matrix type: full, symmetric, or SPD (affects cuDSS mtype; symmetric/SPD use lower storage)
    CuDSSMatrixType m_cudssMatType;

    // Multi-GPU (MG) mode: when true, use cudssCreateMg; otherwise cudssCreate
    bool m_useMultiGPU;
    std::vector<int> m_deviceIndices;

    // cuDSS initializer (to be used by constructors only)
    void init(CudaPrecision precision);
    
    // Helper function to initialize cuDSS matrices when structure changes
    int setupMatrices();

    #ifdef _CUDSS
    // Per-instance handle and stream (same member for single-GPU and MG)
    cudssHandle_t m_Handle;
    cudaStream_t m_cudaStream;
    
    // cuDSS objects
    cudssConfig_t m_Config;      ///< solver configuration
    cudssData_t m_Data;          ///< solver data
    cudssMatrix_t m_Matrix;      ///< matrix data
    cudssMatrix_t m_RHS;         ///< right-hand side data
    cudssMatrix_t m_Solution;    ///< solution data
    
    // Precision
    cudaDataType_t m_ValueType; ///< SOE value type
    cudaDataType_t m_IndexType; ///< SOE index type
    #endif // _CUDSS
};
#endif