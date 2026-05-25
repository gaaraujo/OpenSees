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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCUDA/AmgXLinSolver.h
//
// Written: gaaraujo 
// Created: 08/2025
//
// Description: This file contains the class definition for AmgXLinSolver.
// An AmgXLinSolver object can be constructed to solve a CudaGenBcsrLinSOE
// object. It obtains the solution by making calls to the
// AmgX library developed by NVIDIA.
//
// The AmgX library provides a high-performance GPU-accelerated solver
// for sparse linear systems of the form AX = B. It supports various
// iterative methods such as Conjugate Gradient (CG), GMRES, and BiCGStab,
// combined with advanced preconditioners including Algebraic Multigrid (AMG).
//
// AmgX is designed for integration into large-scale simulation codes,
// enabling significant acceleration of the linear solution phase when
// running on CUDA-capable GPUs.
//
// This solver will only be available if OpenSees is compiled with AMGX
// support enabled (i.e., with the compile flag -D_AMGX). This wrapper 
// is implemented for single-CPU and single-GPU systems only. 
//

#ifndef AmgXLinSolver_h
#define AmgXLinSolver_h

// OpenSees includes
#include <CudaGenBcsrLinSolver.h>
#include <OPS_Stream.h>

// C++ includes
#include <string>

// AMGX includes
#ifdef _AMGX
#include <amgx_c.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
void AmgXCallback(const char* msg, int length);
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // _AMGX

class AmgXLinSolver : public CudaGenBcsrLinSolver
{
public:
    // Constructor with default config
    AmgXLinSolver(
        const std::string solver = "PCG", const std::string preconditioner = "JACOBI_L1", 
        const std::string smoother = "JACOBI_L1", 
        int max_iters = 1000, double abs_tol = 1e-12, double rel_tol = 1e-6, 
        int monitor_residual = 1, bool verbose = false
    );
    
    // Constructor with config file and options
    AmgXLinSolver(
        const char *configFile = nullptr, const char *configOptions = nullptr,
        CudaPrecision precision = CudaPrecision::dDDI, bool verbose = false, 
        OPS_Stream* callbackStream = (OPS_Stream*)&opserr
    );
    
    // Destructor
    ~AmgXLinSolver();

    // Solver methods
    int solve(void) override;
    int setSize(void) override;
    int getNumIterations(void);
    double getResidualNorm(void);
    int setLinearSOE(CudaGenBcsrLinSOE &theSOE) override;
    
    // Output stream
    static void setCallbackStream(OPS_Stream* output);
    static OPS_Stream* getCallbackStream();

protected:

private:
    // Verbosity flag
    bool m_verbose;
    
    
    // AMGX initializer (to be used by constructors only)
    void init(const char *configFile = nullptr, 
              const char *configOptions = nullptr, 
              const char *precision = "dDDI",
              OPS_Stream* callbackStream = (OPS_Stream*)&opserr);

    #ifdef _AMGX
    // Static members for global state
    static bool m_AmgXInitialized;           ///< Whether AMGX is initialized
    static int m_ActiveSolverInstances;     ///< Count of active solver instances
    static AMGX_resources_handle m_Resources;  ///< Resources handle
    static OPS_Stream* m_CallbackStream;         ///< Callback stream

    // AMGX handles
    AMGX_config_handle    m_Config       = nullptr;  ///< Configuration handle
    AMGX_matrix_handle    m_Matrix       = nullptr;  ///< Matrix handle
    AMGX_vector_handle    m_RHS          = nullptr;  ///< Right-hand side vector handle
    AMGX_vector_handle    m_Solution     = nullptr;  ///< Solution vector handle
    AMGX_solver_handle    m_Solver       = nullptr;  ///< Solver handle
    AMGX_Mode             m_Mode;                    ///< Solver mode
    #endif // _AMGX
};

#endif