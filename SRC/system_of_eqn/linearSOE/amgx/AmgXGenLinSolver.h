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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/amgx/AmgXGenLinSolver.h
//
// Written: gaaraujo 
// Created: 05/2025
//
// Description: This file contains the class definition for AmgXSolver.
// An AmgXSolver object can be constructed to solve a AmgXGenLinSOE
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

#ifndef AmgXGenLinSolver_h
#define AmgXGenLinSolver_h

#include <LinearSOESolver.h>
#include <OPS_Stream.h>
#include <string>
#include <amgx_c.h>

#ifdef __cplusplus
extern "C" {
#endif
void defaultAmgXCallback(const char* msg, int length);
void noAmgXCallback(const char* msg, int length);
#ifdef __cplusplus
}
#endif

class AmgXGenLinSOE;

class AmgXGenLinSolver : public LinearSOESolver
{
    public:
        // Constructor with default config
        AmgXGenLinSolver(const std::string solver = "PCG", const std::string preconditioner = "AMG", 
            std::string smoother = "JACOBI_L1", int max_iters = 1000, double abs_tolerance = 1e-12, 
            double rel_tolerance = 1e-6, int monitor_residual = 1, const std::string mode = "dDDI", 
            bool usePinnedMemory = true, bool verbose = false);
        // Constructor with config file and options
        AmgXGenLinSolver(const char *configFile = nullptr, const char *configOptions = nullptr, 
            const char* mode = "dDDI", bool usePinnedMemory = true, bool verbose = false,
            AMGX_print_callback callback = defaultAmgXCallback);
        // Destructor
        ~AmgXGenLinSolver();

        // Solver methods
        int solve();
        int setSize();
        int setLinearSOE(AmgXGenLinSOE &theSOE);
        int getNumIterations();
        double getResidualNorm();

        // Parallel methods
        int sendSelf(int commitTag, Channel &theChannel);
        int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker);

    protected:

    private:
        AmgXGenLinSOE *theSOE;
        bool m_usePinnedMemory;
        bool m_verbose;

        // AMGX initializer (to be used by constructors only)
        void _init(const char *configFile = nullptr, 
                   const char *configOptions = nullptr, 
                   const char* mode = "dDDI",
                   AMGX_print_callback callback = defaultAmgXCallback);

        // Static members for global state
        static bool m_AmgXInitialized;           ///< Whether AMGX is initialized
        static int m_ActiveSolverInstances;     ///< Count of active solver instances
        static AMGX_resources_handle m_Resources;  ///< Resources handle
        
        // AMGX handles
        AMGX_config_handle    m_Config       = nullptr;  ///< Configuration handle
        AMGX_matrix_handle    m_Matrix       = nullptr;  ///< Matrix handle
        AMGX_vector_handle    m_RHS          = nullptr;  ///< Right-hand side vector handle
        AMGX_vector_handle    m_Solution     = nullptr;  ///< Solution vector handle
        AMGX_solver_handle    m_Solver       = nullptr;  ///< Solver handle
        AMGX_Mode             m_Mode;                    ///< Solver mode
};

#endif