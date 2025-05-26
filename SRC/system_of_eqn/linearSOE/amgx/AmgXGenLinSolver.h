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
        AmgXGenLinSolver(const char *configFile = nullptr, const char *configOptions = nullptr, 
            const char* mode = "dDDI", bool usePinnedMemory = true, bool verbose = false,
            AMGX_print_callback callback = defaultAmgXCallback);
        ~AmgXGenLinSolver();
        
        int solve();
        int setSize();
        int setLinearSOE(AmgXGenLinSOE &theSOE);

        int sendSelf(int commitTag, Channel &theChannel);
        int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker);

        int getNumIterations();
        double getResidualNorm();

        using MatrixStatus = AmgXGenLinSOE::AmgXMatrixStatus;
    protected:

    private:
        AmgXGenLinSOE *theSOE;
        bool _usePinnedMemory;
        bool _verbose;

        // Static members for global state
        static bool _AmgXInitialized;           ///< Whether AMGX is initialized
        static int _ActiveSolverInstances;     ///< Count of active solver instances
        static AMGX_resources_handle _Resources;  ///< Resources handle
        
        // AMGX handles
        AMGX_config_handle    _Config       = nullptr;  ///< Configuration handle
        AMGX_matrix_handle    _Matrix       = nullptr;  ///< Matrix handle
        AMGX_vector_handle    _RHS          = nullptr;  ///< Right-hand side vector handle
        AMGX_vector_handle    _Solution     = nullptr;  ///< Solution vector handle
        AMGX_solver_handle    _Solver       = nullptr;  ///< Solver handle
        AMGX_Mode             _Mode;                    ///< Solver mode
};

#endif