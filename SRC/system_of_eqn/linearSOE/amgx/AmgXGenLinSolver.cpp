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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/amgx/AmgXGenLinSolver.cpp,v $
                                                                        
                                                                        
// Written: gaaraujo 
// Created: 2025
//
// Description: This file contains the class definition for 
// AmgXGenLinSolver. It solves the AmgXGenLinSOEobject by calling
// AMGX routines.
//

#include <AmgXGenLinSOE.h>
#include <AmgXGenLinSolver.h>
#include <math.h>
#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include <OPS_Stream.h>

// Static member initialization
bool AmgXGenLinSolver::_AmgXInitialized = false;
int AmgXGenLinSolver::_ActiveSolverInstances = 0;

void AmgXGenLinSolver::printAmgXNotAvailable() {
    opserr << "ERROR: AmgXGenLinSolver requires OpenSees "
           << "to be compiled with AMGX (-D_AMGX)\n";
}

void* OPS_AmgXGenLinSolver()
{
    #ifdef _AMGX
    char *configFile = nullptr; 
    char *configOptions = nullptr; 
    AMGX_Mode mode = AMGX_mode_dDDI; 
    bool usePinnedMemory = true;
    void (*callback)(const char* msg, int length) = defaultAmgXCallback;
    int blockSize = 1;
    AmgXGenLinSolver *theSolver = new AmgXGenLinSolver(
        configFile, configOptions, mode, usePinnedMemory, callback
    );
    return new AmgXGenLinSOE(*theSolver, blockSize);
    #else
    opserr << "ERROR: AmgXGenLinSolver is only available when OpenSees is compiled with AMGX support (-D_AMGX)\n";
    return nullptr;
    #endif
}

void defaultAmgXCallback(const char* msg, int length) {
    if (msg && length > 0) {
        opserr.write(msg, length);
        opserr << endln;
    }
}

#ifdef _AMGX
AmgXGenLinSolver::AmgXGenLinSolver( 
    char *configFile, char *configOptions, 
    AMGX_Mode mode, bool usePinnedMemory,
    void (*callback)(const char* msg, int length))
    :LinearSOESolver(SOLVER_TAGS_AmgXGenLinSolver), theSOE(0), 
    _matrixStructureHasChanged(true), _usePinnedMemory(usePinnedMemory)
{
    /* Initialize AMGX library - only done once across all instances */
    if (!_AmgXInitialized) {
        AMGX_SAFE_CALL(AMGX_initialize());
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
        _AmgXInitialized = true;
    }
    AMGX_SAFE_CALL(AMGX_register_print_callback(callback));

    if (configFile != nullptr && configOptions != nullptr) {
        AMGX_SAFE_CALL(AMGX_config_create_from_file_and_string(&_Config, configFile, configOptions));
    } else if (configFile != nullptr) {
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&_Config, configFile));
    } else if (configOptions != nullptr) {
        AMGX_SAFE_CALL(AMGX_config_create(&_Config, configOptions));
    } else {
        opserr << "AmgXGenLinSOE: No config file or options provided" << endln;
        opserr << "AmgXGenLinSOE: Using default config" << endln;

        /* The following settings create an Aggregation solver with DILU 
         * smoother, with 1 pre and 1 post sweep. The solver will stop when 
         * the L2 norm has been reduced by 1000 from the initial norm.
         */
        const char* defaultOptions =
            "config_version=2,"
            "algorithm=AGGREGATION,"
            "selector=ONE_PHASE_HANDSHAKING,"
            "cycle=V,"
            "smoother=MULTICOLOR_DILU,"
            "presweeps=1,"
            "postsweeps=1,"
            "coarse_solver=NOSOLVER,"
            "coarsest_sweeps=2,"
            "max_levels=1000,"
            "norm=L2,"
            "convergence=RELATIVE_INI,"
            "max_uncolored_percentage=0.15,"
            "max_iters=1000,"
            "monitor_residual=1,"
            "tolerance=0.001,"
            "print_solve_stats=1,"
            "print_grid_stats=1,"
            "obtain_timings=1";
        AMGX_SAFE_CALL(AMGX_config_create(&_Config, defaultOptions));
    }

    /* Monitor residual and store residual history 
     * (required for AmgXGenLinSolver::getFinalResidualNorm) */
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&_Config, "monitor_residual=1,store_res_history=1"));

    /* Switch on internal error handling 
     * (no need to use AMGX_SAFE_CALL after this point) */
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&_Config, "exception_handling=1"));

    /* Create resources: single-GPU and single-threaded applications only */
    AMGX_resources_create_simple(&_Resources, _Config);

    /* Create solver, matrix, rhs and solution vectors */
    AMGX_solver_create(&_Solver, _Resources, _Mode, _Config);
    AMGX_matrix_create(&_Matrix, _Resources, _Mode);
    AMGX_vector_create(&_RHS, _Resources, _Mode);
    AMGX_vector_create(&_Solution, _Resources, _Mode);

    _ActiveSolverInstances++;
}
#else
AmgXGenLinSolver::AmgXGenLinSolver() : LinearSOESolver(SOLVER_TAGS_AmgXGenLinSolver) {
    printAmgXNotAvailable();
}
#endif

AmgXGenLinSolver::~AmgXGenLinSolver() {
    #ifdef _AMGX
    /* destroy resources, matrix, vector and solver */
    if (_Solution) { AMGX_vector_destroy(_Solution); _Solution = nullptr; }
    if (_RHS) { AMGX_vector_destroy(_RHS); _RHS = nullptr; }
    if (_Matrix) { AMGX_matrix_destroy(_Matrix); _Matrix = nullptr; }
    if (_Solver) { AMGX_solver_destroy(_Solver); _Solver = nullptr; }
    if (_Resources) { AMGX_resources_destroy(_Resources); _Resources = nullptr; }
    
    /* destroy config (need to use AMGX_SAFE_CALL after this point) */
    if (_Config) { AMGX_SAFE_CALL(AMGX_config_destroy(_Config)); _Config = nullptr; }

    if (_ActiveSolverInstances > 0) {
        _ActiveSolverInstances--;
    }

    // Finalize AMGX only when last instance is destroyed
    if (_ActiveSolverInstances == 0 && _AmgXInitialized) {
        AMGX_reset_signal_handler();
        AMGX_SAFE_CALL(AMGX_finalize());
        _AmgXInitialized = false;
    }
    #endif
}

int AmgXGenLinSolver::solve() {
    #ifdef _AMGX
    /* Obtain information about the matrix structure */
    int numRowBlocks = theSOE->_ARowPtrBlock.size() - 1;
    int nnzBlocks = theSOE->_AColIdxBlock.size();
    int totalNNZ = theSOE->_AValuesBlock.size();
    int blockSize = theSOE->_BlockSize;

    /* Do some sanity checks */
    if (numRowBlocks == 0 || nnzBlocks == 0 || totalNNZ == 0) { return 0; }

    if (totalNNZ != nnzBlocks * blockSize * blockSize) {
        opserr << "WARNING: Total number of non-zero elements (";
        opserr << totalNNZ << ") in the matrix does not match the expected number (";
        opserr << nnzBlocks * blockSize * blockSize << ") -- AmgXGenLinSolver::solve" << endln;
        return -1;
    }

    if (theSOE->_B.Size() != numRowBlocks * blockSize) {
        opserr << "WARNING: RHS vector size (";
        opserr << theSOE->_B.Size() << ") does not match the expected number (";
        opserr << numRowBlocks * blockSize << ") -- AmgXGenLinSolver::solve" << endln;
        return -1;
    }

    if (theSOE->_X.Size() != numRowBlocks * blockSize) {
        opserr << "WARNING: Solution vector size (";
        opserr << theSOE->_X.Size() << ") does not match the expected number (";
        opserr << numRowBlocks * blockSize << ") -- AmgXGenLinSolver::solve" << endln;
        return -1;
    }
    
    /* Upload the system matrix to the GPU */
    if (_matrixStructureHasChanged) { /* Entire matrix needs to be reloaded*/
        if (_Mode == AMGX_mode_dDDI && _usePinnedMemory) {
            AMGX_pin_memory(
                (void*)(theSOE->_ARowPtrBlock.data()), 
                sizeof(int) * (numRowBlocks + 1)
            );
            AMGX_pin_memory(
                (void*)(theSOE->_AColIdxBlock.data()), 
                sizeof(int) * (nnzBlocks)
            );
            AMGX_pin_memory(
                (void*)(theSOE->_AValuesBlock.data()), 
                sizeof(double) * (totalNNZ)
            );
        }
        AMGX_matrix_upload_all(
            _Matrix, numRowBlocks, nnzBlocks, 
            blockSize, blockSize, theSOE->_ARowPtrBlock.data(), 
            theSOE->_AColIdxBlock.data(), theSOE->_AValuesBlock.data(), nullptr
        );
        if (_Mode == AMGX_mode_dDDsI && _usePinnedMemory) {
            AMGX_unpin_memory((void*)(theSOE->_ARowPtrBlock.data()));
            AMGX_unpin_memory((void*)(theSOE->_AColIdxBlock.data()));
            AMGX_unpin_memory((void*)(theSOE->_AValuesBlock.data()));
        }
        _matrixStructureHasChanged = false;
    } else { /* Only the matrix coefficients have changed */
        if (_Mode == AMGX_mode_dDDI && _usePinnedMemory) {
            AMGX_pin_memory(
                (void*)(theSOE->_AValuesBlock.data()), 
                sizeof(double) * (totalNNZ)
            );
        }
        AMGX_matrix_replace_coefficients(
            _Matrix, numRowBlocks, nnzBlocks, 
            theSOE->_AValuesBlock.data(), nullptr
        );
        if (_Mode == AMGX_mode_dDDI && _usePinnedMemory) {
            AMGX_unpin_memory((void*)(theSOE->_AValuesBlock.data()));
        }
    }

    AMGX_solver_setup(_Solver, _Matrix);

    /* Upload the RHS vector to the GPU*/
    double* X_ptr = &(theSOE->_X(0));
    double* B_ptr = &(theSOE->_B(0));
    if (_Mode == AMGX_mode_dDDI && _usePinnedMemory) {
        AMGX_pin_memory((void*)(B_ptr), sizeof(double) * (theSOE->_B.Size()));
        AMGX_pin_memory((void*)(X_ptr), sizeof(double) * (theSOE->_X.Size()));
    }
    AMGX_vector_upload(_RHS, numRowBlocks, blockSize, B_ptr);
    
    /* Solve with 0-vector initial guess */
    AMGX_vector_set_zero(_Solution, numRowBlocks, blockSize);
    AMGX_solver_solve_with_0_initial_guess(_Solver, _RHS, _Solution);

    /* Download the solution vector from the GPU */
    AMGX_vector_download(_Solution, X_ptr);
    if (_Mode == AMGX_mode_dDDI && _usePinnedMemory) {
        AMGX_unpin_memory((void*)(X_ptr));
        AMGX_unpin_memory((void*)(B_ptr));
    }

    /* AMGX check status */
    AMGX_SOLVE_STATUS status;
    AMGX_solver_get_status(_Solver, &status);

    if (status == AMGX_SOLVE_DIVERGED) {
        opserr << "WARNING: Solver diverged -- AmgXGenLinSolver::solve" << endln;
        opserr << "AmgXGenLinSolver::getNumIterations() = " << this->getNumIterations() << endln;
        opserr << "AmgXGenLinSolver::getFinalResidualNorm() = " << this->getFinalResidualNorm() << endln;
    }
    if (status != AMGX_SOLVE_SUCCESS) {
        opserr << "WARNING: solving returns " << status << " -- AmgXGenLinSolver::solve" << endln;
        return -1;
    }

    return 0;
    #else
    printAmgXNotAvailable();
    return -1;
    #endif
}

int AmgXGenLinSolver::setSize()
{
    #ifdef _AMGX
    _matrixStructureHasChanged = true;
    return 0;
    #else
    printAmgXNotAvailable();
    return -1;
    #endif
}

int AmgXGenLinSolver::setLinearSOE(AmgXGenLinSOE &theLinearSOE)
{
    #ifdef _AMGX
    theSOE = &theLinearSOE;
    return 0;
    #else
    printAmgXNotAvailable();
    return -1;
    #endif
}

int AmgXGenLinSolver::sendSelf(int cTag, Channel &theChannel)
{
    // nothing to do
    return 0;
}

int AmgXGenLinSolver::recvSelf(int ctag,
			      Channel &theChannel, 
			      FEM_ObjectBroker &theBroker)
{
    // nothing to do
    return 0;
}

int AmgXGenLinSolver::getNumIterations() {
    #ifdef _AMGX
    int numIterations;
    AMGX_solver_get_iterations_number(_Solver, &numIterations);
    return numIterations;
    #else
    printAmgXNotAvailable();
    return -1;
    #endif
}

double AmgXGenLinSolver::getFinalResidualNorm() {
    #ifdef _AMGX
    double residualComponent;
    double finalResidualNorm = 0.0;
    for (int blockIdx = 0; blockIdx < theSOE->_BlockSize; blockIdx++) {
        AMGX_solver_get_iteration_residual(_Solver, this->getNumIterations(), blockIdx, &residualComponent);
        finalResidualNorm += residualComponent * residualComponent;
    }
    return std::sqrt(finalResidualNorm);
    #else
    printAmgXNotAvailable();
    return -1;
    #endif
}