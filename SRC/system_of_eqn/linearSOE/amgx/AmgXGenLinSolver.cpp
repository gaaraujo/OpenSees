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
#include <elementAPI.h>
#include <sstream>
#include <iomanip>

// Track if the matrix in the AmgXGenLinSOE has changed
using MatrixStatus = AmgXGenLinSOE::AmgXMatrixStatus;

// Static member initialization
bool AmgXGenLinSolver::m_AmgXInitialized = false;
int AmgXGenLinSolver::m_ActiveSolverInstances = 0;
AMGX_resources_handle AmgXGenLinSolver::m_Resources = nullptr;

/* AMGX callbacks */
#ifdef __cplusplus
extern "C" {
#endif
void defaultAmgXCallback(const char* msg, int length) {
    if (msg && length > 0) {
        opserr.write(msg, length);
        opserr << endln;
    }
}

void noAmgXCallback(const char* msg, int length) {
    // Do nothing
}

#ifdef __cplusplus
}
#endif

/* Anonymous namespace for helper functions */
namespace {
    void reportAmgXSolveStats(int numIterations, double initialResidualNorm, double finalResidualNorm) {
        opserr << "AmgXGenLinSolver: Number of iterations = " << numIterations << endln;
        opserr << "AmgXGenLinSolver: Initial residual norm = " << initialResidualNorm << endln;
        opserr << "AmgXGenLinSolver: Final residual norm = " << finalResidualNorm << endln;
        if (initialResidualNorm > 0.0) {
            opserr << "AmgXGenLinSolver: Final to initial residual norm ratio = " << finalResidualNorm / initialResidualNorm << endln;
        } else {
            opserr << "AmgXGenLinSolver: Final to initial residual norm ratio = NaN" << endln;
        }
        // opserr << "NOTE: If the AMGX final absolute or residual norms reported here \n";
        // opserr << "are smaller than your specified tolerances and the solver still failed to converge, \n";
        // opserr << "it is likely that you may have specified use_scalar_norm=0 in the AMGX solver config options. \n";
        // opserr << "This results in the AMGX solver using a component-wise L2 norm instead of a scalar L2 norm. \n";
        // opserr << "To fix this, pass the config option use_scalar_norm=1 to your solver config options. \n";
        // opserr << "For more information, see the AMGX documentation." << endln;
    }

    std::string getDefaultConfigOptions(std::string solver = "PCG", 
                                        std::string preconditioner = "AMG", 
                                        std::string smoother = "BLOCK_JACOBI",
                                        int max_iters = 1000,
                                        double abs_tolerance = 1e-12,
                                        double rel_tolerance = 1e-6,
                                        int monitor_residual = 1) 
    {
        /* These settings configure a PCG solver with a V-cycle AMG preconditioner
         * using a BLOCK_JACOBI smoother. Convergence is reached when the L2 norm
         * drops by 6 orders of magnitude relative to the initial norm or below 
         * 1e-12 absolute. Based on MFEM's AmgXWrapper defaults:
         * https://docs.mfem.org/html/amgxsolver_8cpp_source.html
        */
        if (solver != "PCG" && solver != "FGMRES" && solver != "PCGF") {
            opserr << "WARNING: AmgXGenLinSolver: Tried to create default config with solver = " << solver.c_str() << ".";
            opserr << "Default config constructor only works with solver = PCG, PCGF, or FGMRES.";
            opserr << "To use a different solver, pass a config file or config string to the constructor using: " << endln;
            opserr << "system AmgX <-configFile CONFIG_FILE> <-configOptions CONFIG_OPTIONS>" << endln;
            opserr << "For more information, see the AMGX documentation." << endln;
            return "";
        }
        if (preconditioner != "AMG" && preconditioner != "BLOCK_JACOBI" && preconditioner != "JACOBI_L1") {
            opserr << "WARNING: AmgXGenLinSolver: Tried to create default config with preconditioner = " << preconditioner.c_str() << ".";
            opserr << "Default config constructor only works with preconditioner = AMG, BLOCK_JACOBI, or JACOBI_L1.";
            opserr << "To use a different preconditioner, pass a config file or config string to the constructor using: " << endln;
            opserr << "system AmgX <-configFile CONFIG_FILE> <-configOptions CONFIG_OPTIONS>" << endln;
            opserr << "For more information, see the AMGX documentation." << endln;
            return "";
        }
        if (preconditioner == "AMG" && (smoother != "JACOBI_L1" && smoother != "BLOCK_JACOBI")) {
            opserr << "WARNING: AmgXGenLinSolver: Tried to create default AMG preconditioner with smoother = " << smoother.c_str() << ".";
            opserr << "Default AMG preconditioner only works with smoother = JACOBI_L1 or BLOCK_JACOBI.";
            opserr << "To use a different smoother, pass a config file or config string to the constructor using: " << endln;
            opserr << "system AmgX <-configFile CONFIG_FILE> <-configOptions CONFIG_OPTIONS>" << endln;
            opserr << "For more information, see the AMGX documentation." << endln;
            return "";
        }

        if (max_iters <= 0) {
            opserr << "WARNING: AmgXGenLinSolver: Tried to create default config with max_iters = " << max_iters << ".";
            opserr << "Expected max_iters > 0.";
            return "";
        }
        if (abs_tolerance < 0.0) {
            opserr << "WARNING: AmgXGenLinSolver: Tried to create default config with abs_tolerance = " << abs_tolerance << ".";
            opserr << "Expected abs_tolerance >= 0.0.";
            return "";
        }
        if (rel_tolerance < 0.0) {
            opserr << "WARNING: AmgXGenLinSolver: Tried to create default config with rel_tolerance = " << rel_tolerance << ".";
            opserr << "Expected rel_tolerance >= 0.0.";
            return "";
        }
        if (monitor_residual != 0 && monitor_residual != 1) {
            opserr << "WARNING: AmgXGenLinSolver: Tried to create default config with monitor_residual = " << monitor_residual << ".";
            opserr << "Expected monitor_residual = 0 or 1.";
            return "";
        }

        std::ostringstream defaultOptions;
        defaultOptions << std::setprecision(16);
        defaultOptions << "config_version=2,";
        defaultOptions << "solver(main)=" << solver << ",";
        defaultOptions << "main:norm=L2,";
        defaultOptions << "main:convergence=COMBINED_REL_INI_ABS,";
        defaultOptions << "main:max_iters=" << max_iters << ",";
        defaultOptions << "main:tolerance=" << abs_tolerance << ",";
        defaultOptions << "main:alt_rel_tolerance=" << rel_tolerance << ",";
        defaultOptions << "main:monitor_residual=" << monitor_residual << ",";
        defaultOptions << "main:use_scalar_norm=1,"; // Do not use vector norm
        defaultOptions << "main:preconditioner(precond)=" << preconditioner << ",";
        defaultOptions << "precond:max_iters=1,";

        if (preconditioner == "AMG") {
            defaultOptions << "precond:smoother(smoother)=" << smoother << ",";
            defaultOptions << "precond:presweeps=1,";
            defaultOptions << "precond:interpolator=D2,";
            defaultOptions << "precond:max_row_sum=0.9,";
            defaultOptions << "precond:strength_threshold=0.25,";
            defaultOptions << "precond:max_levels=100,";
            defaultOptions << "precond:cycle=V,";
            defaultOptions << "precond:postsweeps=1";
        }
        return defaultOptions.str();
    }
}

AmgXGenLinSolver::AmgXGenLinSolver(
    const std::string solver, const std::string preconditioner, const std::string smoother, 
    int max_iters, double abs_tolerance, double rel_tolerance, int monitor_residual, 
    const std::string mode, bool usePinnedMemory, bool verbose)
    :LinearSOESolver(SOLVER_TAGS_AmgXGenLinSolver), theSOE(0), 
    m_usePinnedMemory(usePinnedMemory), m_verbose(verbose)
{
    std::string configOptions = getDefaultConfigOptions(
        solver, preconditioner, smoother, max_iters, 
        abs_tolerance, rel_tolerance, monitor_residual);
    const char* nullConfigFile = nullptr;
    _init(nullConfigFile, configOptions.c_str(), mode.c_str());
}

AmgXGenLinSolver::AmgXGenLinSolver( 
    const char *configFile, const char *configOptions, 
    const char* mode, bool usePinnedMemory, bool verbose,
    AMGX_print_callback callback)
    :LinearSOESolver(SOLVER_TAGS_AmgXGenLinSolver), theSOE(0), 
    m_usePinnedMemory(usePinnedMemory), m_verbose(verbose)
{
    _init(configFile, configOptions, mode, callback);
}

void AmgXGenLinSolver::_init(const char *configFile, const char *configOptions, 
                        const char* mode, AMGX_print_callback callback) 
{
    /* Initialize AMGX library - only done once across all instances */
    if (!m_AmgXInitialized) {
        AMGX_SAFE_CALL(AMGX_initialize());
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
        m_AmgXInitialized = true;
    }
    AMGX_SAFE_CALL(AMGX_register_print_callback(callback));

    if (configFile != nullptr && strlen(configFile) > 0 && configOptions != nullptr && strlen(configOptions) > 0) {
        AMGX_SAFE_CALL(AMGX_config_create_from_file_and_string(&m_Config, configFile, configOptions));
    } else if (configFile != nullptr && strlen(configFile) > 0) {
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&m_Config, configFile));
    } else if (configOptions != nullptr && strlen(configOptions) > 0) {
        AMGX_SAFE_CALL(AMGX_config_create(&m_Config, configOptions));
    } else {
        std::string defaultOptions = getDefaultConfigOptions();

        AMGX_SAFE_CALL(AMGX_config_create(&m_Config, defaultOptions.c_str()));
    }

    /* Print some stuff to the console */
    if (m_verbose) {
        /* Note: Some parameters are scope-specific and depend on names defined 
         * in the config. Since parsing the config for scope names is 
         * non-trivial, we hardcode common ones: "main" (used in AMGX examples) 
         * and "default_sub_solver" (default if none is specified).
         */
        const std::vector<std::string> scopes = {
            "main", "default_sub_solver"
            };
        /* Main solver settings */
        for (const std::string& scope : scopes) {
            std::string params = 
                "config_version=2," +
                scope + ":obtain_timings=1," +
                scope + ":print_solve_stats=1," +
                scope + ":print_grid_stats=1," +
                scope + ":print_config=1";
            AMGX_SAFE_CALL(AMGX_config_add_parameters(&m_Config, params.c_str()));
        }
    }

    /* Switch on internal error handling 
     * (no need to use AMGX_SAFE_CALL after this point) */
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&m_Config, "exception_handling=1"));

    /* Use scalar norm for the L2 norm */
    AMGX_config_add_parameters(&m_Config, "config_version=2,default:use_scalar_norm=1,main:use_scalar_norm=1,default_sub_folder:use_scalar_norm=1");

    /* Create resources (single-GPU, single-threaded only).
     * AMGX resource objects are not thread-safe, so we ensure only one is created.
     * References:
     * https://github.com/NVIDIA/AMGX/issues/109#issuecomment-674046246
     * https://github.com/barbagroup/AmgXWrapper/blob/ab70b5e7bc1248875040814edef7474ad18349af/src/AmgXSolver.hpp#L335
     */

    if (m_ActiveSolverInstances == 0) {
        AMGX_resources_create_simple(&m_Resources, m_Config);
    }

    /* Set AmgX mode */
    if(strcmp(mode, "dDDI") == 0) {
        m_Mode = AMGX_mode_dDDI;
    } else {
        opserr << "WARNING: AmgXGenLinSolver: Invalid mode (" << mode << "). Only mode=dDDI is supported.\n";
        return;
    }
    /* Create solver, matrix, rhs and solution vectors */
    AMGX_solver_create(&m_Solver, m_Resources, m_Mode, m_Config);
    AMGX_matrix_create(&m_Matrix, m_Resources, m_Mode);
    AMGX_vector_create(&m_RHS, m_Resources, m_Mode);
    AMGX_vector_create(&m_Solution, m_Resources, m_Mode);

    m_ActiveSolverInstances++;
}

AmgXGenLinSolver::~AmgXGenLinSolver() {
    /* AMGX solver destroy must be called prior to AMGX matrix destroy. 
      See AMGX Reference Manual V2.0 (2017), Section 1.4.2 */
    AMGX_solver_destroy(m_Solver);

    /* Destroy matrix and vectors */
    AMGX_vector_destroy(m_Solution);
    AMGX_vector_destroy(m_RHS);
    AMGX_matrix_destroy(m_Matrix);
    
    /* Finalize AMGX only when last instance is destroyed. 
       Otherwise, just destroy the config */
    if (m_ActiveSolverInstances == 1) {
        AMGX_resources_destroy(m_Resources);
        /* need to use AMGX_SAFE_CALL after this point */
        AMGX_SAFE_CALL(AMGX_config_destroy(m_Config));
        AMGX_reset_signal_handler();
        AMGX_SAFE_CALL(AMGX_finalize());
        m_AmgXInitialized = false;
    } else {
        AMGX_config_destroy(m_Config);
    }

    if (m_ActiveSolverInstances > 0) {
        m_ActiveSolverInstances--;
    }
}

int AmgXGenLinSolver::solve() {
    /* Obtain information about the matrix structure */
    int numRowBlocks = theSOE->m_ARowPtrBlock.size() - 1;
    int nnzBlocks = theSOE->m_AColIdxBlock.size();
    int totalNNZ = theSOE->m_AValuesBlock.size();
    int blockSize = theSOE->m_BlockSize;

    /* Do some sanity checks */
    if (numRowBlocks == 0 || nnzBlocks == 0 || totalNNZ == 0) { return 0; }

    if (totalNNZ != nnzBlocks * blockSize * blockSize) {
        opserr << "WARNING: Total number of non-zero elements (";
        opserr << totalNNZ << ") in the matrix does not match the expected number (";
        opserr << nnzBlocks * blockSize * blockSize << ") -- AmgXGenLinSolver::solve" << endln;
        return -1;
    }

    if (theSOE->m_BPadded.size() != numRowBlocks * blockSize) {
        opserr << "WARNING: RHS vector size (";
        opserr << theSOE->m_BPadded.size() << ") does not match the expected number (";
        opserr << numRowBlocks * blockSize << ") -- AmgXGenLinSolver::solve" << endln;
        return -1;
    }

    if (theSOE->m_XPadded.size() != numRowBlocks * blockSize) {
        opserr << "WARNING: Solution vector size (";
        opserr << theSOE->m_XPadded.size() << ") does not match the expected number (";
        opserr << numRowBlocks * blockSize << ") -- AmgXGenLinSolver::solve" << endln;
        return -1;
    }
    
    /* Upload the system matrix to the GPU */
    switch (theSOE->m_matrixStatus) {
        case MatrixStatus::STRUCTURE_CHANGED: /* Entire matrix needs to be reloaded*/
            if (m_Mode == AMGX_mode_dDDI && m_usePinnedMemory) {
                /* WARNING: Even though, internal error handling has been 
                 * requested, AMGX_SAFE_CALL needs to be used on this system 
                 * call. It is an exception to the general rule.
                 */
                AMGX_SAFE_CALL(AMGX_pin_memory(
                    (void*)(theSOE->m_ARowPtrBlock.data()), 
                    sizeof(int) * (numRowBlocks + 1)
                ));
                AMGX_SAFE_CALL(AMGX_pin_memory(
                    (void*)(theSOE->m_AColIdxBlock.data()), 
                    sizeof(int) * (nnzBlocks)
                ));
                AMGX_SAFE_CALL(AMGX_pin_memory(
                    (void*)(theSOE->m_AValuesBlock.data()), 
                    sizeof(double) * (totalNNZ)
                ));
            }
            AMGX_matrix_upload_all(
                m_Matrix, numRowBlocks, nnzBlocks, 
                blockSize, blockSize, theSOE->m_ARowPtrBlock.data(), 
                theSOE->m_AColIdxBlock.data(), theSOE->m_AValuesBlock.data(), nullptr
            );
            if (m_Mode == AMGX_mode_dDDI && m_usePinnedMemory) {
                AMGX_SAFE_CALL(AMGX_unpin_memory((void*)(theSOE->m_ARowPtrBlock.data())));
                AMGX_SAFE_CALL(AMGX_unpin_memory((void*)(theSOE->m_AColIdxBlock.data())));
                AMGX_SAFE_CALL(AMGX_unpin_memory((void*)(theSOE->m_AValuesBlock.data())));
            }
            AMGX_solver_setup(m_Solver, m_Matrix);
            break;
        case MatrixStatus::COEFFICIENTS_CHANGED: /* Only the matrix coefficients have changed */
            if (m_Mode == AMGX_mode_dDDI && m_usePinnedMemory) {
                AMGX_SAFE_CALL(AMGX_pin_memory(
                    (void*)(theSOE->m_AValuesBlock.data()), 
                    sizeof(double) * (totalNNZ)
                ));
            }
            AMGX_matrix_replace_coefficients(
                m_Matrix, numRowBlocks, nnzBlocks, 
                theSOE->m_AValuesBlock.data(), nullptr
            );
            if (m_Mode == AMGX_mode_dDDI && m_usePinnedMemory) {
                AMGX_SAFE_CALL(AMGX_unpin_memory((void*)(theSOE->m_AValuesBlock.data())));
            }
            AMGX_solver_setup(m_Solver, m_Matrix);
            break;
        case MatrixStatus::UNCHANGED: /* Matrix is the same as the previous solve */
            /* Do nothing */
            break;
        default:
            opserr << "WARNING: AmgXGenLinSolver: Invalid matrix status (" << theSOE->m_matrixStatus << ").\n";
            opserr << "Only " << MatrixStatus::STRUCTURE_CHANGED << ", ";
            opserr << MatrixStatus::COEFFICIENTS_CHANGED << ", ";
            opserr << MatrixStatus::UNCHANGED << " are supported -- AmgXGenLinSolver::solve" << endln;
            return -1;
    }

    /* Upload the RHS vector to the GPU*/
    double* X_ptr = theSOE->m_XPadded.data(); //&(theSOE->m_X(0));
    double* B_ptr = theSOE->m_BPadded.data(); //&(theSOE->m_B(0));
    if (m_Mode == AMGX_mode_dDDI && m_usePinnedMemory) {
        AMGX_SAFE_CALL(AMGX_pin_memory((void*)(B_ptr), sizeof(double) * (theSOE->m_BPadded.size())));
        AMGX_SAFE_CALL(AMGX_pin_memory((void*)(X_ptr), sizeof(double) * (theSOE->m_XPadded.size())));
    }
    AMGX_vector_upload(m_RHS, numRowBlocks, blockSize, B_ptr);
    
    /* Solve with 0-vector initial guess */
    AMGX_vector_set_zero(m_Solution, numRowBlocks, blockSize);
    double initialResidualNorm = this->getResidualNorm();
    AMGX_solver_solve_with_0_initial_guess(m_Solver, m_RHS, m_Solution);
    double finalResidualNorm = this->getResidualNorm();

    /* Download the solution vector from the GPU */
    AMGX_vector_download(m_Solution, X_ptr);
    if (m_Mode == AMGX_mode_dDDI && m_usePinnedMemory) {
        AMGX_SAFE_CALL(AMGX_unpin_memory((void*)(X_ptr)));
        AMGX_SAFE_CALL(AMGX_unpin_memory((void*)(B_ptr)));
    }

    /* AMGX check status */
    AMGX_SOLVE_STATUS status;
    AMGX_solver_get_status(m_Solver, &status);

    if (status != AMGX_SOLVE_SUCCESS) {
        opserr << "WARNING: Solver failed with status " << status << " -- AmgXGenLinSolver::solve" << endln;
        if (status != AMGX_SOLVE_FAILED) {
            reportAmgXSolveStats(this->getNumIterations(), initialResidualNorm, finalResidualNorm);
        }
        return -1;
    }

    if (m_verbose) {
        opserr << "AmgXGenLinSolver: Solve successful" << endln;
        reportAmgXSolveStats(this->getNumIterations(), initialResidualNorm, finalResidualNorm);
    }
    return 0;
}

int AmgXGenLinSolver::setSize()
{
    return 0;
}

int AmgXGenLinSolver::setLinearSOE(AmgXGenLinSOE &theLinearSOE)
{
    theSOE = &theLinearSOE;
    return 0;
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
    int numIterations;
    if (m_Solver == nullptr) {
        opserr << "WARNING: AmgXGenLinSolver::getNumIterations: Solver not initialized" << endln;
        return 0;
    }
    AMGX_solver_get_iterations_number(m_Solver, &numIterations);
    return numIterations;
}

double AmgXGenLinSolver::getResidualNorm() {
    if (m_Solver == nullptr || m_Matrix == nullptr || m_RHS == nullptr || m_Solution == nullptr || theSOE == nullptr) {
        opserr << "WARNING: AmgXGenLinSolver::getResidualNorm: Solver, matrix, RHS, solution vector or LinearSOE not initialized" << endln;
        return 0.0;
    }
    double finalResidualNorm = 0.0;
    void* residualComponent = calloc(theSOE->m_BlockSize, sizeof(double));
    AMGX_solver_calculate_residual_norm(m_Solver, m_Matrix, m_RHS, m_Solution, residualComponent);
    for (int i = 0; i < theSOE->m_BlockSize; i++) {
        finalResidualNorm += ((double*)residualComponent)[i] * ((double*)residualComponent)[i];
    }
    free(residualComponent);
    return std::sqrt(finalResidualNorm);
}
