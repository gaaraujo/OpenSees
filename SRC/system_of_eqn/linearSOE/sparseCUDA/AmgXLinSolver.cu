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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCUDA/AmgXLinSolver.cpp,v $
                                                                        
                                                                        
// Written: gaaraujo 
// Created: 08/2025
//
// Description: This file contains the class definition for 
// AmgXLinSolver. It solves the CudaGenBcsrLinSOE object by calling
// AMGX routines.
//

// OpenSees includes
#include <classTags.h>
#include <OPS_Globals.h>

// Solve core classes
#include <CudaGenBcsrLinSOE.h>
#include <AmgXLinSolver.h>

// for parsing command line arguments
#ifdef _AMGX
#include <elementAPI.h>
#include <FileStream.h>
#include <unordered_map>
#include <functional>
#include <memory>
#include "ParameterUtils.h"
#endif // _AMGX

// C++ includes
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

#ifdef _AMGX
// Static member initialization
bool AmgXLinSolver::m_AmgXInitialized = false;

int AmgXLinSolver::m_ActiveSolverInstances = 0;

AMGX_resources_handle AmgXLinSolver::m_Resources = nullptr;

OPS_Stream* AmgXLinSolver::m_CallbackStream = (OPS_Stream*)&opserr;

// Static member function implementations
void AmgXLinSolver::setCallbackStream(OPS_Stream* output) {
    m_CallbackStream = output;
}

OPS_Stream* AmgXLinSolver::getCallbackStream() {
    return m_CallbackStream;
}

/* AMGX callbacks */
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
void AmgXCallback(const char* msg, int length) {
    if (msg && length > 0) {
        OPS_Stream* callbackStream = AmgXLinSolver::getCallbackStream();
        if (callbackStream) {
            *callbackStream << msg;
            *callbackStream << endln;
        }
    }
}
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // _AMGX

/* Anonymous namespace for helper functions */
namespace {
    [[maybe_unused]] void reportAmgXSolveStats(int numIterations, double initialResidualNorm, double finalResidualNorm) {
        opserr << "INFO: AmgXLinSolver::reportAmgXSolveStats() - " << endln;
        opserr << "Number of iterations = " << numIterations << endln;
        opserr << "Initial residual norm = " << initialResidualNorm << endln;
        opserr << "Final residual norm = " << finalResidualNorm << endln;
        if (initialResidualNorm > 0.0) {
            opserr << "Final to initial residual norm ratio = " << finalResidualNorm / initialResidualNorm << endln;
        } else {
            opserr << "Final to initial residual norm ratio = NaN" << endln;
        }
    }

    [[maybe_unused]] std::string getDefaultConfigOptions(std::string solver = "PCG", 
                                        std::string preconditioner = "JACOBI_L1", 
                                        std::string smoother = "JACOBI_L1",
                                        int max_iters = 1000,
                                        double abs_tolerance = 1e-12,
                                        double rel_tolerance = 1e-6,
                                        int monitor_residual = 1) 
    {
        if (solver != "PCG" && solver != "FGMRES" && solver != "PCGF") {
            opserr << "WARNING: AmgXLinSolver::getDefaultConfigOptions() - "
                   << "Tried to create default config with solver = " << solver.c_str() << ".";
            opserr << "Default config constructor only works with solver = PCG, PCGF, or FGMRES.";
            opserr << "To use a different solver, pass a config file or config string to the constructor using: " << endln;
            opserr << "system AmgX <-configFile CONFIG_FILE> <-configOptions CONFIG_OPTIONS>" << endln;
            opserr << "For more information, see the AMGX documentation." << endln;
            return "";
        }
        if (preconditioner != "AMG" && preconditioner != "BLOCK_JACOBI" && preconditioner != "JACOBI_L1") {
            opserr << "WARNING: AmgXLinSolver::getDefaultConfigOptions() - "
                   << "Tried to create default config with preconditioner = " << preconditioner.c_str() << ".";
            opserr << "Default config constructor only works with preconditioner = AMG, BLOCK_JACOBI, or JACOBI_L1.";
            opserr << "To use a different preconditioner, pass a config file or config string to the constructor using: " << endln;
            opserr << "system AmgX <-configFile CONFIG_FILE> <-configOptions CONFIG_OPTIONS>" << endln;
            opserr << "For more information, see the AMGX documentation." << endln;
            return "";
        }
        if (preconditioner == "AMG" && (smoother != "JACOBI_L1" && smoother != "BLOCK_JACOBI")) {
            opserr << "WARNING: AmgXLinSolver::getDefaultConfigOptions() - "
                   << "Tried to create default AMG preconditioner with smoother = " << smoother.c_str() << ".";
            opserr << "Default AMG preconditioner only works with smoother = JACOBI_L1 or BLOCK_JACOBI.";
            opserr << "To use a different smoother, pass a config file or config string to the constructor using: " << endln;
            opserr << "system AmgX <-configFile CONFIG_FILE> <-configOptions CONFIG_OPTIONS>" << endln;
            opserr << "For more information, see the AMGX documentation." << endln;
            return "";
        }

        if (max_iters <= 0) {
            opserr << "WARNING: AmgXLinSolver::getDefaultConfigOptions() - "
                   << "Tried to create default config with maxIters = " << max_iters << ".";
            opserr << "Expected maxIters > 0.";
            return "";
        }
        if (abs_tolerance < 0.0) {
            opserr << "WARNING: AmgXLinSolver::getDefaultConfigOptions() - "
                   << "Tried to create default config with absTolerance = " << abs_tolerance << ".";
            opserr << "Expected absTolerance >= 0.0.";
            return "";
        }
        if (rel_tolerance < 0.0) {
            opserr << "WARNING: AmgXLinSolver::getDefaultConfigOptions() - "
                   << "Tried to create default config with relTolerance = " << rel_tolerance << ".";
            opserr << "Expected relTolerance >= 0.0.";
            return "";
        }
        if (monitor_residual != 0 && monitor_residual != 1) {
            opserr << "WARNING: AmgXLinSolver: Tried to create default config with monitorResidual = " << monitor_residual << ".";
            opserr << "Expected monitorResidual = 0 or 1.";
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

AmgXLinSolver::AmgXLinSolver(
    const std::string solver, const std::string preconditioner, const std::string smoother, 
    int max_iters, double abs_tolerance, double rel_tolerance, int monitor_residual, 
    bool verbose)
    :CudaGenBcsrLinSolver(SOLVER_TAGS_AmgXLinSolver, CudaPrecision::dDDI),  // Simple constructor always uses double
    m_verbose(verbose)
{
    #ifdef _AMGX
    std::string configOptions = getDefaultConfigOptions(
        solver, preconditioner, smoother, max_iters, 
        abs_tolerance, rel_tolerance, monitor_residual);
    const char* nullConfigFile = nullptr;
    // Simple constructor always uses double precision
    const char* precisionStr = "dDDI";
    init(nullConfigFile, configOptions.c_str(), precisionStr);
    #endif // _AMGX
}

AmgXLinSolver::AmgXLinSolver( 
    const char *configFile, const char *configOptions, 
    CudaPrecision precision, bool verbose, OPS_Stream* callbackStream)
    :CudaGenBcsrLinSolver(SOLVER_TAGS_AmgXLinSolver, precision), 
    m_verbose(verbose)
{
    #ifdef _AMGX
    // AmgX currently only supports uniform precision (dDDI or dFFI)
    if (!isUniformPrecision(precision)) {
        opserr << "ERROR: AmgXLinSolver::AmgXLinSolver() - "
               << "Precision " << cudaPrecisionToString(precision) << " is not supported by AmgX. "
               << "AmgX only supports uniform precision: dDDI (double) and dFFI (float)." << endln;
        throw std::invalid_argument("AmgX does not support mixed precision");
    }
    
    // Convert enum to string for AMGX library (which expects string format)
    const char* precisionStr = cudaPrecisionToString(precision);
    init(configFile, configOptions, precisionStr, callbackStream);
    #endif // _AMGX
}


void AmgXLinSolver::init(const char *configFile, const char *configOptions, 
                         const char *precision, OPS_Stream* callbackStream) 
{
    #ifdef _AMGX
    /* Initialize AMGX library - only done once across all instances */
    if (!m_AmgXInitialized) {
        AMGX_SAFE_CALL(AMGX_initialize());
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
        m_AmgXInitialized = true;
    }
    AmgXLinSolver::setCallbackStream(callbackStream);
    AMGX_SAFE_CALL(AMGX_register_print_callback(AmgXCallback));

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

    /* Enforce some parameters for certain solver scopes */
    std::string additionalOptions = "config_version=2,";
    const std::vector<std::string> scopes = {
        "default", "main", "default_sub_solver"
    };
    std::vector<std::string> params = {
        "use_scalar_norm=1" // Use scalar norm for the L2 norm
    };
    /* Verbosity parameters */
    if (m_verbose) {
        params.push_back("obtain_timings=1"); // prints solver timings to the console
        // Other parameters could be added here. For example:
        // "params.push_back("print_config=1");"
        // "params.push_back("print_solve_stats=1");"
        // "params.push_back("print_grid_stats=1");"
        for (const std::string& scope : scopes) {
            for (const std::string& param : params) {
                additionalOptions += scope + ":" + param + ",";
            }
        }
    }
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&m_Config, additionalOptions.c_str()));

    /* Switch on internal error handling 
     * (no need to use AMGX_SAFE_CALL after this point) */
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&m_Config, "exception_handling=1"));

    /* Create resources (single-GPU, single-threaded only).
     * AMGX resource objects are not thread-safe, so we ensure only one is created.
     * References:
     * https://github.com/NVIDIA/AMGX/issues/109#issuecomment-674046246
     * https://github.com/barbagroup/AmgXWrapper/blob/ab70b5e7bc1248875040814edef7474ad18349af/src/AmgXSolver.hpp#L335
     */

    if (m_ActiveSolverInstances == 0) {
        AMGX_resources_create_simple(&m_Resources, m_Config);
    }

    /* Set AmgX precision mode */
    if (strcmp(precision, "dDDI") == 0) {
        m_Mode = AMGX_mode_dDDI; // device, double matrix, double vector, int index
    } else if (strcmp(precision, "dFFI") == 0) {
        m_Mode = AMGX_mode_dFFI; // device, float matrix, float vector, int index
    } else {
        opserr << "WARNING: AmgXLinSolver::init() - "
               << "Invalid precision '" << precision << "'. Only dDDI and dFFI are supported. "
               << "Setting precision to dDDI" << endln;
        m_Mode = AMGX_mode_dDDI; // device, double matrix, double vector, int index
    }

    /* Create solver, matrix, rhs and solution vectors */
    AMGX_solver_create(&m_Solver, m_Resources, m_Mode, m_Config);
    AMGX_matrix_create(&m_Matrix, m_Resources, m_Mode);
    AMGX_vector_create(&m_RHS, m_Resources, m_Mode);
    AMGX_vector_create(&m_Solution, m_Resources, m_Mode);

    m_ActiveSolverInstances++;
    #endif // _AMGX

    return;
}

AmgXLinSolver::~AmgXLinSolver() {
    #ifdef _AMGX
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
    #endif // _AMGX
    return;
}

int AmgXLinSolver::setLinearSOE(CudaGenBcsrLinSOE &theSOE) {
    #ifdef _AMGX
    // Check precision match between solver and SOE
    if (theSOE.getPrecision() != this->getPrecision()) {
        opserr << "WARNING: AmgXLinSolver::setLinearSOE() - "
               << "precision mismatch: SOE is " << cudaPrecisionToString(theSOE.getPrecision())
               << ", solver is " << cudaPrecisionToString(this->getPrecision()) << endln;
        return -1;
    }
    
    return this->CudaGenBcsrLinSolver::setLinearSOE(theSOE);
    #endif // _AMGX

    return 0;
}

int AmgXLinSolver::solve() {
    #ifdef _AMGX
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "WARNING: AmgXLinSolver::solve() - "
               << "LinearSOE not set" << endln;
        return -1;
    }

    // Extract info from the SOE
    CudaGenBcsrLinSOE::MatrixStatus matrixStatus = theSOE->getMatrixStatus();
    int numRowBlocks = theSOE->getNumRowBlocks();
    int numNonZeroBlocks = theSOE->getNumNonZeroBlocks();
    int numNonZeroValues = theSOE->getNumNonZeroValues();
    int blockSize = theSOE->getBlockSize();
    int* rowPtrs = theSOE->getDeviceRowPtrs();
    int* colIndices = theSOE->getDeviceColIndices();
    
    // Check if device pointers are valid
    if (!rowPtrs) {
        opserr << "ERROR: AmgXLinSolver::solve() - getDeviceRowPtrs() returned nullptr" << endln;
        return -1;
    }
    if (!colIndices) {
        opserr << "ERROR: AmgXLinSolver::solve() - getDeviceColIndices() returned nullptr" << endln;
        return -1;
    }
    void* values = theSOE->getDeviceAValues();
    void* x = theSOE->getDeviceX();
    void* b = theSOE->getDeviceB();
    
    if (!values) {
        opserr << "ERROR: AmgXLinSolver::solve() - getDeviceAValues() returned nullptr" << endln;
        return -1;
    }
    if (!x) {
        opserr << "ERROR: AmgXLinSolver::solve() - getDeviceX() returned nullptr" << endln;
        return -1;
    }
    if (!b) {
        opserr << "ERROR: AmgXLinSolver::solve() - getDeviceB() returned nullptr" << endln;
        return -1;
    }

    // Upload the matrix data to the GPU
    if (matrixStatus == CudaGenBcsrLinSOE::MatrixStatus::STRUCTURE_CHANGED) {
        AMGX_matrix_upload_all(
            m_Matrix, numRowBlocks, numNonZeroBlocks, 
            blockSize, blockSize, rowPtrs,
            colIndices, values, nullptr
        );
        // Setup the solver with the new matrix structure
        AMGX_solver_setup(m_Solver, m_Matrix);
    } else if (matrixStatus == CudaGenBcsrLinSOE::MatrixStatus::COEFFICIENTS_CHANGED) {
        AMGX_matrix_replace_coefficients(
            m_Matrix, numRowBlocks, numNonZeroBlocks,
            values, nullptr
        );
        // Setup the solver with the updated matrix coefficients
        AMGX_solver_setup(m_Solver, m_Matrix);
    }

    // Upload the rhs data to the GPU
    AMGX_vector_upload(m_RHS, numRowBlocks, blockSize, b);

    /* Solve with 0-vector initial guess */
    AMGX_vector_set_zero(m_Solution, numRowBlocks, blockSize);

    // Solve the system of equations
    double initialResidualNorm = this->getResidualNorm();
    AMGX_solver_solve_with_0_initial_guess(m_Solver, m_RHS, m_Solution);
    double finalResidualNorm = this->getResidualNorm();

    /* Download the solution vector from the GPU */
    AMGX_vector_download(m_Solution, x);

    /* AMGX check status */
    AMGX_SOLVE_STATUS status;
    AMGX_solver_get_status(m_Solver, &status);

    if (status != AMGX_SOLVE_SUCCESS) {
        opserr << "WARNING: AmgXLinSolver::solve() - "
               << "Solver failed with status " << status << endln;
        if (status != AMGX_SOLVE_FAILED) {
            reportAmgXSolveStats(this->getNumIterations(), initialResidualNorm, finalResidualNorm);
        }
        return -1;
    }

    if (m_verbose) {
        opserr << "INFO: AmgXLinSolver::solve() - "
               << "Solve successful" << endln;
        reportAmgXSolveStats(this->getNumIterations(), initialResidualNorm, finalResidualNorm);
    }
    #endif // _AMGX

    return 0;
}

int AmgXLinSolver::setSize()
{
    return 0;
}

int AmgXLinSolver::getNumIterations() {
    int numIterations = 0;
    #ifdef _AMGX
    if (m_Solver == nullptr) {
        opserr << "WARNING: AmgXLinSolver::getNumIterations() - "
               << "Solver not initialized" << endln;
        return 0;
    }
    AMGX_solver_get_iterations_number(m_Solver, &numIterations);
    #endif // _AMGX
    return numIterations;
}

double AmgXLinSolver::getResidualNorm() {
    double finalResidualNorm = 0.0;
    #ifdef _AMGX
    if (m_Solver == nullptr || m_Matrix == nullptr || m_RHS == nullptr || m_Solution == nullptr) {
        opserr << "WARNING: AmgXLinSolver::getResidualNorm() - "
               << "Solver, matrix, RHS, or solution AMGX handles not initialized" << endln;
        return 0.0;
    }
    
    CudaGenBcsrLinSOE *theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    int blockSize = theSOE->getBlockSize();
    std::vector<double> residualComponent(blockSize, 0.0);
    AMGX_solver_calculate_residual_norm(m_Solver, m_Matrix, m_RHS, m_Solution, (void*)residualComponent.data());
    for (double residual : residualComponent) {
        finalResidualNorm += residual * residual;
    }
    #endif // _AMGX
    return std::sqrt(finalResidualNorm);
}

// OpenSees API for parsing command line arguments
#ifdef _AMGX

struct AmgXGeneralConfig;
struct AmgXSimpleConfig;
CudaGenBcsrLinSolver* createAmgXSolver(const AmgXGeneralConfig& config);
CudaGenBcsrLinSolver* createAmgXSolver(const AmgXSimpleConfig& config);

// Configuration structures for input parsing
struct AmgXGeneralConfig {
    std::string configFile;
    std::string configOptions;
    std::string precision = "dDDI";
    bool verbose = false;
    OPS_Stream* callbackStream = (OPS_Stream*)&opserr;
    std::string callbackFilename;  // Store the filename for FileStream creation
    int blockSize = 1;
    bool paddingEnabled = true;
};

struct AmgXSimpleConfig {
    std::string solver = "PCG";
    std::string preconditioner = "JACOBI_L1";
    std::string smoother = "JACOBI_L1";
    int maxIters = 1000;
    double absTolerance = 1e-12;
    double relTolerance = 1e-6;
    int monitorResidual = 1;
    std::string precision = "dDDI";
    bool verbose = false;
    int blockSize = 1;
    bool paddingEnabled = true;
};

// Parameter parser class
class AmgXParameterParser {
private:
    // Defines valid parameter names
    // Option 1: users can pass a config json file or a config string per AMGX Reference Manual
    static const std::unordered_map<std::string, std::function<void(AmgXGeneralConfig&)>> generalConfigParsers;
    // Option 2: users can pass specify a basic solver configuration using command line arguments
    static const std::unordered_map<std::string, std::function<void(AmgXSimpleConfig&)>> simpleConfigParsers;
    
public:
    // Option 1: users can pass a config json file or a config string per AMGX Reference Manual
    static bool parseGeneralConfigParameters(AmgXGeneralConfig& config);

    // Option 2: users can pass specify a basic solver configuration using command line arguments
    static bool parseSimpleConfigParameters(AmgXSimpleConfig& config);
    
    // Print usage information for both options
    static void printGeneralUsageInfo();
    static void printSimpleUsageInfo();
};

// Option 1: users can pass a config json file or a config string per AMGX Reference Manual
const std::unordered_map<std::string, std::function<void(AmgXGeneralConfig&)>> 
AmgXParameterParser::generalConfigParsers = {
    {"configFile", [](AmgXGeneralConfig& config) { 
        const char* value = OPS_GetString();
        if (value) config.configFile = value;
    }},
    {"configOptions", [](AmgXGeneralConfig& config) { 
        const char* value = OPS_GetString();
        if (value) config.configOptions = value;
    }},
    {"precision", [](AmgXGeneralConfig& config) { 
        const char* value = OPS_GetString();
        if (value && (strcmp(value, "dDDI") == 0 || strcmp(value, "dFFI") == 0)) {
            config.precision = value;
        } else if (value) {
            throw std::invalid_argument("Invalid precision. Only dDDI and dFFI are supported.");
        }
    }},
    {"mode", [](AmgXGeneralConfig& config) {  // Alias for precision (backward compatibility)
        const char* value = OPS_GetString();
        if (value && (strcmp(value, "dDDI") == 0 || strcmp(value, "dFFI") == 0)) {
            config.precision = value;
        } else if (value) {
            throw std::invalid_argument("Invalid mode/precision. Only dDDI and dFFI are supported.");
        }
    }},
    {"verbose", [](AmgXGeneralConfig& config) { 
        int numData = 1;
        int flag = 0;
        if (OPS_GetIntInput(&numData, &flag) == 0) {
            if (flag != 0 && flag != 1) throw std::invalid_argument("verbose must be 0 or 1");
            config.verbose = (flag == 1);
        }
    }},
    {"blockSize", [](AmgXGeneralConfig& config) { 
        int numData = 1;
        int bs = 0;
        if (OPS_GetIntInput(&numData, &bs) == 0) {
            if (bs < 0) throw std::invalid_argument("blockSize cannot be negative");
            config.blockSize = bs;
        }
    }},
    {"callback", [](AmgXGeneralConfig& config) { 
        const char* value = OPS_GetString();
        if (value) {
            if (strcmp(value, "default") == 0 || strcmp(value, "opserr") == 0) {
                config.callbackStream = (OPS_Stream*)&opserr;
            } else if (strcmp(value, "none") == 0 || strcmp(value, "null") == 0) {
                config.callbackStream = nullptr;
            } else {
                // Store the filename for later FileStream creation
                // We'll create the FileStream when we actually need it
                config.callbackFilename = value; // Store the filename
                config.callbackStream = nullptr; // Will be set to FileStream later
            }
        }
    }}
};

// Option 2: users can pass specify a basic solver configuration using command line arguments
const std::unordered_map<std::string, std::function<void(AmgXSimpleConfig&)>> 
AmgXParameterParser::simpleConfigParsers = {
    {"solver", [](AmgXSimpleConfig& config) { 
        const char* value = OPS_GetString();
        if (value) config.solver = value;
    }},
    {"preconditioner", [](AmgXSimpleConfig& config) { 
        const char* value = OPS_GetString();
        if (value) config.preconditioner = value;
    }},
    {"smoother", [](AmgXSimpleConfig& config) { 
        const char* value = OPS_GetString();
        if (value) config.smoother = value;
    }},
    {"maxIters", [](AmgXSimpleConfig& config) { 
        int numData = 1;
        int val = 0;
        if (OPS_GetIntInput(&numData, &val) == 0) {
            if (val <= 0) throw std::invalid_argument("maxIters must be positive");
            config.maxIters = val;
        }
    }},
    {"absTolerance", [](AmgXSimpleConfig& config) { 
        int numData = 1;
        double val = 0.0;
        if (OPS_GetDoubleInput(&numData, &val) == 0) {
            if (val <= 0.0) throw std::invalid_argument("absTolerance must be positive");
            config.absTolerance = val;
        }
    }},
    {"relTolerance", [](AmgXSimpleConfig& config) { 
        int numData = 1;
        double val = 0.0;
        if (OPS_GetDoubleInput(&numData, &val) == 0) {
            if (val <= 0.0) throw std::invalid_argument("relTolerance must be positive");
            config.relTolerance = val;
        }
    }},
    {"monitorResidual", [](AmgXSimpleConfig& config) { 
        int numData = 1;
        int val = 0;
        if (OPS_GetIntInput(&numData, &val) == 0) {
            if (val != 0 && val != 1) throw std::invalid_argument("monitorResidual must be 0 or 1");
            config.monitorResidual = val;
        }
    }},
    {"precision", [](AmgXSimpleConfig& config) { 
        const char* value = OPS_GetString();
        if (value && (strcmp(value, "dDDI") == 0 || strcmp(value, "dFFI") == 0)) {
            config.precision = value;
        } else if (value) {
            throw std::invalid_argument("Invalid precision. Only dDDI and dFFI are supported.");
        }
    }},
    {"mode", [](AmgXSimpleConfig& config) {  // Alias for precision (backward compatibility)
        const char* value = OPS_GetString();
        if (value && (strcmp(value, "dDDI") == 0 || strcmp(value, "dFFI") == 0)) {
            config.precision = value;
        } else if (value) {
            throw std::invalid_argument("Invalid mode/precision. Only dDDI and dFFI are supported.");
        }
    }},
    {"verbose", [](AmgXSimpleConfig& config) { 
        int numData = 1;
        int flag = 0;
        if (OPS_GetIntInput(&numData, &flag) == 0) {
            if (flag != 0 && flag != 1) throw std::invalid_argument("verbose must be 0 or 1");
            config.verbose = (flag == 1);
        }
    }},
    {"blockSize", [](AmgXSimpleConfig& config) { 
        int numData = 1;
        int bs = 0;
        if (OPS_GetIntInput(&numData, &bs) == 0) {
            if (bs < 0) throw std::invalid_argument("blockSize cannot be negative");
            config.blockSize = bs;
        }
    }}
};

// Option 1: users can pass a config json file or a config string per AMGX Reference Manual
bool AmgXParameterParser::parseGeneralConfigParameters(AmgXGeneralConfig& config) {
    try {
        while (OPS_GetNumRemainingInputArgs() > 0) {
            const char* key = OPS_GetString();
            if (!key) {
                opserr << "WARNING: AmgXParameterParser::parseGeneralConfigParameters() - "
                       << "Invalid input argument" << endln;
                return false;
            }
            
            // Use helper function to handle both with and without dashes
            auto it = ParameterUtils::findParameter(key, generalConfigParsers);
            if (it != generalConfigParsers.end()) {
                it->second(config);
            } else {
                continue;
            }
        }
        return true;
    } catch (const std::exception& e) {
        opserr << "WARNING: AmgXParameterParser::parseGeneralConfigParameters() - "
               << e.what() << endln;
        return false;
    }
}

// Option 2: users can pass specify a basic solver configuration using command line arguments
bool AmgXParameterParser::parseSimpleConfigParameters(AmgXSimpleConfig& config) {
    try {
        while (OPS_GetNumRemainingInputArgs() > 0) {
            const char* key = OPS_GetString();
            if (!key) {
                opserr << "WARNING: AmgXParameterParser::parseSimpleConfigParameters() - "
                       << "Invalid input argument" << endln;
                return false;
            }
            
            // Use helper function to handle both with and without dashes
            auto it = ParameterUtils::findParameter(key, simpleConfigParsers);
            if (it != simpleConfigParsers.end()) {
                it->second(config);
            } else {
                continue;
            }
        }
        return true;
    } catch (const std::exception& e) {
        opserr << "WARNING: AmgXParameterParser::parseSimpleConfigParameters() - "
               << e.what() << endln;
        return false;
    }
}

// Print usage information for both options
void AmgXParameterParser::printGeneralUsageInfo() {
    opserr << "AmgXParameterParser::printGeneralUsageInfo() - " << endln;
    opserr << "Usage: system AmgX [options]" << endln;
    opserr << "Options:" << endln;
    opserr << "  -configFile <path>              AmgX JSON config file" << endln;
    opserr << "  -configOptions <string>         AmgX config string" << endln;
    opserr << "  -precision <dDDI|dFFI>          Precision mode (default: dDDI)" << endln;
    opserr << "  -blockSize <int>                Block size (default: 1)" << endln;
    opserr << "  -verbose <0|1>                  Enable verbose output (default: 0)" << endln;
    opserr << "  -callback <stream|file>         Callback output (default|opserr|none|filename)" << endln;
}

void AmgXParameterParser::printSimpleUsageInfo() {
    opserr << "AmgXParameterParser::printSimpleUsageInfo() - " << endln;
    opserr << "Usage: system AmgX [options]" << endln;
    opserr << "Options:" << endln;
    opserr << "  -solver <name>                  Solver type (e.g., PCG, GMRES, BiCGSTAB)" << endln;
    opserr << "  -preconditioner <name>          Preconditioner (e.g., JACOBI_L1, AMG, BLOCK_JACOBI)" << endln;
    opserr << "  -smoother <name>                Smoother (e.g., JACOBI_L1, BLOCK_JACOBI)" << endln;
    opserr << "  -maxIters <int>                 Max iterations (default: 1000)" << endln;
    opserr << "  -absTolerance <double>          Absolute tolerance (default: 1e-12)" << endln;
    opserr << "  -relTolerance <double>          Relative tolerance (default: 1e-6)" << endln;
    opserr << "  -monitorResidual <0|1>          Monitor residual (default: 1)" << endln;
    opserr << "  -precision <dDDI|dFFI>          Precision mode (default: dDDI)" << endln;
    opserr << "  -blockSize <int>                Block size (default: 1)" << endln;
    opserr << "  -verbose <0|1>                  Enable verbose output (default: 0)" << endln;
}

// Factory functions for creating solvers and SOEs
CudaGenBcsrLinSolver* createAmgXSolver(const AmgXGeneralConfig& config) {
    // Convert string precision to enum
    CudaPrecision precision;
    if (!cudaPrecisionFromString(config.precision.c_str(), precision)) {
        opserr << "WARNING: createAmgXSolver() - "
               << "Invalid precision '" << config.precision.c_str() << "', defaulting to dDDI" << endln;
        precision = CudaPrecision::dDDI;
    }
    
    try {
        return new AmgXLinSolver(
            config.configFile.c_str(), 
            config.configOptions.c_str(), 
            precision,
            config.verbose, 
            config.callbackStream
        );
    } catch (const std::exception& e) {
        opserr << "ERROR: createAmgXSolver() - " << e.what() << endln;
        return nullptr;
    }
}

CudaGenBcsrLinSolver* createAmgXSolver(const AmgXSimpleConfig& config) {
    return new AmgXLinSolver(
        config.solver.c_str(), 
        config.preconditioner.c_str(), 
        config.smoother.c_str(), 
        config.maxIters, 
        config.absTolerance, 
        config.relTolerance, 
        config.monitorResidual, 
        config.verbose
    );
}

// Helper struct to return both solver and SOE config from parsing
struct AmgXSolverAndConfig {
    CudaGenBcsrLinSolver* solver;
    int blockSize;
    std::string precision;
    bool paddingEnabled;
    bool verbose;
};

// Internal function for OPS_AmgXLinSolver to parse and create solver with full config
static AmgXSolverAndConfig createAmgXSolverAndConfigFromParser() {
    AmgXSolverAndConfig result = {nullptr, 1, "dDDI", false, false};
    
    // Handle case with no arguments - use default configuration
    if (OPS_GetNumRemainingInputArgs() == 0) {
        AmgXSimpleConfig simpleConfig;
        result.solver = createAmgXSolver(simpleConfig);
        result.blockSize = simpleConfig.blockSize;
        result.precision = simpleConfig.precision;
        result.paddingEnabled = simpleConfig.paddingEnabled;
        result.verbose = simpleConfig.verbose;
        return result;
    }
    
    // Check argument count
    if (OPS_GetNumRemainingInputArgs() % 2 != 0) {
        opserr << "WARNING: OPS_AmgXLinSolver() - "
               << "Incorrect number of arguments for AmgXLinSolver. " << endln;
        AmgXParameterParser::printGeneralUsageInfo();
        opserr << "Alternatively, use the simple constructor: " << endln;
        AmgXParameterParser::printSimpleUsageInfo();
        return result;
    }

    // Try to parse as general config parameters first
    AmgXGeneralConfig generalConfig;
    generalConfig.callbackStream = (OPS_Stream*)&opserr;
    
    int originalArgCount = OPS_GetNumRemainingInputArgs();
    
    if (AmgXParameterParser::parseGeneralConfigParameters(generalConfig)) {
        if (!generalConfig.configFile.empty() || !generalConfig.configOptions.empty()) {
            // Handle file callback if specified
            if (!generalConfig.callbackFilename.empty()) {
                static FileStream callbackFile;
                if (callbackFile.setFile(generalConfig.callbackFilename.c_str()) != 0) {
                    opserr << "WARNING: OPS_AmgXLinSolver() - "
                           << "Failed to open callback file: " 
                           << generalConfig.callbackFilename.c_str() << endln;
                    return result;
                }
                generalConfig.callbackStream = &callbackFile;
            }
            
            result.solver = createAmgXSolver(generalConfig);
            result.blockSize = generalConfig.blockSize;
            result.precision = generalConfig.precision;
            result.paddingEnabled = generalConfig.paddingEnabled;
            result.verbose = generalConfig.verbose;
            return result;
        }
    }
    
    // Reset arguments and try simple config
    OPS_ResetCurrentInputArg(-originalArgCount);
    
    AmgXSimpleConfig simpleConfig;
    if (AmgXParameterParser::parseSimpleConfigParameters(simpleConfig)) {
        // Validate block size for JACOBI_L1
        if ((simpleConfig.preconditioner == "JACOBI_L1" || 
             simpleConfig.smoother == "JACOBI_L1") && simpleConfig.blockSize != 1) {
            opserr << "WARNING: OPS_AmgXLinSolver() - "
                   << "JACOBI_L1 smoother/preconditioner only supports blockSize = 1. "
                   << "Setting blockSize to 1..." << endln;
            simpleConfig.blockSize = 1;
        }
        
        result.solver = createAmgXSolver(simpleConfig);
        result.blockSize = simpleConfig.blockSize;
        result.precision = simpleConfig.precision;
        result.paddingEnabled = simpleConfig.paddingEnabled;
        result.verbose = simpleConfig.verbose;
        return result;
    }
    
    // If we get here, parsing failed
    opserr << "WARNING: OPS_AmgXLinSolver() - "
           << "Failed to parse AmgXLinSolver parameters" << endln;
    AmgXParameterParser::printGeneralUsageInfo();
    opserr << "Alternatively, use the simple constructor: " << endln;
    AmgXParameterParser::printSimpleUsageInfo();
    return result;
}

void* OPS_AmgXLinSolver()
{
    // Use existing parser-based factory; Python bindings can use OPS_ExpandDictArgs
    // Expand dict to CLI args if present ({"key": val} -> "-key", val)
    if (OPS_GetNumRemainingInputArgs() == 1) {
        (void)OPS_ExpandDictArgs();
    }
    AmgXSolverAndConfig result = createAmgXSolverAndConfigFromParser();
    
    if (result.solver == nullptr) {
        return nullptr;
    }
    
    // Create and return SOE based on precision (only uniform modes reach here)
    CudaPrecision precision = result.solver->getPrecision();
    switch(precision) {
        case CudaPrecision::dDDI:
            return CudaGenBcsrLinSOE::createDouble(
                *result.solver, result.blockSize, 
                result.paddingEnabled, result.verbose
            );
        case CudaPrecision::dFFI:
            return CudaGenBcsrLinSOE::createFloat(
                *result.solver, result.blockSize, 
                result.paddingEnabled, result.verbose
            );
        default:
            // Should never reach here - constructor rejects mixed precision
            opserr << "ERROR: OPS_AmgXLinSolver() - Unexpected precision mode" << endln;
            delete result.solver;
            return nullptr;
    }
}
#else // _AMGX
void* OPS_AmgXLinSolver() {
    opserr << "WARNING: OPS_AmgXLinSolver() - "
           << "AMGX not available" << endln;
    return nullptr;
}
#endif // _AMGX
