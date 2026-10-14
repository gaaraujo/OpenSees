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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCUDA/CudaDirectSparseSolver.cpp,v $
                                                                        
                                                                        
// Written: gaaraujo 
// Created: 10/2025
//
// Description: This file contains the class definition for 
// CudaDirectSparseSolver. It solves the CudaBcsrLinSOE object by calling
// CuDSS routines.
//

// OpenSees includes
#include <classTags.h>
#include <OPS_Globals.h>

// Solve core classes
#include <CudaBcsrLinSOE.h>
#include <CudaDirectSparseSolver.h>
#include "ParameterUtils.h"

// CUDA utilities
#include "CudaUtils.h"

// for parsing command line arguments
#include <elementAPI.h>
#include <FileStream.h>
#include <unordered_map>

// C++ includes
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace CudaUtils;

#ifdef _WIN32
static void printCuDSSPathHint()
{
    opserr << "  Add cuDSS and CUDA bin directories to PATH, for example:" << endln;
    const char* cudssDir = std::getenv("CUDSS_DIR");
    if (cudssDir != nullptr && cudssDir[0] != '\0') {
        opserr << "    " << cudssDir << "\\bin\\12" << endln;
    } else {
        opserr << "    <cuDSS install>\\bin\\12"
               << "  (e.g. C:\\Program Files\\NVIDIA cuDSS\\v0.8\\bin\\12)" << endln;
    }
    const char* cudaRoot = std::getenv("CUDAToolkit_ROOT");
    if (cudaRoot != nullptr && cudaRoot[0] != '\0') {
        opserr << "    " << cudaRoot << "\\bin" << endln;
    } else {
        opserr << "    <CUDA toolkit>\\bin"
               << "  (e.g. C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin)"
               << endln;
    }
    opserr << "  Set CUDSS_DIR / CUDAToolkit_ROOT if installs are non-default." << endln;
}
#endif

// Fail fast at "system CuDSS" time instead of silently crashing on the first analyze().
static bool verifyCuDSSRuntime()
{
#ifdef _WIN32
    HMODULE hCudss = LoadLibraryA("cudss64_0.dll");
    if (hCudss == nullptr) {
        hCudss = LoadLibraryA("cudss64.dll");
    }
    if (hCudss == nullptr) {
        const DWORD err = GetLastError();
        opserr << "ERROR: verifyCuDSSRuntime() - cannot load cuDSS runtime DLL (GetLastError="
               << err << ")" << endln;
        printCuDSSPathHint();
        return false;
    }
#endif

    int deviceCount = 0;
    const cudaError_t cudaErr = cudaGetDeviceCount(&deviceCount);
    if (cudaErr != cudaSuccess) {
        opserr << "ERROR: verifyCuDSSRuntime() - cudaGetDeviceCount failed: "
               << cudaGetErrorString(cudaErr) << endln;
        opserr << "  Check CUDA driver/runtime and PATH to CUDAToolkit bin." << endln;
        return false;
    }
    if (deviceCount <= 0) {
        opserr << "ERROR: verifyCuDSSRuntime() - no CUDA devices found" << endln;
        return false;
    }

    cudssHandle_t handle = nullptr;
    cudssStatus_t status = cudssCreate(&handle);
    if (status != CUDSS_STATUS_SUCCESS || handle == nullptr) {
        opserr << "ERROR: verifyCuDSSRuntime() - cudssCreate failed with status "
               << status << endln;
#ifdef _WIN32
        printCuDSSPathHint();
#endif
        return false;
    }

    status = cudssDestroy(handle);
    if (status != CUDSS_STATUS_SUCCESS) {
        opserr << "WARNING: verifyCuDSSRuntime() - cudssDestroy failed with status "
               << status << endln;
    }
    return true;
}

CudaDirectSparseSolver::CudaDirectSparseSolver(CudaPrecision precision, bool verbose, 
                               bool hybridMemoryMode, const std::vector<size_t>& hybridDeviceMemoryLimits, 
                               bool hybridExecuteMode, bool multiThreadingMode,
                               const char* threadingLibPath,
                               CuDSSMatrixType cudssMatType,
                               bool useMultiGPU,
                               const std::vector<int>& deviceIndices,
                               int irNSteps,
                               double irTol,
                               int deviceId)
    :CudaBcsrLinSolver(SOLVER_TAGS_CudaDirectSparseSolver, precision), 
    m_verbose(verbose),
    m_hybridMemoryMode(hybridMemoryMode),
    m_hybridDeviceMemoryLimits(hybridDeviceMemoryLimits),
    m_hybridExecuteMode(hybridExecuteMode),
    m_multiThreadingMode(multiThreadingMode),
    m_threadingLibPath(threadingLibPath ? threadingLibPath : ""),
    m_cudssMatType(cudssMatType),
    m_useMultiGPU(useMultiGPU),
    m_deviceIndices(deviceIndices),
    m_deviceId(deviceId),
    m_irNSteps(irNSteps),
    m_irTol(irTol),
    m_matrix(nullptr)
{
    if (!isUniformPrecision(precision)) {
        opserr << "ERROR: CudaDirectSparseSolver::CudaDirectSparseSolver() - "
               << "Precision " << cudaPrecisionToString(precision) << " is not supported by cuDSS. "
               << "cuDSS only supports uniform precision: dDDI (double) and dFFI (float)." << endln;
        throw std::invalid_argument("CuDSS does not support mixed precision");
    }
    init(precision);
}

void CudaDirectSparseSolver::init(CudaPrecision /*precision*/)
{
    return;
}

CudaCsrMatrix::SolverConfig CudaDirectSparseSolver::makeSolverConfig(CudaPrecision precision) const
{
    CudaCsrMatrix::SolverConfig solver;
    solver.precision = precision;
    solver.syncAfterSolve = true;
    solver.verbose = m_verbose;
    solver.hybridMemoryMode = m_hybridMemoryMode;
    solver.hybridDeviceMemoryLimits = m_hybridDeviceMemoryLimits;
    solver.hybridExecuteMode = m_hybridExecuteMode;
    solver.multiThreadingMode = m_multiThreadingMode;
    solver.threadingLibPath = m_threadingLibPath;
    solver.matType = m_cudssMatType;
    solver.useMultiGPU = m_useMultiGPU;
    solver.deviceIndices = m_deviceIndices;
    solver.deviceId = m_deviceId;
    solver.irNSteps = m_irNSteps;
    solver.irTol = m_irTol;
    return solver;
}

int CudaDirectSparseSolver::ensureMatrix(CudaBcsrLinSOE *theSOE)
{
    if (theSOE == nullptr) {
        return -1;
    }
    cudaStream_t cudaStream = static_cast<cudaStream_t>(theSOE->getCudaStream());
    if (cudaStream == nullptr) {
        opserr << "ERROR: CudaDirectSparseSolver::ensureMatrix() - SOE CUDA stream unavailable\n";
        return -1;
    }
    if (m_matrix != nullptr) {
        if (m_matrix->getStream() != cudaStream) {
            opserr << "ERROR: CudaDirectSparseSolver::ensureMatrix() - SOE CUDA stream changed\n";
            return -1;
        }
        return 0;
    }
    CudaCsrMatrix::ExecutionContext exec;
    exec.stream = cudaStream;
    m_matrix = new CudaCsrMatrix(makeSolverConfig(getPrecision()), CudaCsrMatrix::SpmvConfig{}, exec);
    return 0;
}

void CudaDirectSparseSolver::releaseDeviceResources(void)
{
    // CudaCsrMatrix / CuDSSBackend hold the SOE stream as externalStream.
    delete m_matrix;
    m_matrix = nullptr;
}

CudaDirectSparseSolver::~CudaDirectSparseSolver()
{
    releaseDeviceResources();
}

int CudaDirectSparseSolver::setLinearSOE(CudaBcsrLinSOE &theSOE) {
    // cuDSS only supports scalar CSR (blockSize = 1), not BSR
    if (theSOE.getBlockSize() != 1) {
        opserr << "WARNING: CudaDirectSparseSolver::setLinearSOE() - "
               << "cuDSS only supports scalar CSR (blockSize = 1), got blockSize = " 
               << theSOE.getBlockSize() << endln;
        return -1;
    }
    
    // Check precision match using unified enum
    if (theSOE.getPrecision() != this->getPrecision()) {
        opserr << "WARNING: CudaDirectSparseSolver::setLinearSOE() - "
               << "precision mismatch: SOE is " << cudaPrecisionToString(theSOE.getPrecision())
               << ", solver is " << cudaPrecisionToString(this->getPrecision()) << endln;
        return -1;
    }
    
    return this->CudaBcsrLinSolver::setLinearSOE(theSOE);
}

int CudaDirectSparseSolver::solve(void) {
    CudaBcsrLinSOE* theSOE = this->CudaBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "WARNING: CudaDirectSparseSolver::solve() - LinearSOE not set" << endln;
        return -1;
    }
    if (ensureMatrix(theSOE) != 0) {
        opserr << "ERROR: CudaDirectSparseSolver::solve() - ensureMatrix() failed" << endln;
        return -1;
    }

    CudaBcsrLinSOE::MatrixStatus matrixStatus = theSOE->getMatrixStatus();
    void* AValues = theSOE->getDeviceAValues();
    void* xValues = theSOE->getDeviceX();
    void* bValues = theSOE->getDeviceB();

    if (!AValues || !xValues || !bValues) {
        opserr << "ERROR: CudaDirectSparseSolver::solve() - null device pointer(s)" << endln;
        return -1;
    }

    if (setupMatrices() != 0) {
        opserr << "ERROR: CudaDirectSparseSolver::solve() - setupMatrices() failed" << endln;
        return -1;
    }

    m_matrix->bindValues(AValues);

    if (!m_matrix->isFactored()) {
        if (m_matrix->factorize(bValues, xValues) != 0) {
            return -1;
        }
    } else if (matrixStatus == CudaBcsrLinSOE::MatrixStatus::STRUCTURE_CHANGED) {
        if (m_matrix->factorize(bValues, xValues) != 0) {
            return -1;
        }
    } else if (matrixStatus == CudaBcsrLinSOE::MatrixStatus::COEFFICIENTS_CHANGED) {
        if (m_matrix->refactorize(bValues, xValues) != 0) {
            return -1;
        }
    }

    if (m_matrix->solve(bValues, xValues) != 0) {
        opserr << "WARNING: CudaDirectSparseSolver::solve() - cuDSS solve failed" << endln;
        return -1;
    }

    if (m_verbose) {
        opserr << "INFO: CudaDirectSparseSolver::solve() - cuDSS solve successful" << endln;
    }

    return 0;
}

int CudaDirectSparseSolver::setSize() {
    // In OpenSees, setSize() is called before data is ready on the GPU
    // Matrix initialization is done in solve() via setupMatrices()
    return 0;
}

LinearSOESolver *
CudaDirectSparseSolver::getCopy(void) const
{
    return new CudaDirectSparseSolver(
        getPrecision(),
        m_verbose,
        m_hybridMemoryMode,
        m_hybridDeviceMemoryLimits,
        m_hybridExecuteMode,
        m_multiThreadingMode,
        m_threadingLibPath.empty() ? nullptr : m_threadingLibPath.c_str(),
        m_cudssMatType,
        m_useMultiGPU,
        m_deviceIndices,
        m_irNSteps,
        m_irTol,
        m_deviceId);
}

int CudaDirectSparseSolver::setupMatrices() {
    CudaBcsrLinSOE* theSOE = this->CudaBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "WARNING: CudaDirectSparseSolver::setupMatrices() - LinearSOE not set" << endln;
        return -1;
    }
    if (ensureMatrix(theSOE) != 0) {
        return -1;
    }
    if (m_matrix == nullptr) {
        opserr << "ERROR: CudaDirectSparseSolver::setupMatrices() - direct solver not initialized\n";
        return -1;
    }

    CudaBcsrLinSOE::MatrixStatus matrixStatus = theSOE->getMatrixStatus();
    if (matrixStatus != CudaBcsrLinSOE::MatrixStatus::STRUCTURE_CHANGED &&
        m_matrix->isStructureBound()) {
        return 0;
    }

    int numRows = theSOE->getNumRowBlocks();
    int numNZ = theSOE->getNumNonZeroValues();
    if (theSOE->getBlockSize() != 1) {
        opserr << "ERROR: CudaDirectSparseSolver::setupMatrices() - Only blockSize = 1 is supported" << endln;
        return -1;
    }

    int* rowPtrs = theSOE->getDeviceRowPtrs();
    int* colIndices = theSOE->getDeviceColIndices();
    void* AValues = theSOE->getDeviceAValues();
    if (!rowPtrs || !colIndices || !AValues) {
        opserr << "ERROR: CudaDirectSparseSolver::setupMatrices() - null device pointer(s)" << endln;
        return -1;
    }

    if (m_matrix->bindStructure(numRows, numNZ, rowPtrs, colIndices) != 0) {
        return -1;
    }
    if (m_matrix->bindValues(AValues) != 0) {
        return -1;
    }

    if (m_verbose) {
        opserr << "INFO: CudaDirectSparseSolver::setupMatrices() - CSR structure bound" << endln;
    }

    return 0;
}

// OpenSees API for creating CuDSS solver

struct CuDSSConfig {
    std::string precision = "dDDI";
    bool verbose = false;
    bool hybridMemoryMode = false;        // Hybrid host/device memory mode
    std::vector<size_t> hybridDeviceMemoryLimits;   // Per-device limit for hybrid memory (empty = use heuristic; one value = same for all)
    bool hybridExecuteMode = false;       // Hybrid host/device execute mode
    bool multiThreadingMode = false;      // cuDSS MT mode via runtime cudssSetThreadingLayer()
#ifdef _WIN32
    std::string threadingLibPath = "cudss_mtlayer_vcomp14064_0.dll";
#else
    std::string threadingLibPath = "/usr/lib/x86_64-linux-gnu/libcudss_mtlayer_gomp.so";
#endif
    std::string cudssMatTypeStr = "full"; // full | symmetric | spd (symmetric/spd use lower storage in SOE)
    int deviceId = -1;                    // Preferred single-GPU device (-1 = driver current/default)
    bool verifyRuntime = false;           // Optional early cuda/cuDSS smoke test at system time
    int irNSteps = 0;                     // Iterative refinement steps (0 = disabled)
    double irTol = 0.0;                   // IR tolerance (0 = fixed-step, no convergence check)
};

class CuDSSParameterParser {
private:
    static std::unordered_map<std::string, std::function<void(CuDSSConfig&)>> const configParsers;

public:
    static bool parseParameters(CuDSSConfig& config);
    static void printUsageInfo();
};

const std::unordered_map<std::string, std::function<void(CuDSSConfig&)>> 
CuDSSParameterParser::configParsers = {
    {"precision", [](CuDSSConfig& config) { 
        const char* value = OPS_GetString();
        if (value && (strcmp(value, "dDDI") == 0 || strcmp(value, "dFFI") == 0)) {
            config.precision = value;
        } else if (value) {
            throw std::invalid_argument("Invalid precision. Only dDDI and dFFI are supported.");
        }
    }},
    {"verbose", [](CuDSSConfig& config) { 
        int numData = 1;
        int flag = 0;
        if (OPS_GetIntInput(&numData, &flag) == 0) {
            if (flag != 0 && flag != 1) throw std::invalid_argument("verbose must be 0 or 1");
            config.verbose = (flag == 1);
        }
    }},
    {"hybridMemoryMode", [](CuDSSConfig& config) { 
        int numData = 1;
        int flag = 0;
        if (OPS_GetIntInput(&numData, &flag) == 0) {
            if (flag != 0 && flag != 1) throw std::invalid_argument("hybridMemoryMode must be 0 or 1");
            config.hybridMemoryMode = (flag == 1);
        }
    }},
    {"hybridDeviceMemoryLimit", [](CuDSSConfig& config) { 
        config.hybridDeviceMemoryLimits.clear();
        int numData = 1;
        double limit = 0.0;
        if (OPS_GetDoubleInput(&numData, &limit) != 0) return;
        if (limit < 0.0) throw std::invalid_argument("hybridDeviceMemoryLimit cannot be negative");
        config.hybridDeviceMemoryLimits.push_back(static_cast<size_t>(limit));
        while (OPS_GetNumRemainingInputArgs() > 0) {
            if (OPS_GetDoubleInput(&numData, &limit) != 0) break;
            if (limit < 0.0) throw std::invalid_argument("hybridDeviceMemoryLimit value cannot be negative");
            config.hybridDeviceMemoryLimits.push_back(static_cast<size_t>(limit));
        }
    }},
    {"hybridExecuteMode", [](CuDSSConfig& config) { 
        int numData = 1;
        int flag = 0;
        if (OPS_GetIntInput(&numData, &flag) == 0) {
            if (flag != 0 && flag != 1) throw std::invalid_argument("hybridExecuteMode must be 0 or 1");
            config.hybridExecuteMode = (flag == 1);
        }
    }},
    {"multiThreadingMode", [](CuDSSConfig& config) { 
        int numData = 1;
        int flag = 0;
        if (OPS_GetIntInput(&numData, &flag) == 0) {
            if (flag != 0 && flag != 1) throw std::invalid_argument("multiThreadingMode must be 0 or 1");
            config.multiThreadingMode = (flag == 1);
        }
    }},
    {"threadingLibPath", [](CuDSSConfig& config) { 
        const char* value = OPS_GetString();
        if (value) config.threadingLibPath = value;
    }},
    {"matrixType", [](CuDSSConfig& config) { 
        const char* value = OPS_GetString();
        if (value) {
            std::string s(value);
            if (s == "full" || s == "symmetric" || s == "spd") {
                config.cudssMatTypeStr = s;
            } else {
                throw std::invalid_argument("matrixType must be full, symmetric, or spd");
            }
        }
    }},
    {"device", [](CuDSSConfig& config) {
        int numData = 1;
        int id = 0;
        if (OPS_GetIntInput(&numData, &id) == 0) {
            if (id < 0) throw std::invalid_argument("device must be >= 0");
            config.deviceId = id;
        }
    }},
    {"verifyRuntime", [](CuDSSConfig& config) {
        int numData = 1;
        int flag = 0;
        if (OPS_GetIntInput(&numData, &flag) == 0) {
            if (flag != 0 && flag != 1) throw std::invalid_argument("verifyRuntime must be 0 or 1");
            config.verifyRuntime = (flag == 1);
        }
    }},
    {"irNSteps", [](CuDSSConfig& config) {
        int numData = 1;
        int steps = 0;
        if (OPS_GetIntInput(&numData, &steps) == 0) {
            if (steps < 0) throw std::invalid_argument("irNSteps must be >= 0");
            config.irNSteps = steps;
        }
    }},
    {"irTol", [](CuDSSConfig& config) {
        int numData = 1;
        double tol = 0.0;
        if (OPS_GetDoubleInput(&numData, &tol) == 0) {
            if (tol < 0.0) throw std::invalid_argument("irTol must be >= 0");
            config.irTol = tol;
        }
    }}
};

bool CuDSSParameterParser::parseParameters(CuDSSConfig& config) {
    try {
        while (OPS_GetNumRemainingInputArgs() > 0) {
            const char* key = OPS_GetString();
            if (!key) {
                opserr << "WARNING: CuDSSParameterParser::parseParameters() - "
                       << "Invalid input argument" << endln;
                return false;
            }

            auto it = ParameterUtils::findParameter(key, configParsers);
            if (it != configParsers.end()) {
                it->second(config);
            } else {
                continue;
            }
        }

        return true;
    } catch (const std::exception& e) {
        opserr << "WARNING: CuDSSParameterParser::parseParameters() - "
               << e.what() << endln;
        return false;
    }
}

void CuDSSParameterParser::printUsageInfo() {
    opserr << "CuDSSParameterParser::printUsageInfo() - " << endln;
    opserr << "Usage: system CuDSS [options]" << endln;
    opserr << "Options:" << endln;
    opserr << "  -precision <dDDI|dFFI>          Precision (default: dDDI)" << endln;
    opserr << "  -verbose <0|1>                  Enable verbose output (default: 0)" << endln;
    opserr << "  -device <id>                    Preferred CUDA device (default: current)" << endln;
    opserr << "  -verifyRuntime <0|1>            Early cudaGetDeviceCount/cudssCreate check at system time (default: 0)" << endln;
    opserr << "  -hybridMemoryMode <0|1>         Hybrid host/device memory mode (default: 0)" << endln;
    opserr << "  -hybridDeviceMemoryLimit <bytes1 [bytes2 ...]> Per-device limit for hybrid memory (one value=all devices; 0=min)" << endln;
    opserr << "  -hybridExecuteMode <0|1>        Hybrid host/device execute mode (default: 0)" << endln;
    opserr << "  -multiThreadingMode <0|1>       cuDSS MT mode via cudssSetThreadingLayer (default: 0)" << endln;
    opserr << "  -threadingLibPath <path|NULL>   Path to cuDSS mtlayer shim (when multiThreadingMode is enabled)" << endln;
#ifdef _WIN32
    opserr << "                                  (default: cudss_mtlayer_vcomp14064_0.dll in CUDSS_DIR\\bin\\12," << endln;
#else
    opserr << "                                  (default: /usr/lib/x86_64-linux-gnu/libcudss_mtlayer_gomp.so," << endln;
#endif
    opserr << "                                   use 'NULL' to let cuDSS use CUDSS_THREADING_LIB env var)" << endln;
    opserr << "  -matrixType <full|symmetric|spd> Matrix type: full (default), symmetric, or spd" << endln;
    opserr << "                                  (symmetric and spd use lower storage; halves matrix memory)" << endln;
    opserr << "  -irNSteps <int>                 Iterative refinement steps (default: 0, disabled)" << endln;
    opserr << "  -irTol <double>                 IR relative-residual tolerance (default: 0)" << endln;
    opserr << "                                  irTol=0: exactly irNSteps iterations, no convergence check" << endln;
    opserr << "                                  irTol>0: early stop when ||r||/||b|| < irTol (max irNSteps)" << endln;
    opserr << "Notes:" << endln;
    opserr << "  - hybridMemoryMode and hybridExecuteMode are mutually exclusive" << endln;
    opserr << "  - When multiThreadingMode is enabled: set OMP_NUM_THREADS or CUDSS_THREADING_LIB as needed" << endln;
    opserr << "  - OpenSeesMP: use system DistributedCuDSS (rank-0 GPU + gather-add); cuDSS multi-GPU/MGMN is not exposed yet" << endln;
}

// Factory function to create CuDSS solver from parsed config
CudaBcsrLinSolver* createCuDSSSolverFromConfig(const CuDSSConfig& config) {
    // Convert string precision to enum
    CudaPrecision precision;
    if (!cudaPrecisionFromString(config.precision.c_str(), precision)) {
        opserr << "WARNING: createCuDSSSolverFromConfig() - "
               << "Invalid precision '" << config.precision.c_str() << "', defaulting to dDDI" << endln;
        precision = CudaPrecision::dDDI;
    }
    
    CuDSSMatrixType cudssMatType = CuDSSMatrixType::FULL;
    if (config.cudssMatTypeStr == "symmetric") cudssMatType = CuDSSMatrixType::SYMMETRIC;
    else if (config.cudssMatTypeStr == "spd") cudssMatType = CuDSSMatrixType::SPD;

    return new CudaDirectSparseSolver(
        precision, 
        config.verbose,
        config.hybridMemoryMode,
        config.hybridDeviceMemoryLimits,
        config.hybridExecuteMode,
        config.multiThreadingMode,
        config.threadingLibPath.c_str(),
        cudssMatType,
        /*useMultiGPU=*/false,
        /*deviceIndices=*/{},
        config.irNSteps,
        config.irTol,
        config.deviceId
    );
}

// Factory function that parses OPS arguments and creates solver
CudaBcsrLinSolver* createCuDSSSolverFromParser() {
    CuDSSConfig config;
    
    // Parse command-line arguments
    if (!CuDSSParameterParser::parseParameters(config)) {
        opserr << "WARNING: createCuDSSSolverFromParser() - "
               << "Failed to parse parameters, using defaults" << endln;
        opserr << "For valid parameters, use:" << endln;
        CuDSSParameterParser::printUsageInfo();
    }
    
    // Validate that hybrid modes are mutually exclusive
    if (config.hybridMemoryMode && config.hybridExecuteMode) {
        opserr << "ERROR: createCuDSSSolverFromParser() - "
               << "hybridMemoryMode and hybridExecuteMode are mutually exclusive. "
               << "Only one can be enabled at a time." << endln;
        return nullptr;
    }
    
    // Validate that hybridDeviceMemoryLimits is only used with hybridMemoryMode
    if (!config.hybridDeviceMemoryLimits.empty() && !config.hybridMemoryMode) {
        opserr << "WARNING: createCuDSSSolverFromParser() - "
               << "hybridDeviceMemoryLimit is only valid with hybridMemoryMode enabled. "
               << "Ignoring hybridDeviceMemoryLimit." << endln;
    }
    
    return createCuDSSSolverFromConfig(config);
}

#include <DistributedCudaBcsrLinSOE.h>
#include <classTags.h>

namespace {

/** Host-only placeholder for DistributedCuDSS worker ranks (no CUDA/cuDSS). */
class CudaHostWorkerSolver : public CudaBcsrLinSolver {
public:
    CudaHostWorkerSolver()
        : CudaBcsrLinSolver(SOLVER_TAGS_CudaHostWorkerSolver, CudaPrecision::dDDI)
    {
    }

    int solve(void) override
    {
        opserr << "ERROR: CudaHostWorkerSolver::solve() - GPU solve is rank-0 only "
               << "for DistributedCuDSS\n";
        return -1;
    }

    int setSize(void) override { return 0; }

    void releaseDeviceResources(void) override {}

    LinearSOESolver *getCopy(void) const override
    {
        return new CudaHostWorkerSolver();
    }
};

bool validateCuDSSConfig(const CuDSSConfig &config)
{
    if (config.hybridMemoryMode && config.hybridExecuteMode) {
        opserr << "ERROR: CuDSS - hybridMemoryMode and hybridExecuteMode are mutually exclusive. "
               << "Only one can be enabled at a time." << endln;
        return false;
    }
    if (!config.hybridDeviceMemoryLimits.empty() && !config.hybridMemoryMode) {
        opserr << "WARNING: CuDSS - hybridDeviceMemoryLimit is only valid with hybridMemoryMode enabled. "
               << "Ignoring hybridDeviceMemoryLimit." << endln;
    }
    return true;
}

bool parseCuDSSConfigFromOPS(CuDSSConfig &config)
{
    if (OPS_GetNumRemainingInputArgs() == 1) {
        (void)OPS_ExpandDictArgs();
    }

    if (!CuDSSParameterParser::parseParameters(config)) {
        opserr << "WARNING: CuDSS - failed to parse parameters, using defaults" << endln;
        opserr << "For valid parameters, use:" << endln;
        CuDSSParameterParser::printUsageInfo();
    }
    return validateCuDSSConfig(config);
}

/**
 * Build a CudaBcsrLinSOE for CuDSS.
 * hostOnly: DistributedCuDSS workers — no verifyCuDSSRuntime, no cuDSS/CUDA device use.
 * Rank-0 / serial: optional verifyRuntime, then full CudaDirectSparseSolver.
 */
CudaBcsrLinSOE *createCudaBcsrSOEFromCuDSSConfig(const CuDSSConfig &config, bool hostOnly)
{
    // Workers must use the same storage as rank 0: with symmetric storage rank 0 keeps only
    // the lower triangle, so upper-triangle triplets from a full-storage worker have no slot.
    const bool symmetricStorage =
        (config.cudssMatTypeStr == "symmetric" || config.cudssMatTypeStr == "spd");

    if (hostOnly) {
        CudaBcsrLinSolver *workerSolver = new CudaHostWorkerSolver();
        CudaBcsrLinSOE *soe =
            CudaBcsrLinSOE::createDouble(*workerSolver, /*blockSize=*/1,
                                         /*paddingEnabled=*/false,
                                         config.verbose,
                                         symmetricStorage);
        if (soe == nullptr) {
            delete workerSolver;
            opserr << "ERROR: CuDSS - failed to create host-only worker SOE\n";
            return nullptr;
        }
        soe->setCudaDeviceEnabled(false);
        return soe;
    }

    if (config.verifyRuntime && !verifyCuDSSRuntime()) {
        opserr << "ERROR: CuDSS - cuDSS/CUDA runtime verification failed" << endln;
        return nullptr;
    }

    // Select device early so subsequent stream/cuDSS create use it.
    if (config.deviceId >= 0) {
        const cudaError_t err = cudaSetDevice(config.deviceId);
        if (err != cudaSuccess) {
            opserr << "ERROR: CuDSS - cudaSetDevice(" << config.deviceId << ") failed: "
                   << cudaGetErrorString(err) << endln;
            return nullptr;
        }
    }

    CudaBcsrLinSolver *solver = nullptr;
    try {
        solver = createCuDSSSolverFromConfig(config);
    } catch (const std::exception &e) {
        opserr << "ERROR: CuDSS - failed to create solver: " << e.what() << endln;
        return nullptr;
    }
    if (solver == nullptr)
        return nullptr;

    const int blockSize = 1;
    const bool paddingEnabled = false;
    const CudaPrecision precision = solver->getPrecision();

    CudaBcsrLinSOE *soe = nullptr;
    switch (precision) {
        case CudaPrecision::dDDI:
            soe = CudaBcsrLinSOE::createDouble(*solver, blockSize, paddingEnabled, config.verbose,
                                               symmetricStorage);
            break;
        case CudaPrecision::dFFI:
            soe = CudaBcsrLinSOE::createFloat(*solver, blockSize, paddingEnabled, config.verbose,
                                              symmetricStorage);
            break;
        default:
            opserr << "ERROR: CuDSS - unexpected precision mode" << endln;
            delete solver;
            return nullptr;
    }
    if (soe == nullptr)
        delete solver;
    return soe;
}

} // namespace

void* OPS_CudaDirectSparseSolver()
{
    CuDSSConfig config;
    if (!parseCuDSSConfigFromOPS(config))
        return nullptr;

    // Serial CuDSS: every process that requests it gets a full CUDA/cuDSS SOE.
    return createCudaBcsrSOEFromCuDSSConfig(config, /*hostOnly=*/false);
}

void *OPS_DistributedCudaDirectSparseSolver(void)
{
    CuDSSConfig config;
    // All ranks must parse the same argv (OpenSeesMP runs the command on each process).
    if (!parseCuDSSConfigFromOPS(config))
        return nullptr;

    // Workers: host-only SOE (no CUDA). Matches commands.cpp setProcessID(OPS_rank).
    extern int OPS_rank;
    const bool hostOnly = (OPS_rank != 0);

    CudaBcsrLinSOE *cudaSOE =
        createCudaBcsrSOEFromCuDSSConfig(config, hostOnly);
    if (cudaSOE == nullptr)
        return nullptr;

    return new DistributedCudaBcsrLinSOE(cudaSOE);
}

