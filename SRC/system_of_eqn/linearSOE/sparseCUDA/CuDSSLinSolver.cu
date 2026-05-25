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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCUDA/CuDSSLinSolver.cpp,v $
                                                                        
                                                                        
// Written: gaaraujo 
// Created: 10/2025
//
// Description: This file contains the class definition for 
// CuDSSLinSolver. It solves the CudaGenBcsrLinSOE object by calling
// CuDSS routines.
//

// OpenSees includes
#include <classTags.h>
#include <OPS_Globals.h>

// Solve core classes
#include <CudaGenBcsrLinSOE.h>
#include <CuDSSLinSolver.h>
#include "ParameterUtils.h"

// CUDA utilities
#include "CudaUtils.h"

// for parsing command line arguments
#ifdef _CUDSS
#include <elementAPI.h>
#include <FileStream.h>
#include <unordered_map>
#include "ParameterUtils.h"
#endif // _CUDSS

// C++ includes
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <stdexcept>

#ifdef _CUDSS
// Use CudaUtils namespace for error checking
using namespace CudaUtils;
#endif // _CUDSS

CuDSSLinSolver::CuDSSLinSolver(CudaPrecision precision, bool verbose, 
                               bool hybridMemoryMode, const std::vector<size_t>& hybridDeviceMemoryLimits, 
                               bool hybridExecuteMode, bool multiThreadingMode,
                               const char* threadingLibPath,
                               CuDSSMatrixType cudssMatType,
                               bool useMultiGPU,
                               const std::vector<int>& deviceIndices)
    :CudaGenBcsrLinSolver(SOLVER_TAGS_CuDSSLinSolver, precision), 
    m_verbose(verbose),
    m_hybridMemoryMode(hybridMemoryMode),
    m_hybridDeviceMemoryLimits(hybridDeviceMemoryLimits),
    m_hybridExecuteMode(hybridExecuteMode),
    m_multiThreadingMode(multiThreadingMode),
    m_threadingLibPath(threadingLibPath ? threadingLibPath : ""),
    m_cudssMatType(cudssMatType),
    m_useMultiGPU(useMultiGPU),
    m_deviceIndices(deviceIndices)
{
    #ifdef _CUDSS
    m_Handle = nullptr;
    m_cudaStream = nullptr;
    #endif
    #ifdef _CUDSS
    // CuDSS currently only supports uniform precision (dDDI or dFFI)
    if (!isUniformPrecision(precision)) {
        opserr << "ERROR: CuDSSLinSolver::CuDSSLinSolver() - "
               << "Precision " << cudaPrecisionToString(precision) << " is not supported by cuDSS. "
               << "cuDSS only supports uniform precision: dDDI (double) and dFFI (float)." << endln;
        throw std::invalid_argument("CuDSS does not support mixed precision");
    }
    init(precision);
    #endif // _CUDSS
}

void CuDSSLinSolver::init(CudaPrecision precision)
{
    #ifdef _CUDSS
    /* Per-instance stream and handle (single-GPU or MG) */
    cudaCheckError(cudaStreamCreate(&m_cudaStream), "create CUDA stream");

    if (m_useMultiGPU) {
        std::vector<int> devs = m_deviceIndices;
        if (devs.empty()) {
            int count = 0;
            cudaCheckError(cudaGetDeviceCount(&count), "get device count");
            if (count <= 0) {
                opserr << "ERROR: CuDSSLinSolver::init() - no CUDA devices available for multi-GPU" << endln;
                throw std::runtime_error("no CUDA devices");
            }
            devs.resize(static_cast<size_t>(count));
            for (int i = 0; i < count; i++) devs[i] = i;
            m_deviceIndices = devs;
        }
        int deviceCount = static_cast<int>(devs.size());
        int firstDevice = devs[0];
        cudaCheckError(cudaSetDevice(firstDevice), "set device for multi-GPU");
        cuDSSCheckError(cudssCreateMg(&m_Handle, deviceCount, devs.data()), "create cuDSS MG handle");
        cuDSSCheckError(cudssSetStream(m_Handle, m_cudaStream), "set CUDA stream on MG handle");
        if (m_verbose) {
            opserr << "INFO: CuDSSLinSolver::init() - multi-GPU mode, devices:";
            for (int d : m_deviceIndices) opserr << " " << d;
            opserr << endln;
        }
    } else {
        cuDSSCheckError(cudssCreate(&m_Handle), "create cuDSS handle");
        cuDSSCheckError(cudssSetStream(m_Handle, m_cudaStream), "set CUDA stream");
    }

    /* Setup OpenMP multi-threading (single-GPU and MG) */
    #ifdef CUDSS_USE_OPENMP
    if (m_multiThreadingMode) {
        const char* threadingLib = (m_threadingLibPath == "NULL") ? nullptr : m_threadingLibPath.c_str();
        cudssStatus_t status = cudssSetThreadingLayer(m_Handle, threadingLib);
        if (status != CUDSS_STATUS_SUCCESS) {
            opserr << "WARNING: CuDSSLinSolver::init() - "
                   << "cudssSetThreadingLayer failed with status " << status << endln;
            opserr << "Continuing without multi-threading support" << endln;
        } else if (m_verbose) {
            opserr << "INFO: CuDSSLinSolver::init() - OpenMP multi-threading mode enabled" << endln;
        }
    }
    #else
    if (m_multiThreadingMode && m_verbose) {
        opserr << "INFO: CuDSSLinSolver::init() - OpenMP not available at build time" << endln;
    }
    #endif

    /* Create cuDSS solver configuration and data objects */
    cuDSSCheckError(cudssConfigCreate(&m_Config), "create cuDSS solver configuration");
    cuDSSCheckError(cudssDataCreate(m_Handle, &m_Data), "create cuDSS solver data");

    if (m_useMultiGPU) {
        int deviceCount = static_cast<int>(m_deviceIndices.empty() ? 0 : m_deviceIndices.size());
        std::vector<int> devs = m_deviceIndices;
        if (devs.empty()) {
            int count = 0;
            cudaGetDeviceCount(&count);
            devs.resize(static_cast<size_t>(count));
            for (int i = 0; i < count; i++) devs[i] = i;
            deviceCount = count;
        }
        cuDSSCheckError(cudssConfigSet(m_Config, CUDSS_CONFIG_DEVICE_COUNT, &deviceCount, sizeof(deviceCount)), "set device count");
        cuDSSCheckError(cudssConfigSet(m_Config, CUDSS_CONFIG_DEVICE_INDICES, devs.data(), deviceCount * sizeof(int)), "set device indices");
    }
    
    // (optional) Modifying solver settings, e.g., reordering algorithm
    cudssAlgType_t reorderAlgorithm = CUDSS_ALG_DEFAULT;
    cudssConfigSet(m_Config, CUDSS_CONFIG_REORDERING_ALG, &reorderAlgorithm, sizeof(cudssAlgType_t));
    
    /* Configure hybrid mode (must be set before analysis phase).
     * Device memory limit is set after ANALYSIS in setupMatrices(). */
    if (m_hybridMemoryMode) {
        int hybridMemoryModeEnabled = 1;
        cuDSSCheckError(cudssConfigSet(m_Config, CUDSS_CONFIG_HYBRID_MODE, &hybridMemoryModeEnabled, sizeof(int)), 
                       "enable hybrid memory mode");
        
        if (m_verbose) {
            opserr << "INFO: CuDSSLinSolver::init() - "
                   << "Hybrid memory mode enabled (device limit set after analysis)" << endln;
        }
    }
    
    if (m_hybridExecuteMode) {
        int hybridExecuteModeEnabled = 1;
        cuDSSCheckError(cudssConfigSet(m_Config, CUDSS_CONFIG_HYBRID_EXECUTE_MODE, &hybridExecuteModeEnabled, sizeof(int)), 
                       "enable hybrid execute mode");
        
        if (m_verbose) {
            opserr << "INFO: CuDSSLinSolver::init() - "
                   << "Hybrid execute mode enabled" << endln;
        }
    }

    m_Matrix = nullptr;
    m_RHS = nullptr;
    m_Solution = nullptr;
    
    /* Set precision (format: dXYI where X=matrix type, Y=vector type, I=index type)
     * d = device (always required for GPU)
     * X,Y = D (double) or F (float)
     * I = I (int32) - currently only int32 is supported
     */
    m_IndexType = CUDA_R_32I; // Currently only 32-bit integers supported
    
    if (precision == CudaPrecision::dFFI) {
        m_ValueType = CUDA_R_32F; // Float precision
    } else {  // CudaPrecision::dDDI
        m_ValueType = CUDA_R_64F; // Double precision
    }
    #endif // _CUDSS

    return;
}

CuDSSLinSolver::~CuDSSLinSolver()
{
    #ifdef _CUDSS
    if (m_Matrix != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_Matrix), "destroy cuDSS matrix", false);
    }
    if (m_RHS != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_RHS), "destroy cuDSS right-hand side", false);
    }
    if (m_Solution != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_Solution), "destroy cuDSS solution", false);
    }
    cuDSSCheckError(cudssDataDestroy(m_Handle, m_Data), "destroy cuDSS solver data", false);
    cuDSSCheckError(cudssConfigDestroy(m_Config), "destroy cuDSS config", false);
    cuDSSCheckError(cudssDestroy(m_Handle), "destroy cuDSS handle", false);
    cudaCheckError(cudaStreamSynchronize(m_cudaStream), "synchronize CUDA stream", false);
    cudaStreamDestroy(m_cudaStream);
    #endif // _CUDSS

    return;
}

int CuDSSLinSolver::setLinearSOE(CudaGenBcsrLinSOE &theSOE) {
    #ifdef _CUDSS
    // cuDSS only supports scalar CSR (blockSize = 1), not BSR
    if (theSOE.getBlockSize() != 1) {
        opserr << "WARNING: CuDSSLinSolver::setLinearSOE() - "
               << "cuDSS only supports scalar CSR (blockSize = 1), got blockSize = " 
               << theSOE.getBlockSize() << endln;
        return -1;
    }
    
    // Check precision match using unified enum
    if (theSOE.getPrecision() != this->getPrecision()) {
        opserr << "WARNING: CuDSSLinSolver::setLinearSOE() - "
               << "precision mismatch: SOE is " << cudaPrecisionToString(theSOE.getPrecision())
               << ", solver is " << cudaPrecisionToString(this->getPrecision()) << endln;
        return -1;
    }
    
    return this->CudaGenBcsrLinSolver::setLinearSOE(theSOE);
    #endif // _CUDSS

    return 0;
}

int CuDSSLinSolver::solve(void) {
    #ifdef _CUDSS
    if (m_useMultiGPU && !m_deviceIndices.empty()) {
        cudaCheckError(cudaSetDevice(m_deviceIndices[0]), "set device for solve");
    }
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "WARNING: CuDSSLinSolver::solve() - "
               << "LinearSOE not set" << endln;
        return -1;
    }

    // Extract info from the SOE
    CudaGenBcsrLinSOE::MatrixStatus matrixStatus = theSOE->getMatrixStatus();
    int numRows = theSOE->getNumRowBlocks();
    void* AValues = theSOE->getDeviceAValues();
    void* xValues = theSOE->getDeviceX();
    void* bValues = theSOE->getDeviceB();

    // Check if device pointers are valid
    if (!AValues) {
        opserr << "ERROR: CuDSSLinSolver::solve() - getDeviceAValues() returned nullptr" << endln;
        return -1;
    }
    if (!xValues) {
        opserr << "ERROR: CuDSSLinSolver::solve() - getDeviceX() returned nullptr" << endln;
        return -1;
    }
    if (!bValues) {
        opserr << "ERROR: CuDSSLinSolver::solve() - getDeviceB() returned nullptr" << endln;
        return -1;
    }

    // Setup matrices if structure has changed (setupMatrices checks internally)
    if (setupMatrices() != 0) {
        opserr << "ERROR: CuDSSLinSolver::solve() - setupMatrices() failed" << endln;
        return -1;
    }

    // Handle matrix updates and factorization
    if (matrixStatus == CudaGenBcsrLinSOE::MatrixStatus::STRUCTURE_CHANGED) {
        /* Update the RHS and solution pointers */
        cuDSSCheckError(cudssMatrixSetValues(m_RHS, bValues), "update cuDSS RHS values");
        cuDSSCheckError(cudssMatrixSetValues(m_Solution, xValues), "update cuDSS solution values");

        /* Numerical factorization (first time) */
        cuDSSCheckError(cudssExecute(
            m_Handle, CUDSS_PHASE_FACTORIZATION, m_Config, m_Data,
            m_Matrix, m_Solution, m_RHS
        ), "cuDSS numerical factorization");
    } else if (matrixStatus == CudaGenBcsrLinSOE::MatrixStatus::COEFFICIENTS_CHANGED) {
        /* Update coefficients */
        cuDSSCheckError(cudssMatrixSetValues(m_Matrix, AValues), "set cuDSS matrix values");
        
        /* Update the RHS and solution pointers */
        cuDSSCheckError(cudssMatrixSetValues(m_RHS, bValues), "update cuDSS RHS values");
        cuDSSCheckError(cudssMatrixSetValues(m_Solution, xValues), "update cuDSS solution values");

        /* Numerical factorization (refactorize with updated coefficients) */
        cuDSSCheckError(cudssExecute(
            m_Handle, CUDSS_PHASE_REFACTORIZATION, m_Config, m_Data,
            m_Matrix, m_Solution, m_RHS
        ), "cuDSS numerical factorization");
    } else {
        /* No changes, just update RHS and solution pointers */
        cuDSSCheckError(cudssMatrixSetValues(m_RHS, bValues), "update cuDSS RHS values");
        cuDSSCheckError(cudssMatrixSetValues(m_Solution, xValues), "update cuDSS solution values");
    }

    // Solve the system of equations
    cudssStatus_t status;
    status = cudssExecute(
        m_Handle, CUDSS_PHASE_SOLVE, m_Config, m_Data,
        m_Matrix, m_Solution, m_RHS
    );
    
    if (status != CUDSS_STATUS_SUCCESS) {
        opserr << "WARNING: CuDSSLinSolver::solve() - "
               << "cuDSS solve failed with status " << status << endln;
        return -1;
    }

    // Synchronize the CUDA stream to ensure solve is complete before returning
    cudaCheckError(cudaStreamSynchronize(m_cudaStream), "synchronize CUDA stream after solve");

    if (m_verbose) {
        opserr << "INFO: CuDSSLinSolver::solve() - "
               << "cuDSS solve successful" << endln;
    }

    #endif // _CUDSS

    return 0;
}

int CuDSSLinSolver::setSize() {
    // In OpenSees, setSize() is called before data is ready on the GPU
    // Matrix initialization is done in solve() via setupMatrices()
    return 0;
}

int CuDSSLinSolver::setupMatrices() {
    #ifdef _CUDSS
    if (m_useMultiGPU && !m_deviceIndices.empty()) {
        cudaCheckError(cudaSetDevice(m_deviceIndices[0]), "set device for setupMatrices");
    }
    CudaGenBcsrLinSOE* theSOE = this->CudaGenBcsrLinSolver::getLinearSOE();
    if (theSOE == nullptr) {
        opserr << "WARNING: CuDSSLinSolver::setupMatrices() - "
               << "LinearSOE not set" << endln;
        return -1;
    }

    // Check matrix status - only setup if structure has changed
    CudaGenBcsrLinSOE::MatrixStatus matrixStatus = theSOE->getMatrixStatus();
    if (matrixStatus != CudaGenBcsrLinSOE::MatrixStatus::STRUCTURE_CHANGED) {
        // Matrices already set up, nothing to do
        return 0;
    }

    // Extract info from the SOE
    int numRows = theSOE->getNumRowBlocks();
    int numCols = numRows;
    int numNZ = theSOE->getNumNonZeroValues();
    int blockSize = theSOE->getBlockSize();
    int* rowPtrs = theSOE->getDeviceRowPtrs();
    int* colIndices = theSOE->getDeviceColIndices();

    // Check that system is scalar CSR (block CSR not supported)
    if (blockSize != 1) {
        opserr << "ERROR: CuDSSLinSolver::setupMatrices() - "
               << "Only blockSize = 1 is supported" << endln;
        return -1;
    }
    
    // Check if device pointers are valid
    if (!rowPtrs) {
        opserr << "ERROR: CuDSSLinSolver::setupMatrices() - getDeviceRowPtrs() returned nullptr" << endln;
        return -1;
    }
    if (!colIndices) {
        opserr << "ERROR: CuDSSLinSolver::setupMatrices() - getDeviceColIndices() returned nullptr" << endln;
        return -1;
    }
    void* AValues = theSOE->getDeviceAValues();
    if (!AValues) {
        opserr << "ERROR: CuDSSLinSolver::setupMatrices() - getDeviceAValues() returned nullptr" << endln;
        return -1;
    }

    // Get x and b pointers for dense matrix creation
    void* xValues = theSOE->getDeviceX();
    void* bValues = theSOE->getDeviceB();
    if (!xValues) {
        opserr << "ERROR: CuDSSLinSolver::setupMatrices() - getDeviceX() returned nullptr" << endln;
        return -1;
    }
    if (!bValues) {
        opserr << "ERROR: CuDSSLinSolver::setupMatrices() - getDeviceB() returned nullptr" << endln;
        return -1;
    }

    /* Destroy existing matrices if they exist */
    if (m_Matrix != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_Matrix), "destroy existing cuDSS matrix");
        m_Matrix = nullptr;
    }
    if (m_RHS != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_RHS), "destroy existing cuDSS RHS");
        m_RHS = nullptr;
    }
    if (m_Solution != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_Solution), "destroy existing cuDSS solution");
        m_Solution = nullptr;
    }
    
    /* Create the cuDSS CSR matrix (full, symmetric, or SPD; symmetric/SPD use lower storage) */
    cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
    if (m_cudssMatType == CuDSSMatrixType::SYMMETRIC) {
        mtype = CUDSS_MTYPE_SYMMETRIC;
        mview = CUDSS_MVIEW_LOWER;
    } else if (m_cudssMatType == CuDSSMatrixType::SPD) {
        mtype = CUDSS_MTYPE_SPD;
        mview = CUDSS_MVIEW_LOWER;
    }
    cudssIndexBase_t ibase = CUDSS_BASE_ZERO;
    cuDSSCheckError(cudssMatrixCreateCsr(
        &m_Matrix, numRows, numCols, numNZ, 
        rowPtrs, nullptr, colIndices, AValues, 
        m_IndexType, m_ValueType, 
        mtype, mview, ibase
    ), "create cuDSS matrix");
    
    /* Create the cuDSS dense matrices for RHS and solution */
    cuDSSCheckError(cudssMatrixCreateDn(
        &m_RHS, (int64_t)numRows, 1, (int64_t)numRows, bValues, m_ValueType, CUDSS_LAYOUT_COL_MAJOR
    ), "create cuDSS right-hand side");
    cuDSSCheckError(cudssMatrixCreateDn(
        &m_Solution, (int64_t)numRows, 1, (int64_t)numRows, xValues, m_ValueType, CUDSS_LAYOUT_COL_MAJOR
    ), "create cuDSS solution");
    
    /* Symbolic factorization */
    cuDSSCheckError(cudssExecute(
        m_Handle, CUDSS_PHASE_ANALYSIS, m_Config, m_Data,
        m_Matrix, m_Solution, m_RHS
    ), "cuDSS symbolic factorization");

    /* Set hybrid device memory limit after ANALYSIS, before FACTORIZATION.
     * CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN and CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT must be
     * queried/set per device: call cudaSetDevice(dev) before each cudssDataGet/cudssConfigSet. */
    if (m_hybridMemoryMode) {
        const int numDev = m_useMultiGPU ? static_cast<int>(m_deviceIndices.size()) : 1;
        std::vector<int64_t> minPerDevice(static_cast<size_t>(numDev), 0);
        for (int i = 0; i < numDev; i++) {
            int dev = m_useMultiGPU ? m_deviceIndices[i] : 0;
            if (m_useMultiGPU) {
                cudaCheckError(cudaSetDevice(dev), "set device for hybrid memory query");
            }
            size_t sizeWritten = 0;
            int64_t minThisDev = 0;
            cudssStatus_t status = cudssDataGet(m_Handle, m_Data, CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN,
                                                &minThisDev, sizeof(minThisDev), &sizeWritten);
            if (status == CUDSS_STATUS_SUCCESS) {
                minPerDevice[static_cast<size_t>(i)] = minThisDev;
                if (m_verbose) {
                    opserr << "INFO: CuDSSLinSolver::setupMatrices() - "
                           << "dev " << dev << " CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN = " << minThisDev << " bytes" << endln;
                }
            }
        }
        if (m_useMultiGPU && numDev > 0) {
            cudaCheckError(cudaSetDevice(m_deviceIndices[0]), "restore first device after hybrid query");
        }

        for (int i = 0; i < numDev; i++) {
            int64_t limitBytes = minPerDevice[static_cast<size_t>(i)];
            // One value in list = same limit for all devices; else per-device by index
            size_t limitIdx = (m_hybridDeviceMemoryLimits.size() == 1) ? 0 : static_cast<size_t>(i);
            if (limitIdx < m_hybridDeviceMemoryLimits.size() && m_hybridDeviceMemoryLimits[limitIdx] > 0) {
                limitBytes = static_cast<int64_t>(m_hybridDeviceMemoryLimits[limitIdx]);
                if (limitBytes < minPerDevice[static_cast<size_t>(i)] && m_verbose) {
                    opserr << "WARNING: CuDSSLinSolver::setupMatrices() - dev " << i
                           << " user limit (" << limitBytes << ") < min (" << minPerDevice[static_cast<size_t>(i)]
                           << "); using min" << endln;
                    limitBytes = minPerDevice[static_cast<size_t>(i)];
                }
            }
            if (limitBytes <= 0) continue;
            if (m_useMultiGPU) {
                cudaCheckError(cudaSetDevice(m_deviceIndices[i]), "set device for hybrid limit");
            }
            cuDSSCheckError(cudssConfigSet(m_Config, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT,
                                          &limitBytes, sizeof(limitBytes)),
                           "set hybrid device memory limit");
            if (m_verbose) {
                opserr << "INFO: CuDSSLinSolver::setupMatrices() - dev " << (m_useMultiGPU ? m_deviceIndices[i] : 0)
                       << " hybrid limit = " << limitBytes << " bytes" << endln;
            }
        }
        if (m_useMultiGPU && numDev > 0) {
            cudaCheckError(cudaSetDevice(m_deviceIndices[0]), "restore first device after hybrid limit");
        }
    }

    if (m_verbose) {
        opserr << "INFO: CuDSSLinSolver::setupMatrices() - "
               << "All matrices created and symbolic factorization complete" << endln;
    }
    #endif // _CUDSS

    return 0;
}

// OpenSees API for creating CuDSS solver
#ifdef _CUDSS

struct CuDSSConfig {
    std::string precision = "dDDI";
    bool verbose = false;
    bool hybridMemoryMode = false;        // Hybrid host/device memory mode
    std::vector<size_t> hybridDeviceMemoryLimits;   // Per-device limit for hybrid memory (empty = use heuristic; one value = same for all)
    bool hybridExecuteMode = false;       // Hybrid host/device execute mode
    bool multiThreadingMode = false;      // OpenMP multi-threading mode (requires OpenMP at build time)
    std::string threadingLibPath = "/usr/lib/x86_64-linux-gnu/libcudss_mtlayer_gomp.so";  // Threading layer library path ("NULL" = pass NULL to cuDSS)
    std::string cudssMatTypeStr = "full"; // full | symmetric | spd (symmetric/spd use lower storage in SOE)
    std::string parallelMode = "single";  // single | multiGPU | MGMN (parallelism across processes/GPUs)
    int distributed = 0;                  // For MGMN: 0 = root-only (gather-scatter), 1 = row-wise distributed
    std::vector<int> deviceIndices;       // For multiGPU: empty = use all devices; else list of device IDs
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
    {"parallelMode", [](CuDSSConfig& config) { 
        const char* value = OPS_GetString();
        if (value) {
            std::string s(value);
            if (s == "single" || s == "multiGPU" || s == "MGMN") {
                config.parallelMode = s;
            } else {
                throw std::invalid_argument("parallelMode must be single, multiGPU, or MGMN");
            }
        }
    }},
    {"distributed", [](CuDSSConfig& config) { 
        int numData = 1;
        int flag = 0;
        if (OPS_GetIntInput(&numData, &flag) == 0) {
            if (flag != 0 && flag != 1) throw std::invalid_argument("distributed must be 0 or 1");
            config.distributed = flag;
        }
    }},
    {"devices", [](CuDSSConfig& config) { 
        const char* value = OPS_GetString();
        if (!value) return;
        config.deviceIndices.clear();
        if (strcmp(value, "all") == 0) return;  // empty = use all devices
        int id = atoi(value);
        config.deviceIndices.push_back(id);
        int numData = 1;
        while (OPS_GetNumRemainingInputArgs() > 0) {
            int next = 0;
            if (OPS_GetIntInput(&numData, &next) != 0) break;
            config.deviceIndices.push_back(next);
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

        // Validation: parallelMode vs single-process vs OpenSeesMP (works for Tcl and Python)
        int np = getNumProcesses();
        bool isParallel = (np > 1);
        if (config.parallelMode == "MGMN" && !isParallel) {
            opserr << "ERROR: CuDSSParameterParser::parseParameters() - "
                   << "parallelMode MGMN requires OpenSeesMP (multiple processes). Use single or multiGPU for single-process runs." << endln;
            return false;
        }
        if (config.parallelMode == "multiGPU" && isParallel) {
            opserr << "ERROR: CuDSSParameterParser::parseParameters() - "
                   << "parallelMode multiGPU is for single-process multi-GPU only. In OpenSeesMP use parallelMode MGMN." << endln;
            return false;
        }
        if (config.parallelMode == "MGMN" && (config.hybridMemoryMode || config.hybridExecuteMode)) {
            opserr << "ERROR: CuDSSParameterParser::parseParameters() - "
                   << "hybridMemoryMode and hybridExecuteMode are not allowed with parallelMode MGMN." << endln;
            return false;
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
    opserr << "  -parallelMode <single|multiGPU|MGMN>  Parallelism across processes/GPUs (default: single)" << endln;
    opserr << "                                  single: one process, one GPU" << endln;
    opserr << "                                  multiGPU: one process, multiple GPUs" << endln;
    opserr << "                                  MGMN: OpenSeesMP, multi-GPU multi-node (requires getNP > 1)" << endln;
    opserr << "  -distributed <0|1>               For MGMN only: 0 = root-only gather-scatter (default), 1 = row-wise distributed" << endln;
    opserr << "  -devices <all|id1 [id2 ...]>     For multiGPU only: GPU IDs to use (default: all)" << endln;
    opserr << "  -hybridMemoryMode <0|1>         Hybrid host/device memory mode (default: 0)" << endln;
    opserr << "  -hybridDeviceMemoryLimit <bytes1 [bytes2 ...]> Per-device limit for hybrid memory (one value=all devices; 0=min)" << endln;
    opserr << "  -hybridExecuteMode <0|1>        Hybrid host/device execute mode (default: 0)" << endln;
    opserr << "  -multiThreadingMode <0|1>       OpenMP multi-threading mode (default: 0; requires OpenMP at build time)" << endln;
    opserr << "  -threadingLibPath <path|NULL>   Path to threading layer (when multiThreadingMode is enabled)" << endln;
    opserr << "                                  (default: /usr/lib/x86_64-linux-gnu/libcudss_mtlayer_gomp.so," << endln;
    opserr << "                                   use 'NULL' to let cuDSS use CUDSS_THREADING_LIB env var)" << endln;
    opserr << "  -matrixType <full|symmetric|spd> Matrix type: full (default), symmetric, or spd" << endln;
    opserr << "                                  (symmetric and spd use lower storage; halves matrix memory)" << endln;
    opserr << "Notes:" << endln;
    opserr << "  - hybridMemoryMode and hybridExecuteMode are mutually exclusive; hybridExecute mode is not allowed with parallelMode MGMN" << endln;
    opserr << "  - MGMN is only valid in OpenSeesMP (getNP > 1); multiGPU is only valid for single process" << endln;
    opserr << "  - When multiThreadingMode is enabled: export OMP_NUM_THREADS=<n> to control thread count" << endln;
}

// Factory function to create CuDSS solver from parsed config
CudaGenBcsrLinSolver* createCuDSSSolverFromConfig(const CuDSSConfig& config) {
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

    bool useMultiGPU = (config.parallelMode == "multiGPU");
    return new CuDSSLinSolver(
        precision, 
        config.verbose,
        config.hybridMemoryMode,
        config.hybridDeviceMemoryLimits,
        config.hybridExecuteMode,
        config.multiThreadingMode,
        config.threadingLibPath.c_str(),
        cudssMatType,
        useMultiGPU,
        config.deviceIndices
    );
}

// Factory function that parses OPS arguments and creates solver
CudaGenBcsrLinSolver* createCuDSSSolverFromParser() {
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

void* OPS_CuDSSLinSolver()
{
    CuDSSConfig config;

    // Expand dict to CLI args if present ({"key": val} -> "-key", val)
    if (OPS_GetNumRemainingInputArgs() == 1) {
        (void)OPS_ExpandDictArgs();
    }

    // Parse CLI-style parameters (after any normalization).
    if (!CuDSSParameterParser::parseParameters(config)) {
        opserr << "WARNING: OPS_CuDSSLinSolver() - "
               << "Failed to parse parameters, using defaults" << endln;
        opserr << "For valid parameters, use:" << endln;
        CuDSSParameterParser::printUsageInfo();
    }

    // Validate that hybrid modes are mutually exclusive
    if (config.hybridMemoryMode && config.hybridExecuteMode) {
        opserr << "ERROR: OPS_CuDSSLinSolver() - "
               << "hybridMemoryMode and hybridExecuteMode are mutually exclusive. "
               << "Only one can be enabled at a time." << endln;
        return nullptr;
    }

    // Validate that hybridDeviceMemoryLimits is only used with hybridMemoryMode
    if (!config.hybridDeviceMemoryLimits.empty() && !config.hybridMemoryMode) {
        opserr << "WARNING: OPS_CuDSSLinSolver() - "
               << "hybridDeviceMemoryLimit is only valid with hybridMemoryMode enabled. "
               << "Ignoring hybridDeviceMemoryLimit." << endln;
    }

    CudaGenBcsrLinSolver* solver = nullptr;
    try {
        solver = createCuDSSSolverFromConfig(config);
    } catch (const std::exception& e) {
        opserr << "ERROR: OPS_CuDSSLinSolver() - "
               << "Failed to create solver: " << e.what() << endln;
        return nullptr;
    }
    if (solver == nullptr) return nullptr;

    const int blockSize = 1;
    const bool paddingEnabled = false;
    CudaPrecision precision = solver->getPrecision();
    const bool symmetricStorage = (config.cudssMatTypeStr == "symmetric" || config.cudssMatTypeStr == "spd");
    switch (precision) {
        case CudaPrecision::dDDI:
            return CudaGenBcsrLinSOE::createDouble(*solver, blockSize, paddingEnabled, config.verbose, symmetricStorage);
        case CudaPrecision::dFFI:
            return CudaGenBcsrLinSOE::createFloat(*solver, blockSize, paddingEnabled, config.verbose, symmetricStorage);
        default:
            opserr << "ERROR: OPS_CuDSSLinSolver() - Unexpected precision mode" << endln;
            delete solver;
            return nullptr;
    }
}
#else // _CUDSS
void* OPS_CuDSSLinSolver() {
    opserr << "WARNING: OPS_CuDSSLinSolver() - "
           << "cuDSS not available" << endln;
    return nullptr;
}
#endif // _CUDSS

