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
** Description: cuDSS backend for device-CSR direct solve. **
** ****************************************************************** */

#include "CuDSSBackend.h"
#include <OPS_Globals.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
// A multi-RHS block must start on a 16-byte boundary; its later columns are kept
// aligned by denseLeadingDim. Checked here so that a misaligned block fails with a
// message instead of an asynchronous "misaligned address" at the next stream sync.
bool denseBlockAligned(const char *where, const void *rhs, const void *solution, int numRhs)
{
    if (numRhs <= 1) {
        return true;
    }
    const bool ok = reinterpret_cast<std::uintptr_t>(rhs) % 16 == 0 &&
                    reinterpret_cast<std::uintptr_t>(solution) % 16 == 0;
    if (!ok) {
        opserr << "ERROR CuDSSBackend::" << where << "() - " << numRhs
               << "-RHS block is not 16-byte aligned\n";
    }
    return ok;
}
}  // namespace

#if CUDSS_VERSION < 800
#error "OpenSees-CUDA requires cuDSS 0.8.0 or later"
#endif

using namespace CudaUtils;

CuDSSBackend::CuDSSBackend(const Config &config)
    : m_config(config),
      m_Handle(nullptr),
      m_cudaStream(nullptr),
      m_Config(nullptr),
      m_Data(nullptr),
      m_Matrix(nullptr),
      m_RHS(nullptr),
      m_Solution(nullptr),
      m_OffsetType(CUDSS_R_32I),
      m_IndexType(CUDSS_R_32I),
      m_numRows(0),
      m_numNnz(0),
      m_numRhs(1),
      m_phaseStatus(PhaseStatus::Unbound),
      m_ownsStream(false)
{
    if (!isUniformPrecision(m_config.precision)) {
        throw std::invalid_argument("CuDSSBackend only supports uniform precision (dDDI or dFFI)");
    }
    m_ValueType = (m_config.precision == CudaPrecision::dFFI) ? CUDSS_R_32F : CUDSS_R_64F;
    initHandle();
}

CuDSSBackend::~CuDSSBackend()
{
    // Sync only streams we own. An external stream may already have been
    // destroyed by its owner; cudaDeviceSynchronize still drains device work.
    if (m_cudaStream != nullptr && m_ownsStream) {
        cudaStreamSynchronize(m_cudaStream);
    }
    cudaDeviceSynchronize();
    destroyMatrixObjects();
    if (m_Data != nullptr) {
        cuDSSCheckError(cudssDataDestroy(m_Handle, m_Data), "destroy cuDSS solver data", false);
        m_Data = nullptr;
    }
    if (m_Config != nullptr) {
        cuDSSCheckError(cudssConfigDestroy(m_Config), "destroy cuDSS config", false);
        m_Config = nullptr;
    }
    if (m_Handle != nullptr) {
        cuDSSCheckError(cudssDestroy(m_Handle), "destroy cuDSS handle", false);
        m_Handle = nullptr;
    }
    if (m_cudaStream != nullptr && m_ownsStream) {
        cudaStreamSynchronize(m_cudaStream);
        cudaStreamDestroy(m_cudaStream);
    }
    m_cudaStream = nullptr;
}

void CuDSSBackend::initHandle()
{
    if (m_config.externalStream != nullptr) {
        m_cudaStream = m_config.externalStream;
        m_ownsStream = false;
    } else {
        cudaCheckError(cudaStreamCreate(&m_cudaStream), "create CUDA stream");
        m_ownsStream = true;
    }

    if (m_config.useMultiGPU) {
        m_deviceIndices = m_config.deviceIndices;
        if (m_deviceIndices.empty()) {
            int count = 0;
            cudaCheckError(cudaGetDeviceCount(&count), "get device count");
            if (count <= 0) {
                throw std::runtime_error("no CUDA devices for multi-GPU cuDSS");
            }
            m_deviceIndices.resize(static_cast<std::size_t>(count));
            for (int i = 0; i < count; ++i) {
                m_deviceIndices[static_cast<std::size_t>(i)] = i;
            }
        }
        cudaCheckError(cudaSetDevice(m_deviceIndices.front()), "set device for multi-GPU");
        cuDSSCheckError(
            cudssCreateMg(&m_Handle, static_cast<int>(m_deviceIndices.size()), m_deviceIndices.data()),
            "create cuDSS MG handle");
    } else {
        if (m_config.deviceId >= 0) {
            cudaCheckError(cudaSetDevice(m_config.deviceId), "set preferred CUDA device");
        } else if (!m_config.deviceIndices.empty()) {
            cudaCheckError(cudaSetDevice(m_config.deviceIndices.front()), "set CUDA device from -devices");
        }
        cuDSSCheckError(cudssCreate(&m_Handle), "create cuDSS handle");
    }
    cuDSSCheckError(cudssSetStream(m_Handle, m_cudaStream), "set CUDA stream");

    if (m_config.multiThreadingMode) {
        const char *threadingLib =
            (m_config.threadingLibPath == "NULL") ? nullptr : m_config.threadingLibPath.c_str();
        cudssStatus_t status = cudssSetThreadingLayer(m_Handle, threadingLib);
        if (status != CUDSS_STATUS_SUCCESS && m_config.verbose) {
            opserr << "WARNING CuDSSBackend::initHandle() - cudssSetThreadingLayer failed\n";
        }
    }

    cuDSSCheckError(cudssConfigCreate(&m_Config), "create cuDSS solver configuration");
    cuDSSCheckError(cudssDataCreate(m_Handle, &m_Data), "create cuDSS solver data");

    if (m_config.useMultiGPU) {
        int deviceCount = static_cast<int>(m_deviceIndices.size());
        cuDSSCheckError(
            cudssConfigSet(m_Config, CUDSS_CONFIG_DEVICE_COUNT, &deviceCount, sizeof(deviceCount)),
            "set device count");
        cuDSSCheckError(
            cudssConfigSet(m_Config, CUDSS_CONFIG_DEVICE_INDICES, m_deviceIndices.data(),
                           deviceCount * static_cast<int>(sizeof(int))),
            "set device indices");
    }

    cudssReorderingAlg_t reorderAlgorithm = CUDSS_REORDERING_ALG_DEFAULT;
    cuDSSCheckError(
        cudssConfigSet(m_Config, CUDSS_CONFIG_REORDERING_ALG, &reorderAlgorithm, sizeof(cudssReorderingAlg_t)),
        "set cuDSS reordering algorithm");

    if (m_config.hybridMemoryMode) {
        int hybridMemoryModeEnabled = 1;
        cuDSSCheckError(
            cudssConfigSet(m_Config, CUDSS_CONFIG_HYBRID_MEMORY_MODE, &hybridMemoryModeEnabled, sizeof(int)),
            "enable hybrid memory mode");
    }
    if (m_config.hybridExecuteMode) {
        int hybridExecuteModeEnabled = 1;
        cuDSSCheckError(
            cudssConfigSet(m_Config, CUDSS_CONFIG_HYBRID_EXECUTE_MODE, &hybridExecuteModeEnabled, sizeof(int)),
            "enable hybrid execute mode");
    }
    if (m_config.irNSteps > 0) {
        const int irSteps = m_config.irNSteps;
        cuDSSCheckError(
            cudssConfigSet(m_Config, CUDSS_CONFIG_IR_N_STEPS, &irSteps, sizeof(irSteps)),
            "set cuDSS iterative refinement steps");
        const double irTol = m_config.irTol;
        cuDSSCheckError(
            cudssConfigSet(m_Config, CUDSS_CONFIG_IR_TOL, &irTol, sizeof(irTol)),
            "set cuDSS iterative refinement tolerance");
    }
}

void CuDSSBackend::destroyMatrixObjects()
{
    if (m_Matrix != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_Matrix), "destroy cuDSS matrix", false);
        m_Matrix = nullptr;
    }
    if (m_RHS != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_RHS), "destroy cuDSS RHS", false);
        m_RHS = nullptr;
    }
    if (m_Solution != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_Solution), "destroy cuDSS solution", false);
        m_Solution = nullptr;
    }
    m_phaseStatus = PhaseStatus::Unbound;
    m_rowPtr = nullptr;
    m_colIdx = nullptr;
    m_numNnz = 0;
}

bool CuDSSBackend::matchesSparsityPattern(const CuDSSBackend &other, int numRows, int numNnz, const int *rowPtr,
                                          const int *colIdx) const
{
    return other.getPhaseStatus() >= PhaseStatus::AnalysisComplete && other.m_numRows == numRows &&
           other.m_numNnz == numNnz && other.m_rowPtr == rowPtr && other.m_colIdx == colIdx;
}

int CuDSSBackend::importSymbolicAnalysisFrom(const CuDSSBackend &source)
{
    if (source.getPhaseStatus() < PhaseStatus::AnalysisComplete) {
        return -1;
    }

    static const cudssDataParam_t kCopyParams[] = {CUDSS_DATA_PERM_REORDER_ROW, CUDSS_DATA_PERM_REORDER_COL,
                                                   CUDSS_DATA_PERM_ROW, CUDSS_DATA_PERM_COL, CUDSS_DATA_LU_NNZ,
                                                   CUDSS_DATA_NSUPERPANELS, CUDSS_DATA_ND_PARTITION_TREE};

    for (cudssDataParam_t param : kCopyParams) {
        size_t sizeWritten = 0;
        int64_t probe = 0;
        const cudssStatus_t probeStatus =
            cudssDataGet(source.m_Handle, source.m_Data, param, &probe, sizeof(probe), &sizeWritten);
        if (probeStatus != CUDSS_STATUS_SUCCESS || sizeWritten == 0) {
            continue;
        }

        std::vector<char> buffer(sizeWritten);
        if (cudssDataGet(source.m_Handle, source.m_Data, param, buffer.data(), sizeWritten, &sizeWritten) !=
            CUDSS_STATUS_SUCCESS) {
            continue;
        }
        if (cudssDataSet(m_Handle, m_Data, param, buffer.data(), sizeWritten) != CUDSS_STATUS_SUCCESS) {
            if (m_config.verbose) {
                opserr << "WARNING CuDSSBackend::importSymbolicAnalysisFrom() - cudssDataSet failed for param "
                       << static_cast<int>(param) << endln;
            }
            return -1;
        }
    }

    m_phaseStatus = PhaseStatus::AnalysisComplete;
    return 0;
}

int CuDSSBackend::bindPattern(int numRows, int numNnz, int *rowPtr, int *colIdx, void *values, void *rhs,
                              void *solution, int numRhs)
{
    if (m_config.useMultiGPU && !m_deviceIndices.empty()) {
        cudaCheckError(cudaSetDevice(m_deviceIndices.front()), "set device for bindPattern");
    }

    destroyMatrixObjects();
    m_numRows = numRows;
    m_numNnz = numNnz;
    m_rowPtr = rowPtr;
    m_colIdx = colIdx;

    cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
    if (m_config.matType == CuDSSMatrixType::SYMMETRIC) {
        mtype = CUDSS_MTYPE_SYMMETRIC;
        mview = CUDSS_MVIEW_LOWER;
    } else if (m_config.matType == CuDSSMatrixType::SPD) {
        mtype = CUDSS_MTYPE_SPD;
        mview = CUDSS_MVIEW_LOWER;
    }

    cuDSSCheckError(cudssMatrixCreateCsr(&m_Matrix, numRows, numRows, numNnz, rowPtr, nullptr, colIdx, values,
                                         m_OffsetType, m_IndexType, m_ValueType, mtype, mview, CUDSS_BASE_ZERO),
                    "create cuDSS CSR matrix");

    if (!denseBlockAligned("bindStructure", rhs, solution, numRhs) ||
        recreateDenseDescriptors(numRows, rhs, solution, numRhs) != 0) {
        return -1;
    }

    m_phaseStatus = PhaseStatus::PatternBound;
    return 0;
}

int CuDSSBackend::runSymbolicAnalysis()
{
    if (m_phaseStatus != PhaseStatus::PatternBound) {
        opserr << "ERROR CuDSSBackend::runSymbolicAnalysis() - CSR pattern not bound\n";
        return -1;
    }

    cuDSSCheckError(cudssExecute(m_Handle, CUDSS_PHASE_ANALYSIS, m_Config, m_Data, m_Matrix, m_Solution, m_RHS),
                    "cuDSS symbolic analysis");
    m_phaseStatus = PhaseStatus::AnalysisComplete;

    if (applyHybridMemoryLimits() != 0) {
        return -1;
    }

    if (m_config.verbose) {
        opserr << "INFO CuDSSBackend::runSymbolicAnalysis() - symbolic analysis complete\n";
    }

    return 0;
}

int CuDSSBackend::bindStructure(int numRows, int numNnz, int *rowPtr, int *colIdx, void *values, void *rhs,
                                void *solution, int numRhs, const CuDSSBackend *symbolicSource)
{
    if (symbolicSource != nullptr && matchesSparsityPattern(*symbolicSource, numRows, numNnz, rowPtr, colIdx)) {
        if (bindPattern(numRows, numNnz, rowPtr, colIdx, values, rhs, solution, numRhs) != 0) {
            return -1;
        }
        if (importSymbolicAnalysisFrom(*symbolicSource) != 0) {
            if (m_config.verbose) {
                opserr << "WARNING CuDSSBackend::bindStructure() - symbolic import failed; "
                          "running full analysis\n";
            }
            destroyMatrixObjects();
            return bindStructure(numRows, numNnz, rowPtr, colIdx, values, rhs, solution, numRhs, nullptr);
        }
        if (applyHybridMemoryLimits() != 0) {
            return -1;
        }
        if (m_config.verbose) {
            opserr << "INFO CuDSSBackend::bindStructure() - reused symbolic analysis\n";
        }
        return 0;
    }

    if (bindPattern(numRows, numNnz, rowPtr, colIdx, values, rhs, solution, numRhs) != 0) {
        return -1;
    }
    return runSymbolicAnalysis();
}

int CuDSSBackend::applyHybridMemoryLimits()
{
    if (!m_config.hybridMemoryMode) {
        return 0;
    }

    const int numDev = m_config.useMultiGPU ? static_cast<int>(m_deviceIndices.size()) : 1;
    std::vector<int64_t> minPerDevice(static_cast<std::size_t>(numDev), 0);
    for (int i = 0; i < numDev; ++i) {
        int dev = m_config.useMultiGPU ? m_deviceIndices[static_cast<std::size_t>(i)] : 0;
        if (m_config.useMultiGPU) {
            cudaCheckError(cudaSetDevice(dev), "set device for hybrid memory query");
        }
        size_t sizeWritten = 0;
        int64_t minThisDev = 0;
        cudssStatus_t status = cudssDataGet(m_Handle, m_Data, CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN,
                                            &minThisDev, sizeof(minThisDev), &sizeWritten);
        if (status == CUDSS_STATUS_SUCCESS) {
            minPerDevice[static_cast<std::size_t>(i)] = minThisDev;
        }
    }
    if (m_config.useMultiGPU && numDev > 0) {
        cudaCheckError(cudaSetDevice(m_deviceIndices.front()), "restore first device after hybrid query");
    }

    for (int i = 0; i < numDev; ++i) {
        int64_t limitBytes = minPerDevice[static_cast<std::size_t>(i)];
        const std::size_t limitIdx =
            (m_config.hybridDeviceMemoryLimits.size() == 1) ? 0 : static_cast<std::size_t>(i);
        if (limitIdx < m_config.hybridDeviceMemoryLimits.size() &&
            m_config.hybridDeviceMemoryLimits[limitIdx] > 0) {
            limitBytes = static_cast<int64_t>(m_config.hybridDeviceMemoryLimits[limitIdx]);
            if (limitBytes < minPerDevice[static_cast<std::size_t>(i)]) {
                limitBytes = minPerDevice[static_cast<std::size_t>(i)];
            }
        }
        if (limitBytes <= 0) {
            continue;
        }
        if (m_config.useMultiGPU) {
            cudaCheckError(cudaSetDevice(m_deviceIndices[static_cast<std::size_t>(i)]), "set device for hybrid limit");
        }
        cuDSSCheckError(
            cudssConfigSet(m_Config, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT, &limitBytes, sizeof(limitBytes)),
            "set hybrid device memory limit");
    }
    if (m_config.useMultiGPU && numDev > 0) {
        cudaCheckError(cudaSetDevice(m_deviceIndices.front()), "restore first device after hybrid limit");
    }
    return 0;
}

int CuDSSBackend::recreateDenseDescriptors(int numRows, void *rhs, void *solution, int numRhs)
{
    if (m_RHS != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_RHS), "destroy cuDSS RHS");
        m_RHS = nullptr;
    }
    if (m_Solution != nullptr) {
        cuDSSCheckError(cudssMatrixDestroy(m_Solution), "destroy cuDSS solution");
        m_Solution = nullptr;
    }

    m_numRows = numRows;
    m_numRhs = numRhs;

    const std::size_t valueBytes = (m_ValueType == CUDSS_R_32F) ? sizeof(float) : sizeof(double);
    const int64_t ld = denseLeadingDim(numRows, numRhs, valueBytes);
    cuDSSCheckError(cudssMatrixCreateDn(&m_RHS, static_cast<int64_t>(numRows), numRhs, ld, rhs, m_ValueType,
                                        CUDSS_LAYOUT_COL_MAJOR),
                    "create cuDSS RHS");
    cuDSSCheckError(cudssMatrixCreateDn(&m_Solution, static_cast<int64_t>(numRows), numRhs, ld, solution,
                                        m_ValueType, CUDSS_LAYOUT_COL_MAJOR),
                    "create cuDSS solution");
    return 0;
}

int CuDSSBackend::setMatrixValues(void *values)
{
    if (getPhaseStatus() < PhaseStatus::PatternBound) {
        opserr << "ERROR CuDSSBackend::setMatrixValues() - structure not bound\n";
        return -1;
    }
    cuDSSCheckError(cudssMatrixSetValues(m_Matrix, values), "set cuDSS matrix values");
    return 0;
}

int CuDSSBackend::factorize(void *values, void *rhs, void *solution, int numRhs)
{
    if (getPhaseStatus() < PhaseStatus::AnalysisComplete) {
        opserr << "ERROR CuDSSBackend::factorize() - symbolic analysis not complete\n";
        return -1;
    }

    if (!denseBlockAligned("factorize", rhs, solution, numRhs)) {
        return -1;
    }
    if (numRhs != m_numRhs) {
        if (recreateDenseDescriptors(m_numRows, rhs, solution, numRhs) != 0) {
            return -1;
        }
    } else {
        cuDSSCheckError(cudssMatrixSetValues(m_RHS, rhs), "set cuDSS RHS values for factorization");
        cuDSSCheckError(cudssMatrixSetValues(m_Solution, solution), "set cuDSS solution values for factorization");
    }

    cuDSSCheckError(cudssMatrixSetValues(m_Matrix, values), "set cuDSS matrix values for factorization");

    cuDSSCheckError(cudssExecute(m_Handle, CUDSS_PHASE_FACTORIZATION, m_Config, m_Data, m_Matrix, m_Solution,
                                 m_RHS),
                    "cuDSS numeric factorization");
    m_phaseStatus = PhaseStatus::FactorizationComplete;
    return 0;
}

int CuDSSBackend::refactorize(void *values, void *rhs, void *solution, int numRhs)
{
    if (getPhaseStatus() < PhaseStatus::FactorizationComplete) {
        opserr << "ERROR CuDSSBackend::refactorize() - not factored\n";
        return -1;
    }

    if (!denseBlockAligned("refactorize", rhs, solution, numRhs)) {
        return -1;
    }
    if (numRhs != m_numRhs) {
        if (recreateDenseDescriptors(m_numRows, rhs, solution, numRhs) != 0) {
            return -1;
        }
    } else {
        cuDSSCheckError(cudssMatrixSetValues(m_RHS, rhs), "update cuDSS RHS values");
        cuDSSCheckError(cudssMatrixSetValues(m_Solution, solution), "update cuDSS solution values");
    }

    cuDSSCheckError(cudssMatrixSetValues(m_Matrix, values), "set cuDSS matrix values for refactorization");
    cuDSSCheckError(cudssExecute(m_Handle, CUDSS_PHASE_REFACTORIZATION, m_Config, m_Data, m_Matrix,
                                 m_Solution, m_RHS),
                    "cuDSS refactorization");
    return 0;
}

int CuDSSBackend::solve(void *rhs, void *solution, int numRhs)
{
    if (getPhaseStatus() < PhaseStatus::FactorizationComplete) {
        opserr << "ERROR CuDSSBackend::solve() - matrix not factored\n";
        return -1;
    }

    if (!denseBlockAligned("solve", rhs, solution, numRhs)) {
        return -1;
    }
    if (numRhs != m_numRhs) {
        if (recreateDenseDescriptors(m_numRows, rhs, solution, numRhs) != 0) {
            return -1;
        }
    } else {
        cuDSSCheckError(cudssMatrixSetValues(m_RHS, rhs), "update cuDSS RHS for solve");
        cuDSSCheckError(cudssMatrixSetValues(m_Solution, solution), "update cuDSS solution for solve");
    }

    cuDSSCheckError(cudssExecute(m_Handle, CUDSS_PHASE_SOLVE, m_Config, m_Data, m_Matrix, m_Solution, m_RHS),
                    "cuDSS solve");
    if (m_config.syncAfterSolve) {
        cudaCheckError(cudaStreamSynchronize(m_cudaStream), "synchronize after cuDSS solve");
    }
    return 0;
}
