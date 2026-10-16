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
** Description: cuDSS backend for device-CSR direct sparse            **
** factorization/solve. Composed by CudaCsrMatrix; not an OpenSees      **
** LinearSOESolver.                                                   **
** ****************************************************************** */

#ifndef CuDSSBackend_h
#define CuDSSBackend_h

#include "CudaUtils.h"

#include <cuda_runtime.h>
#include <cudss.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class CuDSSMatrixType { FULL, SYMMETRIC, SPD };

class CuDSSBackend
{
public:
    struct Config {
        CudaPrecision precision;
        bool verbose;
        bool hybridMemoryMode;
        std::vector<std::size_t> hybridDeviceMemoryLimits;
        bool hybridExecuteMode;
        bool multiThreadingMode;
        std::string threadingLibPath;
        CuDSSMatrixType matType;
        bool useMultiGPU;
        std::vector<int> deviceIndices;
        int deviceId; // preferred CUDA device for single-GPU (-1 = driver default)
        int irNSteps;
        double irTol;
        // When set, cuDSS work is enqueued on this caller-owned stream and the
        // caller is responsible for synchronization (pair with syncAfterSolve=false).
        cudaStream_t externalStream;
        // If false, solve() returns without synchronizing the stream.
        bool syncAfterSolve;

        Config()
            : precision(CudaPrecision::dDDI),
              verbose(false),
              hybridMemoryMode(false),
              hybridExecuteMode(false),
              multiThreadingMode(false),
#ifdef _WIN32
              threadingLibPath("cudss_mtlayer_vcomp14064_0.dll"),
#else
              threadingLibPath("/usr/lib/x86_64-linux-gnu/libcudss_mtlayer_gomp.so"),
#endif
              matType(CuDSSMatrixType::FULL),
              useMultiGPU(false),
              deviceId(-1),
              irNSteps(0),
              irTol(0.0),
              externalStream(nullptr),
              syncAfterSolve(true)
        {
        }

        // Caller-owned stream; caller synchronizes when syncAfterSolve is false.
        Config(CudaPrecision prec, cudaStream_t stream, bool syncAfterSolveIn)
            : Config()
        {
            precision = prec;
            externalStream = stream;
            syncAfterSolve = syncAfterSolveIn;
        }
    };

    explicit CuDSSBackend(const Config &config = Config());
    ~CuDSSBackend();

    CuDSSBackend(const CuDSSBackend &) = delete;
    CuDSSBackend &operator=(const CuDSSBackend &) = delete;

    cudaStream_t stream() const { return m_cudaStream; }

    /**
     * Bind CSR structure (device pointers, not owned) and prepare symbolic analysis.
     * When \a symbolicSource is null, runs full symbolic analysis on this pattern.
     * When non-null and the sparsity pattern matches, imports symbolic data from
     * \a symbolicSource; otherwise falls back to full symbolic analysis.
     */
    int bindStructure(int numRows, int numNnz, int *rowPtr, int *colIdx, void *values, void *rhs, void *solution,
                      int numRhs = 1, const CuDSSBackend *symbolicSource = nullptr);

    /** Update sparse values pointer (same structure). */
    int setMatrixValues(void *values);

    /** First numeric factorization after bindStructure. */
    int factorize(void *values, void *rhs, void *solution, int numRhs = 1);

    /** Refactorize with updated coefficients. */
    int refactorize(void *values, void *rhs, void *solution, int numRhs = 1);

    /** Solve using existing factorization; updates RHS/solution pointers. */
    int solve(void *rhs, void *solution, int numRhs = 1);

    /**
     * Column stride (leading dimension, in values) of the dense RHS and solution
     * blocks for  numRhs right-hand sides. One RHS:  numRows. Several:
     *  numRows rounded up so that every column starts on a 16-byte boundary
     * (cuDSS faults with "misaligned address" when column 2 of a double block
     * starts at an odd offset). Callers lay out multi-RHS buffers with this stride.
     */
    static int64_t denseLeadingDim(int64_t numRows, int numRhs, std::size_t valueBytes)
    {
        if (numRhs <= 1) {
            return numRows;
        }
        const int64_t perAlign = static_cast<int64_t>(16 / valueBytes);
        return (numRows + perAlign - 1) / perAlign * perAlign;
    }

    /**
     * Highest completed setup milestone (ordered for >= comparisons).
     * Names mirror cuDSS execute phases in cudssPhase_t; PatternBound is the
     * pre-ANALYSIS state used while importing shared symbolic data.
     */
    enum class PhaseStatus : std::uint8_t {
        Unbound,
        PatternBound,
        AnalysisComplete,       // CUDSS_PHASE_ANALYSIS
        FactorizationComplete,  // CUDSS_PHASE_FACTORIZATION
    };

    PhaseStatus getPhaseStatus() const { return m_phaseStatus; }

private:
    Config m_config;

    void initHandle();
    void destroyHandle();
    int applyHybridMemoryLimits();
    int recreateDenseDescriptors(int numRows, void *rhs, void *solution, int numRhs);
    void destroyMatrixObjects();

    cudssHandle_t m_Handle;
    cudaStream_t m_cudaStream;
    cudssConfig_t m_Config;
    cudssData_t m_Data;
    cudssMatrix_t m_Matrix;
    cudssMatrix_t m_RHS;
    cudssMatrix_t m_Solution;

    cudssDataType_t m_ValueType;
    cudssDataType_t m_OffsetType;
    cudssDataType_t m_IndexType;

    int m_numRows;
    int m_numNnz;
    int m_numRhs;
    const int *m_rowPtr = nullptr;
    const int *m_colIdx = nullptr;
    PhaseStatus m_phaseStatus = PhaseStatus::Unbound;
    bool m_ownsStream;
    std::vector<int> m_deviceIndices;

    int bindPattern(int numRows, int numNnz, int *rowPtr, int *colIdx, void *values, void *rhs, void *solution,
                    int numRhs);
    int runSymbolicAnalysis();
    bool matchesSparsityPattern(const CuDSSBackend &other, int numRows, int numNnz, const int *rowPtr,
                                const int *colIdx) const;
    int importSymbolicAnalysisFrom(const CuDSSBackend &source);
};

#endif
