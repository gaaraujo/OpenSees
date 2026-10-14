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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCuda/CudaBcsrLinSOE.h
                                                                        
// Written: gaaraujo 
// Created: 08/2025
//
// Description: This file contains the class definition for 
// CudaBcsrLinSOE. It stores the sparse matrix A in a fashion
// required by the CudaBcsrLinSolver object.
// Supports full and symmetric (lower-triangle) storage; solvers can query
// and exploit symmetry via MatrixStorageMode / backend matrix type.
//
// This is the ONLY public interface users should interact with.
// We use type erasure to hide the template implementation details from users.
// The actual template code lives in CudaBcsrLinSOEImpl.h, which users never see.
//
// Design contract:
//   - This base class owns host-side CSR/BCSR assembly (graph -> sparse structure, addA/addB).
//   - All device storage and host/device transfers live in the *Impl subclass.
//   - A new backend (e.g. ROCm) is added by implementing a new Impl subclass.
//

#ifndef CudaBcsrLinSOE_h
#define CudaBcsrLinSOE_h

// OpenSees includes
#include <LinearSOE.h>
#include <LinearSOESolver.h>
#include <Vector.h>
#include <vector>
#include "CudaUtils.h"

#ifndef _CUDA
#error "CudaBcsrLinSOE requires a CUDA build"
#endif

// CUDA includes
#include <cuda_runtime.h>

// Thrust includes (host pinned vectors only; device buffers live in *Impl)
#include <thrust/host_vector.h>
#include <thrust/mr/device_memory_resource.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/mr/allocator.h>

// Forward declarations
class CudaBcsrLinSolver;
class CuSparseBackend;

// Pinned memory allocators for improved host-device transfer performance
using pinned_mr = thrust::universal_host_pinned_memory_resource;

template <typename T>
using pinned_allocator = thrust::mr::stateless_resource_allocator<T, pinned_mr>;

template <typename T>
using pinned_host_vector = thrust::host_vector<T, pinned_allocator<T>>;

class CudaBcsrLinSOE : public LinearSOE
{
public:
    // Constants for block size limits and efficiency thresholds
    static constexpr int MAX_BLOCK_SIZE = 32;
    static constexpr int DEFAULT_BLOCK_SIZE = 1;
    static constexpr double DEFAULT_EFFICIENCY_THRESHOLD = 0.7;
    static constexpr double MIN_DIAGONAL_VALUE_FACTOR = 1e-3;

    // Storage mode: full matrix or symmetric (lower triangle only; matches Matrix Market and Cholesky).
    enum class MatrixStorageMode {
        FULL,              // Store full matrix (default)
        SYMMETRIC_LOWER    // Store only lower triangle; addA(i,j) and addA(j,i) both update (max(i,j), min(i,j))
    };

    CudaBcsrLinSOE(int classTag, CudaBcsrLinSolver &theSolver, 
                      int blockSize = DEFAULT_BLOCK_SIZE, 
                      bool paddingEnabled = true,
                      bool verbose = false,
                      bool symmetricStorage = false);
    CudaBcsrLinSOE(int classTag);

    ~CudaBcsrLinSOE();
    
    // Factory methods - no templates exposed to users
    // Uniform precision modes (matrix and vector types match)
    static CudaBcsrLinSOE* createDouble(
        CudaBcsrLinSolver &theSolver, 
        int blockSize = DEFAULT_BLOCK_SIZE, 
        bool paddingEnabled = true,
        bool verbose = false,
        bool symmetricStorage = false
    );
    
    static CudaBcsrLinSOE* createFloat(
        CudaBcsrLinSolver &theSolver,
        int blockSize = DEFAULT_BLOCK_SIZE, 
        bool paddingEnabled = true,
        bool verbose = false,
        bool symmetricStorage = false
    );
    
    // Mixed-precision modes (available, but most solvers don't support these yet)
    static CudaBcsrLinSOE* createDoubleFloat(
        CudaBcsrLinSolver &theSolver,
        int blockSize = DEFAULT_BLOCK_SIZE, 
        bool paddingEnabled = true,
        bool verbose = false,
        bool symmetricStorage = false
    );
    
    static CudaBcsrLinSOE* createFloatDouble(
        CudaBcsrLinSolver &theSolver,
        int blockSize = DEFAULT_BLOCK_SIZE, 
        bool paddingEnabled = true,
        bool verbose = false,
        bool symmetricStorage = false
    );
    
    // This method is used by the object broker to create instances from class tags.
    static LinearSOE* createCudaLinearSOE(int classTagSOE);
    
    // Core LinearSOE interface methods
    int getNumEqn(void) const override;
    int setSize(Graph &theGraph) override;
    int addA(const Matrix &, const ID &, double fact = 1.0) override;
    int addA(const Matrix &) override;
    int addB(const Vector &, const ID &, double fact = 1.0) override;
    int setB(const Vector &, double fact = 1.0) override;
    /** Device-side setters (D2D on getCudaStream(); mirror getDevice* accessors). */
    int setDeviceB(const void *deviceSrc, int numEqn);
    int setDeviceX(const void *deviceSrc, int numEqn);
    int setDeviceAValues(const void *deviceSrc, int numNnz);
    int setDeviceRowPtrs(const int *deviceSrc);
    int setDeviceColIndices(const int *deviceSrc);
    void zeroAAssembled(void) override;
    void zeroB(void) override;
    const Vector &getX(void) override;
    const Vector &getB(void) override;
    double normRHS(void) override;   
    void setX(int loc, double value) override;
    void setX(const Vector &x) override;
    int solveAssembled(void) override;
    int formApAssembled(const Vector &p, Vector &Ap) override;
    LinearSOE *getCopy(void) const override;
    
    // Set and get the associated CudaBcsrLinSolver object
    int setCudaBcsrLinSolver(CudaBcsrLinSolver &newSolver);
    CudaBcsrLinSolver* getCudaBcsrLinSolver(void);

    // Other getters
    int getBlockSize(void) const;
    int getNumRowBlocks(void) const;
    int getNumNonZeroBlocks(void) const;
    int getNumNonZeroValues(void) const;

    // Symmetric storage: when true, only the lower triangle is stored (Matrix Market / Cholesky convention).
    bool isSymmetricStorage(void) const;
    MatrixStorageMode getMatrixStorageMode(void) const;

    // Track changes in the matrix
    enum class MatrixStatus {
        UNCHANGED, // Matrix is the same as the last solve
        COEFFICIENTS_CHANGED, // Only the coefficients of the matrix have changed
        STRUCTURE_CHANGED // Both the size and coefficients of the matrix have changed
    };
    MatrixStatus getMatrixStatus(void) const;
    void setMatrixStatus(MatrixStatus status);
    /** Shared CUDA stream for SOE device work, solver, and integrators (lazy-created). */
    void *getCudaStream(void);

    // Host CSR accessors used by DistributedCudaBcsrLinSOE gather/merge
    int getPaddedVectorSize(void) const;
    int getNumCsrIndices(void) const;
    const int *getHostCsrIndicesData(void) const;
    int *getHostCsrIndicesData(void);
    double *getHostAValuesData(void);
    const double *getHostAValuesData(void) const;
    Vector &getHostBVector(void);
    Vector &getHostXVector(void);

    /** Install a pre-built host CSR/BCSR structure (distributed setSize on workers). */
    int installHostStructure(int numEqn,
                             int blockSize,
                             int paddedSize,
                             const int *csrIndices,
                             int numIndices,
                             int numValues,
                             bool invokeSolverSetSize);

    /** Worker-only: size host X/B without installing global CSR/A (triplet gather). */
    int resizeHostVectors(int numEqn, int blockSize, int paddedSize);

    /**
     * When false, this SOE must not create CUDA streams, call cudaDeviceSynchronize,
     * allocate pinned host memory, or touch the device. Used by DistributedCuDSS workers.
     */
    void setCudaDeviceEnabled(bool enabled);
    bool isCudaDeviceEnabled(void) const { return m_cudaDeviceEnabled; }

    // Host/device authority tracking for lazy synchronization
    enum class DataLocation {
        Host,   // Host buffer is authoritative; device may be stale
        Device, // Device buffer is authoritative; host may be stale
        Both    // Host and device are in sync
    };

    // Declare primary data location after an in-place write (no copy).
    // Use Host or Device only; Both is set by sync* after a transfer.
    void setBPrimaryLocation(DataLocation loc);
    void setXPrimaryLocation(DataLocation loc);
    void setAValuesPrimaryLocation(DataLocation loc);
    void setAIndicesPrimaryLocation(DataLocation loc);
    DataLocation getAValuesPrimaryLocation(void) const { return m_aLoc; }
    DataLocation getAIndicesPrimaryLocation(void) const { return m_aIndicesLoc; }

    // When true (default), getX() syncs from device; when false, host m_X may be stale.
    void setXSyncMode(bool mode);
    bool getXSyncMode(void) const;

    // Lazy sync: copy when stale and update primary location (implemented in *Impl).
    virtual void syncBToDevice(void) = 0;
    virtual void syncBToHost(void) = 0;
    virtual void syncXToHost(void) = 0;
    virtual void syncAValuesToDevice(void) = 0;
    virtual void syncAValuesToHost(void) = 0;
    virtual void syncIndicesToDevice(void) = 0;

    // Output methods
    int saveSparseA(OPS_Stream& output, int baseIndex = 0);

    // Parallel communication methods
    int sendSelf(int commitTag, Channel &theChannel) override;   
    int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker) override;

    // Friend declarations
    friend class CudaBcsrLinSolver;
    friend class DistributedCudaBcsrLinSOE;
    
    // Required methods for subclasses
    // These are pure virtual methods that provide type-erased access to device data without exposing the template nature.
    // The solver can call these methods without knowing the specific data type (double/float).
    virtual const void* getDeviceAValues(void) const = 0;
    virtual void* getDeviceAValues(void) = 0;
    virtual const void* getDeviceX(void) const = 0;
    virtual void* getDeviceX(void) = 0;
    virtual const void* getDeviceB(void) const = 0;
    virtual void* getDeviceB(void) = 0;
    
    // Precision query method
    virtual CudaPrecision getPrecision() const = 0;

    virtual void ensureDeviceVectorSizes(void) = 0;
    virtual const int* getDeviceRowPtrs(void) const = 0;
    virtual int* getDeviceRowPtrs(void) = 0;
    virtual const int* getDeviceColIndices(void) const = 0;
    virtual int* getDeviceColIndices(void) = 0;

protected:    
    // Track the status of the matrix
    MatrixStatus m_matrixStatus;
    
    // Block size
    int m_blockSize;

    // Host representation
    Vector m_X, m_B; // for interfacing with OpenSees, wraps padded vectors
    
    // Thrust containers for internal data (using pinned memory for faster host-device transfers)
    pinned_host_vector<double> m_hostX, m_hostB, m_hostAValues;
    // Subclasses must provide their own device data vectors
    // thrust::device_vector<DataType> m_deviceX, m_deviceB, m_deviceAValues;
    
    // Thrust containers for index data (using pinned memory; device copy lives in *Impl)
    pinned_host_vector<int> m_hostCsrIndices;
    
    // Whether to pad the matrix with zeros to make it a multiple of the block size
    bool m_paddingEnabled;

    // Whether to print verbose output
    bool m_verbose;

    // Storage mode: full matrix or symmetric lower triangle
    MatrixStorageMode m_storageMode;

    // Lazy cuSPARSE backend for formAp (SpMV only; independent of direct solver / cuDSS)
    CuSparseBackend *m_spmvBackend = nullptr;
    int m_spmvStructureRows = -1;

    // Host/device authority for B, X, and A coefficient values
    DataLocation m_bLoc = DataLocation::Host;
    DataLocation m_xLoc = DataLocation::Host;
    DataLocation m_aLoc = DataLocation::Host;
    DataLocation m_aIndicesLoc = DataLocation::Host;

    // When false, getX() skips device-to-host sync; host m_X may be stale.
    bool m_xSyncMode = true;

    // Single CUDA stream for async GPU work (solver, integrator kernels, cuSPARSE).
    // Thrust host/device sync helpers block the CPU and do not use this stream.
    cudaStream_t m_cudaStream = nullptr;

    // Rank-0 DistributedCuDSS / serial CuDSS: true. Worker ranks: false (pageable X/B only).
    bool m_cudaDeviceEnabled = true;

    // Used only when m_cudaDeviceEnabled is false (avoids CUDA pinned alloc on workers).
    std::vector<double> m_pageableX;
    std::vector<double> m_pageableB;

    // Set size helper methods
    int buildStandardCSR(Graph &theGraph);
    int buildBlockCSR(Graph &theGraph);

    // Block CSR format conversion and utility methods
    int estimateBlockSize(Graph &theGraph, int nnz, double efficiency = DEFAULT_EFFICIENCY_THRESHOLD);
    int fillPaddedDiagonals(double value = 0.0, bool autoCompute = true);
    
    // Helper methods for assembly
    int addAMatrixElement(int globalRow, int globalCol, double value);
    int addAMatrixElementBlock(int globalRow, int globalCol, double value);
    int addAMatrixElementStandard(int globalRow, int globalCol, double value);
           
    // Validation methods
    bool isValidBlockSize(int blockSize) const;
    bool isValidGlobalIndex(int index) const;

    bool isMatrixEmpty(void) const;

    int ensureSpMVOperator(void);

    int copyDeviceAsync(void *dst, const void *src, std::size_t numBytes, const char *label);

    // formAp device scratch (does not use SOE B/X, so solve state is preserved)
    virtual void ensureSpmvScratchSizes(void) = 0;
    virtual void *getDeviceSpmvP(void) = 0;
    virtual void *getDeviceSpmvY(void) = 0;
    virtual void uploadSpmvPFromHost(const Vector &p, int n) = 0;
    virtual void downloadSpmvYToHost(Vector &Ap, int n) = 0;
};

inline OPS_Stream& operator<<(OPS_Stream& os, CudaBcsrLinSOE::MatrixStatus status) {
    switch (status) {
        case CudaBcsrLinSOE::MatrixStatus::UNCHANGED:
            return os << "UNCHANGED";
        case CudaBcsrLinSOE::MatrixStatus::COEFFICIENTS_CHANGED:
            return os << "COEFFICIENTS_CHANGED";
        case CudaBcsrLinSOE::MatrixStatus::STRUCTURE_CHANGED:
            return os << "STRUCTURE_CHANGED";
        default:
            return os << "UNKNOWN";
    }
}

#endif