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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCuda/CudaGenBcsrLinSOE.h
                                                                        
// Written: gaaraujo 
// Created: 08/2025
//
// Description: This file contains the class definition for 
// CudaGenBcsrLinSOE. It stores the sparse matrix A in a fashion
// required by the CudaGenBcsrLinSolver object.
// "Gen" denotes a general interface that supports both full and
// symmetric (lower-triangle) storage; solvers can query and exploit symmetry.
//
// This is the ONLY public interface users should interact with.
// We use type erasure to hide the template implementation details from users.
// The actual template code lives in CudaGenBcsrLinSOEImpl.h, which users never see.
//

#ifndef CudaGenBcsrLinSOE_h
#define CudaGenBcsrLinSOE_h

// OpenSees includes
#include <LinearSOE.h>
#include <LinearSOESolver.h>
#include <Vector.h>
#include "CudaUtils.h"

#ifdef _CUDA
// CUDA includes
#include <cuda_runtime.h>

// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/mr/allocator.h>
#else
#include <vector>
#endif

// Forward declarations
class CudaGenBcsrLinSolver;

#ifdef _CUDA
// Pinned memory allocators for improved host-device transfer performance
using pinned_mr = thrust::universal_host_pinned_memory_resource;

template <typename T>
using pinned_allocator = thrust::mr::stateless_resource_allocator<T, pinned_mr>;

template <typename T>
using pinned_host_vector = thrust::host_vector<T, pinned_allocator<T>>;
#endif

class CudaGenBcsrLinSOE : public LinearSOE
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

    CudaGenBcsrLinSOE(int classTag, CudaGenBcsrLinSolver &theSolver, 
                      int blockSize = DEFAULT_BLOCK_SIZE, 
                      bool paddingEnabled = true,
                      bool verbose = false,
                      bool symmetricStorage = false);
    CudaGenBcsrLinSOE(int classTag);

    ~CudaGenBcsrLinSOE();
    
    // Factory methods - no templates exposed to users
    // Uniform precision modes (matrix and vector types match)
    static CudaGenBcsrLinSOE* createDouble(
        CudaGenBcsrLinSolver &theSolver, 
        int blockSize = DEFAULT_BLOCK_SIZE, 
        bool paddingEnabled = true,
        bool verbose = false,
        bool symmetricStorage = false
    );
    
    static CudaGenBcsrLinSOE* createFloat(
        CudaGenBcsrLinSolver &theSolver,
        int blockSize = DEFAULT_BLOCK_SIZE, 
        bool paddingEnabled = true,
        bool verbose = false,
        bool symmetricStorage = false
    );
    
    // Mixed-precision modes (available, but most solvers don't support these yet)
    static CudaGenBcsrLinSOE* createDoubleFloat(
        CudaGenBcsrLinSolver &theSolver,
        int blockSize = DEFAULT_BLOCK_SIZE, 
        bool paddingEnabled = true,
        bool verbose = false,
        bool symmetricStorage = false
    );
    
    static CudaGenBcsrLinSOE* createFloatDouble(
        CudaGenBcsrLinSolver &theSolver,
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
    void zeroA(void) override;
    void zeroB(void) override;
    const Vector &getX(void) override;
    const Vector &getB(void) override;
    double normRHS(void) override;   
    void setX(int loc, double value) override;
    void setX(const Vector &x) override;
    int solve(void) override;
    
    // Set and get the associated CudaGenBcsrLinSolver object
    int setCudaGenBcsrLinSolver(CudaGenBcsrLinSolver &newSolver);
    CudaGenBcsrLinSolver* getCudaGenBcsrLinSolver(void);

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
    const int* getDeviceRowPtrs(void) const;
    int* getDeviceRowPtrs(void);
    const int* getDeviceColIndices(void) const;
    int* getDeviceColIndices(void);

    // Output methods
    int saveSparseA(OPS_Stream& output, int baseIndex = 0);

    // Parallel communication methods
    int sendSelf(int commitTag, Channel &theChannel) override;   
    int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker) override;

    // Friend declarations
    friend class CudaGenBcsrLinSolver;
    
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

    // Host-device data transfer methods
    virtual void uploadVectorsToDevice(void) = 0;
    virtual void downloadSolutionFromDevice(void) = 0;
    virtual void uploadAValuesToDevice(void) = 0;
    virtual void downloadAValuesFromDevice(void) = 0;
    virtual void ensureDeviceVectorSizes(void) = 0;
    // Copy host B data to device and add into device B (async when stream non-null). For overlapped recv in Distributed.
    virtual void addToDeviceBFromHost(int n, const double* hostData, void* stream = nullptr) = 0;
    void uploadAIndicesToDevice(void);

protected:    
    // Track the status of the matrix
    MatrixStatus m_matrixStatus;
    
    // Block size
    int m_blockSize;

    // Host representation
    Vector m_X, m_B; // for interfacing with OpenSees, wraps padded vectors
    
    #ifdef _CUDA
    // Thrust containers for internal data (using pinned memory for faster host-device transfers)
    pinned_host_vector<double> m_hostX, m_hostB, m_hostAValues;
    // Subclasses must provide their own device data vectors
    // thrust::device_vector<DataType> m_deviceX, m_deviceB, m_deviceAValues;
    
    // Thrust containers for index data (using pinned memory)
    pinned_host_vector<int> m_hostCsrIndices;
    thrust::device_vector<int> m_deviceCsrIndices;
    #else
    std::vector<double> m_hostX, m_hostB, m_hostAValues;
    std::vector<int> m_hostCsrIndices;
    std::vector<int> m_deviceCsrIndices;
    #endif
    
    // Whether to pad the matrix with zeros to make it a multiple of the block size
    bool m_paddingEnabled;

    // Whether to print verbose output
    bool m_verbose;

    // Storage mode: full matrix or symmetric lower triangle
    MatrixStorageMode m_storageMode;

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
};

inline OPS_Stream& operator<<(OPS_Stream& os, CudaGenBcsrLinSOE::MatrixStatus status) {
    switch (status) {
        case CudaGenBcsrLinSOE::MatrixStatus::UNCHANGED:
            return os << "UNCHANGED";
        case CudaGenBcsrLinSOE::MatrixStatus::COEFFICIENTS_CHANGED:
            return os << "COEFFICIENTS_CHANGED";
        case CudaGenBcsrLinSOE::MatrixStatus::STRUCTURE_CHANGED:
            return os << "STRUCTURE_CHANGED";
        default:
            return os << "UNKNOWN";
    }
}

#endif