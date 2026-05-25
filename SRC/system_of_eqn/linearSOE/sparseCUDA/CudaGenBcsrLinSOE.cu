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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCuda/CudaGenBcsrLinSOE.cpp,v $
                                                                        
                                                                        
// Written: gaaraujo 
// Created: 08/2025
//
// Description: This file contains the implementation for 
// CudaGenBcsrLinSOE. It stores the sparse matrix A in a fashion
// required by the CudaGenBcsrLinSolver object.
//

// OpenSees includes
#include <CudaGenBcsrLinSOE.h>
#include <CudaGenBcsrLinSolver.h>
#include <LinearSOE.h>
#include <Matrix.h>
#include <Graph.h>
#include <Vertex.h>
#include <VertexIter.h>
#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include <ID.h>
#include <OPS_Stream.h>

// C++ includes
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

#ifdef _CUDA
#include "CudaGenBcsrLinSOEImpl.h"
#endif

#ifdef _CUDA
// CUDA includes
#include <cuda_runtime.h>

// Thrust includes
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/memory.h>

// Bring thrust::raw_pointer_cast into scope for CUDA builds
using thrust::raw_pointer_cast;
#else
// Define a passthrough raw_pointer_cast for non-CUDA builds
template<typename T>
inline T* raw_pointer_cast(T* ptr) { return ptr; }
#endif

namespace {

    // Count the number of non-zero elements (full or symmetric lower triangle).
    // symmetricLower: when true, nnz = numVertices + numEdges (one entry per edge in lower triangle).
    int countNonZeroElements(Graph &theGraph, bool symmetricLower = false)
    {
        const int numVertices = theGraph.getNumVertex();
        if (numVertices < 0) {
            opserr << "WARNING: CudaGenBcsrLinSOE::countNonZeroElements() - "
                   << "Graph size (" << numVertices << ") < 0" << endln;
            return -1;
        }
        const int numEdges = theGraph.getNumEdge();
        if (numEdges < 0) {
            opserr << "WARNING: CudaGenBcsrLinSOE::countNonZeroElements() - "
                   << "Graph size (" << numEdges << ") < 0" << endln;
            return -1;
        }

        if (symmetricLower) {
            /* Symmetric lower: each vertex contributes 1 (diagonal), each edge 1 (stored in row max(i,j)) */
            return numVertices + numEdges;
        }
        /* Full: each edge contributes 2 non-zero elements, each vertex 1 (diagonal) */
        return 2 * numEdges + numVertices;
    }

    // Count the number of non-zero square blocks in the graph (full or symmetric lower).
    int countNonZeroBlocks(Graph &theGraph, int blockSize, bool paddingEnabled = true, bool symmetricLower = false)
    {
        if (blockSize == 1) {
            return countNonZeroElements(theGraph, symmetricLower);
        }

        int size = theGraph.getNumVertex();
        if (size < 0) {
            opserr << "WARNING: CudaGenBcsrLinSOE::countNonZeroBlocks() - "
                   << "Graph size (" << size << ") < 0" << endln;
            return -1;
        }
        if (blockSize < 1) {
            opserr << "WARNING: CudaGenBcsrLinSOE::countNonZeroBlocks() - "
                   << "Block size (" << blockSize << ") < 1" << endln;
            return -1;
        }
        if (size % blockSize != 0 && !paddingEnabled) {
            opserr << "WARNING: CudaGenBcsrLinSOE::countNonZeroBlocks() - "
                   << "Graph size (" << size << ") not divisible by "
                   << "block size (" << blockSize << ") and "
                   << "padding is disabled" << endln;
            return -1;
        }

        const int numBlockCols = (size + blockSize - 1) / blockSize;
        std::vector<int> mask(numBlockCols, -1);
        int totalNumBlocks = 0;

        for (int i = 0; i < size; i++) {
            const int blockRow = i / blockSize;
            Vertex* theVertex = theGraph.getVertexPtr(i);
            if (theVertex == nullptr) {
                opserr << "WARNING: CudaGenBcsrLinSOE::countNonZeroBlocks() - "
                       << "Vertex (" << i << ") not found in graph!" << endln;
                return -1;
            }
            const ID& adjacency = theVertex->getAdjacency();

            if (mask[blockRow] != blockRow) {
                mask[blockRow] = blockRow;
                totalNumBlocks++;
            }

            for (int j = 0; j < adjacency.Size(); j++) {
                const int blockCol = adjacency(j) / blockSize;
                if (symmetricLower && blockCol > blockRow) continue; /* lower triangle only */
                if (mask[blockCol] != blockRow) {
                    mask[blockCol] = blockRow;
                    totalNumBlocks++;
                }
            }
        }

        return totalNumBlocks;
    }
}

CudaGenBcsrLinSOE::CudaGenBcsrLinSOE(int classTag, CudaGenBcsrLinSolver &theSolver, 
                             int blockSize, bool paddingEnabled,
                             bool verbose, bool symmetricStorage)
    : LinearSOE(theSolver, classTag), 
    m_X(), m_B(), m_hostX(), m_hostB(), m_hostAValues(), 
    m_hostCsrIndices(), m_deviceCsrIndices(),
    m_blockSize(blockSize),
    m_matrixStatus(MatrixStatus::STRUCTURE_CHANGED),
    m_paddingEnabled(paddingEnabled),
    m_verbose(verbose),
    m_storageMode(symmetricStorage ? MatrixStorageMode::SYMMETRIC_LOWER : MatrixStorageMode::FULL)
{   
    // Note: theSolver.setLinearSOE(*this) should be called in derived class constructor
}

CudaGenBcsrLinSOE::CudaGenBcsrLinSOE(int classTag): LinearSOE(classTag), 
    m_X(), m_B(), m_hostX(), m_hostB(), m_hostAValues(), 
    m_hostCsrIndices(), m_deviceCsrIndices(),
    m_blockSize(DEFAULT_BLOCK_SIZE),
    m_matrixStatus(MatrixStatus::STRUCTURE_CHANGED),
    m_paddingEnabled(true),
    m_verbose(false),
    m_storageMode(MatrixStorageMode::FULL)
{
}

CudaGenBcsrLinSOE::MatrixStorageMode CudaGenBcsrLinSOE::getMatrixStorageMode(void) const
{
    return m_storageMode;
}

bool CudaGenBcsrLinSOE::isSymmetricStorage(void) const
{
    return m_storageMode == MatrixStorageMode::SYMMETRIC_LOWER;
}

CudaGenBcsrLinSOE::~CudaGenBcsrLinSOE() 
{
    
}

// Validation methods
bool CudaGenBcsrLinSOE::isValidBlockSize(int blockSize) const
{
    return blockSize > 0 && blockSize <= MAX_BLOCK_SIZE;
}

bool CudaGenBcsrLinSOE::isValidGlobalIndex(int index) const
{
    return index >= 0 && index < m_X.Size();
}

int CudaGenBcsrLinSOE::getNumEqn(void) const 
{
    return m_X.Size();
}

int CudaGenBcsrLinSOE::buildStandardCSR(Graph &theGraph)
{
    int size = theGraph.getNumVertex();
    if (size < 0) {
        opserr << "WARNING: CudaGenBcsrLinSOE::buildStandardCSR() - "
               << "Graph size (" << size << ") < 0" << endln;
        return -1;
    }

    const bool symmetricLower = (m_storageMode == MatrixStorageMode::SYMMETRIC_LOWER);
    int nnz = countNonZeroElements(theGraph, symmetricLower);
    if (nnz <= 0) {
        return nnz;
    }

    m_hostCsrIndices.resize(size + 1 + nnz);
    int *ArowPtr = raw_pointer_cast(m_hostCsrIndices.data());
    int *AcolIdx = raw_pointer_cast(m_hostCsrIndices.data()) + size + 1;

    ArowPtr[0] = 0;
    for (int row = 0; row < size; ++row) {
        Vertex *theVertex = theGraph.getVertexPtr(row);
        if (theVertex == nullptr) {
            opserr << "WARNING: CudaGenBcsrLinSOE::buildStandardCSR() - "
                   << "Vertex (" << row << ") not found in graph!" << endln;
            return -1;
        }

        const ID& theAdjacency = theVertex->getAdjacency();
        ID localColIdx(0, theAdjacency.Size() + 1);

        localColIdx.insert(theVertex->getTag());

        for (int j = 0; j < theAdjacency.Size(); ++j) {
            const int col = theAdjacency(j);
            if (symmetricLower && col > row) continue; /* lower triangle: only col <= row */
            localColIdx.insert(col);
        }

        std::copy_n(&localColIdx(0), localColIdx.Size(), AcolIdx + ArowPtr[row]);
        ArowPtr[row + 1] = ArowPtr[row] + localColIdx.Size();
    }

    if (nnz != ArowPtr[size]) {
        opserr << "WARNING: CudaGenBcsrLinSOE::buildStandardCSR() - "
               << "nnz (" << nnz << ") != ArowPtr[" << size << "]" << endln;
        return -1;
    }

    m_hostAValues.resize(nnz, 0.0);
    m_hostB.resize(size, 0.0);
    m_hostX.resize(size, 0.0);

    return 0;
}

int CudaGenBcsrLinSOE::buildBlockCSR(Graph &theGraph)
{
    // Estimate the block size if not provided
    if (m_blockSize == 0) {
        int nnz = countNonZeroElements(theGraph);
        if (nnz <= 0) {
            return nnz;
        }
        m_blockSize = estimateBlockSize(theGraph, nnz, DEFAULT_EFFICIENCY_THRESHOLD);
        if (m_verbose) {
            opserr << "INFO: CudaGenBcsrLinSOE::buildBlockCSR() - "
                   << "Automatically estimating block size for "
                   << "efficiency >= " << DEFAULT_EFFICIENCY_THRESHOLD << ". "
                   << "Estimated block size: " << m_blockSize << endln;
        }
    }

    // Special case for BlockSize = 1 - treat as regular CSR format
    if (m_blockSize == 1) {
        return buildStandardCSR(theGraph);
    }

    // Validate block size
    if (!isValidBlockSize(m_blockSize)) {
        opserr << "WARNING: CudaGenBcsrLinSOE::buildBlockCSR() - "
               << "Invalid block size (" << m_blockSize << "). "
               << "Must be between 1 and " << MAX_BLOCK_SIZE << endln;
        return -1;
    }

    // Compute the original number of equations
    const int originalSize = theGraph.getNumVertex();
    if (originalSize < 0) {
        opserr << "WARNING: CudaGenBcsrLinSOE::buildBlockCSR() - "
               << "Graph size (" << originalSize << ") < 0" << endln;
        return -1;
    }
    
    // Compute number of block rows and block columns
    const int numBlockRows = (originalSize + m_blockSize - 1) / m_blockSize;
    const int numBlockCols = numBlockRows;  // Square matrix
    
    // Compute the padded number of equations (in DOFs, not blocks)
    const int paddedSize = numBlockRows * m_blockSize;
    const bool symmetricLower = (m_storageMode == MatrixStorageMode::SYMMETRIC_LOWER);
    const int nnzBlock = countNonZeroBlocks(theGraph, m_blockSize, m_paddingEnabled, symmetricLower);
    if (nnzBlock <= 0) {
        return nnzBlock;
    }

    // Reserve space for row pointers and column indices of matrix A in block CSR format
    m_hostCsrIndices.resize(numBlockRows + 1 + nnzBlock);
    int *ArowPtr = raw_pointer_cast(m_hostCsrIndices.data());
    int *AcolIdx = raw_pointer_cast(m_hostCsrIndices.data()) + numBlockRows + 1;

    // Helper vectors used to build the block CSR format
    std::vector<int> mask(numBlockCols, -1);

    /* Note: graph vertices need to be processed ordered by their tags for 
     * the following loop to work correctly.
     */
    
    ArowPtr[0] = 0; // Start of first block row
    for (int blockRow = 0; blockRow < numBlockRows; ++blockRow) {
        ID localColIdx(0, m_blockSize);

        // Insert the diagonal block
        mask[blockRow] = blockRow;
        localColIdx.insert(blockRow);
        
        // Loop over rows within block row
        const int startRow = blockRow * m_blockSize;
        const int endRow = std::min(startRow + m_blockSize, originalSize);
        
        for (int row = startRow; row < endRow; ++row) {
            Vertex *theVertex = theGraph.getVertexPtr(row);
            if (theVertex == nullptr) {
                opserr << "WARNING: CudaGenBcsrLinSOE::buildBlockCSR() - "
                       << "Vertex (" << row << ") not found in graph!" << endln; 
                return -1;
            }
            
            const ID& theAdjacency = theVertex->getAdjacency();  // connected columns

            for (int k = 0; k < theAdjacency.Size(); ++k) {
                const int col = theAdjacency(k);
                const int blockCol = col / m_blockSize;
                if (symmetricLower && blockCol > blockRow) continue; /* lower triangle only */
                if (mask[blockCol] != blockRow) {
                    mask[blockCol] = blockRow;
                    localColIdx.insert(blockCol);
                }
            }
        }

        // Append this block row's block col indices to the global block col indices
        std::copy_n(&localColIdx(0), localColIdx.Size(), AcolIdx + ArowPtr[blockRow]);

        // Update row pointer
        ArowPtr[blockRow + 1] = ArowPtr[blockRow] + localColIdx.Size();
    }

    // Check that we built row pointers correctly
    if (nnzBlock != ArowPtr[numBlockRows]) {
        opserr << "WARNING: CudaGenBcsrLinSOE::buildBlockCSR() - "
               << "nnzBlock (" << nnzBlock << ") != ArowPtr[" << numBlockRows << "]" << endln;
        return -1;
    }

    // Reserve space for values of matrix A
    m_hostAValues.resize(nnzBlock * m_blockSize * m_blockSize, 0.0);

    // Reserve space for vectors b and x
    m_hostB.resize(paddedSize, 0.0);
    m_hostX.resize(paddedSize, 0.0);

    return 0;
}

int CudaGenBcsrLinSOE::setSize(Graph &theGraph) 
{
    // Get the original size of the system of equations
    const int originalSize = theGraph.getNumVertex();
    if (originalSize < 0) {
        opserr << "WARNING: CudaGenBcsrLinSOE::setSize() - "
               << "Graph size (" << originalSize << ") < 0" << endln;
        return -1;
    }

    // Build data structures for the matrix in either standard or block CSR format
    if (m_blockSize == 1) {
        if (buildStandardCSR(theGraph) != 0) {
            opserr << "WARNING: CudaGenBcsrLinSOE::setSize() - "
                   << "buildStandardCSR() failed" << endln;
            return -1;
        }
    } else {
        if (buildBlockCSR(theGraph) != 0) {
            opserr << "WARNING: CudaGenBcsrLinSOE::setSize() - "
                   << "buildBlockCSR() failed" << endln;
            return -1;
        }
    }

    // Create OpenSees vectors wrapping the host vectors
    m_B.setData(raw_pointer_cast(m_hostB.data()), originalSize);
    m_X.setData(raw_pointer_cast(m_hostX.data()), originalSize);
    
    // Update matrix status
    m_matrixStatus = MatrixStatus::STRUCTURE_CHANGED;


    // Get the solver
    LinearSOESolver *the_Solver = this->getSolver();
    if (the_Solver == nullptr) {
        opserr << "WARNING: CudaGenBcsrLinSOE::setSize() - "
               << "No solver set" << endln;
        return -1;
    }

    // invoke setSize() on the Solver
    int solverOK = the_Solver->setSize();
    if (solverOK < 0) {
        opserr << "WARNING: CudaGenBcsrLinSOE::setSize() - "
               << "Solver failed setSize()" << endln;
        return solverOK;
    }

    return 0;
}

// Helper methods for matrix assembly
int CudaGenBcsrLinSOE::addAMatrixElement(int globalRow, int globalCol, double value)
{
    if (!isValidGlobalIndex(globalRow) || !isValidGlobalIndex(globalCol)) {
        return -1;
    }

    // Symmetric lower storage: only store entries where row >= col.
    if (m_storageMode == MatrixStorageMode::SYMMETRIC_LOWER) {
        if (globalRow < globalCol) {
            return 0; // Skip upper triangle
        }
    }

    if (m_blockSize > 1) {
        return addAMatrixElementBlock(globalRow, globalCol, value);
    } else {
        return addAMatrixElementStandard(globalRow, globalCol, value);
    }
}

int CudaGenBcsrLinSOE::addAMatrixElementBlock(int globalRow, int globalCol, double value)
{
    const int numBlockRows = m_hostB.size() / m_blockSize;
    const int *ArowPtr = raw_pointer_cast(m_hostCsrIndices.data());
    const int *AcolIdx = raw_pointer_cast(m_hostCsrIndices.data()) + numBlockRows + 1;
    
    const int blockRow = globalRow / m_blockSize;
    const int localRow = globalRow % m_blockSize;
    const int blockCol = globalCol / m_blockSize;
    const int localCol = globalCol % m_blockSize;
    
    // Find the block index in m_AColIdxBlock where m_AColIdxBlock[k] == blockCol 
    // and k is in [m_ARowPtrBlock[blockRow], m_ARowPtrBlock[blockRow + 1])
    const int startRowPtr = ArowPtr[blockRow];
    const int endRowPtr = ArowPtr[blockRow + 1];
    for (int k = startRowPtr; k < endRowPtr; ++k) {
        if (AcolIdx[k] == blockCol) {
            // Block k holds the blockRow, blockCol block
            const int blockOffset = k * m_blockSize * m_blockSize;
            const int localOffset = localRow * m_blockSize + localCol; // row-major
            m_hostAValues[blockOffset + localOffset] += value;
            return 0;
        }
    }
    
    opserr << "WARNING: CudaGenBcsrLinSOE::addAMatrixElementBlock() - "
           << "Could not find block for row (" << globalRow << "), "
           << "col (" << globalCol << ")" << endln;
    return -1;
}

int CudaGenBcsrLinSOE::addAMatrixElementStandard(int globalRow, int globalCol, double value)
{
    int size = m_hostB.size();
    const int *ArowPtr = raw_pointer_cast(m_hostCsrIndices.data());
    const int *AcolIdx = raw_pointer_cast(m_hostCsrIndices.data()) + size + 1;
    // Find the column index in m_AColIdxBlock where m_AColIdxBlock[k] == globalCol 
    // and k is in [m_ARowPtrBlock[globalRow], m_ARowPtrBlock[globalRow + 1])
    const int startRowPtr = ArowPtr[globalRow];
    const int endRowPtr = ArowPtr[globalRow + 1];
    for (int k = startRowPtr; k < endRowPtr; ++k) {
        if (AcolIdx[k] == globalCol) {
            m_hostAValues[k] += value;
            return 0;
        }
    }
    
    opserr << "WARNING: CudaGenBcsrLinSOE::addAMatrixElementStandard() - "
           << "Could not find element for row (" << globalRow << "), "
           << "col (" << globalCol << ")" << endln;
    return -1;
}

int CudaGenBcsrLinSOE::addA(const Matrix &m, const ID &id, double fact)
{   
    // Check for a quick return
    if (fact == 0.0) {
        return 0;
    }

    const int idSize = id.Size();

    // Check that m and id are of similar size
    if (idSize != m.noRows() && idSize != m.noCols()) {
        opserr << "WARNING: CudaGenBcsrLinSOE::addA() - "
               << "Matrix and ID not of similar sizes" << endln;
        return -1;
    }
    
    auto reportAddError = [](const Matrix &m, int i, int j, int globalRow, int globalCol) {
        opserr << "WARNING: CudaGenBcsrLinSOE::addA() - "
               << "Failed to add m(" << i << ", " << j << ") = " << m(i, j)
               << " to the system of equations at location "
               << "(" << globalRow << ", " << globalCol << ")" << endln;
    };

    // Add matrix elements using the helper methods
    if (fact == 1.0) {
        for (int i = 0; i < idSize; ++i) {
            const int globalRow = id(i);
            if (!isValidGlobalIndex(globalRow)) continue;

            for (int j = 0; j < idSize; ++j) {
                const int globalCol = id(j);
                if (!isValidGlobalIndex(globalCol)) continue;
                
                if (addAMatrixElement(globalRow, globalCol, m(i, j)) != 0) {
                    reportAddError(m, i, j, globalRow, globalCol);
                    return -1;
                }
            }
        }
    } else if (fact == -1.0) {
        for (int i = 0; i < idSize; ++i) {
            const int globalRow = id(i);
            if (!isValidGlobalIndex(globalRow)) continue;

            for (int j = 0; j < idSize; ++j) {
                const int globalCol = id(j);
                if (!isValidGlobalIndex(globalCol)) continue;
                
                if (addAMatrixElement(globalRow, globalCol, -m(i, j)) != 0) {
                    reportAddError(m, i, j, globalRow, globalCol);
                    return -1;
                }
            }
        }
    } else {
        for (int i = 0; i < idSize; ++i) {
            const int globalRow = id(i);
            if (!isValidGlobalIndex(globalRow)) continue;

            for (int j = 0; j < idSize; ++j) {
                const int globalCol = id(j);
                if (!isValidGlobalIndex(globalCol)) continue;
                
                if (addAMatrixElement(globalRow, globalCol, fact * m(i, j)) != 0) {
                    reportAddError(m, i, j, globalRow, globalCol);
                    return -1;
                }
            }
        }
    }
    // Update matrix status
    if (m_matrixStatus == MatrixStatus::UNCHANGED) {
        m_matrixStatus = MatrixStatus::COEFFICIENTS_CHANGED;
    }

    return 0;
}

int CudaGenBcsrLinSOE::addA(const Matrix &m)
{
    // This method adds the entire matrix to the system
    // We need to create an ID with all the DOFs and call the main addA method
    const int numRows = m.noRows();
    const int numCols = m.noCols();
    
    if (numRows != numCols || numRows != getNumEqn()) {
        opserr << "CudaGenBcsrLinSOE::addA(Matrix) - matrix size mismatch\n";
        return -1;
    }
    
    // Create ID with all DOFs (0 to numRows-1)
    ID allDOFs(numRows);
    for (int i = 0; i < numRows; i++) {
        allDOFs(i) = i;
    }
    
    return addA(m, allDOFs, 1.0);
}

int CudaGenBcsrLinSOE::addB(const Vector &v, const ID &id, double fact)
{
    // Check for a quick return
    if (fact == 0.0) {
        return 0;
    }

    const int idSize = id.Size();

    // Check that v and id are of similar size
    if (idSize != v.Size()) {
        opserr << "WARNING: CudaGenBcsrLinSOE::addB() - "
               << "Vector and ID not of similar sizes" << endln;
        return -1;
    }

    // Add vector elements
    if (fact == 1.0) {
        for (int i = 0; i < idSize; ++i) {
            const int globalRow = id(i);
            if (isValidGlobalIndex(globalRow)) {
                m_B(globalRow) += v(i);
            }
        }
    } else if (fact == -1.0) {
        for (int i = 0; i < idSize; ++i) {
            const int globalRow = id(i);
            if (isValidGlobalIndex(globalRow)) {
                m_B(globalRow) -= v(i);
            }
        }
    } else {
        for (int i = 0; i < idSize; ++i) {
            const int globalRow = id(i);
            if (isValidGlobalIndex(globalRow)) {
                m_B(globalRow) += fact * v(i);
            }
        }
    }

    return 0;
}

int CudaGenBcsrLinSOE::setB(const Vector &v, double fact)
{
    // Check for a quick return
    if (fact == 0.0) {
        zeroB();
        return 0;
    }

    const int size = m_B.Size();
    if (size != v.Size()) {
        opserr << "WARNING: CudaGenBcsrLinSOE::setB() - "
               << "Vector size mismatch" << endln;
        return -1;
    }

    // Set vector elements
    if (fact == 1.0) {
        #ifdef _CUDA
        // Cast away const-ness to access non-const operator() that returns reference
        Vector& nonConstV = const_cast<Vector&>(v);
        thrust::copy_n(thrust::host, &nonConstV(0), size, m_hostB.begin());
        #else
        for (int i = 0; i < size; i++) {
            m_hostB[i] = v(i);
        }
        #endif
    } else if (fact == -1.0) {
        #ifdef _CUDA
        // Cast away const-ness to access non-const operator() that returns reference
        Vector& nonConstV = const_cast<Vector&>(v);
        thrust::transform(
            thrust::host, // execution policy
            &nonConstV(0), &nonConstV(0) + size, // input range
            m_hostB.begin(), // output range
            thrust::negate<double>() // unary operation
        );
        #else
        for (int i = 0; i < size; i++) {
            m_hostB[i] = -v(i);
        }
        #endif
    } else {
        #ifdef _CUDA
        // Cast away const-ness to access non-const operator() that returns reference
        Vector& nonConstV = const_cast<Vector&>(v);
        thrust::transform(
            thrust::host, // execution policy
            &nonConstV(0), &nonConstV(0) + size, // input1 range
            thrust::make_constant_iterator(fact), // input2 range
            m_hostB.begin(), // output range
            thrust::multiplies<double>() // binary operation
        );
        #else
        for (int i = 0; i < size; i++) {
            m_hostB[i] = fact * v(i);
        }
        #endif
    }

    return 0;
}

void CudaGenBcsrLinSOE::zeroA(void)
{
    #ifdef _CUDA
    thrust::fill(thrust::host, m_hostAValues.begin(), m_hostAValues.end(), 0.0);
    #else
    for (int i = 0; i < m_hostAValues.size(); i++) {
        m_hostAValues[i] = 0.0;
    }
    #endif
    
    // Update matrix status
    if (m_matrixStatus == MatrixStatus::UNCHANGED) {
        m_matrixStatus = MatrixStatus::COEFFICIENTS_CHANGED;
    }
}

void CudaGenBcsrLinSOE::zeroB(void)
{
    #ifdef _CUDA
    thrust::fill(thrust::host, m_hostB.begin(), m_hostB.end(), 0.0);
    #else
    m_B.Zero();
    #endif
}

void CudaGenBcsrLinSOE::setX(int loc, double value)
{
    if (isValidGlobalIndex(loc)) {
        m_X(loc) = value;
    }
}

void CudaGenBcsrLinSOE::setX(const Vector &x)
{
    const int size = m_X.Size();
    if (size != x.Size()) {
        opserr << "WARNING: CudaGenBcsrLinSOE::setX() - "
               << "Vector size mismatch" << endln;
        return;
    }

    m_X = x;
}

const Vector & CudaGenBcsrLinSOE::getX(void)
{
    return m_X;
}   

const Vector & CudaGenBcsrLinSOE::getB(void)
{
    return m_B;
}

double CudaGenBcsrLinSOE::normRHS(void)
{
    return m_B.Norm();
}

int CudaGenBcsrLinSOE::setCudaGenBcsrLinSolver(CudaGenBcsrLinSolver &newSolver)
{
    newSolver.setLinearSOE(*this);
    return this->LinearSOE::setSolver(newSolver);
}

// Fill padded diagonals with a user-supplied value plus an automatically computed value
int CudaGenBcsrLinSOE::fillPaddedDiagonals(double value, bool autoCompute) {
    if (m_blockSize == 1 || m_X.Size() == m_hostX.size()) {
        return 0;
    }

    const size_t blockOffset = m_hostAValues.size() - m_blockSize * m_blockSize;
    const size_t startRow = m_X.Size() % m_blockSize;

    if (startRow == 0) {
        opserr << "WARNING: CudaGenBcsrLinSOE::fillPaddedDiagonals() - "
               << "Invalid start row" << endln;
        return -1;
    }

    double repDiagValue = value;

    if (autoCompute) {
        double avgDiag = 0.0;
        double maxAbsDiag = 0.0;

        for (size_t localRow = 0; localRow < startRow; ++localRow) {
            const size_t idx = blockOffset + localRow * m_blockSize + localRow;
            const double diag = m_hostAValues[idx];
            const double absDiag = std::abs(diag);
            avgDiag += absDiag;
            if (absDiag > maxAbsDiag) maxAbsDiag = absDiag;
        }

        avgDiag /= static_cast<double>(startRow);
        const double minDiag = MIN_DIAGONAL_VALUE_FACTOR * maxAbsDiag;
        // Add the average diagonal value to the user-supplied value
        repDiagValue += (avgDiag > minDiag) ? avgDiag : minDiag;
    }

    for (size_t localRow = startRow; localRow < m_blockSize; ++localRow) {
        const size_t idx = blockOffset + localRow * m_blockSize + localRow;
        m_hostAValues[idx] = repDiagValue;
    }

    return 0;
}

int CudaGenBcsrLinSOE::solve(void)
{
    // Quick sanity check
    if (m_X.Size() == 0 || isMatrixEmpty()) {
        return 0;
    }

    // Fill diagonal entries in rows beyond the original size
    if (m_matrixStatus != MatrixStatus::UNCHANGED && m_paddingEnabled) {
        if (fillPaddedDiagonals(0.0, true) != 0) {
            opserr << "WARNING: CudaGenBcsrLinSOE::solve() - "
                   << "Failed to fill padded diagonals" << endln;
            return -1;
        }
    }

    uploadVectorsToDevice();
    if (m_matrixStatus == MatrixStatus::STRUCTURE_CHANGED) {
        uploadAValuesToDevice();
        uploadAIndicesToDevice();
    } else if (m_matrixStatus == MatrixStatus::COEFFICIENTS_CHANGED) {
        uploadAValuesToDevice();
    } else { /* pass */ }
    
    // Get the cuda solver
    CudaGenBcsrLinSolver* theCudaSolver = getCudaGenBcsrLinSolver();
    
    if (theCudaSolver != nullptr) {
        // Solve the system of equations
        int solverOk = theCudaSolver->solve();

        // Update matrix status for future solves
        if (solverOk == 0) {
            m_matrixStatus = MatrixStatus::UNCHANGED;
        }

        // Copy solution back to host
        downloadSolutionFromDevice();
        return solverOk;
    } else {
        opserr << "WARNING: CudaGenBcsrLinSOE::solve() - "
               << "No CudaGenBcsrLinSolver available" << endln;
        return -1;
    }
}

int CudaGenBcsrLinSOE::saveSparseA(OPS_Stream& output, int baseIndex)
{
    if (isMatrixEmpty()) {
        opserr << "WARNING: CudaGenBcsrLinSOE::saveSparseA() - "
               << "Matrix data is empty" << endln;
        return 0;
    }

    // Pad matrix before printing
    if (m_matrixStatus != MatrixStatus::UNCHANGED && m_paddingEnabled) {
        if (fillPaddedDiagonals(0.0, true) != 0) {
            opserr << "WARNING: CudaGenBcsrLinSOE::saveSparseA() - "
                   << "Failed to fill padded diagonals" << endln;
            return -1;
        }
    }

    const int paddedSize = m_hostB.size();
    const int numBlockRows = paddedSize / m_blockSize;
    const int originalSize = m_X.Size();
    const int paddedNnz = m_hostAValues.size();

    // Assume the header is already written to output stream
    output << "%% Block size: " << m_blockSize << "\n";
    output << "%% Original number of equations: " << originalSize << "\n";
    output << "%% Padded number of equations: " << paddedSize << "\n";
    output << paddedSize << " " << paddedSize << " " << paddedNnz << "\n";

    // Write the sparse matrix entries
    int nnz_written = 0;
    const int *ArowPtr = raw_pointer_cast(m_hostCsrIndices.data());
    const int *AcolIdx = raw_pointer_cast(m_hostCsrIndices.data()) + numBlockRows + 1;
    const double* AValues = raw_pointer_cast(m_hostAValues.data());
    if (m_blockSize > 1) { // Block CSR format
        for (int blockRow = 0; blockRow < numBlockRows; blockRow++) {
            const int rowStart = ArowPtr[blockRow];
            const int rowEnd = ArowPtr[blockRow + 1];
            for (int blockIdx = rowStart; blockIdx < rowEnd; blockIdx++) {
                const int blockCol = AcolIdx[blockIdx];
                const int blockOffset = blockIdx * m_blockSize * m_blockSize;
                const double* theBlock = AValues + blockOffset;
                for (int i = 0; i < m_blockSize; i++) {
                    const int row = blockRow * m_blockSize + i + baseIndex;
                    for (int j = 0; j < m_blockSize; j++) {
                        const int col = blockCol * m_blockSize + j + baseIndex;
                        const double val = theBlock[i * m_blockSize + j];
                        output << row << " " << col << " " << val << "\n";
                        nnz_written++;
                    }
                }
            }
        }
    } else { // Standard CSR format
        for (int i = 0, row = baseIndex; i < originalSize; i++, row++) {
            const int rowStart = ArowPtr[i];
            const int rowEnd = ArowPtr[i + 1];
            for (int idx = rowStart; idx < rowEnd; idx++) {
                const int col = AcolIdx[idx] + baseIndex;
                const double val = m_hostAValues[idx];
                output << row << " " << col << " " << val << "\n";
                nnz_written++;
            }
        }
    }

    if (nnz_written != paddedNnz) {
        opserr << "WARNING: CudaGenBcsrLinSOE::saveSparseA() - "
               << "written nnz (" << nnz_written << ") != "
               << "actual nnz (" << paddedNnz << ")" << endln;
        return -1;
    }

    return 0;
}

int CudaGenBcsrLinSOE::sendSelf(int commitTag, Channel &theChannel)
{
    return 0;
}

int CudaGenBcsrLinSOE::recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    return 0;
}

/* Utility functions for block size estimation and counting.
 * This algorithm is adapted from the implementation in SciPy's `sparsetools`:
 * https://github.com/scipy/scipy/blob/0f1fd4a7268b813fa2b844ca6038e4dfdf90084a/scipy/sparse/sparsetools/csr.h#L205-L254
 */

// Estimate the optimal block size based on sparsity pattern efficiency
int CudaGenBcsrLinSOE::estimateBlockSize(Graph &theGraph, int nnz, double efficiency)
{
    const int size = theGraph.getNumVertex();
    if (size < 0) {
        opserr << "WARNING: CudaGenBcsrLinSOE::estimateBlockSize() - "
               << "size of soe < 0" << endln;
        return -1;
    }

    if (nnz == 0) {
        return DEFAULT_BLOCK_SIZE;
    }

    if (efficiency <= 0.0 || efficiency >= 1.0) {
        opserr << "WARNING: CudaGenBcsrLinSOE::estimateBlockSize() - "
               << "efficiency must satisfy 0.0 < efficiency < 1.0" << endln;
        return DEFAULT_BLOCK_SIZE;
    }

    int listOfBlockSizes[] = {4, 3, 2};

    for (int blockSize : listOfBlockSizes) {
        if (m_paddingEnabled || size % blockSize == 0) {
            int nb = countNonZeroBlocks(theGraph, blockSize);
            if (nb > 0) {
                double e = nnz / static_cast<double>(blockSize * blockSize * nb);
                if (e > efficiency) {
                    return blockSize;
                }
            }
        }
    }

    return DEFAULT_BLOCK_SIZE;
}

// Additional method implementations
int CudaGenBcsrLinSOE::getBlockSize(void) const
{
    return m_blockSize;
}

int CudaGenBcsrLinSOE::getNumRowBlocks(void) const
{
    return m_hostB.size() / m_blockSize;
}

int CudaGenBcsrLinSOE::getNumNonZeroBlocks(void) const
{
    return m_hostCsrIndices.size() - getNumRowBlocks() - 1;
}

int CudaGenBcsrLinSOE::getNumNonZeroValues(void) const
{
    return m_hostAValues.size();
}

CudaGenBcsrLinSOE::MatrixStatus CudaGenBcsrLinSOE::getMatrixStatus(void) const
{
    return m_matrixStatus;
}

const int* CudaGenBcsrLinSOE::getDeviceRowPtrs(void) const
{
    #ifdef _CUDA
    return m_deviceCsrIndices.empty() ? nullptr : m_deviceCsrIndices.data().get();
    #else
    return nullptr;
    #endif
}

int* CudaGenBcsrLinSOE::getDeviceRowPtrs(void)
{
    #ifdef _CUDA
    return m_deviceCsrIndices.empty() ? nullptr : m_deviceCsrIndices.data().get();
    #else
    return nullptr;
    #endif
}

const int* CudaGenBcsrLinSOE::getDeviceColIndices(void) const
{
    #ifdef _CUDA
    return m_deviceCsrIndices.empty() ? nullptr : m_deviceCsrIndices.data().get() + getNumRowBlocks() + 1;
    #else
    return nullptr;
    #endif
}

int* CudaGenBcsrLinSOE::getDeviceColIndices(void)
{
    #ifdef _CUDA
    return m_deviceCsrIndices.empty() ? nullptr : m_deviceCsrIndices.data().get() + getNumRowBlocks() + 1;
    #else
    return nullptr;
    #endif
}

void CudaGenBcsrLinSOE::uploadAIndicesToDevice(void)
{
    // Use thrust for transfer
    m_deviceCsrIndices = m_hostCsrIndices;
}

bool CudaGenBcsrLinSOE::isMatrixEmpty(void) const
{
    return m_hostAValues.size() == 0 || m_hostCsrIndices.size() <= 1;
}

CudaGenBcsrLinSolver* CudaGenBcsrLinSOE::getCudaGenBcsrLinSolver(void)
{
    return dynamic_cast<CudaGenBcsrLinSolver*>(this->LinearSOE::getSolver());
}

// Factory method implementations
#ifdef _CUDA
CudaGenBcsrLinSOE* CudaGenBcsrLinSOE::createDouble(
    CudaGenBcsrLinSolver &theSolver, 
    int blockSize, 
    bool paddingEnabled,
    bool verbose,
    bool symmetricStorage
) {
    return new CudaGenBcsrLinSOEImpl<double, double>(
        theSolver, blockSize, paddingEnabled, verbose, symmetricStorage
    );
}

CudaGenBcsrLinSOE* CudaGenBcsrLinSOE::createFloat(
    CudaGenBcsrLinSolver &theSolver,
    int blockSize, 
    bool paddingEnabled,
    bool verbose,
    bool symmetricStorage
) {
    return new CudaGenBcsrLinSOEImpl<float, float>(
        theSolver, blockSize, paddingEnabled, verbose, symmetricStorage
    );
}

CudaGenBcsrLinSOE* CudaGenBcsrLinSOE::createDoubleFloat(
    CudaGenBcsrLinSolver &theSolver,
    int blockSize, 
    bool paddingEnabled,
    bool verbose,
    bool symmetricStorage
) {
    return new CudaGenBcsrLinSOEImpl<double, float>(
        theSolver, blockSize, paddingEnabled, verbose, symmetricStorage
    );
}

CudaGenBcsrLinSOE* CudaGenBcsrLinSOE::createFloatDouble(
    CudaGenBcsrLinSolver &theSolver,
    int blockSize, 
    bool paddingEnabled,
    bool verbose,
    bool symmetricStorage
) {
    return new CudaGenBcsrLinSOEImpl<float, double>(
        theSolver, blockSize, paddingEnabled, verbose, symmetricStorage
    );
}

LinearSOE* CudaGenBcsrLinSOE::createCudaLinearSOE(int classTag) {
    switch(classTag) {
        case LinSOE_TAGS_CudaBcsrLinSOE_DOUBLE:
            return new CudaGenBcsrLinSOEImpl<double, double>();  // dDDI
        case LinSOE_TAGS_CudaBcsrLinSOE_FLOAT:
            return new CudaGenBcsrLinSOEImpl<float, float>();    // dFFI
        case LinSOE_TAGS_CudaBcsrLinSOE_DOUBLE_FLOAT:
            return new CudaGenBcsrLinSOEImpl<double, float>();   // dDFI
        case LinSOE_TAGS_CudaBcsrLinSOE_FLOAT_DOUBLE:
            return new CudaGenBcsrLinSOEImpl<float, double>();   // dFDI
        default:
            return nullptr;
    }
}
#else
// Non-CUDA fallbacks
CudaGenBcsrLinSOE* CudaGenBcsrLinSOE::createDouble(
    CudaGenBcsrLinSolver &theSolver, 
    int blockSize, 
    bool paddingEnabled,
    bool verbose,
    bool symmetricStorage
) {
    (void)symmetricStorage;
    opserr << "WARNING: CudaGenBcsrLinSOE::createDouble() - CUDA not available, cannot create SOE\n";
    return nullptr;
}

CudaGenBcsrLinSOE* CudaGenBcsrLinSOE::createFloat(
    CudaGenBcsrLinSolver &theSolver,
    int blockSize, 
    bool paddingEnabled,
    bool verbose,
    bool symmetricStorage
) {
    (void)symmetricStorage;
    opserr << "WARNING: CudaGenBcsrLinSOE::createFloat() - CUDA not available, cannot create SOE\n";
    return nullptr;
}

CudaGenBcsrLinSOE* CudaGenBcsrLinSOE::createDoubleFloat(
    CudaGenBcsrLinSolver &theSolver,
    int blockSize, 
    bool paddingEnabled,
    bool verbose,
    bool symmetricStorage
) {
    (void)symmetricStorage;
    opserr << "WARNING: CudaGenBcsrLinSOE::createDoubleFloat() - CUDA not available, cannot create SOE\n";
    return nullptr;
}

CudaGenBcsrLinSOE* CudaGenBcsrLinSOE::createFloatDouble(
    CudaGenBcsrLinSolver &theSolver,
    int blockSize, 
    bool paddingEnabled,
    bool verbose,
    bool symmetricStorage
) {
    (void)symmetricStorage;
    opserr << "WARNING: CudaGenBcsrLinSOE::createFloatDouble() - CUDA not available, cannot create SOE\n";
    return nullptr;
}

LinearSOE* CudaGenBcsrLinSOE::createCudaLinearSOE(int classTag) {
    opserr << "WARNING: CudaGenBcsrLinSOE::createFromClassTag() - CUDA not available, cannot create SOE\n";
    return nullptr;
}
#endif

