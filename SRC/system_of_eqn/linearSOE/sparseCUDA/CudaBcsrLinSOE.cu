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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCuda/CudaBcsrLinSOE.cpp,v $
                                                                        
                                                                        
// Written: gaaraujo 
// Created: 08/2025
//
// Description: This file contains the implementation for 
// CudaBcsrLinSOE. It stores the sparse matrix A in a fashion
// required by the CudaBcsrLinSolver object.
//

// OpenSees includes
#include <CudaBcsrLinSOE.h>
#include <CudaBcsrLinSolver.h>
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

#include "CudaBcsrLinSOEImpl.h"
#include "CuSparseBackend.h"
#include "CudaUtils.h"

// CUDA includes
#include <cuda_runtime.h>

// Thrust (raw_pointer_cast for pinned host vectors)
#include <thrust/memory.h>

using thrust::raw_pointer_cast;
using namespace CudaUtils;

namespace {

    // Count the number of non-zero elements (full or symmetric lower triangle).
    // symmetricLower: when true, nnz = numVertices + numEdges (one entry per edge in lower triangle).
    int countNonZeroElements(Graph &theGraph, bool symmetricLower = false)
    {
        const int numVertices = theGraph.getNumVertex();
        if (numVertices < 0) {
            opserr << "WARNING: CudaBcsrLinSOE::countNonZeroElements() - "
                   << "Graph size (" << numVertices << ") < 0" << endln;
            return -1;
        }
        const int numEdges = theGraph.getNumEdge();
        if (numEdges < 0) {
            opserr << "WARNING: CudaBcsrLinSOE::countNonZeroElements() - "
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
            opserr << "WARNING: CudaBcsrLinSOE::countNonZeroBlocks() - "
                   << "Graph size (" << size << ") < 0" << endln;
            return -1;
        }
        if (blockSize < 1) {
            opserr << "WARNING: CudaBcsrLinSOE::countNonZeroBlocks() - "
                   << "Block size (" << blockSize << ") < 1" << endln;
            return -1;
        }
        if (size % blockSize != 0 && !paddingEnabled) {
            opserr << "WARNING: CudaBcsrLinSOE::countNonZeroBlocks() - "
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
                opserr << "WARNING: CudaBcsrLinSOE::countNonZeroBlocks() - "
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

CudaBcsrLinSOE::CudaBcsrLinSOE(int classTag, CudaBcsrLinSolver &theSolver, 
                             int blockSize, bool paddingEnabled,
                             bool verbose, bool symmetricStorage)
    : LinearSOE(theSolver, classTag), 
    m_X(), m_B(), m_hostX(), m_hostB(), m_hostAValues(), 
    m_hostCsrIndices(),
    m_blockSize(blockSize),
    m_matrixStatus(MatrixStatus::STRUCTURE_CHANGED),
    m_paddingEnabled(paddingEnabled),
    m_verbose(verbose),
    m_storageMode(symmetricStorage ? MatrixStorageMode::SYMMETRIC_LOWER : MatrixStorageMode::FULL)
{   
    // Note: theSolver.setLinearSOE(*this) should be called in derived class constructor
}

CudaBcsrLinSOE::CudaBcsrLinSOE(int classTag): LinearSOE(classTag), 
    m_X(), m_B(), m_hostX(), m_hostB(), m_hostAValues(), 
    m_hostCsrIndices(),
    m_blockSize(DEFAULT_BLOCK_SIZE),
    m_matrixStatus(MatrixStatus::STRUCTURE_CHANGED),
    m_paddingEnabled(true),
    m_verbose(false),
    m_storageMode(MatrixStorageMode::FULL)
{
}

CudaBcsrLinSOE::MatrixStorageMode CudaBcsrLinSOE::getMatrixStorageMode(void) const
{
    return m_storageMode;
}

bool CudaBcsrLinSOE::isSymmetricStorage(void) const
{
    return m_storageMode == MatrixStorageMode::SYMMETRIC_LOWER;
}

void CudaBcsrLinSOE::setMatrixStatus(MatrixStatus status)
{
    m_matrixStatus = status;
}

int CudaBcsrLinSOE::getPaddedVectorSize(void) const
{
    return static_cast<int>(m_hostB.size());
}

int CudaBcsrLinSOE::getNumCsrIndices(void) const
{
    return static_cast<int>(m_hostCsrIndices.size());
}

const int *CudaBcsrLinSOE::getHostCsrIndicesData(void) const
{
    return m_hostCsrIndices.empty() ? nullptr : raw_pointer_cast(m_hostCsrIndices.data());
}

int *CudaBcsrLinSOE::getHostCsrIndicesData(void)
{
    return m_hostCsrIndices.empty() ? nullptr : raw_pointer_cast(m_hostCsrIndices.data());
}

double *CudaBcsrLinSOE::getHostAValuesData(void)
{
    return m_hostAValues.empty() ? nullptr : raw_pointer_cast(m_hostAValues.data());
}

const double *CudaBcsrLinSOE::getHostAValuesData(void) const
{
    return m_hostAValues.empty() ? nullptr : raw_pointer_cast(m_hostAValues.data());
}

Vector &CudaBcsrLinSOE::getHostBVector(void)
{
    return m_B;
}

Vector &CudaBcsrLinSOE::getHostXVector(void)
{
    return m_X;
}

int CudaBcsrLinSOE::installHostStructure(int numEqn,
                                         int blockSize,
                                         int paddedSize,
                                         const int *csrIndices,
                                         int numIndices,
                                         int numValues,
                                         bool invokeSolverSetSize)
{
    if (numEqn < 0 || blockSize < 1 || paddedSize < numEqn ||
        csrIndices == nullptr || numIndices < 0 || numValues < 0) {
        opserr << "WARNING: CudaBcsrLinSOE::installHostStructure() - invalid arguments\n";
        return -1;
    }

    m_blockSize = blockSize;
    m_hostCsrIndices.resize(numIndices);
    if (numIndices > 0) {
        std::copy_n(csrIndices, numIndices, raw_pointer_cast(m_hostCsrIndices.data()));
    }
    m_hostAValues.assign(numValues, 0.0);
    m_hostB.assign(paddedSize, 0.0);
    m_hostX.assign(paddedSize, 0.0);

    m_B.setData(raw_pointer_cast(m_hostB.data()), numEqn);
    m_X.setData(raw_pointer_cast(m_hostX.data()), numEqn);

    m_matrixStatus = MatrixStatus::STRUCTURE_CHANGED;
    setBPrimaryLocation(DataLocation::Host);
    setXPrimaryLocation(DataLocation::Host);
    setAValuesPrimaryLocation(DataLocation::Host);
    setAIndicesPrimaryLocation(DataLocation::Host);

    if (!invokeSolverSetSize) {
        return 0;
    }

    LinearSOESolver *the_Solver = this->getSolver();
    if (the_Solver == nullptr) {
        opserr << "WARNING: CudaBcsrLinSOE::installHostStructure() - no solver set\n";
        return -1;
    }

    int solverOK = the_Solver->setSize();
    if (solverOK < 0) {
        opserr << "WARNING: CudaBcsrLinSOE::installHostStructure() - solver setSize failed\n";
        return solverOK;
    }
    return 0;
}

int CudaBcsrLinSOE::resizeHostVectors(int numEqn, int blockSize, int paddedSize)
{
    if (numEqn < 0 || blockSize < 1 || paddedSize < numEqn) {
        opserr << "WARNING: CudaBcsrLinSOE::resizeHostVectors() - invalid arguments\n";
        return -1;
    }

    m_blockSize = blockSize;
    m_hostCsrIndices.clear();
    m_hostCsrIndices.shrink_to_fit();
    m_hostAValues.clear();
    m_hostAValues.shrink_to_fit();

    if (!m_cudaDeviceEnabled) {
        // Pageable host storage: workers must not allocate CUDA pinned memory.
        m_hostX.clear();
        m_hostX.shrink_to_fit();
        m_hostB.clear();
        m_hostB.shrink_to_fit();
        m_pageableX.assign(static_cast<std::size_t>(paddedSize), 0.0);
        m_pageableB.assign(static_cast<std::size_t>(paddedSize), 0.0);
        m_B.setData(m_pageableB.data(), numEqn);
        m_X.setData(m_pageableX.data(), numEqn);
    } else {
        m_pageableX.clear();
        m_pageableX.shrink_to_fit();
        m_pageableB.clear();
        m_pageableB.shrink_to_fit();
        m_hostB.assign(paddedSize, 0.0);
        m_hostX.assign(paddedSize, 0.0);
        m_B.setData(raw_pointer_cast(m_hostB.data()), numEqn);
        m_X.setData(raw_pointer_cast(m_hostX.data()), numEqn);
    }

    m_matrixStatus = MatrixStatus::STRUCTURE_CHANGED;
    setBPrimaryLocation(DataLocation::Host);
    setXPrimaryLocation(DataLocation::Host);
    setAValuesPrimaryLocation(DataLocation::Host);
    setAIndicesPrimaryLocation(DataLocation::Host);
    return 0;
}

void CudaBcsrLinSOE::setCudaDeviceEnabled(bool enabled)
{
    m_cudaDeviceEnabled = enabled;
    if (!enabled) {
        m_xSyncMode = false;
    }
}

CudaBcsrLinSOE::~CudaBcsrLinSOE() 
{
    // Solver and SpMV backend may hold cuDSS/cuSPARSE state bound to
    // m_cudaStream as an external stream. Release those before destroying the
    // stream; ~LinearSOE deletes the solver object afterward.
    if (CudaBcsrLinSolver *solver = getCudaBcsrLinSolver()) {
        solver->releaseDeviceResources();
    }
    delete m_spmvBackend;
    m_spmvBackend = nullptr;

    if (!m_cudaDeviceEnabled) {
        // Worker / host-only SOE: never created a CUDA context on purpose.
        return;
    }

    if (m_cudaStream != nullptr) {
        cudaStreamSynchronize(m_cudaStream);
        cudaStreamDestroy(m_cudaStream);
        m_cudaStream = nullptr;
    }
    cudaDeviceSynchronize();
}

// Validation methods
bool CudaBcsrLinSOE::isValidBlockSize(int blockSize) const
{
    return blockSize > 0 && blockSize <= MAX_BLOCK_SIZE;
}

bool CudaBcsrLinSOE::isValidGlobalIndex(int index) const
{
    return index >= 0 && index < m_X.Size();
}

int CudaBcsrLinSOE::getNumEqn(void) const 
{
    return m_X.Size();
}

int CudaBcsrLinSOE::buildStandardCSR(Graph &theGraph)
{
    int size = theGraph.getNumVertex();
    if (size < 0) {
        opserr << "WARNING: CudaBcsrLinSOE::buildStandardCSR() - "
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
            opserr << "WARNING: CudaBcsrLinSOE::buildStandardCSR() - "
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
        opserr << "WARNING: CudaBcsrLinSOE::buildStandardCSR() - "
               << "nnz (" << nnz << ") != ArowPtr[" << size << "]" << endln;
        return -1;
    }

    m_hostAValues.resize(nnz, 0.0);
    m_hostB.resize(size, 0.0);
    m_hostX.resize(size, 0.0);

    return 0;
}

int CudaBcsrLinSOE::buildBlockCSR(Graph &theGraph)
{
    // Estimate the block size if not provided
    if (m_blockSize == 0) {
        int nnz = countNonZeroElements(theGraph);
        if (nnz <= 0) {
            return nnz;
        }
        m_blockSize = estimateBlockSize(theGraph, nnz, DEFAULT_EFFICIENCY_THRESHOLD);
        if (m_verbose) {
            opserr << "INFO: CudaBcsrLinSOE::buildBlockCSR() - "
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
        opserr << "WARNING: CudaBcsrLinSOE::buildBlockCSR() - "
               << "Invalid block size (" << m_blockSize << "). "
               << "Must be between 1 and " << MAX_BLOCK_SIZE << endln;
        return -1;
    }

    // Compute the original number of equations
    const int originalSize = theGraph.getNumVertex();
    if (originalSize < 0) {
        opserr << "WARNING: CudaBcsrLinSOE::buildBlockCSR() - "
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
                opserr << "WARNING: CudaBcsrLinSOE::buildBlockCSR() - "
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
        opserr << "WARNING: CudaBcsrLinSOE::buildBlockCSR() - "
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

int CudaBcsrLinSOE::setSize(Graph &theGraph) 
{
    // Get the original size of the system of equations
    const int originalSize = theGraph.getNumVertex();
    if (originalSize < 0) {
        opserr << "WARNING: CudaBcsrLinSOE::setSize() - "
               << "Graph size (" << originalSize << ") < 0" << endln;
        return -1;
    }

    // Build data structures for the matrix in either standard or block CSR format
    if (m_blockSize == 1) {
        if (buildStandardCSR(theGraph) != 0) {
            opserr << "WARNING: CudaBcsrLinSOE::setSize() - "
                   << "buildStandardCSR() failed" << endln;
            return -1;
        }
    } else {
        if (buildBlockCSR(theGraph) != 0) {
            opserr << "WARNING: CudaBcsrLinSOE::setSize() - "
                   << "buildBlockCSR() failed" << endln;
            return -1;
        }
    }

    // Create OpenSees vectors wrapping the host vectors
    m_B.setData(raw_pointer_cast(m_hostB.data()), originalSize);
    m_X.setData(raw_pointer_cast(m_hostX.data()), originalSize);
    
    // Update matrix status
    m_matrixStatus = MatrixStatus::STRUCTURE_CHANGED;

    // Reset host/device authority after structural rebuild
    setBPrimaryLocation(DataLocation::Host);
    setXPrimaryLocation(DataLocation::Host);
    setAValuesPrimaryLocation(DataLocation::Host);
    setAIndicesPrimaryLocation(DataLocation::Host);

    // Get the solver
    LinearSOESolver *the_Solver = this->getSolver();
    if (the_Solver == nullptr) {
        opserr << "WARNING: CudaBcsrLinSOE::setSize() - "
               << "No solver set" << endln;
        return -1;
    }

    // invoke setSize() on the Solver
    int solverOK = the_Solver->setSize();
    if (solverOK < 0) {
        opserr << "WARNING: CudaBcsrLinSOE::setSize() - "
               << "Solver failed setSize()" << endln;
        return solverOK;
    }

    return 0;
}

// Helper methods for matrix assembly
int CudaBcsrLinSOE::addAMatrixElement(int globalRow, int globalCol, double value)
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

int CudaBcsrLinSOE::addAMatrixElementBlock(int globalRow, int globalCol, double value)
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
    
    opserr << "WARNING: CudaBcsrLinSOE::addAMatrixElementBlock() - "
           << "Could not find block for row (" << globalRow << "), "
           << "col (" << globalCol << ")" << endln;
    return -1;
}

int CudaBcsrLinSOE::addAMatrixElementStandard(int globalRow, int globalCol, double value)
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
    
    opserr << "WARNING: CudaBcsrLinSOE::addAMatrixElementStandard() - "
           << "Could not find element for row (" << globalRow << "), "
           << "col (" << globalCol << ")" << endln;
    return -1;
}

int CudaBcsrLinSOE::addA(const Matrix &m, const ID &id, double fact)
{   
    // Check for a quick return
    if (fact == 0.0) {
        return 0;
    }

    if (m_aLoc == DataLocation::Device) {
        syncAValuesToHost();
    }

    const int idSize = id.Size();

    // Check that m and id are of similar size
    if (idSize != m.noRows() && idSize != m.noCols()) {
        opserr << "WARNING: CudaBcsrLinSOE::addA() - "
               << "Matrix and ID not of similar sizes" << endln;
        return -1;
    }
    
    auto reportAddError = [](const Matrix &m, int i, int j, int globalRow, int globalCol) {
        opserr << "WARNING: CudaBcsrLinSOE::addA() - "
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
    setAValuesPrimaryLocation(DataLocation::Host);

    return 0;
}

int CudaBcsrLinSOE::addA(const Matrix &m)
{
    // This method adds the entire matrix to the system
    // We need to create an ID with all the DOFs and call the main addA method
    const int numRows = m.noRows();
    const int numCols = m.noCols();
    
    if (numRows != numCols || numRows != getNumEqn()) {
        opserr << "CudaBcsrLinSOE::addA(Matrix) - matrix size mismatch\n";
        return -1;
    }
    
    // Create ID with all DOFs (0 to numRows-1)
    ID allDOFs(numRows);
    for (int i = 0; i < numRows; i++) {
        allDOFs(i) = i;
    }
    
    return addA(m, allDOFs, 1.0);
}

int CudaBcsrLinSOE::addB(const Vector &v, const ID &id, double fact)
{
    // Check for a quick return
    if (fact == 0.0) {
        return 0;
    }

    if (m_bLoc == DataLocation::Device) {
        syncBToHost();
    }

    const int idSize = id.Size();

    // Check that v and id are of similar size
    if (idSize != v.Size()) {
        opserr << "WARNING: CudaBcsrLinSOE::addB() - "
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

    setBPrimaryLocation(DataLocation::Host);
    return 0;
}

int CudaBcsrLinSOE::setB(const Vector &v, double fact)
{
    // Check for a quick return
    if (fact == 0.0) {
        zeroB();
        return 0;
    }

    const int size = m_B.Size();
    if (size != v.Size()) {
        opserr << "WARNING: CudaBcsrLinSOE::setB() - "
               << "Vector size mismatch" << endln;
        return -1;
    }

    // Set vector elements
    if (fact == 1.0) {
        for (int i = 0; i < size; i++) {
            m_hostB[i] = v(i);
        }
    } else if (fact == -1.0) {
        for (int i = 0; i < size; i++) {
            m_hostB[i] = -v(i);
        }
    } else {
        for (int i = 0; i < size; i++) {
            m_hostB[i] = fact * v(i);
        }
    }

    setBPrimaryLocation(DataLocation::Host);
    return 0;
}

int CudaBcsrLinSOE::copyDeviceAsync(void *dst, const void *src, std::size_t numBytes, const char *label)
{
    if (dst == nullptr || src == nullptr) {
        opserr << "WARNING: CudaBcsrLinSOE::" << label << " - null pointer\n";
        return -1;
    }
    if (numBytes == 0) {
        return 0;
    }
    cudaStream_t stream = static_cast<cudaStream_t>(getCudaStream());
    if (stream == nullptr) {
        opserr << "WARNING: CudaBcsrLinSOE::" << label << " - CUDA stream unavailable\n";
        return -1;
    }
    cudaCheckError(cudaMemcpyAsync(dst, src, numBytes, cudaMemcpyDeviceToDevice, stream), label);
    return 0;
}

int CudaBcsrLinSOE::setDeviceB(const void *deviceSrc, int numEqn)
{
    if (numEqn <= 0 || numEqn > getNumEqn()) {
        opserr << "WARNING: CudaBcsrLinSOE::setDeviceB() - invalid numEqn (" << numEqn << ")\n";
        return -1;
    }
    if (!isUniformPrecision(getPrecision())) {
        opserr << "WARNING: CudaBcsrLinSOE::setDeviceB() - mixed precision not supported\n";
        return -1;
    }
    ensureDeviceVectorSizes();
    const std::size_t valueBytes =
        (getPrecision() == CudaPrecision::dFFI) ? sizeof(float) : sizeof(double);
    if (copyDeviceAsync(getDeviceB(), deviceSrc, static_cast<std::size_t>(numEqn) * valueBytes,
                        "setDeviceB") != 0) {
        return -1;
    }
    setBPrimaryLocation(DataLocation::Device);
    return 0;
}

int CudaBcsrLinSOE::setDeviceX(const void *deviceSrc, int numEqn)
{
    if (numEqn <= 0 || numEqn > getNumEqn()) {
        opserr << "WARNING: CudaBcsrLinSOE::setDeviceX() - invalid numEqn (" << numEqn << ")\n";
        return -1;
    }
    if (!isUniformPrecision(getPrecision())) {
        opserr << "WARNING: CudaBcsrLinSOE::setDeviceX() - mixed precision not supported\n";
        return -1;
    }
    ensureDeviceVectorSizes();
    const std::size_t valueBytes =
        (getPrecision() == CudaPrecision::dFFI) ? sizeof(float) : sizeof(double);
    if (copyDeviceAsync(getDeviceX(), deviceSrc, static_cast<std::size_t>(numEqn) * valueBytes,
                        "setDeviceX") != 0) {
        return -1;
    }
    setXPrimaryLocation(DataLocation::Device);
    return 0;
}

int CudaBcsrLinSOE::setDeviceAValues(const void *deviceSrc, int numNnz)
{
    if (numNnz <= 0 || numNnz > getNumNonZeroValues()) {
        opserr << "WARNING: CudaBcsrLinSOE::setDeviceAValues() - invalid numNnz (" << numNnz << ")\n";
        return -1;
    }
    ensureDeviceVectorSizes();
    const std::size_t matrixBytes =
        (getPrecision() == CudaPrecision::dFFI || getPrecision() == CudaPrecision::dDFI) ? sizeof(float)
                                                                                         : sizeof(double);
    if (copyDeviceAsync(getDeviceAValues(), deviceSrc, static_cast<std::size_t>(numNnz) * matrixBytes,
                        "setDeviceAValues") != 0) {
        return -1;
    }
    if (m_matrixStatus == MatrixStatus::UNCHANGED) {
        m_matrixStatus = MatrixStatus::COEFFICIENTS_CHANGED;
    }
    setAValuesPrimaryLocation(DataLocation::Device);
    return 0;
}

int CudaBcsrLinSOE::setDeviceRowPtrs(const int *deviceSrc)
{
    if (deviceSrc == nullptr) {
        opserr << "WARNING: CudaBcsrLinSOE::setDeviceRowPtrs() - null deviceSrc\n";
        return -1;
    }
    const int numRowPtrs = getNumRowBlocks() + 1;
    if (numRowPtrs <= 1) {
        opserr << "WARNING: CudaBcsrLinSOE::setDeviceRowPtrs() - matrix structure empty\n";
        return -1;
    }
    ensureDeviceVectorSizes();
    if (copyDeviceAsync(getDeviceRowPtrs(), deviceSrc, static_cast<std::size_t>(numRowPtrs) * sizeof(int),
                        "setDeviceRowPtrs") != 0) {
        return -1;
    }
    setAIndicesPrimaryLocation(DataLocation::Device);
    return 0;
}

int CudaBcsrLinSOE::setDeviceColIndices(const int *deviceSrc)
{
    if (deviceSrc == nullptr) {
        opserr << "WARNING: CudaBcsrLinSOE::setDeviceColIndices() - null deviceSrc\n";
        return -1;
    }
    const int numColIdx = getNumNonZeroBlocks();
    if (numColIdx <= 0) {
        opserr << "WARNING: CudaBcsrLinSOE::setDeviceColIndices() - matrix structure empty\n";
        return -1;
    }
    ensureDeviceVectorSizes();
    if (copyDeviceAsync(getDeviceColIndices(), deviceSrc, static_cast<std::size_t>(numColIdx) * sizeof(int),
                        "setDeviceColIndices") != 0) {
        return -1;
    }
    setAIndicesPrimaryLocation(DataLocation::Device);
    return 0;
}

void CudaBcsrLinSOE::zeroAAssembled(void)
{
    for (size_t i = 0; i < m_hostAValues.size(); i++) {
        m_hostAValues[i] = 0.0;
    }
    
    // Update matrix status
    if (m_matrixStatus == MatrixStatus::UNCHANGED) {
        m_matrixStatus = MatrixStatus::COEFFICIENTS_CHANGED;
    }
    setAValuesPrimaryLocation(DataLocation::Host);
}

void CudaBcsrLinSOE::zeroB(void)
{
    for (size_t i = 0; i < m_hostB.size(); i++) {
        m_hostB[i] = 0.0;
    }
    setBPrimaryLocation(DataLocation::Host);
}

void CudaBcsrLinSOE::setX(int loc, double value)
{
    if (isValidGlobalIndex(loc)) {
        m_X(loc) = value;
        setXPrimaryLocation(DataLocation::Host);
    }
}

void CudaBcsrLinSOE::setX(const Vector &x)
{
    const int size = m_X.Size();
    if (size != x.Size()) {
        opserr << "WARNING: CudaBcsrLinSOE::setX() - "
               << "Vector size mismatch" << endln;
        return;
    }

    m_X = x;
    setXPrimaryLocation(DataLocation::Host);
}

void CudaBcsrLinSOE::setXSyncMode(bool mode)
{
    m_xSyncMode = mode;
}

bool CudaBcsrLinSOE::getXSyncMode(void) const
{
    return m_xSyncMode;
}

const Vector & CudaBcsrLinSOE::getX(void)
{
    if (m_xSyncMode) {
        syncXToHost();
    }
    return m_X;
}   

const Vector & CudaBcsrLinSOE::getB(void)
{
    syncBToHost();
    return m_B;
}

double CudaBcsrLinSOE::normRHS(void)
{
    syncBToHost();
    return m_B.Norm();
}

int CudaBcsrLinSOE::setCudaBcsrLinSolver(CudaBcsrLinSolver &newSolver)
{
    newSolver.setLinearSOE(*this);
    return this->LinearSOE::setSolver(newSolver);
}

// Fill padded diagonals with a user-supplied value plus an automatically computed value
int CudaBcsrLinSOE::fillPaddedDiagonals(double value, bool autoCompute) {
    if (m_blockSize == 1 || m_X.Size() == m_hostX.size()) {
        return 0;
    }

    const size_t blockOffset = m_hostAValues.size() - m_blockSize * m_blockSize;
    const size_t startRow = m_X.Size() % m_blockSize;

    if (startRow == 0) {
        opserr << "WARNING: CudaBcsrLinSOE::fillPaddedDiagonals() - "
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

int CudaBcsrLinSOE::solveAssembled(void)
{
    // Quick sanity check
    if (m_X.Size() == 0 || isMatrixEmpty()) {
        return 0;
    }

    // Fill diagonal entries in rows beyond the original size
    if (m_matrixStatus != MatrixStatus::UNCHANGED && m_paddingEnabled) {
        if (fillPaddedDiagonals(0.0, true) != 0) {
            opserr << "WARNING: CudaBcsrLinSOE::solveAssembled() - "
                   << "Failed to fill padded diagonals" << endln;
            return -1;
        }
        setAValuesPrimaryLocation(DataLocation::Host);
    }

    syncBToDevice();
    if (m_matrixStatus == MatrixStatus::STRUCTURE_CHANGED && m_aIndicesLoc == DataLocation::Host) {
        syncIndicesToDevice();
    }
    if (m_aLoc == DataLocation::Host) {
        syncAValuesToDevice();
    }
    
    // Get the cuda solver
    CudaBcsrLinSolver* theCudaSolver = getCudaBcsrLinSolver();
    
    if (theCudaSolver != nullptr) {
        // Solve the system of equations
        int solverOk = theCudaSolver->solve();

        // Update matrix status for future solves
        if (solverOk == 0) {
            m_matrixStatus = MatrixStatus::UNCHANGED;
            setXPrimaryLocation(DataLocation::Device);
        }

        return solverOk;
    } else {
        opserr << "WARNING: CudaBcsrLinSOE::solveAssembled() - "
               << "No CudaBcsrLinSolver available" << endln;
        return -1;
    }
}

int CudaBcsrLinSOE::saveSparseA(OPS_Stream& output, int baseIndex)
{
    if (isMatrixEmpty()) {
        opserr << "WARNING: CudaBcsrLinSOE::saveSparseA() - "
               << "Matrix data is empty" << endln;
        return 0;
    }

    syncAValuesToHost();

    // Pad matrix before printing
    if (m_matrixStatus != MatrixStatus::UNCHANGED && m_paddingEnabled) {
        if (fillPaddedDiagonals(0.0, true) != 0) {
            opserr << "WARNING: CudaBcsrLinSOE::saveSparseA() - "
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
        opserr << "WARNING: CudaBcsrLinSOE::saveSparseA() - "
               << "written nnz (" << nnz_written << ") != "
               << "actual nnz (" << paddedNnz << ")" << endln;
        return -1;
    }

    return 0;
}

int CudaBcsrLinSOE::sendSelf(int commitTag, Channel &theChannel)
{
    return 0;
}

int CudaBcsrLinSOE::recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    return 0;
}

/* Utility functions for block size estimation and counting.
 * This algorithm is adapted from the implementation in SciPy's `sparsetools`:
 * https://github.com/scipy/scipy/blob/0f1fd4a7268b813fa2b844ca6038e4dfdf90084a/scipy/sparse/sparsetools/csr.h#L205-L254
 */

// Estimate the optimal block size based on sparsity pattern efficiency
int CudaBcsrLinSOE::estimateBlockSize(Graph &theGraph, int nnz, double efficiency)
{
    const int size = theGraph.getNumVertex();
    if (size < 0) {
        opserr << "WARNING: CudaBcsrLinSOE::estimateBlockSize() - "
               << "size of soe < 0" << endln;
        return -1;
    }

    if (nnz == 0) {
        return DEFAULT_BLOCK_SIZE;
    }

    if (efficiency <= 0.0 || efficiency >= 1.0) {
        opserr << "WARNING: CudaBcsrLinSOE::estimateBlockSize() - "
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
int CudaBcsrLinSOE::getBlockSize(void) const
{
    return m_blockSize;
}

int CudaBcsrLinSOE::getNumRowBlocks(void) const
{
    return m_hostB.size() / m_blockSize;
}

int CudaBcsrLinSOE::getNumNonZeroBlocks(void) const
{
    return m_hostCsrIndices.size() - getNumRowBlocks() - 1;
}

int CudaBcsrLinSOE::getNumNonZeroValues(void) const
{
    return m_hostAValues.size();
}

CudaBcsrLinSOE::MatrixStatus CudaBcsrLinSOE::getMatrixStatus(void) const
{
    return m_matrixStatus;
}

void *CudaBcsrLinSOE::getCudaStream(void)
{
    if (!m_cudaDeviceEnabled) {
        opserr << "ERROR: CudaBcsrLinSOE::getCudaStream() - CUDA disabled on this SOE "
               << "(DistributedCuDSS worker ranks must not create device streams)\n";
        return nullptr;
    }
    if (m_cudaStream == nullptr) {
        cudaCheckError(cudaStreamCreate(&m_cudaStream), "create SOE CUDA stream");
    }
    return static_cast<void *>(m_cudaStream);
}

// set*PrimaryLocation: declare authority after an in-place write.
// Prefer Host or Device; Both is reserved for sync* after a transfer.
void CudaBcsrLinSOE::setBPrimaryLocation(DataLocation loc)
{
    m_bLoc = loc;
}

void CudaBcsrLinSOE::setXPrimaryLocation(DataLocation loc)
{
    m_xLoc = loc;
}

void CudaBcsrLinSOE::setAValuesPrimaryLocation(DataLocation loc)
{
    m_aLoc = loc;
}

void CudaBcsrLinSOE::setAIndicesPrimaryLocation(DataLocation loc)
{
    m_aIndicesLoc = loc;
}

bool CudaBcsrLinSOE::isMatrixEmpty(void) const
{
    return m_hostAValues.size() == 0 || m_hostCsrIndices.size() <= 1;
}

CudaBcsrLinSolver* CudaBcsrLinSOE::getCudaBcsrLinSolver(void)
{
    return dynamic_cast<CudaBcsrLinSolver*>(this->LinearSOE::getSolver());
}

int
CudaBcsrLinSOE::ensureSpMVOperator(void)
{
    if (!m_cudaDeviceEnabled) {
        opserr << "WARNING: CudaBcsrLinSOE::ensureSpMVOperator() - CUDA disabled on this SOE\n";
        return -1;
    }

    const int numRows = getNumRowBlocks();
    if (numRows <= 0) {
        return -1;
    }

    const int numNnz = (m_blockSize == 1) ? getNumNonZeroValues() : getNumNonZeroBlocks();

    ensureDeviceVectorSizes();
    syncIndicesToDevice();

    const int *rowPtrs = getDeviceRowPtrs();
    const int *colIndices = getDeviceColIndices();
    if (rowPtrs == nullptr || colIndices == nullptr) {
        opserr << "WARNING: CudaBcsrLinSOE::ensureSpMVOperator() - null CSR indices\n";
        return -1;
    }

    if (m_spmvBackend == nullptr) {
        CuSparseBackend::Config cfg;
        cfg.precision = getPrecision();
        m_spmvBackend = new CuSparseBackend(cfg);
        m_spmvStructureRows = -1;
    }

    if (m_spmvStructureRows != numRows ||
        m_matrixStatus == MatrixStatus::STRUCTURE_CHANGED) {
        if (m_spmvBackend->bindStructure(m_blockSize, numRows, numNnz,
                                         const_cast<int *>(rowPtrs),
                                         const_cast<int *>(colIndices)) != 0) {
            return -1;
        }
        m_spmvStructureRows = numRows;
    }

    return 0;
}

int
CudaBcsrLinSOE::formApAssembled(const Vector &p, Vector &Ap)
{
    const int n = getNumEqn();
    if (p.Size() != n || Ap.Size() != n) {
        opserr << "WARNING: CudaBcsrLinSOE::formAp() - vector size mismatch\n";
        return -1;
    }
    if (n <= 0) {
        return 0;
    }

    if (ensureSpMVOperator() != 0) {
        return -1;
    }

    ensureSpmvScratchSizes();
    syncAValuesToDevice();
    uploadSpmvPFromHost(p, n);

    void *deviceA = getDeviceAValues();
    void *deviceP = getDeviceSpmvP();
    void *deviceAp = getDeviceSpmvY();
    if (deviceA == nullptr || deviceP == nullptr || deviceAp == nullptr) {
        opserr << "WARNING: CudaBcsrLinSOE::formAp() - null device pointer(s)\n";
        return -1;
    }

    if (m_spmvBackend->bindValues(deviceA) != 0) {
        return -1;
    }
    if (m_spmvBackend->spmv(deviceP, deviceAp, 1.0, 0.0) != 0) {
        opserr << "WARNING: CudaBcsrLinSOE::formAp() - SpMV failed\n";
        return -1;
    }

    downloadSpmvYToHost(Ap, n);
    return 0;
}

LinearSOE *
CudaBcsrLinSOE::getCopy(void) const
{
    const CudaBcsrLinSolver *baseSolver =
        dynamic_cast<const CudaBcsrLinSolver *>(this->LinearSOE::getSolver());
    if (baseSolver == nullptr) {
        return nullptr;
    }

    LinearSOESolver *newSolver = baseSolver->getCopy();
    if (newSolver == nullptr) {
        return nullptr;
    }

    CudaBcsrLinSolver *cudaSolver = dynamic_cast<CudaBcsrLinSolver *>(newSolver);
    if (cudaSolver == nullptr) {
        delete newSolver;
        return nullptr;
    }

    const bool symmetricStorage = isSymmetricStorage();
    CudaBcsrLinSOE *out = nullptr;
    switch (getPrecision()) {
        case CudaPrecision::dDDI:
            out = createDouble(*cudaSolver, m_blockSize, m_paddingEnabled, m_verbose, symmetricStorage);
            break;
        case CudaPrecision::dFFI:
            out = createFloat(*cudaSolver, m_blockSize, m_paddingEnabled, m_verbose, symmetricStorage);
            break;
        case CudaPrecision::dDFI:
            out = createDoubleFloat(*cudaSolver, m_blockSize, m_paddingEnabled, m_verbose, symmetricStorage);
            break;
        case CudaPrecision::dFDI:
            out = createFloatDouble(*cudaSolver, m_blockSize, m_paddingEnabled, m_verbose, symmetricStorage);
            break;
        default:
            delete newSolver;
            return nullptr;
    }

    if (out == nullptr) {
        delete newSolver;
        return nullptr;
    }
    out->setCudaDeviceEnabled(m_cudaDeviceEnabled);
    return out;
}

// Factory method implementations
CudaBcsrLinSOE* CudaBcsrLinSOE::createDouble(
    CudaBcsrLinSolver &theSolver, 
    int blockSize, 
    bool paddingEnabled,
    bool verbose,
    bool symmetricStorage
) {
    return new CudaBcsrLinSOEImpl<double, double>(
        theSolver, blockSize, paddingEnabled, verbose, symmetricStorage
    );
}

CudaBcsrLinSOE* CudaBcsrLinSOE::createFloat(
    CudaBcsrLinSolver &theSolver,
    int blockSize, 
    bool paddingEnabled,
    bool verbose,
    bool symmetricStorage
) {
    return new CudaBcsrLinSOEImpl<float, float>(
        theSolver, blockSize, paddingEnabled, verbose, symmetricStorage
    );
}

CudaBcsrLinSOE* CudaBcsrLinSOE::createDoubleFloat(
    CudaBcsrLinSolver &theSolver,
    int blockSize, 
    bool paddingEnabled,
    bool verbose,
    bool symmetricStorage
) {
    return new CudaBcsrLinSOEImpl<double, float>(
        theSolver, blockSize, paddingEnabled, verbose, symmetricStorage
    );
}

CudaBcsrLinSOE* CudaBcsrLinSOE::createFloatDouble(
    CudaBcsrLinSolver &theSolver,
    int blockSize, 
    bool paddingEnabled,
    bool verbose,
    bool symmetricStorage
) {
    return new CudaBcsrLinSOEImpl<float, double>(
        theSolver, blockSize, paddingEnabled, verbose, symmetricStorage
    );
}

LinearSOE* CudaBcsrLinSOE::createCudaLinearSOE(int classTag) {
    switch(classTag) {
        case LinSOE_TAGS_CudaBcsrLinSOE_DOUBLE:
            return new CudaBcsrLinSOEImpl<double, double>();  // dDDI
        case LinSOE_TAGS_CudaBcsrLinSOE_FLOAT:
            return new CudaBcsrLinSOEImpl<float, float>();    // dFFI
        case LinSOE_TAGS_CudaBcsrLinSOE_DOUBLE_FLOAT:
            return new CudaBcsrLinSOEImpl<double, float>();   // dDFI
        case LinSOE_TAGS_CudaBcsrLinSOE_FLOAT_DOUBLE:
            return new CudaBcsrLinSOEImpl<float, double>();   // dFDI
        default:
            return nullptr;
    }
}

