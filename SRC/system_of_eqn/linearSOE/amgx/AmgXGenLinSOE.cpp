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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/amgx/AmgXGenLinSOE.cpp,v $
                                                                        
                                                                        
// Written: gaaraujo 
// Created: 02/2025
//
// Description: This file contains the class definition for 
// AmgXGenLinSolver. It solves the AmgXGenLinSOEobject by calling
// AMGX routines.
//

#include <AmgXGenLinSOE.h>
#include <AmgXGenLinSolver.h>
#include <Matrix.h>
#include <Graph.h>
#include <Vertex.h>
#include <VertexIter.h>
#include <math.h>
#include <stdlib.h>


#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include <ID.h>

AmgXGenLinSOE::AmgXGenLinSOE(LinearSOESolver &the_Solver, 
                             int blockSize, bool paddingEnabled,
                             bool verbose)
    : LinearSOE(the_Solver, LinSOE_TAGS_AmgXGenLinSOE), 
    m_X(), m_B(), m_XPadded(), m_BPadded(), m_ARowPtrBlock(), m_AColIdxBlock(), m_AValuesBlock(), 
    m_BlockSize(blockSize),
    m_matrixStatus(AmgXMatrixStatus::STRUCTURE_CHANGED),
    m_paddingEnabled(paddingEnabled),
    m_verbose(verbose)
{
    // Validate block size
    if (!isValidBlockSize(blockSize)) {
        opserr << "WARNING: AmgXGenLinSOE constructor: Invalid block size " << blockSize 
               << ". Using default block size " << DEFAULT_BLOCK_SIZE << endln;
        m_BlockSize = DEFAULT_BLOCK_SIZE;
    }
    
    // Try to cast to AmgXGenLinSolver<double> first
    AmgXGenLinSolver<double>* doubleSolver = dynamic_cast<AmgXGenLinSolver<double>*>(&the_Solver);
    if (doubleSolver) {
        doubleSolver->setLinearSOE(*this);
        return;
    }
    
    // Try to cast to AmgXGenLinSolver<float>
    AmgXGenLinSolver<float>* floatSolver = dynamic_cast<AmgXGenLinSolver<float>*>(&the_Solver);
    if (floatSolver) {
        floatSolver->setLinearSOE(*this);
        return;
    }
    
    // If neither cast works, this is not an AmgXGenLinSolver
    opserr << "WARNING: AmgXGenLinSOE constructor: Solver is not an AmgXGenLinSolver\n";
}

AmgXGenLinSOE::AmgXGenLinSOE(): LinearSOE(LinSOE_TAGS_AmgXGenLinSOE), 
    m_X(), m_B(), m_XPadded(), m_BPadded(), m_ARowPtrBlock(), m_AColIdxBlock(), m_AValuesBlock(), 
    m_BlockSize(DEFAULT_BLOCK_SIZE),
    m_matrixStatus(AmgXMatrixStatus::STRUCTURE_CHANGED),
    m_paddingEnabled(true),
    m_verbose(false)
{
    
}

AmgXGenLinSOE::~AmgXGenLinSOE() 
{
    
}

// Validation methods
bool AmgXGenLinSOE::isValidBlockSize(int blockSize) const
{
    return blockSize > 0 && blockSize <= MAX_BLOCK_SIZE;
}

bool AmgXGenLinSOE::isValidGlobalIndex(int index) const
{
    return index >= 0 && index < m_X.Size();
}

int AmgXGenLinSOE::getNumEqn(void) const 
{
    return m_X.Size();
}

int AmgXGenLinSOE::setSize(Graph &theGraph) 
{
    int size = theGraph.getNumVertex();
    if (size < 0) {
        opserr << "ERROR: AmgXGenLinSOE::setSize: size of soe < 0\n";
        return -1;
    }

    // Keep track of the original size of the system of equations
    const int originalSize = size;

    // First iterate through the vertices of the graph to get nnz
    Vertex *theVertex;
    int nnz = 0;
    VertexIter &theVertices = theGraph.getVertices();
    while ((theVertex = theVertices()) != 0) {
        const ID &theAdjacency = theVertex->getAdjacency();
        nnz += theAdjacency.Size() + 1; // the +1 is for the diag entry
    }

    // Estimate the block size if not provided
    if (m_BlockSize == 0) {
        m_BlockSize = estimateBlockSize(theGraph, nnz, DEFAULT_EFFICIENCY_THRESHOLD);
        if (m_verbose) {
            opserr << "INFO: AmgXGenLinSOE::setSize: Automatically estimating block size for efficiency >= " 
                   << DEFAULT_EFFICIENCY_THRESHOLD << endln;
            opserr << "      Estimated block size: " << m_BlockSize << endln;
        }
    }

    // Validate the block size
    if (!isValidBlockSize(m_BlockSize)) {
        opserr << "ERROR: AmgXGenLinSOE::setSize: Invalid block size " << m_BlockSize 
               << ". Must be between 1 and " << MAX_BLOCK_SIZE << endln;
        return -1;
    }
    
    // Check if padding is needed
    if (size % m_BlockSize != 0 && !m_paddingEnabled) {
        opserr << "ERROR: AmgXGenLinSOE::setSize: The number of equations (" << size 
               << ") is not divisible by the block size (" << m_BlockSize << ").\n";
        opserr << "      Please provide a block size that divides the number of equations evenly, ";
        opserr << "or set the block size to 0 to automatically estimate it." << endln;
        return -1;
    }

    // Clear the matrix structure
    m_ARowPtrBlock.clear();
    m_AColIdxBlock.clear();
    m_AValuesBlock.clear();

    // If padding is enabled, pad the matrix with zeros to make it a multiple of the block size
    if (m_paddingEnabled && size % m_BlockSize != 0) {
        size = ((originalSize + m_BlockSize - 1) / m_BlockSize) * m_BlockSize;
        if (m_verbose) {
            opserr << "INFO: AmgXGenLinSOE::setSize: Padding enabled.\n";
            opserr << "      Original size: " << originalSize << ", Padded size: " << size << endln;
        }
    }

    // Special case for BlockSize = 1 - treat as regular CSR format
    if (m_BlockSize == 1) {
        // Reserve space for matrix A
        m_ARowPtrBlock.reserve(size + 1);
        m_AColIdxBlock.reserve(nnz);
        m_AValuesBlock.resize(nnz, 0.0);

        // Fill in m_ARowPtrBlock and m_AColIdxBlock
        m_ARowPtrBlock.push_back(0); // Start of first row

        for (int row = 0; row < size; ++row) {
            theVertex = theGraph.getVertexPtr(row);
            if (theVertex == nullptr) {
                opserr << "ERROR: AmgXGenLinSOE::setSize: vertex " << row << " not in graph!\n";
                return -1;
            }

            const ID& theAdjacency = theVertex->getAdjacency();
            ID colIdx(0, theAdjacency.Size() + 1); // +1 for the diagonal

            // Add diagonal first
            colIdx.insert(theVertex->getTag());

            // Add adjacency entries in order
            for (int j = 0; j < theAdjacency.Size(); ++j) {
                colIdx.insert(theAdjacency(j));
            }

            // Append to global col index block
            for (int i = 0; i < colIdx.Size(); ++i) {
                m_AColIdxBlock.push_back(colIdx(i));
            }

            // Update row pointer
            m_ARowPtrBlock.push_back(m_AColIdxBlock.size());
        }
    } else {
        // Block size > 1 -> block CSR format
        const int numBlockRows = size / m_BlockSize;
        const int numBlockCols = size / m_BlockSize;
        
        // Prepare block structure
        m_ARowPtrBlock.resize(numBlockRows + 1, 0);
        std::vector<int> mask(numBlockCols, -1);
        std::vector<ID> colIdxPerBlockRow(numBlockRows); // using ID because it automatically sorts the entries

        /* Note: graph vertices need to be processed ordered by their tags for 
         * the following loop to work correctly.
         */
        for (int row = 0; row < size; ++row) {
            const int blockRow = row / m_BlockSize;

            // Insert the diagonal block
            if (mask[blockRow] != blockRow) {
                mask[blockRow] = blockRow;
                colIdxPerBlockRow[blockRow].insert(blockRow);
            }

            if (m_paddingEnabled && row >= originalSize) {
                // padded diagonal block is already inserted, so we can break
                break;
            }
            
            theVertex = theGraph.getVertexPtr(row);
            if (theVertex == nullptr) {
                opserr << "ERROR: AmgXGenLinSOE::setSize: vertex " << row << " not in graph!\n";
                return -1;
            }
            
            const ID& theAdjacency = theVertex->getAdjacency();  // connected columns

            // Insert other adjacency blocks
            for (int k = 0; k < theAdjacency.Size(); ++k) {
                const int col = theAdjacency(k);
                const int blockCol = col / m_BlockSize;
                if (mask[blockCol] != blockRow) {
                    mask[blockCol] = blockRow;
                    colIdxPerBlockRow[blockRow].insert(blockCol);
                }
            }
        }

        // Fill m_ARowPtrBlock
        for (int blockRow = 0; blockRow < numBlockRows; ++blockRow) {
            m_ARowPtrBlock[blockRow + 1] = m_ARowPtrBlock[blockRow] + colIdxPerBlockRow[blockRow].Size();
        }

        // Fill m_AColIdxBlock and allocate m_AValuesBlock
        const int nnzBlock = m_ARowPtrBlock[numBlockRows];
        m_AColIdxBlock.resize(nnzBlock, 0);
        m_AValuesBlock.resize(nnzBlock * m_BlockSize * m_BlockSize, 0.0);

        // Copy the block column indices
        int blockIdx = 0;
        for (int blockRow = 0; blockRow < numBlockRows; ++blockRow) {
            const int rowStart = m_ARowPtrBlock[blockRow];
            const int rowEnd = m_ARowPtrBlock[blockRow + 1];
            const ID& blockColIdx = colIdxPerBlockRow[blockRow];
            const int numBlockCols = blockColIdx.Size();
            for (int blockCol = 0; blockCol < numBlockCols; ++blockCol) {
                m_AColIdxBlock[blockIdx] = blockColIdx(blockCol);
                blockIdx++;
            }
        }
    }

    // Allocate solution and RHS vectors
    m_XPadded.resize(size, 0.0); 
    m_BPadded.resize(size, 0.0);
    m_X.setData(m_XPadded.data(), originalSize);
    m_B.setData(m_BPadded.data(), originalSize);
    
    // Update matrix status
    m_matrixStatus = AmgXMatrixStatus::STRUCTURE_CHANGED;

    // invoke setSize() on the Solver
    LinearSOESolver *the_Solver = this->getSolver();
    int solverOK = the_Solver->setSize();
    if (solverOK < 0) {
        opserr << "ERROR: AmgXGenLinSOE::setSize :";
        opserr << " solver failed setSize()\n";
        return solverOK;
    }
    return 0;
}

// Helper methods for matrix assembly to reduce code duplication
int AmgXGenLinSOE::addAMatrixElement(int globalRow, int globalCol, double value)
{
    if (!isValidGlobalIndex(globalRow) || !isValidGlobalIndex(globalCol)) {
        return -1;
    }

    if (m_BlockSize > 1) {
        return addAMatrixElementBlock(globalRow, globalCol, value);
    } else {
        return addAMatrixElementStandard(globalRow, globalCol, value);
    }
}

int AmgXGenLinSOE::addAMatrixElementBlock(int globalRow, int globalCol, double value)
{
    const int blockRow = globalRow / m_BlockSize;
    const int localRow = globalRow % m_BlockSize;
    const int blockCol = globalCol / m_BlockSize;
    const int localCol = globalCol % m_BlockSize;
    
    // Find the block index in m_AColIdxBlock where m_AColIdxBlock[k] == blockCol 
    // and k is in [m_ARowPtrBlock[blockRow], m_ARowPtrBlock[blockRow + 1])
    const int startRowPtr = m_ARowPtrBlock[blockRow];
    const int endRowPtr = m_ARowPtrBlock[blockRow + 1];
    for (int k = startRowPtr; k < endRowPtr; ++k) {
        if (m_AColIdxBlock[k] == blockCol) {
            // Block k holds the blockRow, blockCol block
            const int blockOffset = k * m_BlockSize * m_BlockSize;
            const int localOffset = localRow * m_BlockSize + localCol; // row-major
            m_AValuesBlock[blockOffset + localOffset] += value;
            return 0;
        }
    }
    
    opserr << "ERROR: AmgXGenLinSOE::addAMatrixElementBlock: Could not find block for row " 
           << globalRow << ", col " << globalCol << endln;
    return -1;
}

int AmgXGenLinSOE::addAMatrixElementStandard(int globalRow, int globalCol, double value)
{
    // Find the column index in m_AColIdxBlock where m_AColIdxBlock[k] == globalCol 
    // and k is in [m_ARowPtrBlock[globalRow], m_ARowPtrBlock[globalRow + 1])
    const int startRowPtr = m_ARowPtrBlock[globalRow];
    const int endRowPtr = m_ARowPtrBlock[globalRow + 1];
    for (int k = startRowPtr; k < endRowPtr; ++k) {
        if (m_AColIdxBlock[k] == globalCol) {
            m_AValuesBlock[k] += value;
            return 0;
        }
    }
    
    opserr << "ERROR: AmgXGenLinSOE::addAMatrixElementStandard: Could not find element for row " 
           << globalRow << ", col " << globalCol << endln;
    return -1;
}

int AmgXGenLinSOE::addA(const Matrix &m, const ID &id, double fact)
{   
    // Check for a quick return
    if (fact == 0.0) {
        return 0;
    }

    const int idSize = id.Size();

    // Check that m and id are of similar size
    if (idSize != m.noRows() && idSize != m.noCols()) {
        opserr << "ERROR: AmgXGenLinSOE::addA: Matrix and ID not of similar sizes\n";
        return -1;
    }
    
    const int size = m_X.Size();

    // Add matrix elements using the helper methods
    for (int i = 0; i < idSize; ++i) {
        const int globalRow = id(i);
        if (!isValidGlobalIndex(globalRow)) continue;

        for (int j = 0; j < idSize; ++j) {
            const int globalCol = id(j);
            if (!isValidGlobalIndex(globalCol)) continue;
            
            double value = applyFact(m(i, j), fact);
            if (addAMatrixElement(globalRow, globalCol, value) != 0) {
                opserr << "ERROR: AmgXGenLinSOE::addA: Failed to add element at (" 
                       << globalRow << ", " << globalCol << ")\n";
                return -1;
            }
        }
    }

    // Update matrix status
    if (m_matrixStatus == AmgXMatrixStatus::UNCHANGED) {
        m_matrixStatus = AmgXMatrixStatus::COEFFICIENTS_CHANGED;
    }

    return 0;
}

int AmgXGenLinSOE::addB(const Vector &v, const ID &id, double fact)
{
    // Check for a quick return
    if (fact == 0.0) {
        return 0;
    }

    const int idSize = id.Size();

    // Check that v and id are of similar size
    if (idSize != v.Size()) {
        opserr << "ERROR: AmgXGenLinSOE::addB: Vector and ID not of similar sizes\n";
        return -1;
    }

    // Add vector elements
    for (int i = 0; i < idSize; ++i) {
        const int globalRow = id(i);
        if (isValidGlobalIndex(globalRow)) {
            m_B(globalRow) += applyFact(v(i), fact);
        }
    }

    return 0;
}

int AmgXGenLinSOE::setB(const Vector &v, double fact)
{
    // Check for a quick return
    if (fact == 0.0) {
        zeroB();
        return 0;
    }

    const int size = m_B.Size();
    if (size != v.Size()) {
        opserr << "ERROR: AmgXGenLinSOE::setB: Vector size mismatch\n";
        return -1;
    }

    // Set vector elements
    for (int i = 0; i < size; ++i) {
        m_B(i) = applyFact(v(i), fact);
    }

    return 0;
}

void AmgXGenLinSOE::zeroA(void)
{
    std::fill(m_AValuesBlock.begin(), m_AValuesBlock.end(), 0.0);
    
    // Update matrix status
    if (m_matrixStatus == AmgXMatrixStatus::UNCHANGED) {
        m_matrixStatus = AmgXMatrixStatus::COEFFICIENTS_CHANGED;
    }
}

void AmgXGenLinSOE::zeroB(void)
{
    std::fill(m_BPadded.begin(), m_BPadded.end(), 0.0);
}

void AmgXGenLinSOE::setX(int loc, double value)
{
    if (isValidGlobalIndex(loc)) {
        m_X(loc) = value;
    }
}

void AmgXGenLinSOE::setX(const Vector &x)
{
    const int size = m_X.Size();
    if (size != x.Size()) {
        opserr << "ERROR: AmgXGenLinSOE::setX: Vector size mismatch\n";
        return;
    }

    m_X = x;
}

const Vector & AmgXGenLinSOE::getX(void)
{
    return m_X;
}   

const Vector & AmgXGenLinSOE::getB(void)
{
    return m_B;
}

double AmgXGenLinSOE::normRHS(void)
{
    return m_B.Norm();
}

int AmgXGenLinSOE::setAmgXGenLinSolver(LinearSOESolver &newSolver)
{
    // Try to cast to AmgXGenLinSolver<double> first
    AmgXGenLinSolver<double>* doubleSolver = dynamic_cast<AmgXGenLinSolver<double>*>(&newSolver);
    if (doubleSolver) {
        doubleSolver->setLinearSOE(*this);
        if (m_X.Size() != 0) {
            int solverOK = doubleSolver->setSize();
            if (solverOK < 0) {
                opserr << "ERROR: AmgXGenLinSOE::setAmgXGenLinSolver: ";
                opserr << "the new solver could not setSize() - staying with old\n";
                return -1;
            }
        }
        return this->LinearSOE::setSolver(newSolver);
    }
    
    // Try to cast to AmgXGenLinSolver<float>
    AmgXGenLinSolver<float>* floatSolver = dynamic_cast<AmgXGenLinSolver<float>*>(&newSolver);
    if (floatSolver) {
        floatSolver->setLinearSOE(*this);
        if (m_X.Size() != 0) {
            int solverOK = floatSolver->setSize();
            if (solverOK < 0) {
                opserr << "ERROR: AmgXGenLinSOE::setAmgXGenLinSolver: ";
                opserr << "the new solver could not setSize() - staying with old\n";
                return -1;
            }
        }
        return this->LinearSOE::setSolver(newSolver);
    }
    
    // If neither cast works, this is not an AmgXGenLinSolver
    opserr << "ERROR: AmgXGenLinSOE::setAmgXGenLinSolver: Solver is not an AmgXGenLinSolver\n";
    return -1;
}

// Fill padded diagonals with a user-supplied value plus an automatically computed value
// Note: existing values in the padded diagonals are overwritten
int AmgXGenLinSOE::fillPaddedDiagonals(double value, bool autoCompute) {
    if (m_BlockSize == 1 || m_X.Size() == m_XPadded.size()) {
        return 0;
    }

    const int blockOffset = static_cast<int>(m_AValuesBlock.size()) - m_BlockSize * m_BlockSize;
    const int startRow = m_X.Size() % m_BlockSize;

    if (startRow <= 0 || blockOffset < 0) {
        opserr << "ERROR: AmgXGenLinSOE::fillPaddedDiagonals: Invalid block offset or start row\n";
        return -1;
    }

    double repDiagValue = value;

    if (autoCompute) {
        double avgDiag = 0.0;
        double maxAbsDiag = 0.0;

        for (int localRow = 0; localRow < startRow; ++localRow) {
            const int idx = blockOffset + localRow * m_BlockSize + localRow;
            const double diag = m_AValuesBlock[idx];
            const double absDiag = std::abs(diag);
            avgDiag += absDiag;
            if (absDiag > maxAbsDiag) maxAbsDiag = absDiag;
        }

        avgDiag /= static_cast<double>(startRow);
        const double minDiag = MIN_DIAGONAL_VALUE_FACTOR * maxAbsDiag;
        // Add the average diagonal value to the user-supplied value
        repDiagValue += (avgDiag > minDiag) ? avgDiag : minDiag;
    }

    for (int localRow = startRow; localRow < m_BlockSize; ++localRow) {
        const int idx = blockOffset + localRow * m_BlockSize + localRow;
        m_AValuesBlock[idx] = repDiagValue;
    }

    return 0;
}



int AmgXGenLinSOE::solve(void)
{
    // Some basic matrix info
    const int originalSize = m_X.Size();
    const int size = m_XPadded.size();

    // Quick sanity check
    if (originalSize == 0 || size == 0) {
        return 0;
    }

    // Fill diagonal entries in rows beyond the original size
    // NOTE: We estimate a representative stiffness value. 
    // Alternatively, we could just fill out with ones in the diagonal, 
    // but that may cause numerical problems.
    if (m_matrixStatus != AmgXMatrixStatus::UNCHANGED && m_paddingEnabled) {
        if (fillPaddedDiagonals() != 0) {
            opserr << "ERROR: AmgXGenLinSOE::solve: Failed to fill padded diagonals\n";
            return -1;
        }
    }

    LinearSOESolver *the_Solver = this->getSolver();
    if (the_Solver != 0) {
        int solverOk = the_Solver->solve();
        if (solverOk == 0) {
            /* Update matrix status for future solves */
            m_matrixStatus = AmgXMatrixStatus::UNCHANGED;
        }
        return solverOk;
    } else {
        opserr << "ERROR: AmgXGenLinSOE::solve: No solver available\n";
        return -1;
    }
}

int AmgXGenLinSOE::saveSparseA(OPS_Stream& output, int baseIndex)
{
    if (m_AValuesBlock.empty() || m_ARowPtrBlock.empty() || m_AColIdxBlock.empty()) {
        opserr << "WARNING: AmgXGenLinSOE::saveSparseA: Matrix data is empty\n";
        return 0;
    }

    // Pad matrix before printing
    if (m_matrixStatus != AmgXMatrixStatus::UNCHANGED && m_paddingEnabled) {
        if (fillPaddedDiagonals() != 0) {
            opserr << "ERROR: AmgXGenLinSOE::saveSparseA: Failed to fill padded diagonals\n";
            return -1;
        }
    }

    const int numBlockRows = m_ARowPtrBlock.size() - 1;
    const int size = numBlockRows * m_BlockSize;
    const int nnz = m_AValuesBlock.size();

    // Assume the header is already written to output stream
    output << "%% Block size: " << m_BlockSize << "\n";
    output << size << " " << size << " " << nnz << "\n";

    // Write the sparse matrix entries
    int nnz_written = 0;
    if (m_BlockSize > 1) { // Block CSR format
        for (int blockRow = 0; blockRow < numBlockRows; blockRow++) {
            const int rowStart = m_ARowPtrBlock[blockRow];
            const int rowEnd = m_ARowPtrBlock[blockRow + 1];
            for (int blockIdx = rowStart; blockIdx < rowEnd; blockIdx++) {
                const int blockCol = m_AColIdxBlock[blockIdx];
                double* theBlock = m_AValuesBlock.data() + blockIdx * m_BlockSize * m_BlockSize;
                for (int i = 0; i < m_BlockSize; i++) {
                    const int row = blockRow * m_BlockSize + i + baseIndex;
                    for (int j = 0; j < m_BlockSize; j++) {
                        const int col = blockCol * m_BlockSize + j + baseIndex;
                        const double val = theBlock[i * m_BlockSize + j];
                        output << row << " " << col << " " << val << "\n";
                        nnz_written++;
                    }
                }
            }
        }
    } else { // Standard CSR format
        for (int i = 0, row = baseIndex; i < size; i++, row++) {
            const int rowStart = m_ARowPtrBlock[i];
            const int rowEnd = m_ARowPtrBlock[i + 1];
            for (int idx = rowStart; idx < rowEnd; idx++) {
                const int col = m_AColIdxBlock[idx] + baseIndex;
                const double val = m_AValuesBlock[idx];
                output << row << " " << col << " " << val << "\n";
                nnz_written++;
            }
        }
    }

    if (nnz_written != nnz) {
        opserr << "WARNING: AmgXGenLinSOE::saveSparseA: nnz_written (" << nnz_written 
               << ") != nnz (" << nnz << ")\n";
        return -1;
    }

    return 0;
}

int AmgXGenLinSOE::sendSelf(int commitTag, Channel &theChannel)
{
    return 0;
}

int AmgXGenLinSOE::recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    return 0;
}

/* Utility functions for block size estimation and counting.
 * This algorithm is adapted from the implementation in SciPy's `sparsetools`:
 * https://github.com/scipy/scipy/blob/0f1fd4a7268b813fa2b844ca6038e4dfdf90084a/scipy/sparse/sparsetools/csr.h#L205-L254
 */

// Estimate the optimal block size based on sparsity pattern efficiency
int AmgXGenLinSOE::estimateBlockSize(Graph &theGraph, int nnz, double efficiency)
{
    const int size = theGraph.getNumVertex();
    if (size < 0) {
        opserr << "ERROR: AmgXGenLinSOE::estimateBlockSize: size of soe < 0\n";
        return -1;
    }

    if (nnz == 0) {
        return DEFAULT_BLOCK_SIZE;
    }

    if (efficiency <= 0.0 || efficiency >= 1.0) {
        opserr << "WARNING: AmgXGenLinSOE::estimateBlockSize: efficiency must satisfy 0.0 < efficiency < 1.0\n";
        return DEFAULT_BLOCK_SIZE;
    }

    int listOfBlockSizes[] = {4, 3, 2};

    for (int blockSize : listOfBlockSizes) {
        if (m_paddingEnabled || size % blockSize == 0) {
            int nb = countBlocks(theGraph, blockSize);
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

// Count the number of square blocks in the graph.
// Assumes the system of equations forms a square sparse matrix 
// and the graph is undirected.
int AmgXGenLinSOE::countBlocks(Graph &theGraph, int blockSize)
{
    int size = theGraph.getNumVertex();
    if (size < 0) {
        opserr << "ERROR: AmgXGenLinSOE::countBlocks: size of soe < 0\n";
        return -1;
    }

    if (blockSize < 1) {
        opserr << "ERROR: AmgXGenLinSOE::countBlocks: blockSize must be >= 1\n";
        return -1;
    }

    if (size % blockSize != 0 && !m_paddingEnabled) {
        opserr << "ERROR: AmgXGenLinSOE::countBlocks: size of soe must be divisible by blockSize\n";
        return -1;
    }

    const int numBlockCols = (size + blockSize - 1) / blockSize;
    std::vector<int> mask(numBlockCols, -1);
    int totalNumBlocks = 0;

    // Note: graph vertices need to be processed ordered by their tags for 
    // the following loop to work correctly.
    for (int i = 0; i < size; i++) {
        const int blockRow = i / blockSize;
        Vertex* theVertex = theGraph.getVertexPtr(i);
        if (theVertex == nullptr) {
            opserr << "ERROR: AmgXGenLinSOE::countBlocks: vertex " << i << " not in graph!\n";
            return -1;
        }
        const ID& adjacency = theVertex->getAdjacency(); // col indices
            
        // Insert the diagonal block
        if (mask[blockRow] != blockRow) {
            mask[blockRow] = blockRow;
            totalNumBlocks++;
        }

        // Insert other adjacency blocks
        for (int j = 0; j < adjacency.Size(); j++) {
            const int blockCol = adjacency(j) / blockSize; 
            if (mask[blockCol] != blockRow) {
                // if this block-col hasn't seen this block-row
                mask[blockCol] = blockRow;
                totalNumBlocks++;
            }
        }
    }

    return totalNumBlocks;
}