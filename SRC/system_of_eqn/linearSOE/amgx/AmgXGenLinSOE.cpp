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

AmgXGenLinSOE::AmgXGenLinSOE(AmgXGenLinSolver &the_Solver, 
                             int blockSize, bool paddingEnabled,
                             bool verbose)
    : LinearSOE(the_Solver, LinSOE_TAGS_AmgXGenLinSOE), 
    m_X(), m_B(), m_XPadded(), m_BPadded(), m_ARowPtrBlock(), m_AColIdxBlock(), m_AValuesBlock(), 
    m_BlockSize(blockSize),
    m_matrixStatus(AmgXMatrixStatus::STRUCTURE_CHANGED),
    m_paddingEnabled(paddingEnabled),
    m_verbose(verbose)
{
    the_Solver.setLinearSOE(*this);
}

AmgXGenLinSOE::AmgXGenLinSOE(): LinearSOE(LinSOE_TAGS_AmgXGenLinSOE), 
    m_X(), m_B(), m_XPadded(), m_BPadded(), m_ARowPtrBlock(), m_AColIdxBlock(), m_AValuesBlock(), 
    m_BlockSize(0),
    m_matrixStatus(AmgXMatrixStatus::STRUCTURE_CHANGED),
    m_paddingEnabled(true),
    m_verbose(false)
{
    
}

AmgXGenLinSOE::~AmgXGenLinSOE() 
{
    
}

int AmgXGenLinSOE::getNumEqn(void) const 
{
    return m_X.Size();
}

int AmgXGenLinSOE::setSize(Graph &theGraph) 
{
    int size = theGraph.getNumVertex();
    if (size < 0) {
        opserr << "size of soe < 0\n";
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
        nnz += theAdjacency.Size() +1; // the +1 is for the diag entry
    }

    // Estimate the block size
    if (m_BlockSize == 0) {
        const double efficiency = 0.7;
        m_BlockSize = estimateBlockSize(theGraph, nnz, efficiency);
        if (m_verbose) {
            opserr << "WARNING: AmgXGenLinSOE::setSize : Provided block size is 0. \n";
            opserr << "- Automatically estimating block size that results in storage efficiency of at least " << efficiency << "...\n";
            opserr << "- Estimated block size: " << m_BlockSize << endln;
        }
    }

    if (m_BlockSize <= 0 || m_BlockSize > 32) {
        opserr << "WARNING: either block size could not be estimated ";
        opserr << "or provided block size is invalid. ";
        opserr << "Please manually provide a positive block size. -- AmgXGenLinSOE::setSize" << endln;
        return -1;
    }

    if (m_BlockSize != 1 && m_BlockSize != 4) {
        opserr << "WARNING: AmgXGenLinSOE::setSize : Most AmgX solvers only support block size 1 or 4. \n";
        opserr << "- Watch out for any AMGX errors in the output. \n";
        opserr << "- If you get errors related to the blockSize, try passing -blockSize 1 or -blockSize 4 to system AmgX. \n";
    }
    
    if (size % m_BlockSize != 0 && m_paddingEnabled == false) {
        opserr << "WARNING: the number of equations (" << size << ") is not divisible by the block size (" << m_BlockSize << "). \n";
        opserr << "Please provide a block size that divides the number of equations evenly, ";
        opserr << "or set the block size to 0 to automatically estimate it. -- AmgXGenLinSOE::setSize" << endln;
        return -1;
    }

    // Clear the matrix structure
    m_ARowPtrBlock.clear();
    m_AColIdxBlock.clear();
    m_AValuesBlock.clear();

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
                opserr << "WARNING: AmgXGenLinSOE::setSize :"
                    << " vertex " << row << " not in graph! - size set to 0\n";
                size = 0;
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
        // If padding is enabled, pad the matrix with zeros to make it a multiple of the block size
        if (m_paddingEnabled) {
            size = ((originalSize + m_BlockSize - 1) / m_BlockSize) * m_BlockSize;
            if (size > originalSize && m_verbose) {
                opserr << "WARNING: AmgXGenLinSOE::setSize : Padding is enabled. \n";
                opserr << "- Padding the matrix with zeros to make it a multiple of the block size. \n";
                opserr << "- Original size: " << originalSize << ", \n";
                opserr << "- Padded size: " << size << endln;
            }
        }

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
                opserr << "WARNING: AmgXGenLinSOE::setSize :"
                    << " vertex " << row << " not in graph! - size set to 0\n";
                size = 0;
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
        m_AColIdxBlock.reserve(nnzBlock);
        m_AValuesBlock.resize(nnzBlock * m_BlockSize * m_BlockSize, 0.0);

        for (int blockRow = 0; blockRow < numBlockRows; ++blockRow) {
            const ID& theColIdx = colIdxPerBlockRow[blockRow];
            for (int blockCol = 0; blockCol < theColIdx.Size(); ++blockCol) {
                m_AColIdxBlock.push_back(theColIdx(blockCol));
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
        opserr << "WARNING: AmgXGenLinSOE::setSize :";
        opserr << " solver failed setSize()\n";
        return solverOK;
    }
    return 0;
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
        opserr << "WARNING: AmgXGenLinSOE::addA() ";
        opserr << " - Matrix and ID not of similar sizes\n";
        return -1;
    }
    
    const int size = m_X.Size();

    if (m_BlockSize > 1) { // Block CSR format
        const int blockSizeSquared = m_BlockSize * m_BlockSize;
        if (fact == 1.0) { // Do not need to multiply by fact
            for (int i = 0; i < idSize; ++i) {
                const int globalRow = id(i);
                if (globalRow < 0 || globalRow >= size) continue;
                const int blockRow = globalRow / m_BlockSize;
                const int localRow = globalRow % m_BlockSize;

                for (int j = 0; j < idSize; ++j) {
                    const int globalCol = id(j);
                    if (globalCol < 0 || globalCol >= size) continue;
                    const int blockCol = globalCol / m_BlockSize;
                    const int localCol = globalCol % m_BlockSize;
                    // Find the block index in m_AColIdxBlock where m_AColIdxBlock[k] == blockCol 
                    // and k is in [m_ARowPtrBlock[blockRow], m_ARowPtrBlock[blockRow + 1])
                    for (int k = m_ARowPtrBlock[blockRow]; k < m_ARowPtrBlock[blockRow + 1]; ++k) {
                        if (m_AColIdxBlock[k] == blockCol) {
                            // Block k holds the blockRow, blockCol block
                            const int blockOffset = k * blockSizeSquared;
                            const int localOffset = localRow * m_BlockSize + localCol; // row-major
                            m_AValuesBlock[blockOffset + localOffset] += m(i, j);
                            break;
                        }
                    }
                }
            }
        } else { // Multiply by fact
            for (int i = 0; i < idSize; ++i) {
                const int globalRow = id(i);
                if (globalRow < 0 || globalRow >= size) continue;
                const int blockRow = globalRow / m_BlockSize;
                const int localRow = globalRow % m_BlockSize;

                for (int j = 0; j < idSize; ++j) {
                    const int globalCol = id(j);
                    if (globalCol < 0 || globalCol >= size) continue;
                    const int blockCol = globalCol / m_BlockSize;
                    const int localCol = globalCol % m_BlockSize;
                    // Find the block index in m_AColIdxBlock where m_AColIdxBlock[k] == blockCol 
                    // and k is in [m_ARowPtrBlock[blockRow], m_ARowPtrBlock[blockRow + 1])
                    for (int k = m_ARowPtrBlock[blockRow]; k < m_ARowPtrBlock[blockRow + 1]; ++k) {
                        if (m_AColIdxBlock[k] == blockCol) {
                            // Block k holds the blockRow, blockCol block
                            const int blockOffset = k * blockSizeSquared;
                            const int localOffset = localRow * m_BlockSize + localCol; // row-major
                            m_AValuesBlock[blockOffset + localOffset] += fact * m(i, j);
                            break;
                        }
                    }
                }
            }
        }
    } else { // m_BlockSize == 1
        if (fact == 1.0) { // Do not need to multiply by fact
            for (int i = 0; i < idSize; ++i) {
                const int globalRow = id(i);
                if (globalRow < 0 || globalRow >= size) continue;

                for (int j = 0; j < idSize; ++j) {
                    const int globalCol = id(j);
                    if (globalCol < 0 || globalCol >= size) continue;
                    // Find the block index in m_AColIdxBlock where m_AColIdxBlock[k] == blockCol 
                    // and k is in [m_ARowPtrBlock[blockRow], m_ARowPtrBlock[blockRow + 1])
                    for (int k = m_ARowPtrBlock[globalRow]; k < m_ARowPtrBlock[globalRow + 1]; ++k) {
                        if (m_AColIdxBlock[k] == globalCol) {
                            m_AValuesBlock[k] += m(i, j);
                            break;
                        }
                    }
                }
            }
        } else { // Multiply by fact
            for (int i = 0; i < idSize; ++i) {
                const int globalRow = id(i);
                if (globalRow < 0 || globalRow >= size) continue;

                for (int j = 0; j < idSize; ++j) {
                    const int globalCol = id(j);
                    if (globalCol < 0 || globalCol >= size) continue;
                    // Find the block index in m_AColIdxBlock where m_AColIdxBlock[k] == blockCol 
                    // and k is in [m_ARowPtrBlock[blockRow], m_ARowPtrBlock[blockRow + 1])
                    for (int k = m_ARowPtrBlock[globalRow]; k < m_ARowPtrBlock[globalRow + 1]; ++k) {
                        if (m_AColIdxBlock[k] == globalCol) {
                            m_AValuesBlock[k] += fact * m(i, j);
                            break;
                        }
                    }
                }
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
    if (fact == 0.0) return 0;

    const int idSize = id.Size();
    if (idSize != v.Size()) {
        opserr << "WARNING: AmgXGenLinSOE::addB() - Vector and ID not of similar sizes\n";
        return -1;
    }

    const int size = m_B.Size();

    if (fact == 1.0) {
        for (int i = 0; i < idSize; ++i) {
            const int pos = id(i);
            if (pos >= 0 && pos < size) m_B[pos] += v(i);
        }
    } else if (fact == -1.0) {
        for (int i = 0; i < idSize; ++i) {
            const int pos = id(i);
            if (pos >= 0 && pos < size) m_B[pos] -= v(i);
        }
    } else {
        for (int i = 0; i < idSize; ++i) {
            const int pos = id(i);
            if (pos >= 0 && pos < size) m_B[pos] += fact * v(i);
        }
    }

    return 0;
}

int AmgXGenLinSOE::setB(const Vector &v, double fact)
{
    // check for a quick return 
    if (fact == 0.0)  {
        m_B.Zero();
        return 0;
    }

    const int size = m_B.Size();

    if (v.Size() != size) {
        opserr << "WARNING: AmgXGenLinSOE::setB() -";
        opserr << " incompatible sizes " << size << " and " << v.Size() << endln;
        return -1;
    }

    if (fact == 1.0) { // do not need to multiply if fact == 1.0
        for (int i = 0; i < size; i++) {
            m_B[i] = v(i);
        }
    } else if (fact == -1.0) {
        for (int i = 0; i < size; i++) {
            m_B[i] = -v(i);
        }
    } else {
        for (int i = 0; i < size; i++) {
            m_B[i] = v(i) * fact;
        }
    }
    
    return 0;
}

void AmgXGenLinSOE::zeroA(void)
{
    m_AValuesBlock.assign(m_AValuesBlock.size(),0.0);

    // Update matrix status
    if (m_matrixStatus == AmgXMatrixStatus::UNCHANGED) {
        m_matrixStatus = AmgXMatrixStatus::COEFFICIENTS_CHANGED;
    }
}

void AmgXGenLinSOE::zeroB(void)
{
    m_B.Zero();
}

void AmgXGenLinSOE::setX(int loc, double value)
{
    const int size = m_X.Size();

    if (loc < size && loc >= 0) {
        m_X(loc) = value;
    }
}

void AmgXGenLinSOE::setX(const Vector &x)
{
    const int size = m_X.Size();

    if (x.Size() == size) {
        m_X = x;
    }
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

int AmgXGenLinSOE::setAmgXGenLinSolver(AmgXGenLinSolver &newSolver)
{
    newSolver.setLinearSOE(*this);
    if (m_X.Size() != 0) {
        int solverOK = newSolver.setSize();
        if (solverOK < 0) {
            opserr << "WARNING: AmgXGenLinSOE::setSolver :";
            opserr << "the new solver could not setSize() - staying with old\n";
            return -1;
        }
    }
    return this->LinearSOE::setSolver(newSolver);
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
    if (m_matrixStatus != AmgXMatrixStatus::UNCHANGED && m_paddingEnabled && originalSize < size) {
        const int blockOffset = m_AValuesBlock.size() - m_BlockSize * m_BlockSize;
        const int startRow = originalSize % m_BlockSize;

        double avgDiag = 0.0;
        double maxAbsDiag = 0.0;

        // Loop: compute sum and max(abs(diag))
        for (int localRow = 0; localRow < startRow; ++localRow) {
            const int idx = blockOffset + localRow * m_BlockSize + localRow;
            const double diag = m_AValuesBlock[idx];
            const double absDiag = (diag >= 0.0) ? diag : -diag;
            avgDiag += absDiag;
            if (absDiag > maxAbsDiag) maxAbsDiag = absDiag;
        }

        if (startRow > 0) {
            // Compute representative diagonal value
            avgDiag /= static_cast<double>(startRow);
            const double minDiag = 1e-3 * maxAbsDiag;
            double repDiagValue = (avgDiag > minDiag) ? avgDiag : minDiag;

            // Loop to set padded rows
            for (int localRow = startRow; localRow < m_BlockSize; ++localRow) {
                const int idx = blockOffset + localRow * m_BlockSize + localRow;
                m_AValuesBlock[idx] = repDiagValue;
            }
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
        return -1;
    }
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
 * These functions help determine the optimal block size for the block CSR format
 * by analyzing the sparsity pattern of the system of equations.
 * The estimation is based on the efficiency of different block sizes, where
 * efficiency is defined as the ratio of nonzeros to the total number of elements
 * in the blocks. The algorithm tries to find the largest block size that maintains
 * a good efficiency (i.e., not too many zeros within blocks).
 *
 * This algorithm is adapted from the implementation in SciPy's `sparsetools`:
 * https://github.com/scipy/scipy/blob/0f1fd4a7268b813fa2b844ca6038e4dfdf90084a/scipy/sparse/sparsetools/csr.h#L205-L254
 */

// Estimate the number of blocks in the graph.
// Assumes the system of equations forms a square sparse matrix.
int AmgXGenLinSOE::estimateBlockSize(Graph &theGraph, int nnz, double efficiency)
{
    const int size = theGraph.getNumVertex();
    if (size < 0) {
        opserr << "WARNING: size of soe < 0 -- AmgXGenLinSOE::estimateBlockSize" << endln;
        return -1;
    }

    if (nnz == 0) {
        return 1;
    }

    if (efficiency <= 0.0 || efficiency >= 1.0) {
        opserr << "WARNING: efficiency must satisfy 0.0 < efficiency < 1.0 -- AmgXGenLinSOE::estimateBlockSize" << endln;
        return -1;
    }

    double e4 = 0.0, e3 = 0.0, e2 = 0.0;

    // Try block size 4
    if (m_paddingEnabled || size % 4 == 0) {
        int nb4 = countBlocks(theGraph, 4);
        if (nb4 > 0)
            e4 = nnz / static_cast<double>(16 * nb4);
        if (e4 > efficiency)
            return 4;
    }

    // Try block size 3
    if (m_paddingEnabled || size % 3 == 0) {
        int nb3 = countBlocks(theGraph, 3);
        if (nb3 > 0)
            e3 = nnz / static_cast<double>(9 * nb3);
        if (e3 > efficiency)
            return 3;
    }

    // Try block size 2
    if (m_paddingEnabled || size % 2 == 0) {
        int nb2 = countBlocks(theGraph, 2);
        if (nb2 > 0)
            e2 = nnz / static_cast<double>(4 * nb2);
        if (e2 > efficiency)
            return 2;
    }

    // Fallback: block size 1
    return 1;
}
// Count the number of square blocks in the graph.
// Assumes the system of equations forms a square sparse matrix 
// and the graph is undirected.
int AmgXGenLinSOE::countBlocks(Graph &theGraph, int blockSize)
{
    int size = theGraph.getNumVertex();
    if (size < 0) {
        opserr << "WARNING: size of soe < 0 -- AmgXGenLinSOE::countBlocks" << endln;
        return -1;
    }

    if (blockSize < 1) {
        opserr << "WARNING: blockSize must be >= 1 -- AmgXGenLinSOE::countBlocks" << endln;
        return -1;
    }

    if (size % blockSize != 0 && m_paddingEnabled == false) {
        opserr << "WARNING: size of soe must be divisible by blockSize -- AmgXGenLinSOE::countBlocks" << endln;
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
            opserr << "WARNING: AmgXGenLinSOE::setSize :"
                << " vertex " << i << " not in graph! - size set to 0\n";
            size = 0;
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