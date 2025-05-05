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

AmgXGenLinSOE::AmgXGenLinSOE(AmgXGenLinSolver &the_Solver, int blockSize)
    : LinearSOE(the_Solver, LinSOE_TAGS_AmgXGenLinSOE), 
    _X(), _B(), _ARowPtrBlock(), _AColIdxBlock(), _AValuesBlock(), _BlockSize(blockSize)
{
    the_Solver.setLinearSOE(*this);
}

AmgXGenLinSOE::AmgXGenLinSOE(): LinearSOE(LinSOE_TAGS_AmgXGenLinSOE), 
    _X(), _B(), _ARowPtrBlock(), _AColIdxBlock(), _AValuesBlock(), _BlockSize()
{
    
}

AmgXGenLinSOE::~AmgXGenLinSOE() 
{
    
}

int AmgXGenLinSOE::getNumEqn(void) const 
{
    return _X.Size();
}

int AmgXGenLinSOE::setSize(Graph &theGraph) 
{
    int size = theGraph.getNumVertex();
    if (size < 0) {
        opserr << "size of soe < 0\n";
        return -1;
    }

    // First iterate through the vertices of the graph to get nnz
    Vertex *theVertex;
    int nnz = 0;
    VertexIter &theVertices = theGraph.getVertices();
    while ((theVertex = theVertices()) != 0) {
        const ID &theAdjacency = theVertex->getAdjacency();
        nnz += theAdjacency.Size() +1; // the +1 is for the diag entry
    }

    // Step 0: Estimate the block size
    if (_BlockSize == 0) {
        _BlockSize = estimateBlockSize(theGraph, nnz);
    }
    if (_BlockSize == -1) {
        opserr << "WARNING: either block size could not be estimated ";
        opserr << "or provided block size is invalid. ";
        opserr << "Please manually provide a positive block size. -- AmgXGenLinSOE::setSize" << endln;
        return -1;
    }
    
    if (size % _BlockSize != 0) {
        opserr << "WARNING: the number of equations is not divisible by the block size. ";
        opserr << "Please provide a block size that divides the number of equations evenly, ";
        opserr << "or set the block size to 0 to automatically estimate it. -- AmgXGenLinSOE::setSize" << endln;
        return -1;
    }

    // Special case for BlockSize = 1 - treat as regular CSR format
    if (_BlockSize == 1) {
        // Reserve space for matrix A
        _ARowPtrBlock.reserve(size + 1);
        _AColIdxBlock.reserve(nnz);
        _AValuesBlock.resize(nnz, 0.0);

        // Fill in _ARowPtrBlock and _AColIdxBlock
        _ARowPtrBlock.push_back(0); // Start of first row

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
                _AColIdxBlock.push_back(colIdx(i));
            }

            // Update row pointer
            _ARowPtrBlock.push_back(_AColIdxBlock.size());
        }
    } else {
        // Block size > 1 -> block CSR format
        int numBlockRows = size / _BlockSize;
        int numBlockCols = size / _BlockSize;
        
        // Prepare block structure
        _ARowPtrBlock.resize(numBlockRows + 1, 0);
        std::vector<int> mask(numBlockCols, -1);
        std::vector<ID> colIdxPerBlockRow(numBlockRows);

        // Loop over vertices (rows), and their adjacency (columns)
        // Note: this assumes the graph is undirected
        VertexIter &theVertices = theGraph.getVertices();
        Vertex* theVertex = nullptr;

        while ((theVertex = theVertices()) != nullptr) {
            int row = theVertex->getTag();  // global scalar row index
            int blockRow = row / _BlockSize;

            const ID& theAdjacency = theVertex->getAdjacency();  // connected columns

            // Insert the diagonal block
            if (mask[blockRow] != blockRow) {
                mask[blockRow] = blockRow;
                colIdxPerBlockRow[blockRow].insert(blockRow);
            }

            // Insert other adjacency blocks
            for (int k = 0; k < theAdjacency.Size(); ++k) {
                int col = theAdjacency(k);
                int blockCol = col / _BlockSize;
                if (mask[blockCol] != blockRow) {
                    mask[blockCol] = blockRow;
                    colIdxPerBlockRow[blockRow].insert(blockCol);
                }
            }
        }

        // Finalize _ARowPtrBlock and sort
        for (int blockRow = 0; blockRow < numBlockRows; ++blockRow) {
            _ARowPtrBlock[blockRow + 1] = _ARowPtrBlock[blockRow] + colIdxPerBlockRow[blockRow].Size();
        }

        // Fill _AColIdxBlock and allocate _AValuesBlock
        int nnzBlock = _ARowPtrBlock[numBlockRows];
        _AColIdxBlock.reserve(nnzBlock);
        _AValuesBlock.resize(nnzBlock * _BlockSize * _BlockSize, 0.0);

        for (int blockRow = 0; blockRow < numBlockRows; ++blockRow) {
            for (int blockCol = 0; blockCol < colIdxPerBlockRow[blockRow].Size(); ++blockCol) {
                _AColIdxBlock.push_back(colIdxPerBlockRow[blockRow](blockCol));
            }
        }
    }

    // Allocate solution and RHS vectors
    _X.resize(size); _X.Zero();
    _B.resize(size); _B.Zero();

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

    int idSize = id.Size();

    // Check that m and id are of similar size
    if (idSize != m.noRows() && idSize != m.noCols()) {
        opserr << "WARNING: AmgXGenLinSOE::addA() ";
        opserr << " - Matrix and ID not of similar sizes\n";
        return -1;
    }
    

    int size = _X.Size();

    if (_BlockSize > 1) { // Block CSR format
        if (fact == 1.0) { // Do not need to multiply by fact
            for (int i = 0; i < idSize; ++i) {
                int globalRow = id(i);
                if (globalRow < 0 || globalRow >= size) continue;
                int blockRow = globalRow / _BlockSize;
                int localRow = globalRow % _BlockSize;

                for (int j = 0; j < idSize; ++j) {
                    int globalCol = id(j);
                    if (globalCol < 0 || globalCol >= size) continue;
                    int blockCol = globalCol / _BlockSize;
                    int localCol = globalCol % _BlockSize;
                    // Find the block index in _AColIdxBlock where _AColIdxBlock[k] == blockCol 
                    // and k is in [_ARowPtrBlock[blockRow], _ARowPtrBlock[blockRow + 1])
                    for (int k = _ARowPtrBlock[blockRow]; k < _ARowPtrBlock[blockRow + 1]; ++k) {
                        if (_AColIdxBlock[k] == blockCol) {
                            // Block k holds the blockRow, blockCol block
                            int blockOffset = k * _BlockSize * _BlockSize;
                            int localOffset = localRow * _BlockSize + localCol; // row-major
                            _AValuesBlock[blockOffset + localOffset] += m(i, j);
                            break;
                        }
                    }
                }
            }
        } else { // Multiply by fact
            for (int i = 0; i < idSize; ++i) {
                int globalRow = id(i);
                if (globalRow < 0 || globalRow >= size) continue;
                int blockRow = globalRow / _BlockSize;
                int localRow = globalRow % _BlockSize;

                for (int j = 0; j < idSize; ++j) {
                    int globalCol = id(j);
                    if (globalCol < 0 || globalCol >= size) continue;
                    int blockCol = globalCol / _BlockSize;
                    int localCol = globalCol % _BlockSize;
                    // Find the block index in _AColIdxBlock where _AColIdxBlock[k] == blockCol 
                    // and k is in [_ARowPtrBlock[blockRow], _ARowPtrBlock[blockRow + 1])
                    for (int k = _ARowPtrBlock[blockRow]; k < _ARowPtrBlock[blockRow + 1]; ++k) {
                        if (_AColIdxBlock[k] == blockCol) {
                            // Block k holds the blockRow, blockCol block
                            int blockOffset = k * _BlockSize * _BlockSize;
                            int localOffset = localRow * _BlockSize + localCol; // row-major
                            _AValuesBlock[blockOffset + localOffset] += fact * m(i, j);
                            break;
                        }
                    }
                }
            }
        }
    } else { // _BlockSize == 1
        if (fact == 1.0) { // Do not need to multiply by fact
            for (int i = 0; i < idSize; ++i) {
                int globalRow = id(i);
                if (globalRow < 0 || globalRow >= size) continue;

                for (int j = 0; j < idSize; ++j) {
                    int globalCol = id(j);
                    if (globalCol < 0 || globalCol >= size) continue;
                    // Find the block index in _AColIdxBlock where _AColIdxBlock[k] == blockCol 
                    // and k is in [_ARowPtrBlock[blockRow], _ARowPtrBlock[blockRow + 1])
                    for (int k = _ARowPtrBlock[globalRow]; k < _ARowPtrBlock[globalRow + 1]; ++k) {
                        if (_AColIdxBlock[k] == globalCol) {
                            _AValuesBlock[k] += m(i, j);
                            break;
                        }
                    }
                }
            }
        } else { // Multiply by fact
            for (int i = 0; i < idSize; ++i) {
                int globalRow = id(i);
                if (globalRow < 0 || globalRow >= size) continue;

                for (int j = 0; j < idSize; ++j) {
                    int globalCol = id(j);
                    if (globalCol < 0 || globalCol >= size) continue;
                    // Find the block index in _AColIdxBlock where _AColIdxBlock[k] == blockCol 
                    // and k is in [_ARowPtrBlock[blockRow], _ARowPtrBlock[blockRow + 1])
                    for (int k = _ARowPtrBlock[globalRow]; k < _ARowPtrBlock[globalRow + 1]; ++k) {
                        if (_AColIdxBlock[k] == globalCol) {
                            _AValuesBlock[k] += fact * m(i, j);
                            break;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

int AmgXGenLinSOE::addB(const Vector &v, const ID &id, double fact)
{
    if (fact == 0.0) return 0;

    int idSize = id.Size();
    if (idSize != v.Size()) {
        opserr << "WARNING: AmgXGenLinSOE::addB() - Vector and ID not of similar sizes\n";
        return -1;
    }

    int size = _B.Size();

    if (fact == 1.0) {
        for (int i = 0; i < idSize; ++i) {
            int pos = id(i);
            if (pos >= 0 && pos < size) _B[pos] += v(i);
        }
    } else if (fact == -1.0) {
        for (int i = 0; i < idSize; ++i) {
            int pos = id(i);
            if (pos >= 0 && pos < size) _B[pos] -= v(i);
        }
    } else {
        for (int i = 0; i < idSize; ++i) {
            int pos = id(i);
            if (pos >= 0 && pos < size) _B[pos] += fact * v(i);
        }
    }

    return 0;
}

int AmgXGenLinSOE::setB(const Vector &v, double fact)
{
    // check for a quick return 
    if (fact == 0.0)  {
        _B.Zero();
        return 0;
    }

    int size = _B.Size();
    if (v.Size() != size) {
        opserr << "WARNING: AmgXGenLinSOE::setB() -";
        opserr << " incompatible sizes " << size << " and " << v.Size() << endln;
        return -1;
    }

    if (fact == 1.0) { // do not need to multiply if fact == 1.0
        for (int i = 0; i < size; i++) {
            _B[i] = v(i);
        }
    } else if (fact == -1.0) {
        for (int i = 0; i < size; i++) {
            _B[i] = -v(i);
        }
    } else {
        for (int i = 0; i < size; i++) {
            _B[i] = v(i) * fact;
        }
    }
    
    return 0;
}

void AmgXGenLinSOE::zeroA(void)
{
    _AValuesBlock.assign(_AValuesBlock.size(),0.0);
}

void AmgXGenLinSOE::zeroB(void)
{
    _B.Zero();
}

void AmgXGenLinSOE::setX(int loc, double value)
{
    if (loc < _X.Size() && loc >= 0) {
        _X(loc) = value;
    }
}

void AmgXGenLinSOE::setX(const Vector &x)
{
    if (x.Size() == _X.Size()) {
        _X = x;
    }
}

const Vector & AmgXGenLinSOE::getX(void)
{
    return _X;
}   

const Vector & AmgXGenLinSOE::getB(void)
{
    return _B;
}

double AmgXGenLinSOE::normRHS(void)
{
    return _B.Norm();
}

int AmgXGenLinSOE::setAmgXGenLinSolver(AmgXGenLinSolver &newSolver)
{
    newSolver.setLinearSOE(*this);
    if (_X.Size() != 0) {
        int solverOK = newSolver.setSize();
        if (solverOK < 0) {
            opserr << "WARNING: AmgXGenLinSOE::setSolver :";
            opserr << "the new solver could not setSize() - staying with old\n";
            return -1;
        }
    }
    return this->LinearSOE::setSolver(newSolver);
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
    int size = theGraph.getNumVertex();
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

    double high_efficiency = (1.0 + efficiency) / 2.0;
    double e22 = 0, e33 = 0, e44 = 0, e66 = 0;

    if (size % 2 == 0) {
        int nb22 = countBlocks(theGraph, 2);
        e22 = nnz / (static_cast<double>(4 * nb22));
    }

    if (size % 3 == 0) {
        int nb33 = countBlocks(theGraph, 3);
        e33 = nnz / (static_cast<double>(9 * nb33));
    }

    if (e22 > high_efficiency && e33 > high_efficiency) {
        if (size % 6 == 0) {
            int nb66 = countBlocks(theGraph, 6);
            e66 = nnz / (static_cast<double>(36 * nb66));
            if (e66 > efficiency) {
                return 6;
            } else {
                return 3;
            }
        } else {
            return 3;
        }
    } else {
        if (size % 4 == 0) {
            int nb44 = countBlocks(theGraph, 4);
            e44 = nnz / (static_cast<double>(16 * nb44));
        }

        if (e44 > efficiency) {
            return 4;
        } else if (e33 > efficiency) {
            return 3;
        } else if (e22 > efficiency) {
            return 2;
        } else {
            return 1;
        }
    }
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

    if (size % blockSize != 0) {
        opserr << "WARNING: size of soe must be divisible by blockSize -- AmgXGenLinSOE::countBlocks" << endln;
        return -1;
    }

    int numBlockCols = size / blockSize;
    std::vector<int> mask(numBlockCols, -1);
    int totalNumBlocks = 0;

    for (int i = 0; i < size; i++) {
        int blockRow = i / blockSize;
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
            int blockCol = adjacency(j) / blockSize; 
            if (mask[blockCol] != blockRow) {
                // if this block-col hasn't seen this block-row
                mask[blockCol] = blockRow;
                totalNumBlocks++;
            }
        }
    }

    return totalNumBlocks;
}