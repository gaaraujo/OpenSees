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

void defaultAmgXCallback(const char* msg, int length) {
    if (msg && length > 0) {
        opserr.write(msg, length);
        opserr << endln;
    }
}

AmgXGenLinSOE::AmgXGenLinSOE(AmgXGenLinSolver &the_Solver, 
    char *configFile, char *configOptions, 
    AMGX_Mode mode, int blockSize,
    void (*callback)(const char* msg, int length))
    : LinearSOE(the_Solver, LinSOE_TAGS_AmgXGenLinSOE), 
    _X(), _B(), _ARowPtrBlock(), _AColIdxBlock(), _AValuesBlock(), _BlockSize(blockSize)
{
    the_Solver.setLinearSOE(*this);

    /* Initialize AMGX library - only done once across all instances */
    if (!_AmgXInitialized) {
        AMGX_SAFE_CALL(AMGX_initialize());
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
        _AmgXInitialized = true;
    }
    AMGX_SAFE_CALL(AMGX_register_print_callback(callback));

    if (configFile != nullptr && configOptions != nullptr) {
        AMGX_SAFE_CALL(AMGX_config_create_from_file_and_string(&_Config, configFile, configOptions));
    } else if (configFile != nullptr) {
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&_Config, configFile));
    } else if (configOptions != nullptr) {
        AMGX_SAFE_CALL(AMGX_config_create(&_Config, configOptions));
    } else {
        opserr << "AmgXGenLinSOE: No config file or options provided" << endln;
        opserr << "AmgXGenLinSOE: Using default config" << endln;

        /* The following settings create an Aggregation solver with DILU 
         * smoother, with 1 pre and 1 post sweep. The solver will stop when 
         * the L2 norm has been reduced by 1000 from the initial norm.
         */
        const char* defaultOptions =
            "config_version=2, "
            "algorithm=AGGREGATION, "
            "selector=ONE_PHASE_HANDSHAKING, "
            "cycle=V, "
            "smoother=MULTICOLOR_DILU, "
            "presweeps=1, "
            "postsweeps=1, "
            "coarse_solver=NOSOLVER, "
            "coarsest_sweeps=2, "
            "max_levels=1000, "
            "norm=L2, "
            "convergence=RELATIVE_INI, "
            "max_uncolored_percentage=0.15, "
            "max_iters=1000, "
            "monitor_residual=1, "
            "tolerance=0.001, "
            "print_solve_stats=1, "
            "print_grid_stats=1, "
            "obtain_timings=1;";
        AMGX_SAFE_CALL(AMGX_config_create(&_Config, defaultOptions));
    }

    /* Monitor residual and store residual history */
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&_Config, "monitor_residual=1, store_res_history=1"));

    /* Switch on internal error handling 
     * (no need to use AMGX_SAFE_CALL after this point) */
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&_Config, "exception_handling=1"));

    /* Create resources: single-GPU and single-threaded applications only */
    AMGX_resources_create_simple(&_Resources, _Config);

    /* Create solver, matrix, rhs and solution vectors */
    AMGX_solver_create(&_Solver, _Resources, _Mode, _Config);
    AMGX_matrix_create(&_Matrix, _Resources, _Mode);
    AMGX_vector_create(&_RHS, _Resources, _Mode);
    AMGX_vector_create(&_Solution, _Resources, _Mode);

    _ActiveSolverInstances++;
}

AmgXGenLinSOE::AmgXGenLinSOE(): LinearSOE(LinSOE_TAGS_AmgXGenLinSOE), 
    _X(), _B(), _ARowPtrBlock(), _AColIdxBlock(), _AValuesBlock(), _BlockSize()
{
    
}

AmgXGenLinSOE::~AmgXGenLinSOE() 
{
    /* destroy resources, matrix, vector and solver */
    if (_Solution) { AMGX_vector_destroy(_Solution); _Solution = nullptr; }
    if (_RHS) { AMGX_vector_destroy(_RHS); _RHS = nullptr; }
    if (_Matrix) { AMGX_matrix_destroy(_Matrix); _Matrix = nullptr; }
    if (_Solver) { AMGX_solver_destroy(_Solver); _Solver = nullptr; }
    if (_Resources) { AMGX_resources_destroy(_Resources); _Resources = nullptr; }
    
    /* destroy config (need to use AMGX_SAFE_CALL after this point) */
    if (_Config) { AMGX_SAFE_CALL(AMGX_config_destroy(_Config)); _Config = nullptr; }

    if (_ActiveSolverInstances > 0) {
        _ActiveSolverInstances--;
    }

    // Finalize AMGX only when last instance is destroyed
    if (_ActiveSolverInstances == 0 && _AmgXInitialized) {
        AMGX_reset_signal_handler();
        AMGX_SAFE_CALL(AMGX_finalize());
        _AmgXInitialized = false;
    }
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
        opserr << "Error: either block size could not be estimated ";
        opserr << "or provided block size is invalid. ";
        opserr << "Please manually provide a positive block size." << endln;
        return -1;
    }
    
    if (size % _BlockSize != 0) {
        opserr << "Error: the number of equations is not divisible by the block size. ";
        opserr << "Please provide a block size that divides the number of equations evenly, ";
        opserr << "or set the block size to 0 to automatically estimate it. " << endln;
        return -1;
    }

    // Special case for BlockSize = 1 - treat as regular CSR format
    if (_BlockSize == 1) {
        _AColIdxBlock.clear();
        _ARowPtrBlock.clear();
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
            std::vector<int> colIdx;
            colIdx.reserve(theAdjacency.Size() + 1);

            // Add diagonal first
            colIdx.push_back(theVertex->getTag());

            // Add adjacency entries
            for (int j = 0; j < theAdjacency.Size(); ++j) {
                colIdx.push_back(theAdjacency(j));
            }

            // Sort and remove duplicates (in case diagonal == adjacency entry)
            std::sort(colIdx.begin(), colIdx.end());
            colIdx.erase(std::unique(colIdx.begin(), colIdx.end()), colIdx.end());

            // Append to global col index block
            _AColIdxBlock.insert(_AColIdxBlock.end(), colIdx.begin(), colIdx.end());

            // Update row pointer
            _ARowPtrBlock.push_back(_AColIdxBlock.size());
        }
        
        // Allocate _AValuesBlock
        _AValuesBlock.resize(_AColIdxBlock.size(), 0.0); // if using same vector

    } else {
        // Block size > 1 case - original block CSR code
        int numBlockRows = size / _BlockSize;
        int numBlockCols = size / _BlockSize;
        
        // Prepare block structure
        _ARowPtrBlock.resize(numBlockRows + 1, 0);
        std::vector<int> mask(numBlockCols, -1);
        std::vector<std::vector<int>> colIdxPerBlockRow(numBlockRows);

        // Loop over vertices (rows), and their adjacency (columns)
        // Note: this assumes the graph is undirected
        VertexIter &theVertices = theGraph.getVertices();
        Vertex* vertex = nullptr;

        while ((vertex = theVertices()) != nullptr) {
            int row = vertex->getTag();  // global scalar row index
            int blockRow = row / _BlockSize;

            const ID& adjacency = vertex->getAdjacency();  // connected columns
            
            // Insert the diagonal block
            if (mask[blockRow] != blockRow) {
                mask[blockRow] = blockRow;
                colIdxPerBlockRow[blockRow].push_back(blockRow);
            }

            // Insert other adjacency blocks
            for (int k = 0; k < adjacency.Size(); ++k) {
                int col = adjacency(k);
                if (col < 0 || col >= size) continue;
                int blockCol = col / _BlockSize;
                if (mask[blockCol] != blockRow) {
                    mask[blockCol] = blockRow;
                    colIdxPerBlockRow[blockRow].push_back(blockCol);
                }
            }
        }

        // Finalize _ARowPtrBlock and sort
        for (int blockRow = 0; blockRow < numBlockRows; ++blockRow) {
            std::sort(colIdxPerBlockRow[blockRow].begin(), colIdxPerBlockRow[blockRow].end());
            _ARowPtrBlock[blockRow + 1] = _ARowPtrBlock[blockRow] + colIdxPerBlockRow[blockRow].size();
        }

        // Fill _AColIdxBlock and allocate _AValuesBlock
        int nnzBlock = _ARowPtrBlock[numBlockRows];
        _AColIdxBlock.resize(nnzBlock);
        _AValuesBlock.resize(nnzBlock * _BlockSize * _BlockSize, 0.0); // row-major block storage

        int idx = 0;
        for (int blockRow = 0; blockRow < numBlockRows; ++blockRow) {
            for (int blockCol : colIdxPerBlockRow[blockRow]) {
                _AColIdxBlock[idx++] = blockCol;
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
        opserr << "WARNING:AmgXGenLinSOE::setSize :";
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
        opserr << "AmgXGenLinSOE::addA() ";
        opserr << " - Matrix and ID not of similar sizes\n";
        return -1;
    }
    

    int size = _X.Size();

    if (_BlockSize > 1) {
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
        opserr << "AmgXGenLinSOE::addB() - Vector and ID not of similar sizes\n";
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
        opserr << "WARNING AmgXGenLinSOE::setB() -";
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
            opserr << "WARNING:AmgXGenLinSOE::setSolver :";
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

/* Utility functions to convert from CSR to Block CSR format. 
 * This algorithm is adapted from the implementation in SciPy's `sparsetools`. 
 * https://github.com/scipy/scipy/blob/0f1fd4a7268b813fa2b844ca6038e4dfdf90084a/scipy/sparse/sparsetools/csr.h#L205-L254
 */

// Estimate the number of blocks in the graph.
// Assumes the system of equations forms a square sparse matrix.
int AmgXGenLinSOE::estimateBlockSize(Graph &theGraph, int nnz, double efficiency)
{
    int size = theGraph.getNumVertex();
    if (size < 0) {
        opserr << "size of soe < 0\n";
        return -1;
    }

    if (nnz == 0) {
        return 1;
    }

    if (efficiency <= 0.0 || efficiency >= 1.0) {
        opserr << "efficiency must satisfy 0.0 < efficiency < 1.0" << endln;
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
        opserr << "size of soe < 0\n";
        return -1;
    }

    int numBlockCols = size / blockSize;
    std::vector<int> mask(numBlockCols + 1, -1);
    int totalNumBlocks = 0;

    for (int i = 0; i < size; i++) {
        int blockRow = i / blockSize;
        Vertex* theVertex = theGraph.getVertexPtr(i);
        if (theVertex == nullptr) {
            opserr << "WARNING: AmgXGenLinSOE::setSize :"
                << " vertex " << row << " not in graph! - size set to 0\n";
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