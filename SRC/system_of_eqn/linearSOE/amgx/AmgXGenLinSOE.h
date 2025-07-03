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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/amgx/AmgXGenLinSOE.h
                                                                        
// Written: gaaraujo 
// Created: 02/2025
//
// Description: This file contains the class definition for 
// AmgXGenLinSOE. It stores the sparse matrix A in a fashion
// required by the AmgXGenLinSolver object.
//

#ifndef AmgXGenLinSOE_h
#define AmgXGenLinSOE_h

#include <LinearSOE.h>
#include <Vector.h>
#include <vector> // for std::vector
#include <OPS_Stream.h>  // needed for opserr

class AmgXGenLinSolver;

class AmgXGenLinSOE : public LinearSOE
{
    public:
        AmgXGenLinSOE(AmgXGenLinSolver &theSolver, 
                      int blockSize = 1, bool paddingEnabled = true,
                      bool verbose = false);
        AmgXGenLinSOE();

        ~AmgXGenLinSOE();
        
        int getNumEqn(void) const;
        int setSize(Graph &theGraph);
        int addA(const Matrix &, const ID &, double fact = 1.0);
        int addB(const Vector &, const ID &, double fact = 1.0);
        int setB(const Vector &, double fact = 1.0);
        
        void zeroA(void);
        void zeroB(void);

        const Vector &getX(void);
        const Vector &getB(void);
        double normRHS(void);   

        void setX(int loc, double value);
        void setX(const Vector &x);
        int setAmgXGenLinSolver(AmgXGenLinSolver &newSolver);   
        int solve(void);
        int saveSparseA(OPS_Stream& output, int baseIndex = 0);

        int sendSelf(int commitTag, Channel &theChannel);   
        int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker);  

        // Track changes in the matrix
        enum AmgXMatrixStatus {
            UNCHANGED, // Matrix is the same as the last solve
            COEFFICIENTS_CHANGED, // Only the coefficients of the matrix have changed
            STRUCTURE_CHANGED // Both the size and coefficients of the matrix have changed
        };

        friend class AmgXGenLinSolver;

    protected:

    private:    
        // Track the status of the matrix
        AmgXMatrixStatus m_matrixStatus;

        // RHS and solution vectors
        Vector m_X, m_B; // for interfacing with OpenSees
        std::vector<double> m_XPadded, m_BPadded; // for internal use

        // Block CSR format for sparse matrix A
        std::vector<int> m_ARowPtrBlock, m_AColIdxBlock;
        std::vector<double> m_AValuesBlock;
        int m_BlockSize;

        // Whether to pad the matrix with zeros to make it a multiple of the block size
        bool m_paddingEnabled;

        // Whether to print verbose output
        bool m_verbose;

        // Block CSR format conversion
        int estimateBlockSize(Graph &theGraph, int nnz, double efficiency = 0.7);
        int countBlocks(Graph &theGraph, int block_size);
        int fillPaddedDiagonals(double value = 0.0, bool autoCompute = true);
};

#endif