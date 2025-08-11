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
#include <LinearSOESolver.h>
#include <Vector.h>
#include <vector> // for std::vector
#include <OPS_Stream.h>  // needed for opserr

class AmgXGenLinSOE : public LinearSOE
{
    public:
        // Constants for block size limits and efficiency thresholds
        static constexpr int MAX_BLOCK_SIZE = 32;
        static constexpr int DEFAULT_BLOCK_SIZE = 1;
        static constexpr double DEFAULT_EFFICIENCY_THRESHOLD = 0.7;
        static constexpr double MIN_DIAGONAL_VALUE_FACTOR = 1e-3;

        AmgXGenLinSOE(LinearSOESolver &theSolver, 
                      int blockSize = DEFAULT_BLOCK_SIZE, 
                      bool paddingEnabled = true,
                      bool verbose = false);
        AmgXGenLinSOE();

        ~AmgXGenLinSOE();
        
        // Core LinearSOE interface methods
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
        int setAmgXGenLinSolver(LinearSOESolver &newSolver);   
        int solve(void);
        int saveSparseA(OPS_Stream& output, int baseIndex = 0);

        int sendSelf(int commitTag, Channel &theChannel);   
        int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker);  

        // Track changes in the matrix
        enum class AmgXMatrixStatus {
            UNCHANGED, // Matrix is the same as the last solve
            COEFFICIENTS_CHANGED, // Only the coefficients of the matrix have changed
            STRUCTURE_CHANGED // Both the size and coefficients of the matrix have changed
        };

        template<typename DataType>
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

        // Block CSR format conversion and utility methods
        int estimateBlockSize(Graph &theGraph, int nnz, double efficiency = DEFAULT_EFFICIENCY_THRESHOLD);
        int countBlocks(Graph &theGraph, int blockSize);
        int fillPaddedDiagonals(double value = 0.0, bool autoCompute = true);
        
        // Helper methods for assembly
        int addAMatrixElement(int globalRow, int globalCol, double value);
        int addAMatrixElementBlock(int globalRow, int globalCol, double value);
        int addAMatrixElementStandard(int globalRow, int globalCol, double value);
        inline double applyFact(double input, double fact) {
            return (fact == 1.0) ? input : (fact == -1.0) ? -input : fact * input;
        }        
        // Validation methods
        bool isValidBlockSize(int blockSize) const;
        bool isValidGlobalIndex(int index) const;
};

inline OPS_Stream& operator<<(OPS_Stream& os, AmgXGenLinSOE::AmgXMatrixStatus status) {
    switch (status) {
        case AmgXGenLinSOE::AmgXMatrixStatus::UNCHANGED:
            return os << "UNCHANGED";
        case AmgXGenLinSOE::AmgXMatrixStatus::COEFFICIENTS_CHANGED:
            return os << "COEFFICIENTS_CHANGED";
        case AmgXGenLinSOE::AmgXMatrixStatus::STRUCTURE_CHANGED:
            return os << "STRUCTURE_CHANGED";
        default:
            return os << "UNKNOWN";
    }
}

#endif