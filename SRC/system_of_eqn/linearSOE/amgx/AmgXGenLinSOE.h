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

#include <amgx_c.h>
#include <LinearSOE.h>
#include <Vector.h>
#include <vector>

#include <OPS_Stream.h>  // needed for opserr

// Forward declaration of the default callback
void defaultAmgXCallback(const char* msg, int length);

class AmgXGenLinSolver;

class AmgXGenLinSOE : public LinearSOE
{
    public:
        AmgXGenLinSOE(AmgXGenLinSolver &theSolver, 
            char *configFile, char *configOptions, 
            AMGX_Mode mode = AMGX_mode_dDDI, int blockSize = 1,
            void (*callback)(const char* msg, int length) = defaultAmgXCallback);
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
        
        int sendSelf(int commitTag, Channel &theChannel);   
        int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker);  

        friend class AmgXGenLinSolver;

    protected:

    private:    
        // RHS and solution vectors
        Vector _X, _B;

        // Block CSR format for sparse matrix A
        std::vector<int> _ARowPtrBlock, _AColIdxBlock;
        std::vector<double> _AValuesBlock;
        int _BlockSize;

        // Static members for global state
        static bool _AmgXInitialized;           ///< Whether AMGX is initialized
        static int _ActiveSolverInstances;     ///< Count of active solver instances

        // AMGX handles
        AMGX_config_handle    _Config       = nullptr;  ///< Configuration handle
        AMGX_resources_handle _Resources    = nullptr;  ///< Resources handle
        AMGX_matrix_handle    _Matrix       = nullptr;  ///< Matrix handle
        AMGX_vector_handle    _RHS          = nullptr;  ///< Right-hand side vector handle
        AMGX_vector_handle    _Solution     = nullptr;  ///< Solution vector handle
        AMGX_solver_handle    _Solver       = nullptr;  ///< Solver handle
        AMGX_Mode             _Mode;                    ///< Solver mode
        
        // Block CSR format conversion
        int estimateBlockSize(Graph &theGraph, int nnz, double efficiency = 0.7);
        int countBlocks(Graph &theGraph, int block_size);
};
#endif