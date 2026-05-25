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
** ****************************************************************** */                                                                        
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCuda/CudaGenBcsrLinSolver.h
//
// Written: gaaraujo 
// Created: 08/2025
//
// Description: This file contains the class definition for 
// CudaGenBcsrLinSolver. It solves a CudaGenBcsrLinSOE object.

#ifndef CudaGenBcsrLinSolver_h
#define CudaGenBcsrLinSolver_h

// OpenSees includes
#include <LinearSOESolver.h>
#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include "CudaUtils.h"

// Forward declarations
class CudaGenBcsrLinSOE;

class CudaGenBcsrLinSolver : public LinearSOESolver
{
public:
    CudaGenBcsrLinSolver(int classTag, CudaPrecision precision = CudaPrecision::dDDI);    
    virtual ~CudaGenBcsrLinSolver();

    // Set the associated CudaGenBcsrLinSOE object
    virtual int setLinearSOE(CudaGenBcsrLinSOE &theSOE);
    
    // Get the associated SOE
    CudaGenBcsrLinSOE* getLinearSOE(void) const;
    
    // Get solver precision
    CudaPrecision getPrecision(void) const { return m_precision; }

    // Abstract methods that must be implemented by subclasses
    int solve(void) override = 0;
    virtual int setSize(void) override;

    // Parallel communication methods
    int sendSelf(int commitTag, Channel &theChannel) override;   
    int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker) override;  

protected:
    CudaGenBcsrLinSOE* theSOE;
    CudaPrecision m_precision;

private:
};
#endif