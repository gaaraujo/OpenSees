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
                                                                        
// $Revision: 1.1 $
// $Date: 2005-04-08 02:38:18 $
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCuda/CudaGenBcsrLinSolver.cpp,v $
                                                                        
// Written: gaaraujo 
// Created: 08/2025
//
// Description: This file contains the implementation for CudaGenBcsrLinSolver
//
// What: "@(#) CudaGenBcsrLinSolver.C, revA"

// OpenSees includes
#include <CudaGenBcsrLinSolver.h>
#include <CudaGenBcsrLinSOE.h>

CudaGenBcsrLinSolver::CudaGenBcsrLinSolver(int theClassTag, CudaPrecision precision)    
:LinearSOESolver(theClassTag),
 theSOE(nullptr),
 m_precision(precision)
{

}    

CudaGenBcsrLinSolver::~CudaGenBcsrLinSolver()    
{

}    

int CudaGenBcsrLinSolver::setLinearSOE(CudaGenBcsrLinSOE &theCudaGenBcsrLinSOE)
{
    theSOE = &theCudaGenBcsrLinSOE;
    return 0;
}

CudaGenBcsrLinSOE* CudaGenBcsrLinSolver::getLinearSOE(void) const
{
    return theSOE;
}

int CudaGenBcsrLinSolver::setSize(void)
{
    return 0;
}

int CudaGenBcsrLinSolver::sendSelf(int commitTag, Channel &theChannel)
{
    return 0;
}

int CudaGenBcsrLinSolver::recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    return 0;
}
