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

// Written: Gustavo A. Araujo R. (gaaraujor@gmail.com)
// Created: 06/26

#include <WoodburySolver.h>
#include <WoodburySOE.h>
#include <Channel.h>
#include <FEM_ObjectBroker.h>

WoodburySolver::WoodburySolver(LinearSOESolver &innerSolver, WoodburySOE &wrap)
    : LinearSOESolver(SOLVER_TAGS_WoodburySolver),
      innerSolver(&innerSolver),
      theWrapperSOE(&wrap)
{
}

WoodburySolver::~WoodburySolver()
{
    innerSolver = nullptr;
}

int
WoodburySolver::solve(void)
{
    int r = theWrapperSOE->getInnerSOE().solve();
    if (r < 0)
        return r;

    return theWrapperSOE->applyWoodburyCorrection();
}

int
WoodburySolver::setSize(void)
{
    return innerSolver->setSize();
}

int
WoodburySolver::sendSelf(int, Channel &)
{
    return 0;
}

int
WoodburySolver::recvSelf(int, Channel &, FEM_ObjectBroker &)
{
    return 0;
}
