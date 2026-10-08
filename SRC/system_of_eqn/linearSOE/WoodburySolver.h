#ifndef WoodburySolver_h
#define WoodburySolver_h

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
//
// Description: LinearSOESolver wrapper that solves A_s x = b on the inner SOE,
// then applies the Woodbury correction for the modal low-rank update.

#include <LinearSOESolver.h>

class WoodburySOE;
class LinearSOESolver;

class WoodburySolver : public LinearSOESolver
{
  public:
    WoodburySolver(LinearSOESolver &innerSolver, WoodburySOE &wrap);
    ~WoodburySolver() override;

    int solve(void) override;
    int setSize(void) override;

    int sendSelf(int commitTag, Channel &theChannel) override;
    int recvSelf(int commitTag, Channel &theChannel,
                 FEM_ObjectBroker &theBroker) override;

  private:
    LinearSOESolver *innerSolver;
    WoodburySOE *theWrapperSOE;
};

#endif
