#ifndef ModalDamping_h
#define ModalDamping_h

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

// Original implementation: Frank McKenna (03/15)
// Reimplemented: Gustavo A. Araujo R. (gaaraujor@gmail.com, 06/26)
//
// Description: This class implements modal damping. It provides methods
// to add the effects of modal damping to the RHS and the tangent matrix.

class IncrementalIntegrator;
class Vector;
class Matrix;

class ModalDamping
{
  public:
    explicit ModalDamping(IncrementalIntegrator &owner);
    ~ModalDamping();

    int setupModal(const Vector *factors);

    // modal damping force on the RHS: f = C v
    int addToUnbalance(const Vector *factors);

    // legacy MODAL_DAMPING_INCL_MATRIX: add dense modal columns into A
    int addToTangent(const Vector *factors, double cFactor);

    // woodbury: setup modal workspace and fill Q (n×k), diagD (k); returns
    // active mode count (>0), 0 if nothing to add, <0 on error
    int prepareWoodburyLowRank(const Vector *factors, double cFactor, int numDOF,
                               Matrix &Q, Vector &diagD);

    // woodbury: fill Q (n×k) and diagD (k) = cFactor*2*zeta*omega for active modes
    int buildSymmetricLowRank(const Vector *factors, double cFactor,
                              Matrix &Q, Vector &diagD);

    // number of active modes for the given factors (eigenvalue>0, xi!=0)
    int countActiveModes(const Vector *factors) const;

  private:
    IncrementalIntegrator *theIntegrator;
    double *eigenVectors;
    Vector *eigenValues;
    Vector *dampingForces;
};

#endif
