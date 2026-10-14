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

    // M Phi for the current eigenpairs (summed over the ranks in OpenSeesMP)
    int setupModal(const Vector *factors);

    // modal damping force on the RHS: f = C v
    int addToUnbalance(const Vector *factors);

    // c_C C in the coefficient matrix of the integrator's LinearSOE, by the domain's option:
    // quick nothing, legacy dense columns, woodbury the low-rank term of the LinearSOE
    int addToTangent(const Vector *factors, double cFactor);

    // number of active modes for the given factors (eigenvalue>0, xi!=0)
    int countActiveModes(const Vector *factors) const;

  private:
    // Q (n x J) = M Phi of the active modes, diagD (J) = cFactor*2*xi*omega
    int buildSymmetricLowRank(const Vector *factors, double cFactor,
                              Matrix &Q, Vector &diagD);

    IncrementalIntegrator *theIntegrator;
    double *eigenVectors;
    Vector *eigenValues;
    Vector *dampingForces;
    int numDOFModal;
};

#endif
