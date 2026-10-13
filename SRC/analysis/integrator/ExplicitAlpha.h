/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
**                                                                    **
** (C) Copyright 1999, The Regents of the University of California    **
** All Rights Reserved.                                               **
**                                                                    **
** Commercial use of this program without express permission of the   **
** University of California, Berkeley, is strictly prohibited.  See **
** file 'COPYRIGHT'  in main directory for information on usage and   **
** redistribution,  and for a DISCLAIMER OF ALL WARRANTIES.           **
**                                                                    **
** Developed by:                                                      **
**   Frank McKenna (fmckenna@ce.berkeley.edu)                         **
**   Gregory L. Fenves (fenves@ce.berkeley.edu)                       **
**   Filip C. Filippou (filippou@ce.berkeley.edu)                     **
**                                                                    **
** ****************************************************************** */

// $Revision$
// $Date$
// $URL$

// Developed: Chinmoy Kolay (chk311@lehigh.edu)
// Implemented: Andreas Schellenberg (andreas.schellenberg@gmail.com)
// Created: 08/14
// Modified: 07 May 2026, Gustavo A. Araujo R. (gaaraujor@gmail.com): generalized
// the implementation for arbitrary alphaF, alphaM, gamma, and beta.
//
// Description: Transient integrator using the explicit Kolay-Ricles scheme based on
// the midpoint rule, with user-supplied alphaF, alphaM, gamma, and beta (KRAlphaExplicit
// uses the rhoInf mapping from the reference).
//
// Reference: Kolay, C. and J. Ricles (2014). "Development of a family of
// unconditionally stable explicit direct integration algorithms with
// controllable numerical energy dissipation." Earthquake Engineering and
// Structural Dynamics, 43(9):1361-1380.

#ifndef ExplicitAlpha_h
#define ExplicitAlpha_h

#include <cmath>
#include <TransientIntegrator.h>

class DOF_Group;
class FE_Element;
class Vector;
class Matrix;

class ExplicitAlpha : public TransientIntegrator
{
public:
    ExplicitAlpha();
    ExplicitAlpha(double alphaF, double alphaM, double gamma, double beta,
                  bool updElemDisp = false, bool incrementalAccel = false,
                  bool useAlphaCloseCheck = false, bool order2 = false, bool initAccel = false);

    virtual ~ExplicitAlpha();

    int formTangent(int statFlag);

    int formEleTangent(FE_Element *theEle);
    int formNodTangent(DOF_Group *theDof);
    int formUnbalance(void);

    int domainChanged(void);
    int newStep(double deltaT);
    int revertToLastStep(void);
    int update(const Vector &aiPlusOne);
    int commit(void);

    double getCFactor(void) { return c2; }

    const Vector &getVel(void);

    virtual int sendSelf(int commitTag, Channel &theChannel);
    virtual int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker);

    virtual void Print(OPS_Stream &s, int flag = 0);
    int revertToStart();

protected:
    ExplicitAlpha(int classTag, double alphaF, double alphaM, double gamma, double beta,
                  bool updElemDisp, bool incrementalAccel, bool useAlphaCloseCheck, bool order2,
                  bool initAccel);

    double alphaM;
    double alphaF;
    double beta;
    double gamma;

    /** Absolute tolerance on \f$|\alpha_M-\alpha_F|\f$ for the \c areAlphaMFClose shortcut. */
    static constexpr double toleranceAlphaMF = 1.0e-8;

    /** \c true if \c -alphaCloseCheck is set and \f$|\alpha_M-\alpha_F|\le\f$ \ref toleranceAlphaMF. */
    bool areAlphaMFClose() const
    {
        return useAlphaCloseCheck && std::fabs(alphaM - alphaF) <= toleranceAlphaMF;
    }

    bool updElemDisp;
    /** If true, trial accel is Uddot_n and the linear solve returns Delta Uddot. */
    bool incrementalAccel;
    /** If true, use the alpha-close shortcut when \f$|\alpha_M-\alpha_F|\le\f$ tolerance. */
    bool useAlphaCloseCheck;
    /** If true (-order2), apply the second-order corrections with dv = gamma*dt*alpha1*a_n:
     *  start with the algorithmic velocity shifted by -dv(a0), evaluate the resisting force at
     *  v_alpha + dv with C_int*dv added back to the residual, and report v_n = v_alg + dv(a_alg)
     *  (a_n stays the algorithmic value). Implies initAccel. */
    bool order2;
    /** If true (-initAccel, or -order2), start from the equilibrium acceleration
     *  M a0 = p(t0) - C v0 - f(u0, v0) instead of the committed acceleration. */
    bool initAccel;
    /** True until the starting correction (-initAccel / -order2) has been applied. */
    bool startPending;
    double deltaT;

    Matrix *alpha1;
    Matrix *alpha3;
    Matrix *Mhat;

    int updateCount;
    int initAlphaMatrices;
    double c1, c2, c3;
    Vector *Ut, *Utdot, *Utdotdot;
    Vector *U, *Udot, *Udotdot;
    Vector *Ualpha, *Ualphadot, *Ualphadotdot;
    Vector *Utdothat;
    Matrix *Mmat;  // mass matrix (-initAccel / -order2 start)
    Matrix *Cint;  // integrator damping C_int, the C in alpha (-order2 force correction)
    Vector *Vrep;  // reported velocity (-order2)
    Vector *work;  // work vector (-initAccel / -order2)
    Vector *dVres; // velocity shift of the resisting-force argument (-order2)
    double tStepStart;  // domain time at the start of the step (committed-time shift for ops_Dt)

    int applyStartCorrection(void);
};

void *OPS_ExplicitAlpha(void);
void *OPS_KRAlphaExplicit(void);
void *OPS_MKRAlphaExplicit(void);

#endif
