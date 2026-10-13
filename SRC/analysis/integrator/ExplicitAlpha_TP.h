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
// the trapezoidal rule, with user-supplied alphaF, alphaM, gamma, and beta
// (KRAlphaExplicit_TP uses the rhoInf mapping from the reference).
//
// Reference: Kolay, C. and J. Ricles (2014). "Development of a family of
// unconditionally stable explicit direct integration algorithms with
// controllable numerical energy dissipation." Earthquake Engineering and
// Structural Dynamics, 43(9):1361-1380.

#ifndef ExplicitAlpha_TP_h
#define ExplicitAlpha_TP_h

#include <cmath>
#include <TransientIntegrator.h>

class DOF_Group;
class FE_Element;
class Vector;
class Matrix;

class ExplicitAlpha_TP : public TransientIntegrator
{
public:
    ExplicitAlpha_TP();
    ExplicitAlpha_TP(double alphaF, double alphaM, double gamma, double beta,
                     bool incrementalAccel = false, bool useAlphaCloseCheck = false,
                     bool initAccel = false);

    virtual ~ExplicitAlpha_TP();

    int formTangent(int statFlag);
    int formUnbalance(void);

    int formEleTangent(FE_Element *theEle);
    int formNodTangent(DOF_Group *theDof);
    int formEleResidual(FE_Element *theEle);
    int formNodUnbalance(DOF_Group *theDof);

    int domainChanged(void);
    int newStep(double deltaT);
    int revertToLastStep(void);
    int update(const Vector &aiPlusOne);
    int commit(void);

    double getCFactor(void) { return c2; }

    const Vector &getVel(void);

    virtual int sendSelf(int commitTag, Channel &theChannel);
    virtual int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker);

    void Print(OPS_Stream &s, int flag = 0);
    int revertToStart();


protected:
    ExplicitAlpha_TP(int classTag, double alphaF, double alphaM, double gamma, double beta,
                     bool incrementalAccel, bool useAlphaCloseCheck, bool initAccel = false);

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

    bool incrementalAccel;
    /** If true, use the alpha-close shortcut when \f$|\alpha_M-\alpha_F|\le\f$ tolerance. */
    bool useAlphaCloseCheck;
    double deltaT;

    Matrix *alpha1;
    Matrix *alpha3;
    Matrix *Mhat;

    int updateCount;
    int initAlphaMatrices;
    double c1, c2, c3;
    double residM, residD, residR, residP;
    Vector *Ut, *Utdot, *Utdotdot;
    Vector *U, *Udot, *Udotdot;
    Vector *Utdothat;
    Vector *Put;

    /** -initAccel: start from the equilibrium acceleration
     *  M a0 = p(t0) - C v0 - f(u0, v0) instead of the committed acceleration. */
    bool initAccel;
    bool startPending;
    Matrix *Mmat;

    int applyStartCorrection(void);
};

void *OPS_ExplicitAlpha_TP(void);
void *OPS_KRAlphaExplicit_TP(void);
void *OPS_MKRAlphaExplicit_TP(void);

#endif
