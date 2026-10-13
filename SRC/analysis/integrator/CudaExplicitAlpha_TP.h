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

// Developed: Chinmoy Kolay (chk311@lehigh.edu)
// Implemented: Andreas Schellenberg (andreas.schellenberg@gmail.com)
// CUDA implementation: Gustavo A. Araujo R. (garaujor@stanford.edu)
//
// Description: GPU explicit Kolay-Ricles trapezoidal-rule (TP) integrator for
// CudaBcsrLinSOE + cuDSS, based on the CPU ExplicitAlpha_TP family.
//
// Reference: Kolay, C. and J. Ricles (2014). "Development of a family of
// unconditionally stable explicit direct integration algorithms with
// controllable numerical energy dissipation." Earthquake Engineering and
// Structural Dynamics, 43(9):1361-1380.

#ifndef CudaExplicitAlpha_TP_h
#define CudaExplicitAlpha_TP_h

#include <TransientIntegrator.h>

class DOF_Group;
class FE_Element;
class Vector;
class LinearSOE;
class Channel;
class FEM_ObjectBroker;
class CudaBcsrLinSOE;

/** GPU explicit Kolay-Ricles trapezoidal-rule (TP) integrator.
 *  Requires CudaBcsrLinSOE + cuDSS; mirrors ExplicitAlphaMultiSOE_TP control flow. */
class CudaExplicitAlpha_TP : public TransientIntegrator
{
public:
    struct Options {
        bool incrementalAccel = false;   // solve returns Delta Uddot instead of total Uddot
        bool useAlphaCloseCheck = false; // enable alphaM ~= alphaF shortcut when |alphaM-alphaF| small
        bool order2 = false;             // -order2 second-order corrections (implies initAccel)
        bool initAccel = false;          // -initAccel equilibrium initial acceleration
    };

    CudaExplicitAlpha_TP();
    CudaExplicitAlpha_TP(double alphaF, double alphaM, double gamma, double beta);
    CudaExplicitAlpha_TP(double alphaF, double alphaM, double gamma, double beta, const Options &opts);
    ~CudaExplicitAlpha_TP() override;

    int formTangent(int statFlag) override;
    int formEleTangent(FE_Element *theEle) override;
    int formNodTangent(DOF_Group *theDof) override;
    int formUnbalance(void) override;
    int formEleResidual(FE_Element *theEle) override;
    int formNodUnbalance(DOF_Group *theDof) override;

    int domainChanged(void) override;
    int newStep(double deltaT) override;
    int revertToLastStep(void) override;
    int update(const Vector &aiPlusOne) override;
    int commit(void) override;

    double getCFactor(void) override;
    const Vector &getVel(void) override;

    int sendSelf(int commitTag, Channel &theChannel) override;
    int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker) override;
    void Print(OPS_Stream &s, int flag = 0) override;
    int revertToStart(void) override;

protected:
    CudaExplicitAlpha_TP(int classTag, double alphaF, double alphaM, double gamma, double beta, const Options &opts);

    double alphaM;
    double alphaF;
    double beta;
    double gamma;
    bool incrementalAccel;
    bool useAlphaCloseCheck;
    /** -order2: as CudaExplicitAlpha; both force ends use v + dv (the n end through the committed
     *  reported velocity), nodal damping and getVel follow Vforce, and C_int*dv is added to Put.
     *  Implies initAccel. */
    bool order2;
    /** -initAccel (or -order2): start from the equilibrium acceleration M a0 = p - C v0 - f(u0, v0). */
    bool initAccel;
    bool startPending;
    Vector *Vrep;    // reported (committed) velocity
    Vector *Vforce;  // velocity at the current force end
    Vector *dVres;   // dv (host)
    Vector *A1host;  // alpha1 * accel (host)

    int applyStartCorrection(CudaBcsrLinSOE *cudaSOE);

    int updateCount;
    double c1, c2, c3;
    double deltaT;

    Vector *Ut;       // step-start backup (revert)
    Vector *Utdot;
    Vector *Utdotdot;
    Vector *U;        // current / trial kinematics; views pinned host buffers
    Vector *Udot;
    Vector *Udotdot;
    Vector *Put;      // blended TP unbalance from two residual passes

    double residM;    // residual weights for formEleResidual / formNodUnbalance
    double residD;
    double residR;
    double residP;

    bool operatorsBuilt;  // GPU M / alpha / A factorized for current domain and deltaT
    bool motionNeedsGather;  // DistCuDSS: gather U/Udot/Uddot to root once after domainChanged
    /** True when host U/Udot/Uddot were written without updating the GPU mirror
     *  (domainChanged, revert, gather). Cleared after newStepPredictor H2D. */
    bool deviceMotionStale;
    /** -order2, set by a mid-analysis domainChanged: order2PreStep recovers
     *  v_alg = v_rep - gamma*dt*alpha1*a_n (= v_rep - dv) on device and host. */
    bool velRecoverPending;

    static constexpr double toleranceAlphaMF = 1.0e-8;
    bool areAlphaMFClose() const;

    int formTangentIntoSOE(int statFlag, double c1v, double c2v, double c3v);
    int validateCudaSOE(CudaBcsrLinSOE *&cudaSOE) const;
    int formOperators(CudaBcsrLinSOE *cudaSOE);

    template<typename T>
    friend struct ImplT_TP;

private:
    struct ImplBase;
    ImplBase *m_impl;
    bool m_deviceUsesFloat = false;  // follows CudaBcsrLinSOE precision (dFFI vs dDDI)

    void ensureDeviceImpl(CudaBcsrLinSOE *cudaSOE);
    void destroyDeviceImpl();
};

#ifdef _CUDSS
void *OPS_CudaExplicitAlpha_TP(void);
void *OPS_CudaKRAlpha_TP(void);
void *OPS_CudaMKRAlpha_TP(void);
#endif

#endif
