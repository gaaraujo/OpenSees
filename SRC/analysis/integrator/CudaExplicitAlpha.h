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
** Description: GPU explicit Kolay-Ricles integrator (alpha method)   **
** for CudaBcsrLinSOE + cuDSS.                                     **
** ****************************************************************** */

#ifndef CudaExplicitAlpha_h
#define CudaExplicitAlpha_h

#include <TransientIntegrator.h>

class DOF_Group;
class FE_Element;
class Vector;
class LinearSOE;
class Channel;
class FEM_ObjectBroker;
class CudaBcsrLinSOE;

class CudaExplicitAlpha : public TransientIntegrator
{
public:
    struct Options {
        bool updElemDisp = false;
        bool incrementalAccel = false;
        bool useAlphaCloseCheck = false;
        bool order2 = false;
        bool order2Taper = true;
    };

    CudaExplicitAlpha();
    CudaExplicitAlpha(double alphaF, double alphaM, double gamma, double beta);
    CudaExplicitAlpha(double alphaF, double alphaM, double gamma, double beta, const Options &opts);
    ~CudaExplicitAlpha() override;

    int formTangent(int statFlag) override;
    int formEleTangent(FE_Element *theEle) override;
    int formNodTangent(DOF_Group *theDof) override;
    int formUnbalance(void) override;

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
    CudaExplicitAlpha(int classTag, double alphaF, double alphaM, double gamma, double beta, const Options &opts);

    double alphaM;
    double alphaF;
    double beta;
    double gamma;
    bool updElemDisp;
    bool incrementalAccel;
    bool useAlphaCloseCheck;
    /** -order2: equilibrium start with v_alg0 = v0 - gamma*dt*alpha1*a0, resisting force at
     *  v_alpha + gamma*dt*a_n with C_int*gamma*dt*a_n added back to the residual on the GPU,
     *  reported v_n = v_alg + gamma*dt*a_alg. */
    bool order2;
    bool order2Taper;   // -order2 (default) tapers dv and the reported correction by alpha1; -order2NoTaper does not
    bool startPending;  // -order2 starting correction not yet applied
    Vector *Vrep;       // reported velocity (-order2)
    Vector *A1host;     // alpha1*a_{n+1} on the host (-order2 tapered reporting)

    int applyOrder2Start(CudaBcsrLinSOE *cudaSOE);

    int updateCount;
    double c1, c2, c3;
    double deltaT;

    Vector *Ut;
    Vector *Utdot;
    Vector *Utdotdot;   // state at step start (revert backup)
    Vector *U;
    Vector *Udot;
    Vector *Udotdot;      // current state; mirrored to device as d_U*
    Vector *Ualpha;
    Vector *Ualphadot;
    Vector *Ualphadotdot;  // response at t + alphaF*dt (predictor); getVel() returns Ualphadot

    bool operatorsBuilt;  // GPU M / A_alpha / A factorized for current domain and deltaT
    bool motionNeedsGather;  // DistCuDSS: gather U/Udot/Uddot to root once after domainChanged
    /** True when host U/Udot/Uddot were written without updating the GPU mirror
     *  (domainChanged, revert, gather). Cleared after newStepPredictor H2D. */
    bool deviceMotionStale;

    static constexpr double toleranceAlphaMF = 1.0e-8;
    bool areAlphaMFClose() const;

    int formTangentIntoSOE(int statFlag, double c1v, double c2v, double c3v);
    int validateCudaSOE(CudaBcsrLinSOE *&cudaSOE) const;
    int formOperators(CudaBcsrLinSOE *cudaSOE);

    template<typename T>
    friend struct ImplT;

private:
    struct ImplBase;
    ImplBase *m_impl;
    bool m_deviceUsesFloat = false;  // follows CudaBcsrLinSOE precision (dFFI vs dDDI)

    void ensureDeviceImpl(CudaBcsrLinSOE *cudaSOE);
    void destroyDeviceImpl();
};

#ifdef _CUDSS
void *OPS_CudaExplicitAlpha(void);
void *OPS_CudaKRAlpha(void);
void *OPS_CudaMKRAlpha(void);
#endif

#endif
