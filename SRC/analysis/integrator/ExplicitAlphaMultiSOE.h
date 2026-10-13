/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
** ****************************************************************** */

#ifndef ExplicitAlphaMultiSOE_h
#define ExplicitAlphaMultiSOE_h

#include <cmath>
#include <TransientIntegrator.h>
#include <OPS_Stream.h>

class DOF_Group;
class FE_Element;
class Vector;
class LinearSOE;
class Channel;
class FEM_ObjectBroker;

/** Explicit Kolay–Ricles style integrator: the linked analysis \c LinearSOE (locals named \c soeA in
 *  the .cpp) holds operator \b A = \f$\alpha_M M + \alpha_F\gamma\Delta t\,C + \alpha_F\beta\Delta t^2 K\f$
 *  (dense \c ExplicitAlpha::newStep local matrix \c A minus local \c B3) and the transformed RHS for \c solve.
 *  Workspace \c getCopy() systems hold \b M and \b alpha = M + gamma*dt*C + beta*dt^2*K only. Same
 *  OpenSees alphaF, alphaM, gamma, beta convention as ExplicitAlpha. In \c newStep the trial acceleration is
 *  \b alpha3*Uddot_n for inertia in \c formUnbalance. Workspace tangents use \c TransientIntegrator::formTangent
 *  (modal damping matrix when enabled). Workspace SOEs should support \c formAp for matvec where applicable.
 *  Like \c ExplicitAlpha::initAlphaMatrices, a single \c initWorkspaceOperators flag controls when workspace
 *  tangents are assembled: \c newStep does that once per invalidation; other \c formTangent calls then no-op. */
class ExplicitAlphaMultiSOE : public TransientIntegrator
{
  public:
    ExplicitAlphaMultiSOE();
    ExplicitAlphaMultiSOE(double alphaF, double alphaM, double gamma, double beta,
                           bool updElemDisp = false, bool incrementalAccel = false,
                           bool useAlphaCloseCheck = false, bool order2 = false,
                           bool initAccel = false);
    ~ExplicitAlphaMultiSOE() override;

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

    /** OpenSeesMP gather-add SOEs (MumpsParallel, DistributedProfileSPD, DistributedCuDSS):
     *  set to OPS_rank so transformed RHS is installed only on P0. Default -1 = serial. */
    void setProcessID(int pid) { processID = pid; }
    int getProcessID(void) const { return processID; }

  protected:
    ExplicitAlphaMultiSOE(int classTag, double alphaF, double alphaM, double gamma, double beta,
                        bool updElemDisp, bool incrementalAccel, bool useAlphaCloseCheck,
                        bool order2 = false, bool initAccel = false);

    double alphaM;
    double alphaF;
    double beta;
    double gamma;
    bool updElemDisp;
    /** If true, trial accel is Uddot_n and the linear solve returns Delta Uddot. */
    bool incrementalAccel;
    /** If true, use the alpha-close shortcut when \f$|\alpha_M-\alpha_F|\le\f$ tolerance. */
    bool useAlphaCloseCheck;
    /** -order2: with dv = gamma*dt*alpha1*a_n, start with v_alg0 = v0 - dv(a0), resisting force at
     *  v_alpha + dv with C_int*dv added back to the residual, reported v_n = v_alg + dv(a_alg).
     *  Implies initAccel. */
    bool order2;
    /** -initAccel (or -order2): start from the equilibrium acceleration
     *  M a0 = p(t0) - C v0 - f(u0, v0) instead of the committed acceleration. */
    bool initAccel;
    /** True until the starting correction (-initAccel / -order2) has been applied. */
    bool startPending;
    /** Workspace SOE holding C_int (the C in alpha); only used through formAp. */
    LinearSOE *soeC;
    Vector *Vrep;
    Vector *dVres;   // velocity shift of the resisting-force argument
    Vector *A1next;  // alpha1*a_{n+1} from the reported velocity, reused by the next predictor
    bool a1Valid;
    /** -order2, set by a mid-analysis domainChanged: the domain velocity is the reported one; the next
     *  newStep recovers v_alg = v_rep - gamma*dt*alpha1*a_n once it has formed alpha1*a_n. */
    bool velRecoverPending;
    double tStepStart;  // domain time at the start of the step (committed-time shift for ops_Dt)

    int applyStartCorrection(void);

    int updateCount;
    double c1, c2, c3;

    /** Form \f$c_1 K + c_2 C + c_3 M\f$ into \a target via \c TransientIntegrator::formTangent (modal matrix, DOF/FE
     *  tangents); \a statFlag selects Kt vs Ki for K. Temporarily repoints integrator links; restores primary SOE. */
    int formTangentOnSOE(LinearSOE &target, int statFlag, double c1v, double c2v, double c3v);

    /** Solve \a soe for RHS \a b into \a x; \c nullptr \a soe or failure returns negative.
     *  When \c processID > 0 (worker on a gather-add SOE), workers zero B and P0 sets the global RHS. */
    int linearSolve(LinearSOE *soe, const Vector &b, Vector &x) const;
    /** Install a global RHS for a subsequent gather-add \c solve (Woodbury/Arpack pattern). */
    int setGlobalRhs(LinearSOE *soe, const Vector &b) const;
    /** \c soe->formAp(in, out); returns -1 if \a soe is null; otherwise returns \c formAp's result. */
    int applyOperator(LinearSOE *soe, const Vector &in, Vector &out) const;

    int applyAlpha1(const Vector &x, Vector &y) const;
    int applyAlpha3(const Vector &x, Vector &y) const;

    /** Absolute tolerance on \f$|\alpha_M-\alpha_F|\f$ for the \c areAlphaMFClose shortcut (\f$A=\alpha_F\alpha\f$, shorter RHS). */
    static constexpr double toleranceAlphaMF = 1.0e-8;

    /** \c true if \c -alphaCloseCheck is set and \f$|\alpha_M-\alpha_F|\le\f$ \ref toleranceAlphaMF. */
    bool areAlphaMFClose() const
    {
        return useAlphaCloseCheck && std::fabs(alphaM - alphaF) <= toleranceAlphaMF;
    }

    double deltaT;

    LinearSOE *soeM;
    LinearSOE *soeAlpha;

    bool workspaceOk;
    /** When true, \c formTangent assembles M / alpha / A into workspace SOEs; cleared after success (same pattern as
     *  \c ExplicitAlpha::initAlphaMatrices). Set by domain/workspace changes, \c recvSelf, and \c newStep when \f$\Delta t\f$
     *  changes; \c newStep is the intended place that pays assembly cost once per step. */
    bool initWorkspaceOperators;

    Vector *Ut;
    Vector *Utdot;
    Vector *Utdotdot;
    Vector *U;
    Vector *Udot;
    Vector *Udotdot;
    Vector *Ualpha;
    Vector *Ualphadot;
    Vector *Ualphadotdot;
    Vector *Utdothat;
    Vector *Phat;
    Vector *w1;
    Vector *w2;
    Vector *w3;

  private:
    void freeWorkspaceSOEs(void);
    /** Allocate/link workspace SOEs (M, alpha) from \c getCopy and size them; primary SOE is
     *  already sized by the analysis. */
    int setupWorkspaceSOEs(void);

    /** -1 = serial / unset; >0 = OpenSeesMP worker on a gather-add distributed SOE. */
    int processID;
};

void *OPS_ExplicitAlphaMultiSOE(void);
void *OPS_KRAlphaExplicitMultiSOE(void);
void *OPS_MKRAlphaExplicitMultiSOE(void);

#endif
