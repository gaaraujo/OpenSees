/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
** ****************************************************************** */

#include <ExplicitAlphaMultiSOE.h>
#include <classTags.h>
#include <cstring>
#include <cmath>
#include <FE_Element.h>
#include <LinearSOE.h>
#include <AnalysisModel.h>
#include <Domain.h>
#include <Vector.h>
#include <DOF_Group.h>
#include <DOF_GrpIter.h>
#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include <elementAPI.h>

namespace {

bool parseExplicitAlphaMultiSOEOptions(bool &updElemDisp, bool &incrementalAccel, bool &useAlphaCloseCheck)
{
    updElemDisp = false;
    incrementalAccel = false;
    useAlphaCloseCheck = false;
    while (OPS_GetNumRemainingInputArgs() > 0) {
        const char *argvLoc = OPS_GetString();
        if (argvLoc == nullptr) {
            break;
        }
        if (strcmp(argvLoc, "-updateElemDisp") == 0) {
            updElemDisp = true;
        } else if (strcmp(argvLoc, "-incrementalAccel") == 0) {
            incrementalAccel = true;
        } else if (strcmp(argvLoc, "-alphaCloseCheck") == 0) {
            useAlphaCloseCheck = true;
        } else {
            opserr << "WARNING ExplicitAlphaMultiSOE family - unknown flag " << argvLoc
                   << "; want <-updateElemDisp> <-incrementalAccel> <-alphaCloseCheck>\n";
            return false;
        }
    }
    return true;
}

} // namespace

void *
OPS_ExplicitAlphaMultiSOE(void)
{
    TransientIntegrator *theIntegrator = nullptr;

    int argc = OPS_GetNumRemainingInputArgs();
    if (argc < 4) {
        opserr << "WARNING - want: ExplicitAlphaMultiSOE $alphaF $alphaM $gamma $beta "
                  "<-updateElemDisp> <-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }

    double params[4];
    int numData = 4;
    if (OPS_GetDouble(&numData, params) != 0) {
        opserr << "WARNING - invalid args want: ExplicitAlphaMultiSOE $alphaF $alphaM $gamma $beta "
                  "<-updateElemDisp> <-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }

    const double alphaF = params[0];
    const double alphaM = params[1];
    const double gamma = params[2];
    const double beta = params[3];
    if (alphaF < 0.5 || alphaF > 1.0) {
        opserr << "WARNING - invalid alphaF for ExplicitAlphaMultiSOE, want 0.5 <= alphaF <= 1.0\n";
        return nullptr;
    }
    if (gamma <= 0.0 || beta <= 0.0) {
        opserr << "WARNING - invalid gamma/beta for ExplicitAlphaMultiSOE, want gamma > 0 and beta > 0\n";
        return nullptr;
    }
    if (alphaM < 0.5 || alphaM > 2.0) {
        opserr << "WARNING - recommended for unconditional stability (linear): 0.5 <= alphaM <= 2.0\n";
    }
    if (gamma < 0.5) {
        opserr << "WARNING - recommended for unconditional stability (linear): gamma >= 0.5\n";
    }
    if (beta < 0.5 * gamma) {
        opserr << "WARNING - recommended for unconditional stability (linear): beta >= gamma/2\n";
    }

    bool updElemDisp = false;
    bool incrementalAccel = false;
    bool useAlphaCloseCheck = false;
    if (!parseExplicitAlphaMultiSOEOptions(updElemDisp, incrementalAccel, useAlphaCloseCheck)) {
        return nullptr;
    }

    theIntegrator = new ExplicitAlphaMultiSOE(alphaF, alphaM, gamma, beta, updElemDisp, incrementalAccel,
                                              useAlphaCloseCheck);

    if (theIntegrator == nullptr)
        opserr << "WARNING - out of memory creating ExplicitAlphaMultiSOE integrator\n";

    return theIntegrator;
}

void *
OPS_KRAlphaExplicitMultiSOE(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING - want KRAlphaExplicitMultiSOE $rhoInf <-updateElemDisp> <-incrementalAccel> "
                  "<-alphaCloseCheck>\n";
        return nullptr;
    }

    double rhoInf = 0.0;
    int numData = 1;
    if (OPS_GetDouble(&numData, &rhoInf) != 0) {
        opserr << "WARNING - invalid args want KRAlphaExplicitMultiSOE $rhoInf <-updateElemDisp> "
                  "<-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }

    const double den = 1.0 + rhoInf;
    const double alphaF = 1.0 / den;
    const double alphaM = (2.0 - rhoInf) / den;
    const double gamma = 0.5 * (3.0 - rhoInf) / den;
    const double beta = 1.0 / (den * den);

    bool updElemDisp = false;
    bool incrementalAccel = false;
    bool useAlphaCloseCheck = false;
    if (!parseExplicitAlphaMultiSOEOptions(updElemDisp, incrementalAccel, useAlphaCloseCheck)) {
        return nullptr;
    }

    return new ExplicitAlphaMultiSOE(alphaF, alphaM, gamma, beta, updElemDisp, incrementalAccel,
                                     useAlphaCloseCheck);
}

void *
OPS_MKRAlphaExplicitMultiSOE(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING - want MKRAlphaExplicitMultiSOE $rhoInfEquivalent <-updateElemDisp> "
                  "<-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }

    double rhoInfEquivalent = 0.0;
    int numData = 1;
    if (OPS_GetDouble(&numData, &rhoInfEquivalent) != 0) {
        opserr << "WARNING - invalid args want MKRAlphaExplicitMultiSOE $rhoInfEquivalent "
                  "<-updateElemDisp> <-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }

    const double r = (rhoInfEquivalent >= 0.0) ? std::sqrt(rhoInfEquivalent) : rhoInfEquivalent;
    const double alphaF = 1.0 / (r + 1.0);
    const double r2 = r * r;
    const double r3 = r2 * r;
    const double num = (2.0 * r3 + r2 - 1.0);
    const double den = (r3 + r2 + r + 1.0);
    const double alphaM = 1.0 - num / den;
    const double gamma = alphaM - alphaF + 0.5;
    const double beta = 0.5 * (gamma + 0.5);

    bool updElemDisp = false;
    bool incrementalAccel = false;
    bool useAlphaCloseCheck = false;
    if (!parseExplicitAlphaMultiSOEOptions(updElemDisp, incrementalAccel, useAlphaCloseCheck)) {
        return nullptr;
    }

    return new ExplicitAlphaMultiSOE(alphaF, alphaM, gamma, beta, updElemDisp, incrementalAccel,
                                     useAlphaCloseCheck);
}


ExplicitAlphaMultiSOE::ExplicitAlphaMultiSOE(int classTag, double _alphaF, double _alphaM, double _gamma,
                                           double _beta, bool _updElemDisp, bool _incrementalAccel,
                                           bool _useAlphaCloseCheck)
    : TransientIntegrator(classTag),
      alphaM(_alphaM),
      alphaF(_alphaF),
      beta(_beta),
      gamma(_gamma),
      updElemDisp(_updElemDisp),
      incrementalAccel(_incrementalAccel),
      useAlphaCloseCheck(_useAlphaCloseCheck),
      deltaT(0.0),
      soeM(nullptr),
      soeAlpha(nullptr),
      workspaceOk(false),
      initWorkspaceOperators(false),
      updateCount(0),
      c1(0.0),
      c2(0.0),
      c3(0.0),
      Ut(nullptr),
      Utdot(nullptr),
      Utdotdot(nullptr),
      U(nullptr),
      Udot(nullptr),
      Udotdot(nullptr),
      Ualpha(nullptr),
      Ualphadot(nullptr),
      Ualphadotdot(nullptr),
      Utdothat(nullptr),
      Phat(nullptr),
      w1(nullptr),
      w2(nullptr),
      w3(nullptr),
      processID(-1)
{
}


ExplicitAlphaMultiSOE::ExplicitAlphaMultiSOE()
    : ExplicitAlphaMultiSOE(INTEGRATOR_TAGS_ExplicitAlphaMultiSOE, 0.5, 0.5, 0.5, 0.25, false, false, false)
{
}


ExplicitAlphaMultiSOE::ExplicitAlphaMultiSOE(double _alphaF, double _alphaM, double _gamma, double _beta,
                                            bool _updElemDisp, bool _incrementalAccel, bool _useAlphaCloseCheck)
    : ExplicitAlphaMultiSOE(INTEGRATOR_TAGS_ExplicitAlphaMultiSOE, _alphaF, _alphaM, _gamma, _beta, _updElemDisp,
                            _incrementalAccel, _useAlphaCloseCheck)
{
}


ExplicitAlphaMultiSOE::~ExplicitAlphaMultiSOE()
{
    freeWorkspaceSOEs();

    delete Ut;
    delete Utdot;
    delete Utdotdot;
    delete U;
    delete Udot;
    delete Udotdot;
    delete Ualpha;
    delete Ualphadot;
    delete Ualphadotdot;
    delete Utdothat;
    delete Phat;
    delete w1;
    delete w2;
    delete w3;
}


void
ExplicitAlphaMultiSOE::freeWorkspaceSOEs(void)
{
    delete soeM;
    soeM = nullptr;
    delete soeAlpha;
    soeAlpha = nullptr;
    workspaceOk = false;
    initWorkspaceOperators = true;
}


int
ExplicitAlphaMultiSOE::setupWorkspaceSOEs(void)
{
    freeWorkspaceSOEs();

    LinearSOE *soeA = this->getLinearSOE();
    AnalysisModel *theModel = this->getAnalysisModel();
    if (soeA == nullptr || theModel == nullptr) {
        opserr << "WARNING ExplicitAlphaMultiSOE::setupWorkspaceSOEs() - "
                  "missing LinearSOE or AnalysisModel\n";
        return -1;
    }

    soeM = soeA->getCopy();
    soeAlpha = soeA->getCopy();
    if (soeM == nullptr || soeAlpha == nullptr) {
        opserr << "WARNING ExplicitAlphaMultiSOE::setupWorkspaceSOEs() - "
                  "getCopy() failed (solver/SOE must implement getCopy)\n";
        freeWorkspaceSOEs();
        return -2;
    }

    soeM->setLinks(*theModel);
    soeAlpha->setLinks(*theModel);

    Graph &theGraph = theModel->getDOFGraph();
    if (soeM->setSize(theGraph) < 0 || soeAlpha->setSize(theGraph) < 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE::setupWorkspaceSOEs() - "
                  "setSize failed on workspace SOE\n";
        theModel->clearDOFGraph();
        freeWorkspaceSOEs();
        return -3;
    }
    // Same contract as DirectIntegrationAnalysis::handle(): release the DOF graph after setSize
    // (memory; next getDOFGraph() rebuilds if the topology or numbering changes).
    theModel->clearDOFGraph();

    workspaceOk = true;
    initWorkspaceOperators = true;
    return 0;
}


int
ExplicitAlphaMultiSOE::formTangentOnSOE(LinearSOE &target, int statFlag, double c1v, double c2v, double c3v)
{
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr) {
        opserr << "WARNING ExplicitAlphaMultiSOE::formTangentOnSOE() - no AnalysisModel\n";
        return -1;
    }

    LinearSOE *primarySOE = this->getLinearSOE();
    if (primarySOE == nullptr) {
        opserr << "WARNING ExplicitAlphaMultiSOE::formTangentOnSOE() - no LinearSOE\n";
        return -1;
    }

    ConvergenceTest *theTest = this->getConvergenceTest();

    // Same pattern as ExplicitAlpha::newStep: use TransientIntegrator::formTangent so modal damping
    // matrix (inclModalDampingMatrix / addModalDampingMatrix) matches other transient integrators.
    this->IncrementalIntegrator::setLinks(*theModel, target, theTest);
    c1 = c1v;
    c2 = c2v;
    c3 = c3v;

    const int result = this->TransientIntegrator::formTangent(statFlag);

    this->IncrementalIntegrator::setLinks(*theModel, *primarySOE, theTest);

    return result;
}


int
ExplicitAlphaMultiSOE::setGlobalRhs(LinearSOE *soe, const Vector &b) const
{
    if (soe == nullptr)
        return -1;
    soe->zeroB();
    // Gather-add SOEs sum myB across ranks in solve(). A global RHS that already
    // lives on every rank must be installed only on P0 (Woodbury/Arpack pattern).
    if (processID > 0)
        return 0;
    if (soe->setB(b) < 0)
        return -2;
    return 0;
}


int
ExplicitAlphaMultiSOE::linearSolve(LinearSOE *soe, const Vector &b, Vector &x) const
{
    if (soe == nullptr)
        return -1;
    if (setGlobalRhs(soe, b) < 0)
        return -2;
    if (soe->solve() < 0)
        return -3;
    x = soe->getX();
    return 0;
}


int
ExplicitAlphaMultiSOE::applyOperator(LinearSOE *soe, const Vector &in, Vector &out) const
{
    if (soe == nullptr)
        return -1;
    return soe->formAp(in, out);
}


int
ExplicitAlphaMultiSOE::applyAlpha1(const Vector &x, Vector &y) const
{
    // alpha_1 * x = alpha^{-1} * M * x: w1 = M*x, then solve alpha * y = w1 for y
    if (applyOperator(soeM, x, *w1) != 0)
        return -1;
    return linearSolve(soeAlpha, *w1, y);
}


int
ExplicitAlphaMultiSOE::applyAlpha3(const Vector &x, Vector &y) const
{
    // alpha_3 * x = x - alpha^{-1} * A * x. 
    // If alphaM ~= alphaF then A = alphaM * alpha, 
    // hence alpha^{-1} * A * x = alphaM * x and alpha_3 * x = (1 - alphaM) * x.
    if (areAlphaMFClose()) {
        y.addVector(0.0, x, 1.0 - alphaM);
        return 0;
    }

    LinearSOE *soeA = this->getLinearSOE();
    if (applyOperator(soeA, x, *w1) != 0)
        return -1;
    if (linearSolve(soeAlpha, *w1, *w2) != 0)
        return -2;
    y = x;
    y.addVector(1.0, *w2, -1.0);
    return 0;
}


int
ExplicitAlphaMultiSOE::domainChanged()
{
    AnalysisModel *theModel = this->getAnalysisModel();
    LinearSOE *soeA = this->getLinearSOE();
    if (theModel == nullptr || soeA == nullptr)
        return -1;

    const int size = soeA->getNumEqn();

    if (Ut == nullptr || Ut->Size() != size) {
        delete Ut;
        delete Utdot;
        delete Utdotdot;
        delete U;
        delete Udot;
        delete Udotdot;
        delete Ualpha;
        delete Ualphadot;
        delete Ualphadotdot;
        delete Utdothat;
        delete Phat;
        delete w1;
        delete w2;
        delete w3;

        Ut = new Vector(size);
        Utdot = new Vector(size);
        Utdotdot = new Vector(size);
        U = new Vector(size);
        Udot = new Vector(size);
        Udotdot = new Vector(size);
        Ualpha = new Vector(size);
        Ualphadot = new Vector(size);
        Ualphadotdot = new Vector(size);
        Utdothat = new Vector(size);
        Phat = new Vector(size);
        w1 = new Vector(size);
        w2 = new Vector(size);
        w3 = new Vector(size);

        if (Ut == nullptr || Ut->Size() != size || Phat == nullptr || w1 == nullptr || w2 == nullptr ||
            w3 == nullptr) {
            opserr << "WARNING ExplicitAlphaMultiSOE::domainChanged() - allocation failed\n";
            return -1;
        }
    }

    if (setupWorkspaceSOEs() != 0)
        return -2;

    DOF_GrpIter &theDOFs = theModel->getDOFs();
    DOF_Group *dofPtr;
    while ((dofPtr = theDOFs()) != nullptr) {
        const ID &id = dofPtr->getID();
        const int idSize = id.Size();

        const Vector &disp = dofPtr->getCommittedDisp();
        for (int i = 0; i < idSize; i++) {
            const int loc = id(i);
            if (loc >= 0)
                (*U)(loc) = disp(i);
        }

        const Vector &vel = dofPtr->getCommittedVel();
        for (int i = 0; i < idSize; i++) {
            const int loc = id(i);
            if (loc >= 0)
                (*Udot)(loc) = vel(i);
        }

        const Vector &accel = dofPtr->getCommittedAccel();
        for (int i = 0; i < idSize; i++) {
            const int loc = id(i);
            if (loc >= 0)
                (*Udotdot)(loc) = accel(i);
        }
    }

    initWorkspaceOperators = true;
    return 0;
}


int
ExplicitAlphaMultiSOE::newStep(double _deltaT)
{
    updateCount = 0;

    if (alphaF < 0.5 || alphaF > 1.0) {
        opserr << "WARNING ExplicitAlphaMultiSOE::newStep() - invalid alphaF\n";
        return -1;
    }
    if (beta <= 0.0 || gamma <= 0.0) {
        opserr << "WARNING ExplicitAlphaMultiSOE::newStep() - invalid gamma/beta\n";
        return -1;
    }

    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr) {
        opserr << "WARNING ExplicitAlphaMultiSOE::newStep() - no AnalysisModel\n";
        return -2;
    }

    if (!workspaceOk) {
        opserr << "WARNING ExplicitAlphaMultiSOE::newStep() - workspace SOEs not ready\n";
        return -6;
    }

    if (_deltaT <= 0.0) {
        opserr << "WARNING ExplicitAlphaMultiSOE::newStep() - non-positive deltaT\n";
        return -3;
    }

    const double prevDeltaT = deltaT;
    deltaT = _deltaT;
    // Like ExplicitAlpha when _deltaT != deltaT: operators depend on dt.
    if (prevDeltaT != deltaT)
        initWorkspaceOperators = true;

    if (this->formTangent(INITIAL_TANGENT) != 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE::newStep() - formTangent(INITIAL_TANGENT) failed\n";
        return -4;
    }

    const int size = this->getLinearSOE()->getNumEqn();
    if (Ut == nullptr || Ut->Size() != size) {
        opserr << "WARNING ExplicitAlphaMultiSOE::newStep() - domainChanged not successful\n";
        return -5;
    }

    // Current state: Ut = U_n, Utdot = Udot_n, Utdotdot = Uddot_n
    (*Ut) = *U;
    (*Utdot) = *Udot;
    (*Utdotdot) = *Udotdot;

    // Udot_hat = dt * alpha_1 * Uddot_n,   alpha_1 = alpha^{-1} M  (stored in w1 before scaling)
    if (applyAlpha1(*Utdotdot, *w1) != 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE::newStep() - applyAlpha1 failed (formAp / solve)\n";
        return -7;
    }
    Utdothat->addVector(0.0, *w1, deltaT);

    // U_{n+1} = U_n + dt*Udot_n + dt*(1/2+gamma)*Udot_hat
    U->addVector(1.0, *Utdot, deltaT);
    const double a1 = (0.5 + gamma) * deltaT;
    U->addVector(1.0, *Utdothat, a1);

    // Udot_{n+1} = Udot_n + Udot_hat
    Udot->addVector(1.0, *Utdothat, 1.0);

    // U_{n+alphaF} = (1-alphaF)*U_n + alphaF*U_{n+1}
    *Ualpha = *Ut;
    Ualpha->addVector(1.0 - alphaF, *U, alphaF);

    // Udot_{n+alphaF} = (1-alphaF)*Udot_n + alphaF*Udot_{n+1}
    *Ualphadot = *Utdot;
    Ualphadot->addVector(1.0 - alphaF, *Udot, alphaF);

    // Trial accel for inertia in formUnbalance.
    if (incrementalAccel) {
        *Ualphadotdot = *Utdotdot;
    } else if (applyAlpha3(*Utdotdot, *Ualphadotdot) != 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE::newStep() - applyAlpha3 failed (formAp / solve)\n";
        return -8;
    }

    theModel->setResponse(*Ualpha, *Ualphadot, *Ualphadotdot);

    // Equilibrium evaluated at t + alphaF*dt (intermediate time level).
    double time = theModel->getCurrentDomainTime();
    time += alphaF * deltaT;
    // Domain::update(newTime, dT) ignores dT: it sets ops_Dt = newTime - committedTime = alphaF*dt,
    // while element states were committed at the previous alpha level, a full dt earlier. Materials
    // that integrate over ops_Dt (ViscousDamper, Maxwell-type dampers, TDConcrete, friction
    // bearings, ...) then run on a slowed clock. Shift the domain's committed time so ops_Dt = dt
    // at the alpha-level evaluation (Domain::commit resets it; revertToLastStep restores it).
    // -updateElemDisp commits at t_{n+1}, so ops_Dt = alphaF*dt is already right there.
    if (!updElemDisp) {
        Domain *theDomain = theModel->getDomainPtr();
        if (theDomain != 0) {
            tStepStart = time - alphaF * deltaT;
            theDomain->setCommittedTime(time - deltaT);
        }
    }
    if (theModel->updateDomain(time, deltaT) < 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE::newStep() - updateDomain failed\n";
        return -9;
    }

    return 0;
}


int
ExplicitAlphaMultiSOE::revertToLastStep()
{
    // undo the committed-time shift of newStep (Domain::revertToLastCommit used the shifted time)
    if (tStepStart >= 0.0) {
        AnalysisModel *theModel = this->getAnalysisModel();
        Domain *theDomain = theModel ? theModel->getDomainPtr() : 0;
        if (theDomain != 0) {
            theDomain->setCurrentTime(tStepStart);
            theDomain->setCommittedTime(tStepStart);
        }
    }
    if (Ut != nullptr) {
        (*U) = *Ut;
        (*Udot) = *Utdot;
        (*Udotdot) = *Utdotdot;
    }
    return 0;
}


int
ExplicitAlphaMultiSOE::formTangent(int statFlag)
{
    LinearSOE *soeA = this->getLinearSOE();
    if (!workspaceOk || soeM == nullptr || soeAlpha == nullptr || soeA == nullptr)
        return -1;

    statusFlag = statFlag;

    const double dt = deltaT;
    if (dt <= 0.0)
        return -2;

    if (!initWorkspaceOperators)
        return 0;

    // Matrix operators
    //   M      = mass  (soeM)
    //   alpha  = M + gamma*dt*C + beta*dt^2*K  (soeAlpha)
    //   A      = alphaM*M + alphaF*gamma*dt*C + alphaF*beta*dt^2*K (soeA = getLinearSOE())
    // statFlag selects Kt vs Ki for the K contribution.

    const double bdt2 = beta * dt * dt;
    const double gdt = gamma * dt;

    if (formTangentOnSOE(*soeM, statFlag, 0.0, 0.0, 1.0) < 0)
        return -3;

    if (formTangentOnSOE(*soeAlpha, statFlag, bdt2, gdt, 1.0) < 0)
        return -4;

    if (areAlphaMFClose()) {
        // A = alphaM * M; with untransformed Phat in formUnbalance, (alphaM M) uddot = Phat.
        if (formTangentOnSOE(*soeA, statFlag, 0.0, 0.0, alphaM) < 0)
            return -5;
    } else {
        if (formTangentOnSOE(*soeA, statFlag, alphaF * bdt2, alphaF * gdt, alphaM) < 0)
            return -5;
    }

    initWorkspaceOperators = false;
    return 0;
}


int
ExplicitAlphaMultiSOE::formEleTangent(FE_Element *theEle)
{
    theEle->zeroTangent();

    // Assembled tangent: c1*K + c2*C + c3*M (K from Kt or Ki according to statusFlag in formTangentOnSOE).
    if (statusFlag == CURRENT_TANGENT)
        theEle->addKtToTang(c1);
    else if (statusFlag == INITIAL_TANGENT)
        theEle->addKiToTang(c1);

    theEle->addCtoTang(c2);
    theEle->addMtoTang(c3);

    return 0;
}


int
ExplicitAlphaMultiSOE::formNodTangent(DOF_Group *theDof)
{
    theDof->zeroTangent();

    theDof->addCtoTang(c2);
    theDof->addMtoTang(c3);

    return 0;
}


int
ExplicitAlphaMultiSOE::formUnbalance()
{
    if (!workspaceOk)
        return -1;

    if (TransientIntegrator::formUnbalance() < 0)
        return -2;

    if (areAlphaMFClose()) {
        // Primary B already holds the assembled residual; soeA holds alphaF * M only. Phat is unused on this path.
        return 0;
    }

    LinearSOE *soeA = this->getLinearSOE();
    if (soeA == nullptr)
        return -3;

    // Phat = unbalance from equilibrium at t_n + alphaF*dt (trial Ualpha, Ualphadot, Ualphadotdot from newStep).
    *Phat = soeA->getB();

    // Effective equation: A * Uddot_{n+1} = alpha * M^{-1} * Phat
    if (linearSolve(soeM, *Phat, *w2) != 0)
        return -4;

    if (applyOperator(soeAlpha, *w2, *w3) != 0)
        return -5;

    if (setGlobalRhs(soeA, *w3) < 0)
        return -6;

    return 0;
}


int
ExplicitAlphaMultiSOE::update(const Vector &aiPlusOne)
{
    updateCount++;
    if (updateCount > 1) {
        opserr << "WARNING ExplicitAlphaMultiSOE::update() - called more than once; "
                  "use a linear solution algorithm\n";
        return -1;
    }

    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr)
        return -2;
    if (Ut == nullptr)
        return -3;
    if (aiPlusOne.Size() != U->Size()) {
        opserr << "WARNING ExplicitAlphaMultiSOE::update() - incompatible vector size\n";
        return -4;
    }

    if (incrementalAccel) {
        Udotdot->addVector(1.0, aiPlusOne, 1.0);
    } else {
        *Udotdot = aiPlusOne;
    }

    theModel->setVel(*Udot);
    theModel->setAccel(*Udotdot);
    if (theModel->updateDomain() < 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE::update() - updateDomain failed\n";
        return -5;
    }
    theModel->setDisp(*U);

    return 0;
}


int
ExplicitAlphaMultiSOE::commit(void)
{
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr) {
        opserr << "WARNING ExplicitAlphaMultiSOE::commit() - no AnalysisModel\n";
        return -1;
    }

    // Advance domain clock to end of step: t_{n+1} = (t_n + alphaF*dt) + (1-alphaF)*dt.
    double time = theModel->getCurrentDomainTime();
    time += (1.0 - alphaF) * deltaT;
    theModel->setCurrentDomainTime(time);

    if (updElemDisp)
        theModel->updateDomain();

    return theModel->commitDomain();
}


double
ExplicitAlphaMultiSOE::getCFactor(void)
{
    return c2;
}


const Vector &
ExplicitAlphaMultiSOE::getVel()
{
    return *Ualphadot;
}


int
ExplicitAlphaMultiSOE::sendSelf(int cTag, Channel &theChannel)
{
    Vector data(7);
    data(0) = alphaM;
    data(1) = alphaF;
    data(2) = beta;
    data(3) = gamma;
    data(4) = updElemDisp ? 1.0 : 0.0;
    data(5) = incrementalAccel ? 1.0 : 0.0;
    data(6) = useAlphaCloseCheck ? 1.0 : 0.0;

    if (theChannel.sendVector(this->getDbTag(), cTag, data) < 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE::sendSelf() - could not send data\n";
        return -1;
    }
    return 0;
}


int
ExplicitAlphaMultiSOE::recvSelf(int cTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    Vector data(7);
    if (theChannel.recvVector(this->getDbTag(), cTag, data) < 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE::recvSelf() - could not receive data\n";
        return -1;
    }

    alphaM = data(0);
    alphaF = data(1);
    beta = data(2);
    gamma = data(3);
    updElemDisp = (data(4) > 0.5);
    incrementalAccel = (data(5) > 0.5);
    useAlphaCloseCheck = (data(6) > 0.5);

    initWorkspaceOperators = true;
    return 0;
}


void
ExplicitAlphaMultiSOE::Print(OPS_Stream &s, int flag)
{
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel != nullptr) {
        const double currentTime = theModel->getCurrentDomainTime();
        s << "ExplicitAlphaMultiSOE - currentTime: " << currentTime << endln;
        s << "  alphaF: " << alphaF << "  alphaM: " << alphaM << "  gamma: " << gamma << "  beta: " << beta
          << endln;
        s << "  c1: " << c1 << "  c2: " << c2 << "  c3: " << c3 << endln;
        if (updElemDisp)
            s << "  updateElemDisp: yes\n";
        else
            s << "  updateElemDisp: no\n";
        if (incrementalAccel)
            s << "  incrementalAccel: yes\n";
        else
            s << "  incrementalAccel: no\n";
        if (useAlphaCloseCheck)
            s << "  alphaCloseCheck: yes\n";
        else
            s << "  alphaCloseCheck: no\n";
    } else
        s << "ExplicitAlphaMultiSOE - no associated AnalysisModel\n";
}


int
ExplicitAlphaMultiSOE::revertToStart()
{
    if (Ut != nullptr)
        Ut->Zero();
    if (Utdot != nullptr)
        Utdot->Zero();
    if (Utdotdot != nullptr)
        Utdotdot->Zero();
    if (U != nullptr)
        U->Zero();
    if (Udot != nullptr)
        Udot->Zero();
    if (Udotdot != nullptr)
        Udotdot->Zero();
    return 0;
}
