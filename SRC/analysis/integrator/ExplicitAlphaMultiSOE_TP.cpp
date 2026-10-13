/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
** ****************************************************************** */

#include <ExplicitAlphaMultiSOE_TP.h>
#include <classTags.h>
#include <FE_Element.h>
#include <LinearSOE.h>
#include <AnalysisModel.h>
#include <Channel.h>
#include <DOF_Group.h>
#include <FEM_ObjectBroker.h>
#include <Vector.h>
#include <elementAPI.h>
#include <cstring>
#include <cmath>

namespace {

bool parseExplicitAlphaMultiSOETPOptions(bool &incrementalAccel, bool &useAlphaCloseCheck, bool &order2,
                                         bool &initAccel)
{
    incrementalAccel = false;
    useAlphaCloseCheck = false;
    order2 = false;
    initAccel = false;
    while (OPS_GetNumRemainingInputArgs() > 0) {
        const char *argvLoc = OPS_GetString();
        if (argvLoc == nullptr) {
            break;
        }
        if (strcmp(argvLoc, "-incrementalAccel") == 0) {
            incrementalAccel = true;
        } else if (strcmp(argvLoc, "-alphaCloseCheck") == 0) {
            useAlphaCloseCheck = true;
        } else if (strcmp(argvLoc, "-order2") == 0) {
            order2 = true;
        } else if (strcmp(argvLoc, "-initAccel") == 0) {
            initAccel = true;
        } else {
            opserr << "WARNING ExplicitAlphaMultiSOE_TP family - unknown flag " << argvLoc
                   << "; want <-incrementalAccel> <-alphaCloseCheck> <-order2> <-initAccel>\n";
            return false;
        }
    }
    return true;
}

} // namespace

void *
OPS_ExplicitAlphaMultiSOE_TP(void)
{
    TransientIntegrator *theIntegrator = nullptr;

    int argc = OPS_GetNumRemainingInputArgs();
    if (argc < 4) {
        opserr << "WARNING - want: ExplicitAlphaMultiSOE_TP $alphaF $alphaM $gamma $beta "
                  "<-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }

    double params[4];
    int numData = 4;
    if (OPS_GetDouble(&numData, params) != 0) {
        opserr << "WARNING - invalid args want: ExplicitAlphaMultiSOE_TP $alphaF $alphaM $gamma $beta\n";
        return nullptr;
    }

    const double alphaF = params[0];
    const double alphaM = params[1];
    const double gamma = params[2];
    const double beta = params[3];
    if (alphaF < 0.5 || alphaF > 1.0) {
        opserr << "WARNING - invalid alphaF for ExplicitAlphaMultiSOE_TP, want 0.5 <= alphaF <= 1.0\n";
        return nullptr;
    }
    if (gamma <= 0.0 || beta <= 0.0) {
        opserr << "WARNING - invalid gamma/beta for ExplicitAlphaMultiSOE_TP, want gamma > 0 and beta > 0\n";
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

    bool incrementalAccel = false;
    bool useAlphaCloseCheck = false;
    bool order2 = false;
    bool initAccel = false;
    if (!parseExplicitAlphaMultiSOETPOptions(incrementalAccel, useAlphaCloseCheck, order2, initAccel)) {
        return nullptr;
    }

    theIntegrator = new ExplicitAlphaMultiSOE_TP(alphaF, alphaM, gamma, beta, incrementalAccel, useAlphaCloseCheck,
                                                 order2, initAccel);

    if (theIntegrator == nullptr)
        opserr << "WARNING - out of memory creating ExplicitAlphaMultiSOE_TP integrator\n";

    return theIntegrator;
}

void *
OPS_KRAlphaExplicitMultiSOE_TP(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING - want KRAlphaExplicitMultiSOE_TP $rhoInf <-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }

    double rhoInf = 0.0;
    int numData = 1;
    if (OPS_GetDouble(&numData, &rhoInf) != 0) {
        opserr << "WARNING - invalid args want KRAlphaExplicitMultiSOE_TP $rhoInf <-incrementalAccel> "
                  "<-alphaCloseCheck>\n";
        return nullptr;
    }

    const double den = 1.0 + rhoInf;
    const double alphaF = 1.0 / den;
    const double alphaM = (2.0 - rhoInf) / den;
    const double gamma = 0.5 * (3.0 - rhoInf) / den;
    const double beta = 1.0 / (den * den);

    bool incrementalAccel = false;
    bool useAlphaCloseCheck = false;
    bool order2 = false;
    bool initAccel = false;
    if (!parseExplicitAlphaMultiSOETPOptions(incrementalAccel, useAlphaCloseCheck, order2, initAccel)) {
        return nullptr;
    }

    return new ExplicitAlphaMultiSOE_TP(alphaF, alphaM, gamma, beta, incrementalAccel, useAlphaCloseCheck, order2, initAccel);
}

void *
OPS_MKRAlphaExplicitMultiSOE_TP(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING - want MKRAlphaExplicitMultiSOE_TP $rhoInfEquivalent <-incrementalAccel> "
                  "<-alphaCloseCheck>\n";
        return nullptr;
    }

    double rhoInfEquivalent = 0.0;
    int numData = 1;
    if (OPS_GetDouble(&numData, &rhoInfEquivalent) != 0) {
        opserr << "WARNING - invalid args want MKRAlphaExplicitMultiSOE_TP $rhoInfEquivalent "
                  "<-incrementalAccel> <-alphaCloseCheck>\n";
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

    bool incrementalAccel = false;
    bool useAlphaCloseCheck = false;
    bool order2 = false;
    bool initAccel = false;
    if (!parseExplicitAlphaMultiSOETPOptions(incrementalAccel, useAlphaCloseCheck, order2, initAccel)) {
        return nullptr;
    }

    return new ExplicitAlphaMultiSOE_TP(alphaF, alphaM, gamma, beta, incrementalAccel, useAlphaCloseCheck, order2, initAccel);
}


ExplicitAlphaMultiSOE_TP::ExplicitAlphaMultiSOE_TP(int classTag, double _alphaF, double _alphaM, double _gamma,
                                                   double _beta, bool _incrementalAccel, bool _useAlphaCloseCheck,
                                                   bool _order2, bool _initAccel)
    : ExplicitAlphaMultiSOE(classTag, _alphaF, _alphaM, _gamma, _beta, false, _incrementalAccel,
                            _useAlphaCloseCheck, _order2, _initAccel),
      Put(nullptr),
      Vforce(nullptr),
      residM(0.0),
      residD(_alphaF),
      residR(_alphaF),
      residP(_alphaF)
{
}


ExplicitAlphaMultiSOE_TP::ExplicitAlphaMultiSOE_TP()
    : ExplicitAlphaMultiSOE_TP(INTEGRATOR_TAGS_ExplicitAlphaMultiSOE_TP, 0.5, 0.5, 0.5, 0.25, false, false)
{
}


ExplicitAlphaMultiSOE_TP::ExplicitAlphaMultiSOE_TP(double _alphaF, double _alphaM, double _gamma, double _beta,
                                                   bool _incrementalAccel, bool _useAlphaCloseCheck, bool _order2,
                                                   bool _initAccel)
    : ExplicitAlphaMultiSOE_TP(INTEGRATOR_TAGS_ExplicitAlphaMultiSOE_TP, _alphaF, _alphaM, _gamma, _beta,
                               _incrementalAccel, _useAlphaCloseCheck, _order2, _initAccel)
{
}


ExplicitAlphaMultiSOE_TP::~ExplicitAlphaMultiSOE_TP()
{
    delete Put;
    delete Vforce;
}


int
ExplicitAlphaMultiSOE_TP::domainChanged()
{
    if (ExplicitAlphaMultiSOE::domainChanged() != 0)
        return -1;

    LinearSOE *soeA = this->getLinearSOE();
    AnalysisModel *theModel = this->getAnalysisModel();
    if (soeA == nullptr || theModel == nullptr)
        return -2;

    const int size = soeA->getNumEqn();

    // Put holds the TP blended unbalance vector (same length as primary SOE B).
    if (Put == nullptr || Put->Size() != size) {
        delete Put;
        Put = new Vector(size);
        if (Put == nullptr || Put->Size() != size) {
            opserr << "WARNING ExplicitAlphaMultiSOE_TP::domainChanged() - failed to allocate Put\n";
            return -3;
        }
    }
    if (order2 && (Vforce == nullptr || Vforce->Size() != size)) {
        delete Vforce;
        Vforce = new Vector(size);
    }

    return 0;
}


int
ExplicitAlphaMultiSOE_TP::newStep(double _deltaT)
{
    updateCount = 0;

    if (alphaF < 0.5 || alphaF > 1.0) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - invalid alphaF\n";
        return -1;
    }
    if (beta <= 0.0 || gamma <= 0.0) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - invalid gamma/beta\n";
        return -1;
    }

    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - no AnalysisModel\n";
        return -2;
    }

    if (!workspaceOk) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - workspace SOEs not ready\n";
        return -6;
    }

    if (_deltaT <= 0.0) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - non-positive deltaT\n";
        return -3;
    }

    const double prevDeltaT = deltaT;
    deltaT = _deltaT;
    // Like ExplicitAlphaMultiSOE when _deltaT != deltaT: workspace operators M, alpha, A depend on dt.
    if (prevDeltaT != deltaT)
        initWorkspaceOperators = true;

    // Assemble M (soeM), alpha = M + gamma*dt*C + beta*dt^2*K (soeAlpha), and
    // A = alphaM*M + alphaF*gamma*dt*C + alphaF*beta*dt^2*K on the primary SOE (same as ExplicitAlphaMultiSOE::formTangent).
    if (this->ExplicitAlphaMultiSOE::formTangent(INITIAL_TANGENT) != 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - formTangent(INITIAL_TANGENT) failed\n";
        return -4;
    }

    const int size = this->getLinearSOE()->getNumEqn();
    if (Ut == nullptr || Ut->Size() != size || Put == nullptr || Put->Size() != size) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - domainChanged not successful\n";
        return -5;
    }

    // -initAccel / -order2 start (residual weights of the n end; nodal damping at the prescribed v0)
    if (initAccel && startPending) {
        residD = residR = residP = 1.0;
        residM = 0.0;
        if (order2) {
            *Vforce = *Udot;
            *Vrep = *Udot;
        }
        if (this->applyStartCorrection() != 0) {
            opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - starting correction (-initAccel / -order2) failed\n";
            return -11;
        }
        startPending = false;
        if (order2) {
            // n end of the first step: v_alg0 + dv_0 (= v0) and a0 (w3 = alpha1*a0)
            *Vrep = *Udot;
            Vrep->addVector(1.0, *w3, gamma * deltaT);
            theModel->setResponse(*U, *Vrep, *Udotdot);
        } else
            theModel->setResponse(*U, *Udot, *Udotdot);
    }

    // Current state: Ut = U_n, Utdot = Udot_n, Utdotdot = Uddot_n
    (*Ut) = *U;
    (*Utdot) = *Udot;
    (*Utdotdot) = *Udotdot;

    // -order2: alpha1*a_n before the n-end pass (dv needs it). The reported velocity already formed it.
    if (order2) {
        if (a1Valid) {
            *w1 = *A1next;
        } else if (applyAlpha1(*Utdotdot, *w1) != 0) {
            opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - applyAlpha1 failed\n";
            return -7;
        }
        a1Valid = false;
        Utdothat->addVector(0.0, *w1, deltaT);
        if (velRecoverPending) {
            // mid-analysis domainChanged: recover v_alg = v_rep - gamma*dt*alpha1*a_n
            Udot->addVector(1.0, *Utdothat, -gamma);
            *Utdot = *Udot;
            velRecoverPending = false;
        }
        dVres->addVector(0.0, *Utdothat, gamma);
        // n end: committed state, whose velocity is the reported v_alg_n + dv; nodal damping likewise
        *Vforce = *Utdot;
        Vforce->addVector(1.0, *dVres, 1.0);
    }

    // First residual pass (ExplicitAlpha_TP): unit weights on R, P, D; no explicit nodal M term (residM = 0).
    residD = residR = residP = 1.0;
    residM = 0.0;

    double time = theModel->getCurrentDomainTime();
    if (theModel->updateDomain(time, deltaT) < 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - updateDomain failed\n";
        return -7;
    }

    // Assemble first-pass unbalance into soeA->B (then copy to Put).
    if (this->TransientIntegrator::formUnbalance() < 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - formUnbalance failed\n";
        return -9;
    }

    LinearSOE *soeA = this->getLinearSOE();
    // Put <- B^(1): assembled unbalance at t_n after first formUnbalance (residual weights above).
    (*Put) = soeA->getB();

    // Udot_hat = dt * alpha_1 * Uddot_n,   alpha_1 = alpha^{-1} M  (workspace: applyAlpha1, result in w1 before scaling)
    if (!order2) {
        if (applyAlpha1(*Utdotdot, *w1) != 0) {
            opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - applyAlpha1 failed\n";
            return -7;
        }
        Utdothat->addVector(0.0, *w1, deltaT);
    }

    // U_{n+1} = U_n + dt*Udot_n + dt*(1/2+gamma)*Udot_hat  (trial displacement/velocity at end of step)
    U->addVector(1.0, *Utdot, deltaT);
    const double a1 = (0.5 + gamma) * deltaT;
    U->addVector(1.0, *Utdothat, a1);

    // Udot_{n+1} = Udot_n + Udot_hat
    Udot->addVector(1.0, *Utdothat, 1.0);

    // Trial accel for second pass.
    if (incrementalAccel) {
        Udotdot->addVector(0.0, *Utdotdot, 1.0 / alphaF);
    } else {
        if (applyAlpha3(*Utdotdot, *Ualphadotdot) != 0) {
            opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - applyAlpha3 failed\n";
            return -8;
        }
        Udotdot->addVector(0.0, *Ualphadotdot, 1.0 / alphaF);
    }

    // Full-step trial (U, Udot, Uddot), not (U_{n+alphaF}, ...); matches ExplicitAlpha_TP setResponse.
    if (order2) {
        // n+1 end: v_alg_{n+1} + dv (a_n stands in for a_{n+1})
        *Vforce = *Udot;
        Vforce->addVector(1.0, *dVres, 1.0);
        theModel->setResponse(*U, *Vforce, *Udotdot);
    } else
        theModel->setResponse(*U, *Udot, *Udotdot);

    // Second residual pass at t_{n+1} with trial kinematics; residM = 1 for TP nodal M term (ExplicitAlpha_TP).
    time += deltaT;
    if (theModel->updateDomain(time, deltaT) < 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - updateDomain failed (end)\n";
        return -7;
    }

    residM = 1.0;
    if (this->TransientIntegrator::formUnbalance() < 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - second formUnbalance failed\n";
        return -10;
    }

    // Trailing predictor blend: Put <- (1 - alphaF) * Put + alphaF * B^(2),  B^(2) = current SOE unbalance.
    Put->addVector(1.0 - alphaF, soeA->getB(), alphaF);

    // -order2: both ends carried + dv into C v and f(u, v); add back C_int*dv.
    if (order2) {
        if (applyOperator(soeC, *dVres, *w3) != 0) {
            opserr << "WARNING ExplicitAlphaMultiSOE_TP::newStep() - C_int product failed\n";
            return -12;
        }
        Put->addVector(1.0, *w3, 1.0);
    }

    return 0;
}


int
ExplicitAlphaMultiSOE_TP::formUnbalance()
{
    if (!workspaceOk)
        return -1;

    if (Put == nullptr || Phat == nullptr) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::formUnbalance() - missing Put or Phat\n";
        return -2;
    }

    // Phat := Put (effective unbalance for this TP variant; Put holds the blended two-pass vector).
    *Phat = *Put;

    LinearSOE *soeA = this->getLinearSOE();
    if (soeA == nullptr)
        return -3;

    if (areAlphaMFClose()) {
        if (setGlobalRhs(soeA, *Phat) < 0) {
            opserr << "WARNING ExplicitAlphaMultiSOE_TP::formUnbalance() - setB failed (proportional path)\n";
            return -6;
        }
        return 0;
    }

    // Same RHS transform as ExplicitAlphaMultiSOE::formUnbalance: w2 = M^{-1} * Phat, w3 = alpha * w2.
    // Linear solve: soeM holds M; primary equation remains A * Uddot_{n+1} = w3.
    if (linearSolve(soeM, *Phat, *w2) != 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::formUnbalance() - linearSolve(soeM) failed\n";
        return -4;
    }

    if (applyOperator(soeAlpha, *w2, *w3) != 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::formUnbalance() - applyOperator(soeAlpha) failed\n";
        return -5;
    }

    if (setGlobalRhs(soeA, *w3) < 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::formUnbalance() - setB failed\n";
        return -6;
    }

    return 0;
}


int
ExplicitAlphaMultiSOE_TP::formEleResidual(FE_Element *theEle)
{
    theEle->zeroResidual();

    // Same pattern as ExplicitAlpha_TP::formEleResidual (residR on R_inc, (residR-residM) on explicit M*a).
    theEle->addRIncInertiaToResidual(residR);
    theEle->addM_Force(*Udotdot, residR - residM);

    return 0;
}


int
ExplicitAlphaMultiSOE_TP::formNodUnbalance(DOF_Group *theDof)
{
    theDof->zeroUnbalance();

    // Same pattern as ExplicitAlpha_TP::formNodUnbalance.
    theDof->addPtoUnbalance(residP);
    theDof->addD_Force(order2 ? *Vforce : *Udot, -residD);
    theDof->addM_Force(*Udotdot, -residM);

    return 0;
}


int
ExplicitAlphaMultiSOE_TP::update(const Vector &aiPlusOne)
{
    // Like ExplicitAlpha_TP::update: trial U and Udot were already applied in newStep via setResponse;
    // only replace trial acceleration with the solved Uddot_{n+1} and refresh the domain.
    updateCount++;
    if (updateCount > 1) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::update() - called more than once; "
                  "use a linear solution algorithm\n";
        return -1;
    }

    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr)
        return -2;
    if (Ut == nullptr)
        return -3;
    if (aiPlusOne.Size() != U->Size()) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::update() - incompatible vector size\n";
        return -4;
    }

    if (incrementalAccel) {
        Udotdot->addVector(alphaF, aiPlusOne, 1.0);
    } else {
        *Udotdot = aiPlusOne;
    }

    if (order2) {
        // Commit the reported velocity v_alg + gamma*dt*alpha1*a, also the n-end velocity of the
        // next step; alpha1*a_{n+1} is reused by the next predictor.
        *Vrep = *Udot;
        if (applyAlpha1(*Udotdot, *A1next) != 0)
            return -6;
        a1Valid = true;
        Vrep->addVector(1.0, *A1next, gamma * deltaT);
        theModel->setVel(*Vrep);
    }
    theModel->setAccel(*Udotdot);
    if (theModel->updateDomain() < 0) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::update() - updateDomain failed\n";
        return -5;
    }

    return 0;
}


int
ExplicitAlphaMultiSOE_TP::commit(void)
{
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr) {
        opserr << "WARNING ExplicitAlphaMultiSOE_TP::commit() - no AnalysisModel\n";
        return -1;
    }

    return theModel->commitDomain();
}


const Vector &
ExplicitAlphaMultiSOE_TP::getVel()
{
    return order2 ? *Vforce : *Udot;
}


int
ExplicitAlphaMultiSOE_TP::sendSelf(int cTag, Channel &theChannel)
{
    return ExplicitAlphaMultiSOE::sendSelf(cTag, theChannel);
}


int
ExplicitAlphaMultiSOE_TP::recvSelf(int cTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    if (ExplicitAlphaMultiSOE::recvSelf(cTag, theChannel, theBroker) != 0)
        return -1;

    residM = 0.0;
    residD = alphaF;
    residR = alphaF;
    residP = alphaF;

    return 0;
}


void
ExplicitAlphaMultiSOE_TP::Print(OPS_Stream &s, int flag)
{
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel != nullptr) {
        const double currentTime = theModel->getCurrentDomainTime();
        s << "ExplicitAlphaMultiSOE_TP - currentTime: " << currentTime << endln;
        s << "  alphaF: " << alphaF << "  alphaM: " << alphaM << "  gamma: " << gamma << "  beta: " << beta << endln;
        s << "  c1: " << c1 << "  c2: " << c2 << "  c3: " << c3 << endln;
        s << "  order2: " << (order2 ? "yes" : "no") << endln;
        s << "  initAccel: " << (initAccel ? "yes" : "no") << endln;
    } else
        s << "ExplicitAlphaMultiSOE_TP - no associated AnalysisModel\n";
}


int
ExplicitAlphaMultiSOE_TP::revertToStart()
{
    if (Put != nullptr)
        Put->Zero();
    return ExplicitAlphaMultiSOE::revertToStart();
}
