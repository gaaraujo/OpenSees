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

#include <ExplicitAlpha_TP.h>
#include <classTags.h>
#include <FE_Element.h>
#include <LinearSOE.h>
#include "fullGEN/FullGenLinSOE.h"
#include "fullGEN/FullGenLinLapackSolver.h"
#include <AnalysisModel.h>
#include <Vector.h>
#include <DOF_Group.h>
#include <DOF_GrpIter.h>
#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include <elementAPI.h>
#include <cstring>
#include <cmath>
#define OPS_Export

namespace {

bool parseExplicitAlphaTPOptions(bool &incrementalAccel, bool &useAlphaCloseCheck, bool &initAccel)
{
    incrementalAccel = false;
    useAlphaCloseCheck = false;
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
        } else if (strcmp(argvLoc, "-initAccel") == 0) {
            initAccel = true;
        } else {
            opserr << "WARNING ExplicitAlpha_TP family - unknown flag " << argvLoc
                   << "; want <-incrementalAccel> <-alphaCloseCheck> <-initAccel>\n";
            return false;
        }
    }
    return true;
}

} // namespace

void *OPS_ExplicitAlpha_TP(void)
{
    TransientIntegrator *theIntegrator = 0;

    int argc = OPS_GetNumRemainingInputArgs();
    if (argc < 4) {
        opserr << "WARNING - want: ExplicitAlpha_TP $alphaF $alphaM $gamma $beta "
                  "<-incrementalAccel> <-alphaCloseCheck>\n";
        return 0;
    }

    double p[4];
    int num = 4;
    if (OPS_GetDouble(&num, p) != 0) {
        opserr << "WARNING - invalid args want: ExplicitAlpha_TP $alphaF $alphaM $gamma $beta\n";
        return 0;
    }

    // enforce recommended parameter ranges for stability/consistency
    const double alphaF = p[0];
    const double alphaM = p[1];
    const double gamma = p[2];
    const double beta = p[3];
    if (alphaF < 0.5 || alphaF > 1.0) {
        opserr << "WARNING - invalid alphaF for ExplicitAlpha_TP, want 0.5 <= alphaF <= 1.0\n";
        return 0;
    }
    if (gamma <= 0.0 || beta <= 0.0) {
        opserr << "WARNING - invalid gamma/beta for ExplicitAlpha_TP, want gamma > 0 and beta > 0\n";
        return 0;
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
    bool initAccel = false;
    if (!parseExplicitAlphaTPOptions(incrementalAccel, useAlphaCloseCheck, initAccel)) {
        return 0;
    }

    theIntegrator = new ExplicitAlpha_TP(alphaF, alphaM, gamma, beta, incrementalAccel, useAlphaCloseCheck,
                                         initAccel);

    if (theIntegrator == 0)
        opserr << "WARNING - out of memory creating ExplicitAlpha_TP integrator\n";

    return theIntegrator;
}

void *OPS_KRAlphaExplicit_TP(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING - want KRAlphaExplicit_TP $rhoInf <-incrementalAccel> <-alphaCloseCheck>\n";
        return 0;
    }

    double rhoInf = 0.0;
    int numData = 1;
    if (OPS_GetDouble(&numData, &rhoInf) != 0) {
        opserr << "WARNING - invalid args want KRAlphaExplicit_TP $rhoInf <-incrementalAccel> "
                  "<-alphaCloseCheck>\n";
        return 0;
    }

    const double den = 1.0 + rhoInf;
    const double alphaF = 1.0 / den;
    const double alphaM = (2.0 - rhoInf) / den;
    const double gamma = 0.5 * (3.0 - rhoInf) / den;
    const double beta = 1.0 / (den * den);

    bool incrementalAccel = false;
    bool useAlphaCloseCheck = false;
    bool initAccel = false;
    if (!parseExplicitAlphaTPOptions(incrementalAccel, useAlphaCloseCheck, initAccel)) {
        return 0;
    }

    return new ExplicitAlpha_TP(alphaF, alphaM, gamma, beta, incrementalAccel, useAlphaCloseCheck, initAccel);
}

void *OPS_MKRAlphaExplicit_TP(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING - want MKRAlphaExplicit_TP $rhoInfEquivalent <-incrementalAccel> "
                  "<-alphaCloseCheck>\n";
        return 0;
    }

    double rhoInfEquivalent = 0.0;
    int numData = 1;
    if (OPS_GetDouble(&numData, &rhoInfEquivalent) != 0) {
        opserr << "WARNING - invalid args want MKRAlphaExplicit_TP $rhoInfEquivalent <-incrementalAccel> "
                  "<-alphaCloseCheck>\n";
        return 0;
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
    bool initAccel = false;
    if (!parseExplicitAlphaTPOptions(incrementalAccel, useAlphaCloseCheck, initAccel)) {
        return 0;
    }

    return new ExplicitAlpha_TP(alphaF, alphaM, gamma, beta, incrementalAccel, useAlphaCloseCheck, initAccel);
}


ExplicitAlpha_TP::ExplicitAlpha_TP(int classTag, double _alphaF, double _alphaM, double _gamma,
                                   double _beta, bool _incrementalAccel, bool _useAlphaCloseCheck,
                                   bool _initAccel)
    : TransientIntegrator(classTag),
      alphaM(_alphaM), alphaF(_alphaF), beta(_beta), gamma(_gamma),
      incrementalAccel(_incrementalAccel), useAlphaCloseCheck(_useAlphaCloseCheck), deltaT(0.0),
      alpha1(0), alpha3(0), Mhat(0),
      updateCount(0), initAlphaMatrices(1),
      c1(0.0), c2(0.0), c3(0.0),
      residM(0.0), residD(_alphaF), residR(_alphaF), residP(_alphaF),
      Ut(0), Utdot(0), Utdotdot(0),
      U(0), Udot(0), Udotdot(0),
      Utdothat(0), Put(0),
      initAccel(_initAccel), startPending(true),
      Mmat(0)
{
}


ExplicitAlpha_TP::ExplicitAlpha_TP()
    : ExplicitAlpha_TP(INTEGRATOR_TAGS_ExplicitAlpha_TP, 0.5, 0.5, 0.5, 0.25, false, false)
{
}


ExplicitAlpha_TP::ExplicitAlpha_TP(double _alphaF, double _alphaM, double _gamma, double _beta,
                                   bool _incrementalAccel, bool _useAlphaCloseCheck, bool _initAccel)
    : ExplicitAlpha_TP(INTEGRATOR_TAGS_ExplicitAlpha_TP, _alphaF, _alphaM, _gamma, _beta,
                       _incrementalAccel, _useAlphaCloseCheck, _initAccel)
{
}


ExplicitAlpha_TP::~ExplicitAlpha_TP()
{
    if (alpha1 != 0)
        delete alpha1;
    if (alpha3 != 0)
        delete alpha3;
    if (Mhat != 0)
        delete Mhat;
    if (Ut != 0)
        delete Ut;
    if (Utdot != 0)
        delete Utdot;
    if (Utdotdot != 0)
        delete Utdotdot;
    if (U != 0)
        delete U;
    if (Udot != 0)
        delete Udot;
    if (Udotdot != 0)
        delete Udotdot;
    if (Utdothat != 0)
        delete Utdothat;
    if (Put != 0)
        delete Put;
    delete Mmat;
}


int ExplicitAlpha_TP::newStep(double _deltaT)
{
    updateCount = 0;

    if (alphaF < 0.5 || alphaF > 1.0) {
        opserr << "WARNING ExplicitAlpha_TP::newStep() - invalid alphaF\n";
        opserr << "alphaF = " << alphaF << " want 0.5 <= alphaF <= 1.0\n";
        return -1;
    }

    if (beta <= 0.0 || gamma <= 0.0) {
        opserr << "WARNING ExplicitAlpha_TP::newStep() - error in variable\n";
        opserr << "gamma = " << gamma << " beta = " << beta << endln;
        return -1;
    }
    if (alphaM < 0.5 || alphaM > 2.0) {
        opserr << "WARNING ExplicitAlpha_TP::newStep() - recommended for unconditional stability (linear): 0.5 <= alphaM <= 2.0\n";
        opserr << "alphaM = " << alphaM << endln;
    }
    if (gamma < 0.5) {
        opserr << "WARNING ExplicitAlpha_TP::newStep() - recommended for unconditional stability (linear): gamma >= 0.5\n";
        opserr << "gamma = " << gamma << endln;
    }
    if (beta < 0.5 * gamma) {
        opserr << "WARNING ExplicitAlpha_TP::newStep() - recommended for unconditional stability (linear): beta >= gamma/2\n";
        opserr << "beta = " << beta << " gamma/2 = " << 0.5 * gamma << endln;
    }

    LinearSOE *theLinSOE = this->getLinearSOE();
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theLinSOE == 0 || theModel == 0) {
        opserr << "WARNING ExplicitAlpha_TP::newStep() - ";
        opserr << "no LinearSOE or AnalysisModel has been set\n";
        return -2;
    }

    if (_deltaT != deltaT) {
        deltaT = _deltaT;
        initAlphaMatrices = 1;
    }
    if (initAlphaMatrices) {
        if (deltaT <= 0.0) {
            opserr << "WARNING ExplicitAlpha_TP::newStep() - error in variable\n";
            opserr << "dT = " << deltaT << endln;
            return -3;
        }

        ConvergenceTest *theTest = this->getConvergenceTest();

        int size = theLinSOE->getNumEqn();
        if (size <= 0) {
            opserr << "WARNING ExplicitAlpha_TP::newStep() - system size is " << size << endln;
            return -4;
        }

        // Ensure internal vectors/matrices have the correct size.
        // domainChanged() is expected to have been invoked by the analysis framework.
        if (Ut == 0 || Ut->Size() != size || alpha1 == 0 || alpha3 == 0 || Mhat == 0 || Put == 0) {
            opserr << "WARNING ExplicitAlpha_TP::newStep() - domainChange() failed or hasn't been called\n";
            return -6;
        }

        FullGenLinSolver *theFullLinSolver = new FullGenLinLapackSolver();
        LinearSOE *theFullLinSOE = new FullGenLinSOE(size, *theFullLinSolver);
        if (theFullLinSOE == 0) {
            opserr << "WARNING ExplicitAlpha_TP::newStep() - failed to create FullLinearSOE\n";
            delete theFullLinSolver;
            return -4;
        }
        theFullLinSOE->setLinks(*theModel);

        this->IncrementalIntegrator::setLinks(*theModel, *theFullLinSOE, theTest);

        const Matrix *tmp = theFullLinSOE->getA();
        if (tmp == 0) {
            opserr << "WARNING ExplicitAlpha_TP::newStep() - ";
            opserr << "failed to get A matrix of FullGeneral LinearSOE\n";
            this->IncrementalIntegrator::setLinks(*theModel, *theLinSOE, theTest);
            delete theFullLinSOE;
            return -5;
        }

        c1 = beta * deltaT * deltaT;
        c2 = gamma * deltaT;
        c3 = 1.0;
        this->TransientIntegrator::formTangent(INITIAL_TANGENT);
        Matrix A(*tmp);

        if (initAccel) {
            // M (start correction), assembled into the same temporary full SOE
            const double c1s = c1, c2s = c2, c3s = c3;
            if (Mmat == 0 || Mmat->noRows() != size) {
                delete Mmat;
                Mmat = new Matrix(size, size);
            }
            c1 = 0.0;
            c2 = 0.0;
            c3 = 1.0;
            this->TransientIntegrator::formTangent(INITIAL_TANGENT);
            *Mmat = *tmp;
            c1 = c1s;
            c2 = c2s;
            c3 = c3s;
        }

        if (areAlphaMFClose()) {
            alpha3->Zero();
            const double s3 = 1.0 - alphaM;
            for (int i = 0; i < size; i++)
                (*alpha3)(i, i) = s3;

            c1 = 0.0;
            c2 = 0.0;
            c3 = 1.0;
            this->TransientIntegrator::formTangent(INITIAL_TANGENT);
            Matrix B1(*tmp);
            A.Solve(B1, *alpha1);

            Mhat->Zero();
            c1 = 0.0;
            c2 = 0.0;
            c3 = alphaM;
            this->TransientIntegrator::formTangent(INITIAL_TANGENT);
            Mhat->addMatrix(0.0, *tmp, 1.0);
        } else {
            c1 *= (1.0 - alphaF);
            c2 *= (1.0 - alphaF);
            c3 = (1.0 - alphaM);
            this->TransientIntegrator::formTangent(INITIAL_TANGENT);
            Matrix B3(*tmp);

            A.Solve(B3, *alpha3);

            c1 = 0.0;
            c2 = 0.0;
            c3 = 1.0;
            this->TransientIntegrator::formTangent(INITIAL_TANGENT);
            Matrix B1(*tmp);

            A.Solve(B1, *alpha1);

            Mhat->addMatrix(0.0, B1, 1.0);
            Mhat->addMatrixProduct(1.0, B1, *alpha3, -1.0);
        }

        this->IncrementalIntegrator::setLinks(*theModel, *theLinSOE, theTest);

        // FullGenLinSOE is a LinearSOE, whose base class destructor deletes its solver.
        // Therefore we must delete only the SOE here (and not delete the solver twice).
        delete theFullLinSOE;

        this->ExplicitAlpha_TP::formTangent(INITIAL_TANGENT);
        initAlphaMatrices = 0;
    }

    if (initAccel && startPending) {
        if (this->applyStartCorrection() < 0) {
            opserr << "WARNING ExplicitAlpha_TP::newStep() - starting correction (-initAccel) failed\n";
            return -8;
        }
        startPending = false;
    }

    (*Ut) = *U;
    (*Utdot) = *Udot;
    (*Utdotdot) = *Udotdot;

    Utdothat->addMatrixVector(0.0, *alpha1, *Utdotdot, deltaT);

    residD = residR = residP = 1.0;
    residM = 0.0;
    double time = theModel->getCurrentDomainTime();
    if (theModel->updateDomain(time, deltaT) < 0) {
        opserr << "WARNING ExplicitAlpha_TP::newStep() - failed to update the domain\n";
        return -7;
    }
    this->TransientIntegrator::formUnbalance();
    (*Put) = theLinSOE->getB();

    U->addVector(1.0, *Utdot, deltaT);
    double a1 = (0.5 + gamma) * deltaT;
    U->addVector(1.0, *Utdothat, a1);

    Udot->addVector(1.0, *Utdothat, 1.0);

    if (incrementalAccel)
        Udotdot->addVector(0.0, *Utdotdot, 1.0 / alphaF);
    else if (areAlphaMFClose())
        Udotdot->addVector(0.0, *Utdotdot, (1.0 - alphaM) / alphaF);
    else
        Udotdot->addMatrixVector(0.0, *alpha3, *Utdotdot, 1.0 / alphaF);
    theModel->setResponse(*U, *Udot, *Udotdot);

    time += deltaT;
    if (theModel->updateDomain(time, deltaT) < 0) {
        opserr << "WARNING ExplicitAlpha_TP::newStep() - failed to update the domain\n";
        return -7;
    }

    residM = 1.0;
    this->TransientIntegrator::formUnbalance();

    Put->addVector(1.0 - alphaF, theLinSOE->getB(), alphaF);

    return 0;
}


int ExplicitAlpha_TP::applyStartCorrection()
{
    // M a0 = p(t0) - C v0 - f(u0, v0) with zero trial acceleration, a_alg0 = a0; v stays v0.
    AnalysisModel *theModel = this->getAnalysisModel();
    LinearSOE *theLinSOE = this->getLinearSOE();
    if (theModel == 0 || theLinSOE == 0 || Mmat == 0)
        return -1;

    Udotdot->Zero();
    residD = residR = residP = 1.0;
    residM = 0.0;
    // Elements that update from the trial-displacement increment (forceBeamColumn) skip their update
    // when the increment is zero, so after setNodeDisp -commit they still hold the previous state and
    // R0 would miss the imposed displacement. A tiny non-committing trial excursion makes them iterate
    // to the committed configuration before R0 is formed.
    {
        Vector Up(*U);
        for (int i = 0; i < Up.Size(); i++)
            Up(i) += 1.0e-9 * (1.0 + fabs(Up(i)));
        theModel->setResponse(Up, *Udot, *Udotdot);
        if (theModel->updateDomain(theModel->getCurrentDomainTime(), deltaT) < 0)
            return -2;
    }
    theModel->setResponse(*U, *Udot, *Udotdot);
    if (theModel->updateDomain(theModel->getCurrentDomainTime(), deltaT) < 0)
        return -2;
    if (this->TransientIntegrator::formUnbalance() < 0)
        return -3;
    Vector R0(theLinSOE->getB());
    if (Mmat->Solve(R0, *Udotdot) < 0)
        return -4;

    theModel->setResponse(*U, *Udot, *Udotdot);
    return 0;
}


int ExplicitAlpha_TP::revertToLastStep()
{
    if (U != 0) {
        (*U) = *Ut;
        (*Udot) = *Utdot;
        (*Udotdot) = *Utdotdot;
    }

    return 0;
}


int ExplicitAlpha_TP::formTangent(int statFlag)
{
    statusFlag = statFlag;

    if (initAlphaMatrices) {
        LinearSOE *theLinSOE = this->getLinearSOE();
        AnalysisModel *theModel = this->getAnalysisModel();
        if (theLinSOE == 0 || theModel == 0) {
            opserr << "WARNING ExplicitAlpha_TP::formTangent() - ";
            opserr << "no LinearSOE or AnalysisModel has been set\n";
            return -1;
        }

        theLinSOE->zeroA();

        int size = theLinSOE->getNumEqn();
        ID id(size);
        id(0) = 0;
        for (int i = 1; i < size; i++) {
            id(i) = id(i - 1) + 1;
        }
        if (theLinSOE->addA(*Mhat, id) < 0) {
            opserr << "WARNING ExplicitAlpha_TP::formTangent() - ";
            opserr << "failed to add Mhat to A\n";
            return -2;
        }
    }
    return 0;
}


int ExplicitAlpha_TP::formUnbalance()
{
    LinearSOE *theLinSOE = this->getLinearSOE();
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theLinSOE == 0 || theModel == 0) {
        opserr << "WARNING ExplicitAlpha_TP::formUnbalance() - ";
        opserr << "no LinearSOE or AnalysisModel has been set\n";
        return -1;
    }

    theLinSOE->setB(*Put);

    return 0;
}


int ExplicitAlpha_TP::formEleTangent(FE_Element *theEle)
{
    theEle->zeroTangent();

    if (statusFlag == CURRENT_TANGENT)
        theEle->addKtToTang(c1);
    else if (statusFlag == INITIAL_TANGENT)
        theEle->addKiToTang(c1);

    theEle->addCtoTang(c2);
    theEle->addMtoTang(c3);

    return 0;
}


int ExplicitAlpha_TP::formNodTangent(DOF_Group *theDof)
{
    theDof->zeroTangent();

    theDof->addCtoTang(c2);
    theDof->addMtoTang(c3);

    return 0;
}


int ExplicitAlpha_TP::formEleResidual(FE_Element *theEle)
{
    theEle->zeroResidual();

    theEle->addRIncInertiaToResidual(residR);
    theEle->addM_Force(*Udotdot, residR - residM);

    return 0;
}


int ExplicitAlpha_TP::formNodUnbalance(DOF_Group *theDof)
{
    theDof->zeroUnbalance();

    theDof->addPtoUnbalance(residP);
    theDof->addD_Force(*Udot, -residD);
    theDof->addM_Force(*Udotdot, -residM);

    return 0;
}


int ExplicitAlpha_TP::domainChanged()
{
    AnalysisModel *theModel = this->getAnalysisModel();
    LinearSOE *theLinSOE = this->getLinearSOE();
    // Use the number of equations rather than X.Size() as X may not be sized yet
    // when domainChanged() is invoked (which can lead to zero-sized allocations).
    int size = theLinSOE->getNumEqn();

    if (Ut == 0 || Ut->Size() != size) {

        if (alpha1 != 0)
            delete alpha1;
        if (alpha3 != 0)
            delete alpha3;
        if (Mhat != 0)
            delete Mhat;
        if (Ut != 0)
            delete Ut;
        if (Utdot != 0)
            delete Utdot;
        if (Utdotdot != 0)
            delete Utdotdot;
        if (U != 0)
            delete U;
        if (Udot != 0)
            delete Udot;
        if (Udotdot != 0)
            delete Udotdot;
        if (Utdothat != 0)
            delete Utdothat;
        if (Put != 0)
            delete Put;

        alpha1 = new Matrix(size, size);
        alpha3 = new Matrix(size, size);
        Mhat = new Matrix(size, size);
        Ut = new Vector(size);
        Utdot = new Vector(size);
        Utdotdot = new Vector(size);
        U = new Vector(size);
        Udot = new Vector(size);
        Udotdot = new Vector(size);
        Utdothat = new Vector(size);
        Put = new Vector(size);

        if (alpha1 == 0 || alpha1->noRows() != size || alpha1->noCols() != size ||
            alpha3 == 0 || alpha3->noRows() != size || alpha3->noCols() != size ||
            Mhat == 0 || Mhat->noRows() != size || Mhat->noCols() != size ||
            Ut == 0 || Ut->Size() != size ||
            Utdot == 0 || Utdot->Size() != size ||
            Utdotdot == 0 || Utdotdot->Size() != size ||
            U == 0 || U->Size() != size ||
            Udot == 0 || Udot->Size() != size ||
            Udotdot == 0 || Udotdot->Size() != size ||
            Utdothat == 0 || Utdothat->Size() != size ||
            Put == 0 || Put->Size() != size) {

            opserr << "WARNING ExplicitAlpha_TP::domainChanged() - ";
            opserr << "ran out of memory\n";

            if (alpha1 != 0)
                delete alpha1;
            if (alpha3 != 0)
                delete alpha3;
            if (Mhat != 0)
                delete Mhat;
            if (Ut != 0)
                delete Ut;
            if (Utdot != 0)
                delete Utdot;
            if (Utdotdot != 0)
                delete Utdotdot;
            if (U != 0)
                delete U;
            if (Udot != 0)
                delete Udot;
            if (Udotdot != 0)
                delete Udotdot;
            if (Utdothat != 0)
                delete Utdothat;
            if (Put != 0)
                delete Put;

            alpha1 = 0;
            alpha3 = 0;
            Mhat = 0;
            Ut = 0;
            Utdot = 0;
            Utdotdot = 0;
            U = 0;
            Udot = 0;
            Udotdot = 0;
            Utdothat = 0;
            Put = 0;

            return -1;
        }
    }

    DOF_GrpIter &theDOFs = theModel->getDOFs();
    DOF_Group *dofPtr;
    while ((dofPtr = theDOFs()) != 0) {
        const ID &id = dofPtr->getID();
        int idSize = id.Size();

        int i;
        const Vector &disp = dofPtr->getCommittedDisp();
        for (i = 0; i < idSize; i++) {
            int loc = id(i);
            if (loc >= 0) {
                (*U)(loc) = disp(i);
            }
        }

        const Vector &vel = dofPtr->getCommittedVel();
        for (i = 0; i < idSize; i++) {
            int loc = id(i);
            if (loc >= 0) {
                (*Udot)(loc) = vel(i);
            }
        }

        const Vector &accel = dofPtr->getCommittedAccel();
        for (i = 0; i < idSize; i++) {
            int loc = id(i);
            if (loc >= 0) {
                (*Udotdot)(loc) = accel(i);
            }
        }
    }

    initAlphaMatrices = 1;

    return 0;
}


int ExplicitAlpha_TP::update(const Vector &aiPlusOne)
{
    updateCount++;
    if (updateCount > 1) {
        opserr << "WARNING ExplicitAlpha_TP::update() - called more than once -";
        opserr << " ExplicitAlpha_TP integration scheme requires a LINEAR solution algorithm\n";
        return -1;
    }

    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == 0) {
        opserr << "WARNING ExplicitAlpha_TP::update() - no AnalysisModel set\n";
        return -2;
    }

    if (Ut == 0) {
        opserr << "WARNING ExplicitAlpha_TP::update() - domainChange() failed or not called\n";
        return -3;
    }

    if (aiPlusOne.Size() != U->Size()) {
        opserr << "WARNING ExplicitAlpha_TP::update() - Vectors of incompatible size ";
        opserr << " expecting " << U->Size() << " obtained " << aiPlusOne.Size() << endln;
        return -4;
    }

    if (incrementalAccel) {
        Udotdot->addVector(alphaF, aiPlusOne, 1.0);
    } else {
        *Udotdot = aiPlusOne;
    }

    theModel->setAccel(*Udotdot);
    if (theModel->updateDomain() < 0) {
        opserr << "WARNING ExplicitAlpha_TP::update() - failed to update the domain\n";
        return -5;
    }

    return 0;
}


int ExplicitAlpha_TP::commit(void)
{
    LinearSOE *theLinSOE = this->getLinearSOE();
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theLinSOE == 0 || theModel == 0) {
        opserr << "WARNING ExplicitAlpha_TP::commit() - ";
        opserr << "no LinearSOE or AnalysisModel has been set\n";
        return -1;
    }

    return theModel->commitDomain();
}


const Vector &
ExplicitAlpha_TP::getVel()
{
    return *Udot;
}


int ExplicitAlpha_TP::sendSelf(int cTag, Channel &theChannel)
{
    Vector data(7);
    data(0) = alphaM;
    data(1) = alphaF;
    data(2) = beta;
    data(3) = gamma;
    data(4) = incrementalAccel ? 1.0 : 0.0;
    data(5) = useAlphaCloseCheck ? 1.0 : 0.0;
    data(6) = initAccel ? 1.0 : 0.0;

    if (theChannel.sendVector(this->getDbTag(), cTag, data) < 0) {
        opserr << "WARNING ExplicitAlpha_TP::sendSelf() - could not send data\n";
        return -1;
    }

    return 0;
}


int ExplicitAlpha_TP::recvSelf(int cTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    Vector data(7);
    if (theChannel.recvVector(this->getDbTag(), cTag, data) < 0) {
        opserr << "WARNING ExplicitAlpha_TP::recvSelf() - could not receive data\n";
        return -1;
    }

    alphaM = data(0);
    alphaF = data(1);
    beta = data(2);
    gamma = data(3);
    incrementalAccel = (data(4) > 0.5);
    useAlphaCloseCheck = (data(5) > 0.5);
    initAccel = (data(6) > 0.5);

    residM = 0.0;
    residD = alphaF;
    residR = alphaF;
    residP = alphaF;

    return 0;
}


void ExplicitAlpha_TP::Print(OPS_Stream &s, int flag)
{
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel != 0) {
        double currentTime = theModel->getCurrentDomainTime();
        s << "ExplicitAlpha_TP - currentTime: " << currentTime << endln;
        s << "  alphaF: " << alphaF << "  alphaM: " << alphaM << "  gamma: " << gamma << "  beta: " << beta
          << endln;
        s << "  c1: " << c1 << "  c2: " << c2 << "  c3: " << c3 << endln;
        if (incrementalAccel)
            s << "  incrementalAccel: yes\n";
        else
            s << "  incrementalAccel: no\n";
        if (useAlphaCloseCheck)
            s << "  alphaCloseCheck: yes\n";
        else
            s << "  alphaCloseCheck: no\n";
        s << "  initAccel: " << (initAccel ? "yes" : "no") << endln;
    } else
        s << "ExplicitAlpha_TP - no associated AnalysisModel\n";
}


int ExplicitAlpha_TP::revertToStart()
{
    if (Ut != 0)
        Ut->Zero();
    if (Utdot != 0)
        Utdot->Zero();
    if (Utdotdot != 0)
        Utdotdot->Zero();
    if (U != 0)
        U->Zero();
    if (Udot != 0)
        Udot->Zero();
    if (Udotdot != 0)
        Udotdot->Zero();
    startPending = true;

    return 0;
}
