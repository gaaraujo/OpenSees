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

// Written: gaaraujo
// Created: 10/2026

#include <SparseExplicitAlpha.h>
#include <SparseCsrLinSOE.h>
#include <DistributedSparseCsrLinSOE.h>
#include <CsrMatrix.h>
#include <AnalysisModel.h>
#include <Channel.h>
#include <DOF_Group.h>
#include <DOF_GrpIter.h>
#include <FE_Element.h>
#include <FEM_ObjectBroker.h>
#include <LinearSOE.h>
#include <Vector.h>
#include <Domain.h>
#include <classTags.h>
#include <elementAPI.h>
#include <OPS_Globals.h>
#include <cmath>
#include <cstring>

// Kolay-Ricles midpoint-rule integrator on the CPU (SparseCsrLinSOE), step for step as
// CudaExplicitAlpha: predictor -> equilibrium at t + alphaF*dt -> Linear solve -> update.

namespace {

bool parseSparseExplicitAlphaOptions(SparseExplicitAlpha::Options &opts)
{
    opts = SparseExplicitAlpha::Options{};
    while (OPS_GetNumRemainingInputArgs() > 0) {
        const char *tok = OPS_GetString();
        if (tok == nullptr)
            break;
        if (strcmp(tok, "-updateElemDisp") == 0) {
            opts.updElemDisp = true;
        } else if (strcmp(tok, "-incrementalAccel") == 0) {
            opts.incrementalAccel = true;
        } else if (strcmp(tok, "-alphaCloseCheck") == 0) {
            opts.useAlphaCloseCheck = true;
        } else if (strcmp(tok, "-initAccel") == 0) {
            opts.initAccel = true;
        } else if (strcmp(tok, "-noSecondPass") == 0) {
            opts.noSecondPass = true;
        } else if (strcmp(tok, "-tangentAtStart") == 0) {
            opts.tangentAtStart = true;
        } else if (strcmp(tok, "-stiffnessFactor") == 0) {
            int one = 1;
            if (OPS_GetNumRemainingInputArgs() < 1 || OPS_GetDoubleInput(&one, &opts.stiffnessFactor) != 0 ||
                !(opts.stiffnessFactor > 0.0)) {
                opserr << "WARNING SparseExplicitAlpha family - -stiffnessFactor needs a positive value\n";
                return false;
            }
        } else {
            opserr << "WARNING SparseExplicitAlpha family - unknown flag " << tok
                   << "; want <-updateElemDisp> <-incrementalAccel> <-alphaCloseCheck> <-initAccel> <-noSecondPass>"
                      " <-tangentAtStart> <-stiffnessFactor s>\n";
            return false;
        }
    }
    return true;
}

} // namespace

void *OPS_SparseExplicitAlpha(void)
{
    if (OPS_GetNumRemainingInputArgs() < 4) {
        opserr << "WARNING integrator SparseExplicitAlpha alphaF alphaM gamma beta <flags>\n";
        return nullptr;
    }
    double alphaF = 0.0, alphaM = 0.0, gamma = 0.0, beta = 0.0;
    int numData = 1;
    if (OPS_GetDoubleInput(&numData, &alphaF) != 0 || OPS_GetDoubleInput(&numData, &alphaM) != 0 ||
        OPS_GetDoubleInput(&numData, &gamma) != 0 || OPS_GetDoubleInput(&numData, &beta) != 0) {
        opserr << "WARNING SparseExplicitAlpha - invalid alphaF/alphaM/gamma/beta\n";
        return nullptr;
    }
    SparseExplicitAlpha::Options opts;
    if (!parseSparseExplicitAlphaOptions(opts))
        return nullptr;
    return new SparseExplicitAlpha(alphaF, alphaM, gamma, beta, opts);
}

void *OPS_SparseKRAlpha(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING integrator SparseKRAlpha rhoInf <flags>\n";
        return nullptr;
    }
    double rhoInf = 0.0;
    int numData = 1;
    if (OPS_GetDoubleInput(&numData, &rhoInf) != 0) {
        opserr << "WARNING SparseKRAlpha - invalid rhoInf\n";
        return nullptr;
    }
    const double den = 1.0 + rhoInf;
    const double alphaF = 1.0 / den;
    const double alphaM = (2.0 - rhoInf) / den;
    const double gamma = 0.5 * (3.0 - rhoInf) / den;
    const double beta = 1.0 / (den * den);
    SparseExplicitAlpha::Options opts;
    if (!parseSparseExplicitAlphaOptions(opts))
        return nullptr;
    return new SparseExplicitAlpha(alphaF, alphaM, gamma, beta, opts);
}

void *OPS_SparseMKRAlpha(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING integrator SparseMKRAlpha rhoInfEquivalent <flags>\n";
        return nullptr;
    }
    double rhoInfEquivalent = 0.0;
    int numData = 1;
    if (OPS_GetDoubleInput(&numData, &rhoInfEquivalent) != 0) {
        opserr << "WARNING SparseMKRAlpha - invalid rhoInfEquivalent\n";
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
    SparseExplicitAlpha::Options opts;
    if (!parseSparseExplicitAlphaOptions(opts))
        return nullptr;
    return new SparseExplicitAlpha(alphaF, alphaM, gamma, beta, opts);
}

SparseExplicitAlpha::SparseExplicitAlpha()
    : SparseExplicitAlpha(0.5, 0.5, 0.5, 0.25, Options{})
{
}

SparseExplicitAlpha::SparseExplicitAlpha(double _alphaF, double _alphaM, double _gamma, double _beta,
                                         const Options &opts)
    : TransientIntegrator(INTEGRATOR_TAGS_SparseExplicitAlpha),
      alphaM(_alphaM), alphaF(_alphaF), beta(_beta), gamma(_gamma),
      updElemDisp(opts.updElemDisp),
      noSecondPass(opts.noSecondPass),
      tangentAtStart(opts.tangentAtStart),
      stiffnessFactor(opts.stiffnessFactor),
      incrementalAccel(opts.incrementalAccel),
      useAlphaCloseCheck(opts.useAlphaCloseCheck),
      initAccel(opts.initAccel),
      startPending(true),
      tStepStart(-1.0),
      updateCount(0), c1(0.0), c2(0.0), c3(0.0), deltaT(0.0),
      Ut(new Vector(0)), Utdot(new Vector(0)), Utdotdot(new Vector(0)),
      U(new Vector(0)), Udot(new Vector(0)), Udotdot(new Vector(0)),
      Ualpha(new Vector(0)), Ualphadot(new Vector(0)), Ualphadotdot(new Vector(0)),
      operatorsBuilt(false),
      motionNeedsGather(true),
      matM(nullptr), matAlpha(nullptr), matA(nullptr),
      boundRows(0), boundNnz(0), boundRowPtr(nullptr), boundColIdx(nullptr)
{
}

SparseExplicitAlpha::~SparseExplicitAlpha()
{
    destroyOperators();
    delete Ut;
    delete Utdot;
    delete Utdotdot;
    delete U;
    delete Udot;
    delete Udotdot;
    delete Ualpha;
    delete Ualphadot;
    delete Ualphadotdot;
}

void SparseExplicitAlpha::destroyOperators(void)
{
    delete matM;
    delete matAlpha;
    delete matA;
    matM = matAlpha = matA = nullptr;
    boundRows = boundNnz = 0;
    boundRowPtr = boundColIdx = nullptr;
}

bool SparseExplicitAlpha::areAlphaMFClose() const
{
    return useAlphaCloseCheck && std::fabs(alphaM - alphaF) <= toleranceAlphaMF;
}

int SparseExplicitAlpha::validateSOE(SparseCsrLinSOE *&csrSOE, DistributedSparseCsrLinSOE *&distSOE) const
{
    LinearSOE *soe = this->getLinearSOE();
    csrSOE = dynamic_cast<SparseCsrLinSOE *>(soe);
    distSOE = dynamic_cast<DistributedSparseCsrLinSOE *>(soe);
    if (csrSOE == nullptr && distSOE != nullptr)
        csrSOE = distSOE->getSparseCsrLinSOE();
    if (csrSOE == nullptr) {
        opserr << "ERROR SparseExplicitAlpha (SparseKRAlpha/SparseMKRAlpha) requires `system SparseCSR` "
                  "or `system DistributedSparseCSR`.\n";
        return -1;
    }
    return 0;
}

bool SparseExplicitAlpha::isRoot(DistributedSparseCsrLinSOE *distSOE) const
{
    return distSOE == nullptr || distSOE->getProcessID() == 0;
}

int SparseExplicitAlpha::formTangentIntoSOE(int statFlag, double c1v, double c2v, double c3v)
{
    LinearSOE *soe = this->getLinearSOE();
    if (this->getAnalysisModel() == nullptr || soe == nullptr)
        return -1;
    // Assemble through the linked SOE so worker triplets participate, then merge them on rank 0.
    soe->zeroA();
    statusFlag = statFlag;
    c1 = c1v;
    c2 = c2v;
    c3 = c3v;
    if (this->TransientIntegrator::formTangent(statFlag) != 0)
        return -1;
    if (auto *dist = dynamic_cast<DistributedSparseCsrLinSOE *>(soe))
        if (dist->ensureMergedA() != 0)
            return -1;
    return 0;
}

int SparseExplicitAlpha::bindSharedStructure(SparseCsrLinSOE *csrSOE)
{
    // One CSR pattern (the system's) for M, alpha and A; values differ.
    const int n = csrSOE->getNumEqn();
    const int nnz = csrSOE->getNumNonZeroValues();
    const int *rowPtr = csrSOE->getRowPtr();
    const int *colIdx = csrSOE->getColIndices();
    if (rowPtr == nullptr || colIdx == nullptr)
        return -1;
    if (matM == nullptr) {
        const CsrMatrix::Solver s = csrSOE->getCsrSolver();
        const double tol = csrSOE->getProfileMinDiagTol();
        matM = new CsrMatrix(s, tol);
        matAlpha = new CsrMatrix(s, tol);
        matA = new CsrMatrix(s, tol);
        // distributed MUMPS: M and alpha factorized and solved on all ranks (matA: products only)
        matM->setDistributed(csrSOE->isCollectiveSolver());
        matAlpha->setDistributed(csrSOE->isCollectiveSolver());
    }
    const bool samePattern = boundRows == n && boundNnz == nnz && boundRowPtr == rowPtr &&
                             boundColIdx == colIdx && matM->isStructureBound();
    if (!samePattern) {
        if (matM->bindStructure(n, nnz, rowPtr, colIdx) != 0 ||
            matAlpha->bindStructure(n, nnz, rowPtr, colIdx) != 0 ||
            matA->bindStructure(n, nnz, rowPtr, colIdx) != 0)
            return -1;
        boundRows = n;
        boundNnz = nnz;
        boundRowPtr = rowPtr;
        boundColIdx = colIdx;
    }
    return 0;
}

int SparseExplicitAlpha::bindParticipants(SparseCsrLinSOE *csrSOE)
{
    const int n = csrSOE->getNumEqn();
    if (matM == nullptr) {
        matM = new CsrMatrix(CsrMatrix::Solver::Mumps);
        matAlpha = new CsrMatrix(CsrMatrix::Solver::Mumps);
        matM->setDistributed(true);
        matAlpha->setDistributed(true);
    }
    if (boundRows != n || !matM->isStructureBound()) {
        if (matM->bindParticipant(n) != 0 || matAlpha->bindParticipant(n) != 0)
            return -1;
        boundRows = n;
    }
    return 0;
}

int SparseExplicitAlpha::numAlphaRhs(void) const
{
    return 1;  // alpha_3 Uddot_n follows from alpha^{-1} M Uddot_n (see newStepPredictor)
}

int SparseExplicitAlpha::formOperators(SparseCsrLinSOE *csrSOE, DistributedSparseCsrLinSOE *distSOE)
{
    // All three assemblies are collective under OpenSeesMP (each ends with a triplet merge on
    // rank 0); rank 0 copies each operator's values off the system and factorizes after the last
    // merge. The system keeps A, which its own solve() factorizes.
    const double bdt2 = beta * deltaT * deltaT;
    const double gdt = gamma * deltaT;
    const bool areClose = areAlphaMFClose();
    const bool root = isRoot(distSOE);
    const bool collective = csrSOE->isCollectiveSolver();
    int rc = 0;

    if (formTangentIntoSOE(INITIAL_TANGENT, 0.0, 0.0, 1.0) != 0)
        return -1;
    if (root) {
        if (bindSharedStructure(csrSOE) != 0 || matM->copyValues(csrSOE->getAValuesData()) != 0)
            rc = -2;
    }

    if (formTangentIntoSOE(INITIAL_TANGENT, bdt2, gdt, 1.0) != 0)
        return -3;
    if (root && rc == 0 && matAlpha->copyValues(csrSOE->getAValuesData()) != 0)
        rc = -4;

    if (areClose) {
        if (formTangentIntoSOE(INITIAL_TANGENT, 0.0, 0.0, alphaM) != 0)
            return -5;
    } else if (formTangentIntoSOE(INITIAL_TANGENT, alphaF * bdt2, alphaF * gdt, alphaM) != 0) {
        return -5;
    }
    if (root && rc == 0 && matA->copyValues(csrSOE->getAValuesData()) != 0)
        rc = -5;

    if (!root && collective) {
        // join rank 0's factorizations, in the same order (alpha, then M)
        if (bindParticipants(csrSOE) != 0 || matAlpha->refactorize() != 0 || matM->refactorize() != 0)
            rc = -4;
    }
    if (root && collective && rc != 0) {
        // the workers are entering the collective factorizations: enter them too, then report the failure
        if (matAlpha != nullptr && matM != nullptr) {
            matAlpha->refactorize();
            matM->refactorize();
        }
    }
    if (root && rc == 0) {
        const int n = csrSOE->getNumEqn();
        const int numRhs = numAlphaRhs();
        alphaRhs.assign(static_cast<std::size_t>(numRhs) * n, 0.0);
        alphaSol.assign(static_cast<std::size_t>(numRhs) * n, 0.0);
        w2.assign(static_cast<std::size_t>(n), 0.0);
        // both factorizations are always entered (collective under distributed MUMPS)
        const int rcAlpha = matAlpha->refactorize();
        const int rcM = matM->refactorize();
        if (rcAlpha != 0) {
            opserr << "ERROR SparseExplicitAlpha::formOperators - alpha factorization failed\n";
            rc = -4;
        } else if (rcM != 0) {
            opserr << "ERROR SparseExplicitAlpha::formOperators - M factorization failed\n";
            rc = -2;
        }
    }

    if (distSOE != nullptr) {
        // Workers wait here for rank 0's factorization (avoids hanging in the newStep broadcast).
        Vector status(1);
        status(0) = (rc == 0) ? 1.0 : -1.0;
        if (distSOE->broadcastFromRoot(status) < 0)
            return -6;
        if (status(0) < 0.0) {
            if (!root)
                opserr << "ERROR SparseExplicitAlpha::formOperators - rank 0 operator build failed\n";
            return -6;
        }
    }
    return rc;
}

int SparseExplicitAlpha::formTangent(int statFlag)
{
    (void)statFlag;
    return 0;  // operators assembled in formOperators(); Linear::formTangent is a no-op
}

int SparseExplicitAlpha::formEleTangent(FE_Element *theEle)
{
    if (operatorsBuilt)
        return 0;
    theEle->zeroTangent();
    // -tangentAtStart: K from the current tangent at the moment the operators are formed (e.g. the
    // post-gravity state) instead of the initial tangent; -stiffnessFactor s: s*K in every operator.
    const int kFlag = (tangentAtStart && statusFlag == INITIAL_TANGENT) ? CURRENT_TANGENT : statusFlag;
    if (kFlag == CURRENT_TANGENT)
        theEle->addKtToTang(c1 * stiffnessFactor);
    else if (kFlag == INITIAL_TANGENT)
        theEle->addKiToTang(c1 * stiffnessFactor);
    theEle->addCtoTang(c2);
    theEle->addMtoTang(c3);
    return 0;
}

int SparseExplicitAlpha::formNodTangent(DOF_Group *theDof)
{
    if (operatorsBuilt)
        return 0;
    theDof->zeroTangent();
    theDof->addCtoTang(c2);
    theDof->addMtoTang(c3);
    return 0;
}

int SparseExplicitAlpha::domainChanged()
{
    SparseCsrLinSOE *csrSOE = nullptr;
    DistributedSparseCsrLinSOE *distSOE = nullptr;
    operatorsBuilt = false;
    if (validateSOE(csrSOE, distSOE) != 0)
        return -1;
    const int size = csrSOE->getNumEqn();
    if (size <= 0)
        return 0;

    destroyOperators();
    Vector *vecs[] = {Ut, Utdot, Utdotdot, U, Udot, Udotdot, Ualpha, Ualphadot, Ualphadotdot};
    for (Vector *v : vecs) {
        v->resize(size);
        v->Zero();
    }

    // Seed the motion from the committed domain state. Copy each vector before the next getter:
    // TransformationDOF_Group writes disp, vel, and accel into the same buffer.
    AnalysisModel *theModel = this->getAnalysisModel();
    DOF_GrpIter &theDOFs = theModel->getDOFs();
    DOF_Group *dofPtr;
    while ((dofPtr = theDOFs()) != nullptr) {
        const ID &id = dofPtr->getID();
        const int idSize = id.Size();
        const Vector &disp = dofPtr->getCommittedDisp();
        for (int i = 0; i < idSize; ++i)
            if (id(i) >= 0)
                (*U)(id(i)) = disp(i);
        const Vector &vel = dofPtr->getCommittedVel();
        for (int i = 0; i < idSize; ++i)
            if (id(i) >= 0)
                (*Udot)(id(i)) = vel(i);
        const Vector &accel = dofPtr->getCommittedAccel();
        for (int i = 0; i < idSize; ++i)
            if (id(i) >= 0)
                (*Udotdot)(id(i)) = accel(i);
    }
    *Ut = *U;
    *Utdot = *Udot;
    *Utdotdot = *Udotdot;
    motionNeedsGather = true;
    return 0;
}

// -initAccel starting correction (collective in OpenSeesMP): assemble R0 = p - C v0 - f(u0, v0) with
// zero trial acceleration, merge it on rank 0, and solve a0 = M^{-1} R0 there.
int SparseExplicitAlpha::applyStartCorrection(SparseCsrLinSOE *csrSOE, DistributedSparseCsrLinSOE *distSOE)
{
    AnalysisModel *theModel = this->getAnalysisModel();
    Ualphadotdot->Zero();
    // Elements that update from the trial-displacement increment (forceBeamColumn) skip their update
    // when the increment is zero; a tiny non-committing trial excursion makes them iterate to the
    // committed configuration before R0 is formed (as CudaExplicitAlpha).
    {
        Vector Up(*U);
        for (int i = 0; i < Up.Size(); i++)
            Up(i) += 1.0e-9 * (1.0 + std::fabs(Up(i)));
        theModel->setResponse(Up, *Udot, *Ualphadotdot);
        if (theModel->updateDomain(theModel->getCurrentDomainTime(), deltaT) < 0)
            return -2;
    }
    theModel->setResponse(*U, *Udot, *Ualphadotdot);
    if (theModel->updateDomain(theModel->getCurrentDomainTime(), deltaT) < 0)
        return -1;
    if (TransientIntegrator::formUnbalance() < 0)
        return -2;

    int rc = 0;
    if (distSOE != nullptr) {
        rc = distSOE->mergeBToRoot();
        if (rc == 0)
            distSOE->zeroB();
    }
    if (isRoot(distSOE)) {
        Vector &b = csrSOE->getHostBVector();
        const int rcSolve = (b.Size() > 0 && matM != nullptr) ? matM->solve(&b(0), &(*Udotdot)(0), 1) : -5;
        if (rc == 0)
            rc = rcSolve;
    } else if (csrSOE->isCollectiveSolver()) {
        rc = matM->solve(nullptr, nullptr, 1);  // join rank 0's M solve
    }
    if (distSOE != nullptr) {
        Vector status(1);
        status(0) = (rc == 0) ? 1.0 : -1.0;
        if (distSOE->broadcastFromRoot(status) < 0)
            return -3;
        if (status(0) < 0.0)
            return -4;
    }
    return rc;
}

// Predictor on rank 0: alpha_1 Uddot_n in one solve with alpha, then the midpoint update and the
// alpha-interpolated kinematics (kernelNewStep of CudaExplicitAlpha). Since
// A = alphaF*alpha + (alphaM - alphaF)*M, alpha^{-1} A Uddot_n = alphaF*Uddot_n + (alphaM - alphaF)*sol.
int SparseExplicitAlpha::newStepPredictor(void)
{
    const int n = U->Size();
    const bool alphaClose = areAlphaMFClose();
    double *rhs = alphaRhs.data();
    double *sol = alphaSol.data();
    const double *uddot = &(*Udotdot)(0);

    if (matM->spmv(uddot, rhs) != 0)
        return -5;
    if (matAlpha->solve(rhs, sol, 1) != 0)
        return -5;
    const double dt = deltaT;
    for (int i = 0; i < n; ++i) {
        const double ut = (*U)(i);
        const double utdot = (*Udot)(i);
        const double utddot = (*Udotdot)(i);
        const double sol1 = sol[i];
        const double uhat = dt * sol1;  // Udot_hat = dt * alpha_1 * Uddot_n
        const double un1 = ut + dt * utdot + (0.5 + gamma) * dt * uhat;
        const double udotn1 = utdot + uhat;
        (*U)(i) = un1;
        (*Udot)(i) = udotn1;
        (*Ualpha)(i) = (1.0 - alphaF) * ut + alphaF * un1;
        (*Ualphadot)(i) = (1.0 - alphaF) * utdot + alphaF * udotn1;
        if (incrementalAccel)
            (*Ualphadotdot)(i) = utddot;
        else if (alphaClose)
            (*Ualphadotdot)(i) = (1.0 - alphaM) * utddot;
        else
            (*Ualphadotdot)(i) = utddot - (alphaF * utddot + (alphaM - alphaF) * sol1);  // alpha_3 * Uddot_n
    }
    return 0;
}

// Per-step flow: predictor -> setResponse at alpha level -> updateDomain at t + alphaF*dt.
// Linear::solveCurrentStep then calls formUnbalance (equilibrium + RHS) and update (kinematics).
int SparseExplicitAlpha::newStep(double _deltaT)
{
    updateCount = 0;
    if (alphaF < 0.5 || alphaF > 1.0 || beta <= 0.0 || gamma <= 0.0 || _deltaT <= 0.0)
        return -1;
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr) {
        opserr << "WARNING SparseExplicitAlpha::newStep() - no AnalysisModel set\n";
        return -2;
    }
    SparseCsrLinSOE *csrSOE = nullptr;
    DistributedSparseCsrLinSOE *distSOE = nullptr;
    if (validateSOE(csrSOE, distSOE) != 0)
        return -3;
    const bool root = isRoot(distSOE);
    if (distSOE != nullptr)
        distSOE->setGlobalRhsReady(false);  // drop a flag left by a formUnbalance that never reached solve()

    // Rank 0's vectors hold only its own DOFs after domainChanged: gather once.
    if (distSOE != nullptr && motionNeedsGather) {
        if (distSOE->gatherOwnedToRoot(*U) < 0 || distSOE->gatherOwnedToRoot(*Udot) < 0 ||
            distSOE->gatherOwnedToRoot(*Udotdot) < 0) {
            opserr << "ERROR SparseExplicitAlpha::newStep() - motion gather to root failed\n";
            return -5;
        }
        motionNeedsGather = false;
    }

    *Ut = *U;
    *Utdot = *Udot;
    *Utdotdot = *Udotdot;
    if (_deltaT != deltaT) {
        deltaT = _deltaT;
        operatorsBuilt = false;
    }
    if (!operatorsBuilt) {
        if (formOperators(csrSOE, distSOE) != 0) {
            opserr << "ERROR SparseExplicitAlpha::newStep() - formOperators failed\n";
            return -4;
        }
        operatorsBuilt = true;
    }

    if (initAccel && startPending) {
        if (applyStartCorrection(csrSOE, distSOE) != 0) {
            opserr << "ERROR SparseExplicitAlpha::newStep() - starting correction (-initAccel) failed\n";
            return -7;
        }
        startPending = false;
        *Utdotdot = *Udotdot;
    }

    if (root && U->Size() > 0 && newStepPredictor() != 0) {
        opserr << "ERROR SparseExplicitAlpha::newStep() - predictor failed\n";
        return -5;
    }
    if (!root && csrSOE->isCollectiveSolver() && matAlpha->solve(nullptr, nullptr, numAlphaRhs()) != 0) {
        opserr << "ERROR SparseExplicitAlpha::newStep() - predictor failed\n";
        return -5;  // joined rank 0's alpha solve
    }
    if (distSOE != nullptr) {
        // Send the predicted kinematics so workers share rank 0's predictor result: each worker
        // receives only the entries of its own DOFs (all it reads), in one message.
        Vector *motion[6] = {U, Udot, Udotdot, Ualpha, Ualphadot, Ualphadotdot};
        if (distSOE->scatterLocalFromRoot(motion, 6) < 0) {
            opserr << "ERROR SparseExplicitAlpha::newStep() - motion broadcast failed\n";
            return -5;
        }
    }

    theModel->setResponse(*Ualpha, *Ualphadot, *Ualphadotdot);
    // Domain::update(newTime, dT) sets ops_Dt = newTime - committedTime = alphaF*dt, while the
    // element states were committed a full dt earlier; shift the committed time so ops_Dt = dt at the
    // alpha-level evaluation (Domain::commit resets it). -updateElemDisp commits at t_{n+1}.
    Domain *theDomain = updElemDisp ? 0 : theModel->getDomainPtr();
    if (theDomain != 0) {
        tStepStart = theModel->getCurrentDomainTime();
        theDomain->setCommittedTime(tStepStart + alphaF * deltaT - deltaT);
    }
    if (theModel->updateDomain(theModel->getCurrentDomainTime() + alphaF * deltaT, deltaT) < 0) {
        opserr << "ERROR SparseExplicitAlpha::newStep() - updateDomain failed\n";
        return -6;
    }
    return 0;
}

// Rank 0: B <- alpha * M^{-1} * Phat, in the system's B.
int SparseExplicitAlpha::transformRhs(SparseCsrLinSOE *csrSOE)
{
    Vector &b = csrSOE->getHostBVector();
    if (b.Size() == 0)
        return 0;
    if (matM == nullptr || !matM->isFactored())
        return -3;
    if (matM->solve(&b(0), w2.data(), 1) != 0)
        return -3;
    return matAlpha->spmv(w2.data(), &b(0));
}

int SparseExplicitAlpha::formUnbalance()
{
    SparseCsrLinSOE *csrSOE = nullptr;
    DistributedSparseCsrLinSOE *distSOE = nullptr;
    if (validateSOE(csrSOE, distSOE) != 0)
        return -2;
    if (distSOE != nullptr)
        distSOE->setGlobalRhsReady(false);

    if (TransientIntegrator::formUnbalance() < 0)
        return -1;

    if (distSOE != nullptr) {
        // Merge Phat to rank 0, clear the local RHS (no double count), transform on rank 0. The status
        // broadcast keeps every rank's global-RHS flag matched so the solve() collectives agree.
        int rc = distSOE->mergeBToRoot();
        if (rc == 0)
            distSOE->zeroB();
        if (isRoot(distSOE) && !areAlphaMFClose()) {
            const int rcT = transformRhs(csrSOE);
            if (rc == 0)
                rc = rcT;
        } else if (!isRoot(distSOE) && csrSOE->isCollectiveSolver() && !areAlphaMFClose()) {
            rc = matM->solve(nullptr, nullptr, 1);  // join rank 0's M solve
        }
        Vector status(1);
        status(0) = (rc == 0) ? 1.0 : -1.0;
        if (distSOE->broadcastFromRoot(status) < 0)
            return -2;
        if (status(0) < 0.0)
            return -2;
        distSOE->setGlobalRhsReady(true);
        distSOE->setLocalSolution(true);  // update() reads X only at this rank's DOFs
        return 0;
    }

    if (areAlphaMFClose())
        return 0;  // B already holds Phat; the system has A = alphaM*M
    return transformRhs(csrSOE);
}

int SparseExplicitAlpha::update(const Vector &aiPlusOne)
{
    updateCount++;
    if (updateCount > 1) {
        opserr << "ERROR SparseExplicitAlpha::update() - called " << updateCount
               << " times in one step (Linear algorithm only)\n";
        return -1;
    }
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr || U->Size() <= 0 || aiPlusOne.Size() != U->Size())
        return -2;

    // aiPlusOne is the system's X: the solution on rank 0, broadcast to the workers by solve().
    if (incrementalAccel)
        Udotdot->addVector(1.0, aiPlusOne, 1.0);
    else
        *Udotdot = aiPlusOne;

    // -noSecondPass: skip the element update at (u_{n+alpha}, v_{n+1}) (OpenSees' second pass); the
    // alpha-level evaluation of newStep is then the committing one: the method as published.
    theModel->setVel(*Udot);
    theModel->setAccel(*Udotdot);
    if (!noSecondPass && theModel->updateDomain() < 0) {
        opserr << "ERROR SparseExplicitAlpha::update() - updateDomain failed\n";
        return -3;
    }
    theModel->setDisp(*U);
    return 0;
}

int SparseExplicitAlpha::commit()
{
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr)
        return -1;
    // Domain was left at t + alphaF*dt after newStep; advance to t + dt.
    theModel->setCurrentDomainTime(theModel->getCurrentDomainTime() + (1.0 - alphaF) * deltaT);
    if (updElemDisp)
        theModel->updateDomain();
    return theModel->commitDomain();
}

int SparseExplicitAlpha::revertToLastStep()
{
    // undo the committed-time shift of newStep (Domain::revertToLastCommit used the shifted time)
    if (tStepStart >= 0.0) {
        if (Domain *theDomain = this->getAnalysisModel() ? this->getAnalysisModel()->getDomainPtr() : 0) {
            theDomain->setCurrentTime(tStepStart);
            theDomain->setCommittedTime(tStepStart);
        }
    }
    if (U->Size() > 0) {
        *U = *Ut;
        *Udot = *Utdot;
        *Udotdot = *Utdotdot;
    }
    return 0;
}

double SparseExplicitAlpha::getCFactor() { return c2; }

const Vector &SparseExplicitAlpha::getVel() { return *Ualphadot; }  // alpha-level velocity

int SparseExplicitAlpha::sendSelf(int cTag, Channel &theChannel)
{
    Vector data(11);
    data(0) = alphaM;
    data(1) = alphaF;
    data(2) = beta;
    data(3) = gamma;
    data(4) = updElemDisp ? 1.0 : 0.0;
    data(5) = incrementalAccel ? 1.0 : 0.0;
    data(6) = useAlphaCloseCheck ? 1.0 : 0.0;
    data(7) = initAccel ? 1.0 : 0.0;
    data(8) = noSecondPass ? 1.0 : 0.0;
    data(9) = tangentAtStart ? 1.0 : 0.0;
    data(10) = stiffnessFactor;
    return theChannel.sendVector(this->getDbTag(), cTag, data);
}

int SparseExplicitAlpha::recvSelf(int cTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    (void)theBroker;
    Vector data(11);
    if (theChannel.recvVector(this->getDbTag(), cTag, data) < 0)
        return -1;
    alphaM = data(0);
    alphaF = data(1);
    beta = data(2);
    gamma = data(3);
    updElemDisp = data(4) > 0.5;
    incrementalAccel = data(5) > 0.5;
    useAlphaCloseCheck = data(6) > 0.5;
    initAccel = data(7) > 0.5;
    noSecondPass = data(8) > 0.5;
    tangentAtStart = data(9) > 0.5;
    stiffnessFactor = data(10);
    operatorsBuilt = false;
    return 0;
}

void SparseExplicitAlpha::Print(OPS_Stream &s, int flag)
{
    (void)flag;
    s << "SparseExplicitAlpha alphaF=" << alphaF << " alphaM=" << alphaM << " initAccel=" << (initAccel ? 1 : 0)
      << " noSecondPass=" << (noSecondPass ? 1 : 0) << " tangentAtStart=" << (tangentAtStart ? 1 : 0)
      << " stiffnessFactor=" << stiffnessFactor << endln;
}

int SparseExplicitAlpha::revertToStart()
{
    if (U->Size() > 0) {
        Ut->Zero();
        Utdot->Zero();
        Utdotdot->Zero();
        U->Zero();
        Udot->Zero();
        Udotdot->Zero();
    }
    startPending = true;
    return 0;
}
