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
** Description: CPU explicit Kolay-Ricles integrator (alpha method)   **
** for SparseCsrLinSOE, the counterpart of CudaExplicitAlpha.         **
** ****************************************************************** */

// Written: gaaraujo
// Created: 10/2026
//
// Same algorithm and data flow as CudaExplicitAlpha, with the GPU replaced by the CPU: the system
// (`system SparseCSR` / `system DistributedSparseCSR`) holds A and solves it; the integrator owns
// M, alpha = M + gamma dt C + beta dt^2 K, and a copy of A for products, all on the system's CSR
// pattern and factorized with the system's solver (UMFPACK or ProfileSPD). Under OpenSeesMP, rank 0
// holds the operators and the kinematics, and every rank determines the state of its own elements.
//
//   integrator SparseKRAlpha rhoInf | SparseMKRAlpha rhoInfEquivalent | SparseExplicitAlpha aF aM g b
//              <-updateElemDisp> <-incrementalAccel> <-alphaCloseCheck> <-initAccel> <-noSecondPass>

#ifndef SparseExplicitAlpha_h
#define SparseExplicitAlpha_h

#include <TransientIntegrator.h>
#include <vector>

class DOF_Group;
class FE_Element;
class Vector;
class Channel;
class FEM_ObjectBroker;
class CsrMatrix;
class SparseCsrLinSOE;
class DistributedSparseCsrLinSOE;

class SparseExplicitAlpha : public TransientIntegrator
{
public:
    struct Options {
        bool updElemDisp = false;
        bool incrementalAccel = false;
        bool useAlphaCloseCheck = false;
        bool initAccel = false;
        bool noSecondPass = false;
    };

    SparseExplicitAlpha();
    SparseExplicitAlpha(double alphaF, double alphaM, double gamma, double beta, const Options &opts);
    ~SparseExplicitAlpha() override;

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

private:
    double alphaM;
    double alphaF;
    double beta;
    double gamma;
    bool updElemDisp;
    bool noSecondPass;  // -noSecondPass (see update())
    bool incrementalAccel;
    bool useAlphaCloseCheck;
    bool initAccel;     // -initAccel: start from the equilibrium acceleration M a0 = p - C v0 - f
    bool startPending;  // starting correction not yet applied
    double tStepStart;  // domain time at the start of the step (committed-time shift for ops_Dt)

    int updateCount;
    double c1, c2, c3;
    double deltaT;

    Vector *Ut, *Utdot, *Utdotdot;               // state at step start (revert backup)
    Vector *U, *Udot, *Udotdot;                  // current state
    Vector *Ualpha, *Ualphadot, *Ualphadotdot;   // response at t + alphaF*dt; getVel() returns Ualphadot

    bool operatorsBuilt;     // M / alpha / A formed and factorized for the current domain and deltaT
    bool motionNeedsGather;  // OpenSeesMP: gather U/Udot/Uddot to rank 0 once after domainChanged

    // Rank 0 (or serial): operators on the system's CSR pattern, and solve workspaces.
    CsrMatrix *matM;
    CsrMatrix *matAlpha;
    CsrMatrix *matA;  // products only; the system factorizes and solves its own A
    int boundRows, boundNnz;
    const int *boundRowPtr;
    const int *boundColIdx;
    std::vector<double> alphaRhs, alphaSol, w2;  // [M Uddot | A Uddot], [alpha1 Uddot | alpha^{-1} A Uddot]

    static constexpr double toleranceAlphaMF = 1.0e-8;
    bool areAlphaMFClose() const;

    int validateSOE(SparseCsrLinSOE *&csrSOE, DistributedSparseCsrLinSOE *&distSOE) const;
    bool isRoot(DistributedSparseCsrLinSOE *distSOE) const;
    int formTangentIntoSOE(int statFlag, double c1v, double c2v, double c3v);
    int formOperators(SparseCsrLinSOE *csrSOE, DistributedSparseCsrLinSOE *distSOE);
    int bindSharedStructure(SparseCsrLinSOE *csrSOE);
    /** Distributed MUMPS: workers hold participant copies of M and alpha and join their collective
     *  factorizations and solves at the same points as rank 0. */
    int bindParticipants(SparseCsrLinSOE *csrSOE);
    int numAlphaRhs(void) const;
    int applyStartCorrection(SparseCsrLinSOE *csrSOE, DistributedSparseCsrLinSOE *distSOE);
    int newStepPredictor(void);
    int transformRhs(SparseCsrLinSOE *csrSOE);
    void destroyOperators(void);
};

void *OPS_SparseExplicitAlpha(void);
void *OPS_SparseKRAlpha(void);
void *OPS_SparseMKRAlpha(void);

#endif
