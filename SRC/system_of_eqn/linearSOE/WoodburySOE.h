#ifndef WoodburySOE_h
#define WoodburySOE_h

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

// Written: Gustavo A. Araujo R. (gaaraujor@gmail.com)
// Created: 06/26
//
// Description: Wraps a structural LinearSOE for modalDamping -woodbury. The inner
// SOE holds A_s; low-rank update A = A_s + U C V^T (symmetric Q diag(D) Q^T)
// is applied in solve and formAp via Woodbury identity.
//
// OpenSeesMP: when the inner SOE gathers RHS contributions across ranks
// (MumpsParallelSOE, DistributedProfileSPDLinSOE), Z = A_s^{-1} U is built with
// the Arpack P0-only setB pattern so the dense modal columns are not summed.

#include <LinearSOE.h>

class AnalysisModel;
class WoodburySolver;
class Matrix;
class Vector;
class Channel;

class WoodburySOE : public LinearSOE
{
  public:
    explicit WoodburySOE(LinearSOE &innerSOE);
    ~WoodburySOE() override;

    LinearSOE &getInnerSOE(void) const { return *innerSOE; }

    // OpenSeesMP: same pattern as ArpackSOE / MumpsParallelSOE
    int setProcessID(int processTag);
    int getProcessID(void) const { return processID; }
    int setChannels(int nChannels, Channel **theChannels);
    int reduceSumVector(Vector &v);

    int applyWoodburyCorrection(void);

    int setWoodburyLowRank(const Matrix &U, const Matrix &C, const Matrix &V);
    int setWoodburySymmetric(const Matrix &Q, const Vector &diagD);
    int rebuildWoodburyBasis(void);
    void clearWoodburyBasis(void);

    bool woodburyActive(void) const { return lowRankKind != LowRankKind::None; }
    bool woodburyBasisValid(void) const { return basisIsValid; }

    int setSize(Graph &theGraph) override;
    int getNumEqn(void) const override;

    int addA(const Matrix &, const ID &, double fact = 1.0) override;
    int addA(const Matrix &) override;
    int addB(const Vector &, const ID &, double fact = 1.0) override;
    int setB(const Vector &, double fact = 1.0) override;
    int setB(const Vector &v, bool localOnly) override;

    void zeroA(void) override;
    void zeroB(void) override;

    int formAp(const Vector &p, Vector &Ap) override;

    const Vector &getX(void) override;
    const Vector &getB(void) override;
    const Vector &getB(bool localOnly) override;
    const Matrix *getA(void) override;
    double normRHS(void) override;

    void setX(int loc, double value) override;
    void setX(const Vector &x) override;

    int setLinks(AnalysisModel &theModel) override;

    int solve(void) override;
    LinearSOESolver *getSolver(void);

    int addColA(const Vector &col, int colIndex, double fact = 1.0) override;

    int saveSparseA(OPS_Stream &output, int baseIndex = 0) override;
    int getSparseA(ID &rowIndices, ID &colIndices, Vector &values,
                   int baseIndex = 0) override;
    int getSparseA(std::vector<int> &rowIndices, std::vector<int> &colIndices,
                   std::vector<double> &values, int baseIndex = 0) override;

    double getDeterminant(void) override;

    int sendSelf(int commitTag, Channel &theChannel) override;
    int recvSelf(int commitTag, Channel &theChannel,
                 FEM_ObjectBroker &theBroker) override;

  private:
    enum class LowRankKind { None, SymmetricDiag, General };

    void resizeX(int size);
    void hookWoodburySolver(void);
    void unhookWoodburySolver(void);
    void clearLowRankBasis(void);
    void clearLowRank(void);
    int applyLowRankCorrection(Vector &x) const;
    int applyLowRankMatvec(const Vector &p, Vector &Ap) const;

    LinearSOE *innerSOE;
    Vector *vectX;
    WoodburySolver *woodburySolver;
    int processID; // -1 serial; >=0 OpenSeesMP (P0 owns dense RHS for Z solves)
    int numChannels;
    Channel **theChannels;

    LowRankKind lowRankKind;
    bool basisIsValid;
    int lowRankNumDOF;
    int lowRankRank;
    Matrix *lowRankU;
    Matrix *lowRankV;
    Matrix *lowRankC;
    Matrix *lowRankCinv;
    double *lowRankDiagD;
    Matrix *lowRankZ;
    Matrix *lowRankG;
    Vector *lowRankWorkV1;
    Vector *lowRankWorkV2;
};

#endif
