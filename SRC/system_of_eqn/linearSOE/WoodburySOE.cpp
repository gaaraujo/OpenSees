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

#include <WoodburySOE.h>
#include <WoodburySolver.h>
#include <AnalysisModel.h>
#include <Matrix.h>
#include <Vector.h>
#include <Graph.h>
#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include <classTags.h>

// Woodbury: A = A_s + U C V^T  (A_s assembled in inner SOE only).
//
//   Z = A_s^{-1} U
//   G = C^{-1} + V^T Z
//   x = x_s - Z * G^{-1} * V^T * x_s
//   A*p = A_s*p + U * C * (V^T * p)

WoodburySOE::WoodburySOE(LinearSOE &inner)
    : LinearSOE(LinSOE_TAGS_WoodburySOE),
      innerSOE(&inner),
      vectX(nullptr),
      woodburySolver(nullptr),
      processID(-1),
      numChannels(0),
      theChannels(nullptr),
      lowRankKind(LowRankKind::None),
      basisIsValid(false),
      lowRankNumDOF(0),
      lowRankRank(0),
      lowRankU(nullptr),
      lowRankV(nullptr),
      lowRankC(nullptr),
      lowRankCinv(nullptr),
      lowRankDiagD(nullptr),
      lowRankZ(nullptr),
      lowRankG(nullptr),
      lowRankWorkV1(nullptr),
      lowRankWorkV2(nullptr)
{
    hookWoodburySolver();
}

int
WoodburySOE::setProcessID(int processTag)
{
    processID = processTag;
    return 0;
}

int
WoodburySOE::setChannels(int nChannels, Channel **theC)
{
    numChannels = nChannels;

    if (theChannels != nullptr)
        delete[] theChannels;

    theChannels = nullptr;
    if (numChannels > 0 && theC != nullptr) {
        theChannels = new Channel *[numChannels];
        for (int i = 0; i < numChannels; i++)
            theChannels[i] = theC[i];
    }
    return 0;
}

int
WoodburySOE::reduceSumVector(Vector &v)
{
    // Same hub-and-spoke merge as ArpackSolver::myMv
    if (processID < 0 || numChannels <= 0 || theChannels == nullptr)
        return 0;

    const int n = v.Size();
    if (n <= 0)
        return 0;

    if (processID != 0) {
        theChannels[0]->sendVector(0, 0, v);
        theChannels[0]->recvVector(0, 0, v);
    } else {
        Vector other(n);
        for (int i = 0; i < numChannels; i++) {
            theChannels[i]->recvVector(0, 0, other);
            v += other;
        }
        for (int i = 0; i < numChannels; i++)
            theChannels[i]->sendVector(0, 0, v);
    }
    return 0;
}

WoodburySOE::~WoodburySOE()
{
    unhookWoodburySolver();
    clearLowRank();
    if (vectX != nullptr) {
        delete vectX;
        vectX = nullptr;
    }
    // Channels are owned by MachineBroker / interpreter; only free our pointer array.
    if (theChannels != nullptr) {
        delete[] theChannels;
        theChannels = nullptr;
    }
}

void
WoodburySOE::clearLowRankBasis(void)
{
    if (lowRankU != nullptr) {
        delete lowRankU;
        lowRankU = nullptr;
    }
    if (lowRankV != nullptr) {
        delete lowRankV;
        lowRankV = nullptr;
    }
    if (lowRankC != nullptr) {
        delete lowRankC;
        lowRankC = nullptr;
    }
    if (lowRankCinv != nullptr) {
        delete lowRankCinv;
        lowRankCinv = nullptr;
    }
    if (lowRankDiagD != nullptr) {
        delete[] lowRankDiagD;
        lowRankDiagD = nullptr;
    }
    if (lowRankZ != nullptr) {
        delete lowRankZ;
        lowRankZ = nullptr;
    }
    if (lowRankG != nullptr) {
        delete lowRankG;
        lowRankG = nullptr;
    }
    if (lowRankWorkV1 != nullptr) {
        delete lowRankWorkV1;
        lowRankWorkV1 = nullptr;
    }
    if (lowRankWorkV2 != nullptr) {
        delete lowRankWorkV2;
        lowRankWorkV2 = nullptr;
    }
    lowRankNumDOF = 0;
    lowRankRank = 0;
    basisIsValid = false;
}

void
WoodburySOE::clearLowRank(void)
{
    clearLowRankBasis();
    lowRankKind = LowRankKind::None;
}

void
WoodburySOE::clearWoodburyBasis(void)
{
    clearLowRank();
}

void
WoodburySOE::resizeX(int size)
{
    if (size <= 0) {
        if (vectX != nullptr) {
            delete vectX;
            vectX = nullptr;
        }
        return;
    }

    if (vectX == nullptr || vectX->Size() != size) {
        if (vectX != nullptr)
            delete vectX;
        vectX = new Vector(size);
    }
}

int
WoodburySOE::setWoodburyLowRank(const Matrix &uMat, const Matrix &cMat,
                                const Matrix &vMat)
{
    const int n = getNumEqn();
    const int kU = uMat.noCols();
    const int kV = vMat.noCols();
    const int kC = cMat.noRows();

    if (n <= 0 || kU <= 0) {
        opserr << "WARNING WoodburySOE::setWoodburyLowRank() - invalid system size\n";
        return -1;
    }
    if (uMat.noRows() != n || vMat.noRows() != n) {
        opserr << "WARNING WoodburySOE::setWoodburyLowRank() - U/V row dimension mismatch\n";
        return -1;
    }
    if (kU != kV || kC != kU || cMat.noCols() != kC) {
        opserr << "WARNING WoodburySOE::setWoodburyLowRank() - rank/dimension mismatch\n";
        return -1;
    }

    clearLowRankBasis();

    Matrix cInv(kC, kC);
    if (cMat.Invert(cInv) < 0) {
        opserr << "WARNING WoodburySOE::setWoodburyLowRank() - C is singular\n";
        return -2;
    }

    lowRankU = new Matrix(uMat);
    lowRankV = new Matrix(vMat);
    lowRankC = new Matrix(cMat);
    lowRankCinv = new Matrix(cInv);
    lowRankNumDOF = n;
    lowRankRank = kU;
    lowRankKind = LowRankKind::General;
    basisIsValid = false;
    return 0;
}

int
WoodburySOE::setWoodburySymmetric(const Matrix &qMat, const Vector &diagDVec)
{
    const int n = getNumEqn();
    const int k = qMat.noCols();

    if (n <= 0 || k <= 0) {
        opserr << "WARNING WoodburySOE::setWoodburySymmetric() - invalid system size\n";
        return -1;
    }
    if (qMat.noRows() != n || diagDVec.Size() != k) {
        opserr << "WARNING WoodburySOE::setWoodburySymmetric() - Q/diagD dimension mismatch\n";
        return -1;
    }
    for (int i = 0; i < k; ++i) {
        if (diagDVec(i) == 0.0) {
            opserr << "WARNING WoodburySOE::setWoodburySymmetric() - zero diagonal entry\n";
            return -1;
        }
    }

    clearLowRankBasis();

    lowRankU = new Matrix(qMat);
    lowRankDiagD = new double[k];
    for (int i = 0; i < k; ++i) {
        lowRankDiagD[i] = diagDVec(i);
    }
    lowRankNumDOF = n;
    lowRankRank = k;
    lowRankKind = LowRankKind::SymmetricDiag;
    basisIsValid = false;
    return 0;
}

int
WoodburySOE::rebuildWoodburyBasis(void)
{
    basisIsValid = false;

    if (lowRankKind == LowRankKind::None || lowRankRank <= 0 || lowRankU == nullptr)
        return 0;

    if (innerSOE == nullptr)
        return -1;

    const int n = innerSOE->getNumEqn();
    if (n != lowRankNumDOF || n != lowRankU->noRows()) {
        opserr << "WARNING WoodburySOE::rebuildWoodburyBasis() - dimension changed since setWoodbury\n";
        return -1;
    }

    if (lowRankZ != nullptr) {
        delete lowRankZ;
        lowRankZ = nullptr;
    }
    if (lowRankG != nullptr) {
        delete lowRankG;
        lowRankG = nullptr;
    }
    if (lowRankWorkV1 != nullptr) {
        delete lowRankWorkV1;
        lowRankWorkV1 = nullptr;
    }
    if (lowRankWorkV2 != nullptr) {
        delete lowRankWorkV2;
        lowRankWorkV2 = nullptr;
    }

    lowRankZ = new Matrix(n, lowRankRank);
    lowRankG = new Matrix(lowRankRank, lowRankRank);
    lowRankWorkV1 = new Vector(lowRankRank);
    lowRankWorkV2 = new Vector(lowRankRank);

    // Newton does formUnbalance → formTangent (here) → solve, so the residual RHS
    // must be preserved across the temporary multi-RHS solves for Z = A_s^{-1} U.
    Vector bSave(innerSOE->getB(true));

    // OpenSeesMP gather-on-solve: if U is full on every rank, only P0 sets B
    // (Arpack pattern). processID < 0 means serial / unset.
    for (int col = 0; col < lowRankRank; ++col) {
        Vector ucol(n);
        for (int row = 0; row < n; ++row) {
            ucol(row) = (*lowRankU)(row, col);
        }
        if (processID > 0) {
            innerSOE->zeroB();
        } else if (innerSOE->setB(ucol) < 0) {
            innerSOE->setB(bSave, true);
            return -2;
        }
        if (innerSOE->solve() < 0) {
            innerSOE->setB(bSave, true);
            return -3;
        }
        const Vector &xcol = innerSOE->getX();
        for (int row = 0; row < n; ++row) {
            (*lowRankZ)(row, col) = xcol(row);
        }
    }
    if (innerSOE->setB(bSave, true) < 0)
        return -2;

    if (lowRankKind == LowRankKind::General) {
        if (lowRankV == nullptr || lowRankCinv == nullptr) {
            return -1;
        }
        *lowRankG = *lowRankCinv;
        if (lowRankG->addMatrixTransposeProduct(1.0, *lowRankV, *lowRankZ, 1.0) < 0) {
            return -4;
        }
    } else if (lowRankKind == LowRankKind::SymmetricDiag) {
        if (lowRankDiagD == nullptr) {
            return -1;
        }
        lowRankG->Zero();
        if (lowRankG->addMatrixTransposeProduct(0.0, *lowRankU, *lowRankZ, 1.0) < 0) {
            return -4;
        }
        for (int i = 0; i < lowRankRank; ++i) {
            (*lowRankG)(i, i) += 1.0 / lowRankDiagD[i];
        }
    } else {
        return -1;
    }

    basisIsValid = true;
    return 0;
}

int
WoodburySOE::applyLowRankCorrection(Vector &x) const
{
    if (!woodburyBasisValid() || lowRankRank <= 0 || lowRankU == nullptr)
        return 0;

    if (lowRankWorkV1 == nullptr || lowRankWorkV2 == nullptr ||
        lowRankZ == nullptr || lowRankG == nullptr)
        return -1;

    if (lowRankKind == LowRankKind::General) {
        if (lowRankV == nullptr)
            return -1;
        if (lowRankWorkV1->addMatrixTransposeVector(0.0, *lowRankV, x, 1.0) < 0)
            return -1;
    } else if (lowRankKind == LowRankKind::SymmetricDiag) {
        if (lowRankWorkV1->addMatrixTransposeVector(0.0, *lowRankU, x, 1.0) < 0)
            return -1;
    } else {
        return -1;
    }

    if (lowRankG->Solve(*lowRankWorkV1, *lowRankWorkV2) < 0)
        return -1;

    if (x.addMatrixVector(1.0, *lowRankZ, *lowRankWorkV2, -1.0) < 0)
        return -1;

    return 0;
}

int
WoodburySOE::applyLowRankMatvec(const Vector &p, Vector &Ap) const
{
    if (!woodburyBasisValid() || lowRankRank <= 0 || lowRankU == nullptr)
        return 0;

    if (lowRankWorkV1 == nullptr || lowRankWorkV2 == nullptr)
        return -1;

    if (lowRankKind == LowRankKind::General) {
        if (lowRankV == nullptr || lowRankC == nullptr)
            return -1;
        if (lowRankWorkV1->addMatrixTransposeVector(0.0, *lowRankV, p, 1.0) < 0)
            return -1;
        if (lowRankWorkV2->addMatrixVector(0.0, *lowRankC, *lowRankWorkV1, 1.0) < 0)
            return -1;
        if (Ap.addMatrixVector(1.0, *lowRankU, *lowRankWorkV2, 1.0) < 0)
            return -1;
    } else if (lowRankKind == LowRankKind::SymmetricDiag) {
        if (lowRankDiagD == nullptr)
            return -1;
        if (lowRankWorkV1->addMatrixTransposeVector(0.0, *lowRankU, p, 1.0) < 0)
            return -1;
        for (int i = 0; i < lowRankRank; ++i) {
            (*lowRankWorkV1)(i) *= lowRankDiagD[i];
        }
        if (Ap.addMatrixVector(1.0, *lowRankU, *lowRankWorkV1, 1.0) < 0)
            return -1;
    } else {
        return -1;
    }

    return 0;
}

int
WoodburySOE::applyWoodburyCorrection(void)
{
    if (vectX == nullptr) {
        opserr << "WARNING WoodburySOE::applyWoodburyCorrection() - vectX not allocated\n";
        return -1;
    }

    const Vector &innerX = innerSOE->getX();
    if (innerX.Size() != vectX->Size()) {
        opserr << "WARNING WoodburySOE::applyWoodburyCorrection() - inner/wrapper X size mismatch\n";
        return -1;
    }

    *vectX = innerX;

    if (!woodburyBasisValid())
        return 0;

    return applyLowRankCorrection(*vectX);
}

int
WoodburySOE::solve(void)
{
    if (woodburySolver == nullptr)
        hookWoodburySolver();
    if (woodburySolver == nullptr) {
        opserr << "WARNING WoodburySOE::solve() - WoodburySolver not installed\n";
        return -1;
    }
    return woodburySolver->solve();
}

LinearSOESolver *
WoodburySOE::getSolver(void)
{
    return woodburySolver;
}

void
WoodburySOE::hookWoodburySolver(void)
{
    if (woodburySolver != nullptr)
        return;

    LinearSOESolver *innerSolver = innerSOE->getSolver();
    if (innerSolver == nullptr)
        return;

    woodburySolver = new WoodburySolver(*innerSolver, *this);
}

void
WoodburySOE::unhookWoodburySolver(void)
{
    if (woodburySolver == nullptr)
        return;

    delete woodburySolver;
    woodburySolver = nullptr;
}

int
WoodburySOE::addColA(const Vector &col, int colIndex, double fact)
{
    clearLowRank();
    return innerSOE->addColA(col, colIndex, fact);
}

int
WoodburySOE::setSize(Graph &theGraph)
{
    int res = innerSOE->setSize(theGraph);
    clearLowRank();
    resizeX(innerSOE->getNumEqn());
    return res;
}

int
WoodburySOE::getNumEqn(void) const
{
    return innerSOE->getNumEqn();
}

int
WoodburySOE::addA(const Matrix &m, const ID &id, double fact)
{
    clearLowRank();
    return innerSOE->addA(m, id, fact);
}

int
WoodburySOE::addA(const Matrix &m)
{
    clearLowRank();
    return innerSOE->addA(m);
}

int
WoodburySOE::addB(const Vector &v, const ID &id, double fact)
{
    return innerSOE->addB(v, id, fact);
}

int
WoodburySOE::setB(const Vector &v, double fact)
{
    return innerSOE->setB(v, fact);
}

int
WoodburySOE::setB(const Vector &v, bool localOnly)
{
    return innerSOE->setB(v, localOnly);
}

void
WoodburySOE::zeroA(void)
{
    clearLowRank();
    innerSOE->zeroA();
}

void
WoodburySOE::zeroB(void)
{
    innerSOE->zeroB();
}

int
WoodburySOE::formAp(const Vector &p, Vector &Ap)
{
    int res = innerSOE->formAp(p, Ap);
    if (res < 0)
        return res;
    if (woodburyBasisValid())
        res = applyLowRankMatvec(p, Ap);
    return res;
}

const Vector &
WoodburySOE::getX(void)
{
    if (vectX == nullptr) {
        opserr << "FATAL WoodburySOE::getX() - vectX not allocated\n";
        exit(-1);
    }
    return *vectX;
}

const Vector &
WoodburySOE::getB(void)
{
    return innerSOE->getB();
}

const Vector &
WoodburySOE::getB(bool localOnly)
{
    return innerSOE->getB(localOnly);
}

const Matrix *
WoodburySOE::getA(void)
{
    if (innerSOE == nullptr)
        return nullptr;
    return innerSOE->getA();
}

double
WoodburySOE::normRHS(void)
{
    return innerSOE->normRHS();
}

void
WoodburySOE::setX(int loc, double value)
{
    if (vectX != nullptr && loc >= 0 && loc < vectX->Size())
        (*vectX)(loc) = value;
}

void
WoodburySOE::setX(const Vector &x)
{
    if (vectX != nullptr && x.Size() == vectX->Size())
        *vectX = x;
}

int
WoodburySOE::setLinks(AnalysisModel &theModel)
{
    LinearSOE::setLinks(theModel);
    return innerSOE->setLinks(theModel);
}

int
WoodburySOE::saveSparseA(OPS_Stream &output, int baseIndex)
{
    if (innerSOE == nullptr)
        return -1;
    return innerSOE->saveSparseA(output, baseIndex);
}

int
WoodburySOE::getSparseA(ID &rowIndices, ID &colIndices, Vector &values,
                        int baseIndex)
{
    if (innerSOE == nullptr)
        return -1;
    return innerSOE->getSparseA(rowIndices, colIndices, values, baseIndex);
}

int
WoodburySOE::getSparseA(std::vector<int> &rowIndices,
                        std::vector<int> &colIndices,
                        std::vector<double> &values, int baseIndex)
{
    if (innerSOE == nullptr)
        return -1;
    return innerSOE->getSparseA(rowIndices, colIndices, values, baseIndex);
}

double
WoodburySOE::getDeterminant(void)
{
    if (innerSOE == nullptr)
        return 0.0;
    return innerSOE->getDeterminant();
}

int
WoodburySOE::sendSelf(int, Channel &)
{
    return 0;
}

int
WoodburySOE::recvSelf(int, Channel &, FEM_ObjectBroker &)
{
    return 0;
}
