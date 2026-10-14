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
                                                                        
// $Revision: 1.5 $
// $Date: 2009-05-14 22:45:39 $
// $Source: /usr/local/cvs/OpenSees/SRC/system_of_eqn/linearSOE/LinearSOE.cpp,v $
                                                                        
                                                                        
// Written: fmk 
// Created: 11/96
//
// Description: This file contains the implementation of LinearSOE.
//
// What: "@(#) LinearSOE.C, revA"

#include<LinearSOE.h>
#include<LinearSOESolver.h>
#include<Matrix.h>
#include<Vector.h>
#include<ID.h>
#include<Channel.h>

struct LinearSOE::LowRank
{
  Matrix Q, Z, G;
  Vector d, w1, w2;
  bool active = false;
  bool basisValid = false;
};

LinearSOE::LinearSOE(LinearSOESolver &theLinearSOESolver, int classtag)
    :MovableObject(classtag), theModel(0), theLowRank(0), theSolver(&theLinearSOESolver)
{

}

LinearSOE::LinearSOE(int classtag)
:MovableObject(classtag), theModel(0), theLowRank(0), theSolver(0)
{

}


LinearSOE::~LinearSOE()
{
  if (theSolver != 0)
    delete theSolver;
  if (theLowRank != 0)
    delete theLowRank;
}

int
LinearSOE::solveAssembled(void)
{
  if (theSolver != 0)
    return (theSolver->solve());
  else
    return -1;
}

int
LinearSOE::solveAssembled(const Vector &b, Vector &x)
{
  // keep this rank's RHS (e.g. the unbalance, formed before the tangent)
  Vector bSave(this->getB(true));
  // gather-on-solve systems sum the ranks' RHS: b is full on every rank, so only rank 0 sets it
  int res = 0;
  if (this->getProcessID() > 0)
    this->zeroB();
  else
    res = this->setB(b);
  if (res >= 0)
    res = this->solveAssembled();
  if (res >= 0)
    x = this->getX();
  if (this->setB(bSave, true) < 0 && res >= 0)
    res = -2;
  return res;
}

int
LinearSOE::formApAssembled(const Vector &p, Vector &Ap)
{
  return -1;
}

int
LinearSOE::solve(void)
{
  if (theLowRank == 0 || !theLowRank->active)
    return this->solveAssembled();

  if (!theLowRank->basisValid && this->buildLowRankBasis() < 0)
    return -3;

  int res = this->solveAssembled();
  if (res < 0)
    return res;

  // x = x_s - Z G^{-1} Q^T x_s  (Q, Z, G and x_s are full on every rank)
  LowRank &lr = *theLowRank;
  Vector x(this->getX());
  if (lr.w1.addMatrixTransposeVector(0.0, lr.Q, x, 1.0) < 0 ||
      lr.G.Solve(lr.w1, lr.w2) < 0 ||
      x.addMatrixVector(1.0, lr.Z, lr.w2, -1.0) < 0)
    return -4;
  this->setX(x);
  return res;
}

int
LinearSOE::formAp(const Vector &p, Vector &Ap)
{
  int res = this->formApAssembled(p, Ap);
  if (res < 0 || theLowRank == 0 || !theLowRank->active)
    return res;

  // Ap += Q D Q^T p
  LowRank &lr = *theLowRank;
  if (lr.w1.addMatrixTransposeVector(0.0, lr.Q, p, 1.0) < 0)
    return -1;
  for (int i = 0; i < lr.w1.Size(); ++i)
    lr.w1(i) *= lr.d(i);
  return Ap.addMatrixVector(1.0, lr.Q, lr.w1, 1.0);
}

void
LinearSOE::zeroA(void)
{
  this->clearLowRank();
  this->zeroAAssembled();
}

int
LinearSOE::addLowRank(const Matrix &Q, const Vector &d, double fact)
{
  const int n = this->getNumEqn();
  const int J = Q.noCols();
  if (Q.noRows() != n || d.Size() != J) {
    opserr << "WARNING LinearSOE::addLowRank() - Q is " << Q.noRows() << " x " << J
           << ", d has size " << d.Size() << ", system has " << n << " equations\n";
    return -1;
  }
  if (J == 0 || fact == 0.0) {
    this->clearLowRank();
    return 0;
  }
  for (int i = 0; i < J; ++i)
    if (d(i) == 0.0) {
      opserr << "WARNING LinearSOE::addLowRank() - zero entry in d\n";
      return -1;
    }

  if (theLowRank == 0)
    theLowRank = new LowRank();
  LowRank &lr = *theLowRank;
  if (lr.Q.noRows() != n || lr.Q.noCols() != J) {
    lr.Q.resize(n, J);
    lr.Z.resize(n, J);
    lr.G.resize(J, J);
    lr.d.resize(J);
    lr.w1.resize(J);
    lr.w2.resize(J);
  }
  lr.Q = Q;
  lr.d.addVector(0.0, d, fact);
  lr.active = true;
  lr.basisValid = false;
  return 0;
}

bool
LinearSOE::hasLowRank(void) const
{
  return theLowRank != 0 && theLowRank->active;
}

void
LinearSOE::clearLowRank(void)
{
  if (theLowRank != 0) {
    theLowRank->active = false;
    theLowRank->basisValid = false;
  }
}

int
LinearSOE::buildLowRankBasis(void)
{
  LowRank &lr = *theLowRank;
  const int n = this->getNumEqn();
  const int J = lr.Q.noCols();
  if (lr.Q.noRows() != n) {
    opserr << "WARNING LinearSOE::solve() - system size changed after addLowRank()\n";
    return -1;
  }

  // Z = A_s^{-1} Q, one column at a time (the first solve factors A_s)
  Vector q(n), z(n);
  for (int j = 0; j < J; ++j) {
    for (int i = 0; i < n; ++i)
      q(i) = lr.Q(i, j);
    if (this->solveAssembled(q, z) < 0)
      return -2;
    for (int i = 0; i < n; ++i)
      lr.Z(i, j) = z(i);
  }

  // G = D^{-1} + Q^T Z
  if (lr.G.addMatrixTransposeProduct(0.0, lr.Q, lr.Z, 1.0) < 0)
    return -3;
  for (int i = 0; i < J; ++i)
    lr.G(i, i) += 1.0 / lr.d(i);

  lr.basisValid = true;
  return 0;
}

int
LinearSOE::getProcessID(void) const
{
  return -1;
}

int
LinearSOE::reduceSum(Vector &v)
{
  (void)v;
  return 0;
}

int
LinearSOE::reduceSumChannels(Vector &v, int processID, int numChannels, Channel **theChannels)
{
  if (processID < 0 || numChannels <= 0 || theChannels == 0 || v.Size() == 0)
    return 0;

  if (processID != 0) {
    if (theChannels[0]->sendVector(0, 0, v) < 0 || theChannels[0]->recvVector(0, 0, v) < 0)
      return -1;
  } else {
    Vector other(v.Size());
    for (int i = 0; i < numChannels; i++) {
      if (theChannels[i]->recvVector(0, 0, other) < 0)
        return -1;
      v += other;
    }
    for (int i = 0; i < numChannels; i++)
      if (theChannels[i]->sendVector(0, 0, v) < 0)
        return -1;
  }
  return 0;
}

const Vector &
LinearSOE::getB(bool localOnly)
{
  (void)localOnly;
  return this->getB();
}

int
LinearSOE::setB(const Vector &v, bool localOnly)
{
  (void)localOnly;
  return this->setB(v, 1.0);
}

double
LinearSOE::getDeterminant(void)
{
  if (theSolver != 0)
    return theSolver->getDeterminant();
  else 
    return 0;
}



int 
LinearSOE::setSolver(LinearSOESolver &newSolver)
{
    theSolver = &newSolver;
    return 0;
}

LinearSOE *
LinearSOE::getCopy(void) const
{
    return nullptr;
}

LinearSOESolver *
LinearSOE::getSolver(void)
{
    return theSolver;
}

const LinearSOESolver *
LinearSOE::getSolver(void) const
{
    return theSolver;
}

int 
LinearSOE::setLinks(AnalysisModel &theModel)
{
    this->theModel = &theModel;
    return 0;
}


int
LinearSOE::addA(const Matrix &) {
  return -1;
}

int
LinearSOE::addColA(const Vector &col, int colIndex, double fact) {
  return -1;
}

int LinearSOE::saveSparseA(OPS_Stream& output, int baseIndex) {
  const Matrix* A = this->getA();
  
  if (A == nullptr) {
    return -1;
  }
  
  int rows = A->noRows();
  int cols = A->noCols();
  int nnz = rows * cols;
  
  // Assume the header is already written to output stream
  
  output << rows << " " << cols << " " << nnz << "\n";
  
  // Write all elements with base index
  int nnz_written = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      double val = (*A)(i,j);
      output << i + baseIndex << " " << j + baseIndex << " " << val << "\n";
      nnz_written++;
    }
  }
  if (nnz_written != nnz) {
    return -1;
  }
  return 0;
}

int LinearSOE::getSparseA(ID& rowIndices, ID& colIndices, Vector& values, int baseIndex) {
  const Matrix* A = this->getA();
  
  if (A == nullptr) {
    return -1;
  }
  
  int rows = A->noRows();
  int cols = A->noCols();
  int nnz = rows * cols;
  
  // Resize vectors to hold all elements
  rowIndices.resize(nnz);
  colIndices.resize(nnz);
  values.resize(nnz);
  
  // Fill vectors with all elements
  int idx = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      rowIndices(idx) = i + baseIndex;
      colIndices(idx) = j + baseIndex;
      values(idx) = (*A)(i,j);
      idx++;
    }
  }
  
  return 0;
}

int LinearSOE::getSparseA(std::vector<int>& rowIndices, std::vector<int>& colIndices, std::vector<double>& values, int baseIndex) {
  const Matrix* A = this->getA();
  
  if (A == nullptr) {
    return -1;
  }
  
  int rows = A->noRows();
  int cols = A->noCols();
  int nnz = rows * cols;
  
  // Resize vectors to hold all elements
  rowIndices.resize(nnz);
  colIndices.resize(nnz);
  values.resize(nnz);
  
  // Fill vectors with all elements
  int idx = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      rowIndices[idx] = i + baseIndex;
      colIndices[idx] = j + baseIndex;
      values[idx] = (*A)(i,j);
      idx++;
    }
  }
  
  return 0;
}