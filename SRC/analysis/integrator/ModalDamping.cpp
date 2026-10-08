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

// Original implementation: Frank McKenna (03/15)
// Reimplemented: Gustavo A. Araujo R. (gaaraujor@gmail.com, 06/26)
//
// Description: This class implements modal damping. It provides methods
// to add the effects of modal damping to the RHS and the tangent matrix.

#include <ModalDamping.h>
#include <IncrementalIntegrator.h>
#include <AnalysisModel.h>
#include <DirectIntegrationAnalysis.h>
#include <Domain.h>
#include <DOF_Group.h>
#include <DOF_GrpIter.h>
#include <ID.h>
#include <EigenSOE.h>
#include <LinearSOE.h>
#include <OPS_Globals.h>
#include <Vector.h>
#include <Matrix.h>
#include <elementAPI.h>
#include <cmath>
#include <cstring>
#include <string.h>
#include <WoodburySOE.h>
#include <classTags.h>

ModalDamping::ModalDamping(IncrementalIntegrator &owner)
    : theIntegrator(&owner),
      eigenVectors(nullptr),
      eigenValues(nullptr),
      dampingForces(nullptr)
{
}

ModalDamping::~ModalDamping()
{
  if (eigenValues != nullptr)
    delete eigenValues;
  if (eigenVectors != nullptr)
    delete[] eigenVectors;
  if (dampingForces != nullptr)
    delete dampingForces;
}

int
ModalDamping::setupModal(const Vector *factors)
{
  if (factors == nullptr || theIntegrator == nullptr)
    return -1;

  AnalysisModel *theAnalysisModel = theIntegrator->getAnalysisModel();
  LinearSOE *theSOE = theIntegrator->getLinearSOE();
  if (theAnalysisModel == nullptr || theSOE == nullptr)
    return -1;

  int numModes = factors->Size();
  const Vector &eigenvalues = theAnalysisModel->getEigenvalues();
  const int numEigen = eigenvalues.Size();

  if (numEigen < numModes)
    numModes = numEigen;

  const int numDOF = theSOE->getNumEqn();

  if (eigenValues != nullptr && *eigenValues == eigenvalues)
    return 0;

  if (eigenValues != nullptr)
    delete eigenValues;
  if (eigenVectors != nullptr)
    delete[] eigenVectors;
  if (dampingForces != nullptr)
    delete dampingForces;

  eigenValues = new Vector(eigenvalues);
  dampingForces = new Vector(numDOF);
  eigenVectors = new double[static_cast<size_t>(numDOF) * static_cast<size_t>(numModes)];
  for (int i = 0; i < numDOF * numModes; ++i)
    eigenVectors[i] = 0.0;

  DOF_GrpIter &theDOFs = theAnalysisModel->getDOFs();
  DOF_Group *dofPtr;
  while ((dofPtr = theDOFs()) != 0) {
    const Matrix &dofEigenvectors = dofPtr->getEigenvectors();
    const ID &dofID = dofPtr->getID();
    for (int j = 0; j < numModes; j++) {
      for (int i = 0; i < dofID.Size(); i++) {
        const int id = dofID(i);
        if (id >= 0)
          eigenVectors[j * numDOF + id] = dofEigenvectors(i, j);
      }
    }
  }

  double *eigenVectors2 = new double[static_cast<size_t>(numDOF) * static_cast<size_t>(numModes)];

  WoodburySOE *wb = nullptr;
  if (theSOE->getClassTag() == LinSOE_TAGS_WoodburySOE)
    wb = static_cast<WoodburySOE *>(theSOE);

  for (int i = 0; i < numModes; i++) {
    double *eigenVectorI = &eigenVectors[numDOF * i];
    double *mEigenVectorI = &eigenVectors2[numDOF * i];
    Vector v1(eigenVectorI, numDOF);
    Vector v2(mEigenVectorI, numDOF);
    theIntegrator->doMv(v1, v2);
    // Partitioned OpenSeesMP: merge local M*phi like ArpackSolver::myMv
    if (wb != nullptr && wb->reduceSumVector(v2) < 0) {
      delete[] eigenVectors2;
      return -1;
    }
  }

  delete[] eigenVectors;
  eigenVectors = eigenVectors2;

  return 0;
}

int
ModalDamping::addToUnbalance(const Vector *factors)
{
  if (factors == nullptr || theIntegrator == nullptr)
    return 0;

  AnalysisModel *theAnalysisModel = theIntegrator->getAnalysisModel();
  LinearSOE *theSOE = theIntegrator->getLinearSOE();
  if (theAnalysisModel == nullptr || theSOE == nullptr)
    return -1;

  int numModes = factors->Size();
  const Vector &eigenvalues = theAnalysisModel->getEigenvalues();
  const int numEigen = eigenvalues.Size();

  if (numEigen < numModes) {
    numModes = numEigen;
    opserr << "WARNING: HAving to reset numModes to : " << numModes
           << "as not enough eigenvalues. NOTE if 0 you have done something to "
              "require new analysis or have not issued eigen command\n";
  }

  if (setupModal(factors) < 0)
    return -1;

  const int numDOF = theSOE->getNumEqn();
  const Vector &vel = theIntegrator->getVel();

  dampingForces->Zero();

  for (int i = 0; i < numModes; i++) {
    const double eigenvalue = (*eigenValues)(i);
    const double modalDampingValue = (*factors)(i);
    if (eigenvalue > 0 && modalDampingValue != 0.0) {
      const double wn = sqrt(eigenvalue);
      double *eigenVectorI = &eigenVectors[numDOF * i];
      double beta = 0.0;

      for (int j = 0; j < numDOF; j++) {
        const double eij = eigenVectorI[j];
        if (eij != 0)
          beta += eij * vel(j);
      }

      beta = -2.0 * modalDampingValue * wn * beta;

      for (int j = 0; j < numDOF; j++) {
        const double eij = eigenVectorI[j];
        if (eij != 0)
          (*dampingForces)(j) += beta * eij;
      }
    }
  }

  // OpenSeesMP: when Woodbury holds a gather-on-solve SOE with full modal force
  // on every rank, only P0 installs it (same rule as Arpack RHS).
  if (theSOE->getClassTag() == LinSOE_TAGS_WoodburySOE) {
    WoodburySOE *wb = static_cast<WoodburySOE *>(theSOE);
    if (wb->getProcessID() > 0)
      return 0;
  }

  return theSOE->setB(*dampingForces);
}

int
ModalDamping::countActiveModes(const Vector *factors) const
{
  if (factors == nullptr || eigenValues == nullptr)
    return 0;

  int numModes = factors->Size();
  if (numModes > eigenValues->Size())
    numModes = eigenValues->Size();

  int activeCol = 0;
  for (int i = 0; i < numModes; ++i) {
    const double eigenvalue = (*eigenValues)(i);
    const double xi = (*factors)(i);
    if (eigenvalue > 0.0 && xi != 0.0)
      ++activeCol;
  }
  return activeCol;
}

int
ModalDamping::addToTangent(const Vector *factors, double cFactor)
{
  if (factors == nullptr || theIntegrator == nullptr)
    return 0;

  if (cFactor == 0.0)
    return 0;

  LinearSOE *theSOE = theIntegrator->getLinearSOE();
  if (theSOE == nullptr)
    return -1;

  if (setupModal(factors) < 0)
    return -1;

  const int numDOF = theSOE->getNumEqn();
  int numModes = factors->Size();
  if (numModes > eigenValues->Size())
    numModes = eigenValues->Size();

  for (int dof = 0; dof < numDOF; dof++) {
    dampingForces->Zero();
    bool zeroCol = true;

    for (int i = 0; i < numModes; i++) {
      const double eigenvalue = (*eigenValues)(i);
      const double modalDampingValue = (*factors)(i);
      if (eigenvalue > 0 && modalDampingValue != 0.0) {
        const double wn = sqrt(eigenvalue);
        double *eigenVectorI = &eigenVectors[numDOF * i];
        const double ei_dof = eigenVectors[numDOF * i + dof];

        if (ei_dof != 0.0) {
          zeroCol = false;

          const double beta =
              2.0 * modalDampingValue * wn * ei_dof * cFactor;

          for (int j = 0; j < numDOF; j++) {
            const double eij = eigenVectorI[j];
            if (eij != 0)
              (*dampingForces)(j) += beta * eij;
          }
        }
      }
    }

    if (zeroCol == false)
      theSOE->addColA(*dampingForces, dof, 1.0);
  }
  return 0;
}

int
ModalDamping::prepareWoodburyLowRank(const Vector *factors, double cFactor,
                                     int numDOF, Matrix &Q, Vector &diagD)
{
  if (factors == nullptr || theIntegrator == nullptr)
    return 0;

  if (cFactor == 0.0)
    return 0;

  if (setupModal(factors) < 0)
    return -1;

  const int activeCol = countActiveModes(factors);
  if (activeCol <= 0 || numDOF <= 0)
    return 0;

  if (Q.noRows() != numDOF || Q.noCols() != activeCol)
    Q.resize(numDOF, activeCol);
  if (diagD.Size() != activeCol)
    diagD.resize(activeCol);

  return buildSymmetricLowRank(factors, cFactor, Q, diagD);
}

int
ModalDamping::buildSymmetricLowRank(const Vector *factors, double cFactor,
                                    Matrix &Q, Vector &diagD)
{
  if (factors == nullptr || theIntegrator == nullptr)
    return 0;

  if (cFactor == 0.0)
    return 0;

  LinearSOE *theSOE = theIntegrator->getLinearSOE();
  if (theSOE == nullptr)
    return -1;

  const int numDOF = theSOE->getNumEqn();
  if (numDOF <= 0)
    return 0;

  if (setupModal(factors) < 0)
    return -1;

  const int numEigenModes = factors->Size();
  int maxModes = numEigenModes;
  if (maxModes > eigenValues->Size())
    maxModes = eigenValues->Size();

  const int activeCol = countActiveModes(factors);
  if (activeCol <= 0)
    return 0;

  if (Q.noRows() != numDOF || Q.noCols() != activeCol || diagD.Size() != activeCol) {
    opserr << "WARNING ModalDamping::buildSymmetricLowRank() - Q/diagD size mismatch\n";
    return -3;
  }

  int col = 0;
  for (int i = 0; i < numEigenModes; ++i) {
    const double eigenvalue = (*eigenValues)(i);
    const double xi = (*factors)(i);
    if (eigenvalue <= 0.0 || xi == 0.0)
      continue;

    diagD(col) = cFactor * 2.0 * xi * sqrt(eigenvalue);
    memcpy(&Q(0, col), &eigenVectors[i * numDOF],
           static_cast<size_t>(numDOF) * sizeof(double));
    ++col;
  }

  return activeCol;
}

static int
OPS_modalDampingImpl(ModalDampingOption defaultOption, bool allowFlag)
{
  ModalDampingOption option = defaultOption;
  if (allowFlag && OPS_GetNumRemainingInputArgs() >= 1) {
    const char *arg = OPS_GetString();
    if (strcmp(arg, "-woodbury") == 0)
      option = MODAL_DAMPING_WOODBURY;
    else if (strcmp(arg, "-legacy") == 0)
      option = MODAL_DAMPING_INCL_MATRIX;
    else if (strcmp(arg, "-quick") == 0)
      option = MODAL_DAMPING_QUICK;
    else
      OPS_ResetCurrentInputArg(-1);
  }

  if (OPS_GetNumRemainingInputArgs() < 1) {
    opserr << "WARNING modalDamping ?factor - not enough arguments to command\n";
    return -1;
  }

  int *numEigenPtr = OPS_GetNumEigen();
  EigenSOE **eigenSOEPtr = OPS_GetEigenSOE();
  if (numEigenPtr == 0 || eigenSOEPtr == 0) {
    opserr << "WARNING modalDamping - eigen command needs to be called first - NO MODAL DAMPING APPLIED\n ";
    return -1;
  }

  int numEigen = *numEigenPtr;
  EigenSOE *theEigenSOE = *eigenSOEPtr;
  if (numEigen == 0 || theEigenSOE == 0) {
    opserr << "WARNING modalDamping - eigen command needs to be called first - NO MODAL DAMPING APPLIED\n ";
    return -1;
  }

  int numModes = OPS_GetNumRemainingInputArgs();
  if (numModes != 1 && numModes < numEigen) {
    opserr << "WARNING modalDamping - fewer damping factors than modes were specified\n";
    opserr << "                     - zero damping will be applied to un-specified modes" << endln;
  }
  if (numModes > numEigen) {
    opserr << "WARNING modalDamping - more damping factors than modes were specifed\n";
    opserr << "                     - ignoring additional damping factors" << endln;
  }

  double factor;
  Vector modalDampingValues(numEigen);
  int numdata = 1;

  if (numModes == 1) {
    if (OPS_GetDoubleInput(&numdata, &factor) < 0) {
      opserr << "WARNING modalDamping - could not read factor for all modes \n";
      return -1;
    }
    for (int i = 0; i < numEigen; i++)
      modalDampingValues(i) = factor;
  } else {
    for (int i = 0; i < numModes; i++) {
      if (OPS_GetDoubleInput(&numdata, &factor) < 0) {
        opserr << "WARNING modalDamping - could not read factor for mode " << i + 1 << endln;
        return -1;
      }
      modalDampingValues(i) = factor;
    }
    for (int i = numModes; i < numEigen; i++)
      modalDampingValues(i) = 0.0;
  }

  Domain *theDomain = OPS_GetDomain();
  if (theDomain == 0) {
    opserr << "WARNING modalDamping - no domain available\n";
    return -1;
  }

  theDomain->setModalDampingFactors(&modalDampingValues, option);

  DirectIntegrationAnalysis **transPtr = OPS_GetTransientAnalysis();
  LinearSOE **soePtr = OPS_GetSOE();
  if (transPtr != 0 && *transPtr != 0 && soePtr != 0 && *soePtr != 0) {
    if ((*transPtr)->setLinearSOE(**soePtr) < 0) {
      opserr << "WARNING modalDamping - failed to refresh Woodbury analysis links\n";
      return -1;
    }
  }

  return 0;
}

int
OPS_modalDamping(void)
{
  return OPS_modalDampingImpl(MODAL_DAMPING_INCL_MATRIX, true);
}

int
OPS_modalDampingQ(void)
{
  return OPS_modalDampingImpl(MODAL_DAMPING_QUICK, false);
}
