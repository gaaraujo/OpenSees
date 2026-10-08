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
                                                                        
// $Revision: 1.8 $
// $Date: 2007-04-02 23:42:26 $
// $Source: /usr/local/cvs/OpenSees/SRC/analysis/integrator/IncrementalIntegrator.cpp,v $
                                                                        
// Written: fmk 
// Created: 11/96
// Revision: A
//
// Description: This file contains the implementation of IncrementalIntegrator.
//
// What: "@(#) IncrementalIntegrator.C, revA"

#include <IncrementalIntegrator.h>
#include <ModalDamping.h>
#include <WoodburySOE.h>
#include <FE_Element.h>
#include <LinearSOE.h>
#include <AnalysisModel.h>
#include <Vector.h>
#include <DOF_Group.h>
#include <FE_EleIter.h>
#include <DOF_GrpIter.h>
#include <EigenSOE.h>
#include <Matrix.h>
#include <cmath>
#include <cstring>

IncrementalIntegrator::IncrementalIntegrator(int clasTag)
:Integrator(clasTag),
 statusFlag(CURRENT_TANGENT), theEigenSOE(0),
 isDiagonal(false), diagMass(0),
 theSOE(0), theAnalysisModel(0), theTest(0),
 theModalDamping(0)
{
  theModalDamping = new ModalDamping(*this);
}

IncrementalIntegrator::~IncrementalIntegrator()
{
  if (theModalDamping != 0)
    delete theModalDamping;
}

void
IncrementalIntegrator::setLinks(AnalysisModel &theModel, LinearSOE &theLinSOE, ConvergenceTest *theConvergenceTest)
{
    theAnalysisModel = &theModel;
    theSOE = &theLinSOE;
    theTest = theConvergenceTest;
}


void
IncrementalIntegrator::setEigenSOE(EigenSOE *theEigSOE) {
  theEigenSOE = theEigSOE;
}

int 
IncrementalIntegrator::formTangent(int statFlag)
{
    int result = 0;
    statusFlag = statFlag;

    if (theAnalysisModel == 0 || theSOE == 0) {
	opserr << "WARNING IncrementalIntegrator::formTangent() -";
	opserr << " no AnalysisModel or LinearSOE have been set\n";
	return -1;
    }

    // zero the A matrix of the linearSOE
    theSOE->zeroA();

    // the loops to form and add the tangents are broken into two for 
    // efficiency when performing parallel computations - CHANGE

    // loop through the FE_Elements adding their contributions to the tangent
    FE_Element *elePtr;
    FE_EleIter &theEles2 = theAnalysisModel->getFEs();    
    while((elePtr = theEles2()) != 0)     
	if (theSOE->addA(elePtr->getTangent(this),elePtr->getID()) < 0) {
	    opserr << "WARNING IncrementalIntegrator::formTangent -";
	    opserr << " failed in addA for ID " << elePtr->getID();	    
	    result = -3;
	}

    return result;
}

int 
IncrementalIntegrator::formTangent(int statFlag, double iFact, double cFact)
{
    initFactor = iFact;
    curFactor = cFact;
    return this->formTangent(statFlag);
}

int
IncrementalIntegrator::formIndependentSensitivityLHS(int statFlag)
{
    return this->formTangent(statFlag);
}

int 
IncrementalIntegrator::formUnbalance(void)
{
    if (theAnalysisModel == 0 || theSOE == 0) {
	opserr << "WARNING IncrementalIntegrator::formUnbalance -";
	opserr << " no AnalysisModel or LinearSOE has been set\n";
	return -1;
    }
    
    theSOE->zeroB();

    if (this->formElementResidual() < 0) {
	opserr << "WARNING IncrementalIntegrator::formUnbalance ";
	opserr << " - this->formElementResidual failed\n";
	return -1;
    }
    
    if (this->formNodalUnbalance() < 0) {
	opserr << "WARNING IncrementalIntegrator::formUnbalance ";
	opserr << " - this->formNodalUnbalance failed\n";
	return -2;
    }    

    return 0;
}
    
int
IncrementalIntegrator::getLastResponse(Vector &result, const ID &id)
{
  
    if (theSOE == 0) {
	opserr << "WARNING IncrementalIntegrator::getLastResponse() -";
	opserr << "no LineaerSOE object associated with this object\n";	
	return -1;
    }

    int res = 0; 
    int size = theSOE->getNumEqn() -1;
    const Vector &X = theSOE->getX();
    for (int i=0; i<id.Size(); i++) {
	int loc = id(i);
	if (loc < 0 )
	  result(i) = 0.0;
	else if (loc <= size) {
	  result(i) = X(loc);	
	}
	else {
	    opserr << "WARNING IncrementalIntegrator::getLastResponse() -";
	    opserr << "location " << loc << "in ID outside bounds ";
	    opserr << size << "\n";	
	    res = -2;
	}
    }	    
    return res;
}


int
IncrementalIntegrator::newStep(double deltaT)
{
  return 0;
}


int
IncrementalIntegrator::initialize(void)
{
  return 0;
}

int
IncrementalIntegrator::commit(void) 
{
    if (theAnalysisModel == 0) {
	opserr << "WARNING IncrementalIntegrator::commit() -";
	opserr << "no AnalysisModel object associated with this object\n";	
	return -1;
    }    

    return theAnalysisModel->commitDomain();
}   


int
IncrementalIntegrator::revertToLastStep(void) 
{
  return 0;
}   

int
IncrementalIntegrator::revertToStart()
{
  opserr << "ERROR: revertToStart() method not yet implemented " << endln
	 << " for the chosen type of integrator. " << endln;
  
  return 0;
}    

LinearSOE *
IncrementalIntegrator::getLinearSOE(void) const
{
    return theSOE;
}   

ConvergenceTest *
IncrementalIntegrator::getConvergenceTest(void) const
{
    return theTest;
}   

AnalysisModel *
IncrementalIntegrator::getAnalysisModel(void) const
{
    return theAnalysisModel;
}

int 
IncrementalIntegrator::formNodalUnbalance(void)
{
    // loop through the DOF_Groups and add the unbalance
    DOF_GrpIter &theDOFs = theAnalysisModel->getDOFs();
    DOF_Group *dofPtr;
    int res = 0;

    while ((dofPtr = theDOFs()) != 0) { 
      //      opserr << "NODPTR: " << dofPtr->getUnbalance(this);

	if (theSOE->addB(dofPtr->getUnbalance(this),dofPtr->getID()) <0) {
	    opserr << "WARNING IncrementalIntegrator::formNodalUnbalance -";
	    opserr << " failed in addB for ID " << dofPtr->getID();
	    res = -2;
	}
    }
	
    return res;
}

int 
IncrementalIntegrator::formElementResidual(void)
{
    // loop through the FE_Elements and add the residual
    FE_Element *elePtr;

    int res = 0;    

    FE_EleIter &theEles2 = theAnalysisModel->getFEs();    
    while((elePtr = theEles2()) != 0) {

	if (theSOE->addB(elePtr->getResidual(this),elePtr->getID()) <0) {
	    opserr << "WARNING IncrementalIntegrator::formElementResidual -";
	    opserr << " failed in addB for ID " << elePtr->getID();
	    res = -2;
	}
    }

    return res;	    
}

/*
int
IncrementalIntegrator::setModalDampingFactors(const Vector &factors)
{
  if (modalDampingValues != 0)
    delete modalDampingValues;

  modalDampingValues = new Vector(factors);
  if (modalDampingValues == 0 || modalDampingValues->Size() == 0) {
    opserr << "IncrementalIntegrator::setModalDampingFactors(const Vector &factors) - Vector of size 0, out of memory!";
    return -1;
  }
  return 0;
}
*/

 /*
int 
IncrementalIntegrator::addModalDampingForce(const Vector *modalDampingValues)
{
  int res = 0;
  
  if (modalDampingValues == 0)
    return 0;

  int numModes = modalDampingValues->Size();
  const Vector &eigenvalues = theAnalysisModel->getEigenvalues();
  
  if (eigenvalues.Size() < numModes) 
    numModes = eigenvalues.Size();

  Vector dampingForces(theSOE->getNumEqn());

  dampingForces.Zero();

  for (int i=0; i<numModes; i++) {

    DOF_GrpIter &theDOFs1 = theAnalysisModel->getDOFs();
    DOF_Group *dofPtr;
    double beta = 0.0;
    double eigenvalue = eigenvalues(i); // theEigenSOE->getEigenvalue(i+1);
    double wn = 0.;
    if (eigenvalue > 0)
      wn = sqrt(eigenvalue);

    while ((dofPtr = theDOFs1()) != 0) { 
      beta += dofPtr->getDampingBetaFactor(i, (*modalDampingValues)(i), wn);
    }

    DOF_GrpIter &theDOFs2 = theAnalysisModel->getDOFs();
    while ((dofPtr = theDOFs2()) != 0) { 
      if (theSOE->addB(dofPtr->getDampingBetaForce(i, beta),dofPtr->getID()) <0) {
	opserr << "WARNING IncrementalIntegrator::failed in dofPtr";
	res = -1;
      }    
    }
  }

  return res;
}
 */

/*int 
IncrementalIntegrator::addModalDampingForce(void)
{
  int res = 0;
  
  if (modalDampingValues == 0)
    return 0;

  int numModes = modalDampingValues->Size();

  const Vector &eigenvalues = theAnalysisModel->getEigenvalues();
  int numEigen = eigenvalues.Size();

  if (numEigen < numModes) 
    numModes = numEigen;

  int numDOF = theSOE->getNumEqn();

  if (eigenValues == 0 || *eigenValues != eigenvalues) {
    if (eigenValues != 0)
      delete eigenValues;
    if (eigenVectors != 0)
      delete [] eigenVectors;
    if (dampingForces != 0)
      delete dampingForces;
    if (mV != 0)
      delete mV;
    if (tmpV1 != 0)
      delete tmpV1;
    if (tmpV2 != 0)
      delete tmpV2;
    
    eigenValues = new Vector(eigenvalues);
    dampingForces = new Vector(numDOF);
    eigenVectors = new double[numDOF*numModes];
    mV = new Vector(numDOF);
    tmpV1 = new Vector(numDOF);
    tmpV2 = new Vector(numDOF);
    
    DOF_GrpIter &theDOFs2 = theAnalysisModel->getDOFs();
    DOF_Group *dofPtr;
    while ((dofPtr = theDOFs2()) != 0) { 
      const Matrix &dofEigenvectors =dofPtr->getEigenvectors();
      const ID &dofID = dofPtr->getID();
      for (int j=0; j<numModes; j++) {
	for (int i=0; i<dofID.Size(); i++) {
	  int id = dofID(i);
	  if (id >= 0) 
	    eigenVectors[j*numDOF + id] = dofEigenvectors(i,j);
	}
      }
    }
  }

  dampingForces->Zero();


  this->doMv(this->getVel(), *mV);

  for (int i=0; i<numModes; i++) {
    double eigenvalue = (*eigenValues)(i);
    if (eigenvalue > 0) {
      double wn = sqrt(eigenvalue);
      double *eigenVectorI = &eigenVectors[numDOF*i];
      double beta = 0.0;
	for (int j=0; j<numDOF; j++) {
	  double eij = eigenVectorI[j];
	  (*tmpV1)(j) = eij;
	  beta += eij * (*mV)(j);
	}
      beta = -2.0 * (*modalDampingValues)(i) * wn * beta;
      opserr << i << " " << beta << endln;
      *tmpV1 *= beta;
      this->doMv(*tmpV1, *tmpV2);      
      
      *dampingForces += *tmpV2;
      opserr << *dampingForces;
    }
  }
  theSOE->setB(*dampingForces);

  return res;
}
 */


int
IncrementalIntegrator::setupModal(const Vector *modalDampingValues)
{
  if (theModalDamping == 0)
    return -1;
  return theModalDamping->setupModal(modalDampingValues);
}

int
IncrementalIntegrator::addModalDampingForce(const Vector *modalDampingValues)
{
  if (theModalDamping == 0)
    return -1;
  return theModalDamping->addToUnbalance(modalDampingValues);
}

int
IncrementalIntegrator::addModalDampingMatrix(const Vector *modalDampingValues) {
  if (theModalDamping == 0)
    return -1;
  return theModalDamping->addToTangent(modalDampingValues, this->getCFactor());
}

ModalDamping *
IncrementalIntegrator::getModalDamping(void)
{
  return theModalDamping;
}

int
IncrementalIntegrator::addModalDampingWoodbury(const Vector *modalFactors)
{
  if (modalFactors == nullptr || theSOE == nullptr)
    return 0;

  WoodburySOE *wb = dynamic_cast<WoodburySOE *>(theSOE);
  if (wb == nullptr) {
    opserr << "WARNING IncrementalIntegrator::addModalDampingWoodbury() - "
              "modalDamping -woodbury requires WoodburySOE wrapper on the analysis\n";
    return -6;
  }

  if (theModalDamping == nullptr)
    return -1;

  Matrix Q;
  Vector diagD;
  const int built = theModalDamping->prepareWoodburyLowRank(
      modalFactors, this->getCFactor(), theSOE->getNumEqn(), Q, diagD);

  if (built == 0) {
    wb->clearWoodburyBasis();
    return 0;
  }
  if (built < 0)
    return built;

  if (wb->setWoodburySymmetric(Q, diagD) < 0)
    return -7;

  return wb->rebuildWoodburyBasis();
}


const Vector &
IncrementalIntegrator::getVel(void) {
  opserr << "IncrementalIntegrator::getVel() - not implemented for this integrator\n";
  return theSOE->getX();
}

int 
IncrementalIntegrator::doMv(const Vector &v, Vector &res) {

  int n = v.Size();
  if (isDiagonal == true) {
    for (int i=0; i<n; i++)
      res[i] = diagMass[i]*v[i];
    return 0;
  }

  res.Zero();

  // loop over the FE_Elements
  FE_Element *elePtr;
  FE_EleIter &theEles = theAnalysisModel->getFEs();    
  while((elePtr = theEles()) != 0) {
    const Vector &b = elePtr->getM_Force(v, 1.0);
    res.Assemble(b, elePtr->getID(), 1.0);
  }

  // loop over the DOF_Groups
  DOF_Group *dofPtr;
  DOF_GrpIter &theDofs = theAnalysisModel->getDOFs();
  while ((dofPtr = theDofs()) != 0) {
    const Vector &a = dofPtr->getM_Force(v, 1.0);      
    res.Assemble(a, dofPtr->getID(), 1.0);
  }
  return 0;
}

double IncrementalIntegrator::getCFactor(void)
{
  return 0;
}

