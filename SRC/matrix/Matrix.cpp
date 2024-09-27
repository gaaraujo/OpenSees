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
                                                                        
// $Revision: 1.16 $
// $Date: 2009/08/25 21:57:03 $
// $Source: /usr/local/cvs/OpenSees/SRC/matrix/Matrix.cpp,v $
                                                                        
                                                                        
// Written: fmk 
// Created: 11/96
// Revision: A
//
// Description: This file contains the class implementation for Matrix.
//
// What: "@(#) Matrix.h, revA"

#include "Matrix.h"
#include "Vector.h"
#include "ID.h"

#include <stdlib.h>
#include <iostream>
using std::nothrow;

#define MATRIX_WORK_AREA 400
#define INT_WORK_AREA 20

#include <math.h>

int Matrix::sizeDoubleWork = MATRIX_WORK_AREA;
int Matrix::sizeIntWork = INT_WORK_AREA;
double Matrix::MATRIX_NOT_VALID_ENTRY =0.0;
double *Matrix::matrixWork = 0;
int    *Matrix::intWork =0;

//double *Matrix::matrixWork = (double *)malloc(400*sizeof(double));

//
// CONSTRUCTORS
//

Matrix::Matrix()
:numRows(0), numCols(0), dataSize(0), data(0), fromFree(0), 
cachedLUFactor(0), cachedPivot(0), isLUFactorized(false), isLUCacheEnabled(false)
{
  // allocate work areas if the first
  if (matrixWork == 0) {
    matrixWork = new (nothrow) double[sizeDoubleWork];
    intWork = new (nothrow) int[sizeIntWork];
    if (matrixWork == 0 || intWork == 0) {
      opserr << "WARNING: Matrix::Matrix() - out of memory creating work area's\n";
      exit(-1);
    }
  }
}


Matrix::Matrix(int nRows,int nCols)
:numRows(nRows), numCols(nCols), dataSize(0), data(0), fromFree(0), 
cachedLUFactor(0), cachedPivot(0), isLUFactorized(false), isLUCacheEnabled(false)
{

  // allocate work areas if the first matrix
  if (matrixWork == 0) {
    matrixWork = new (nothrow) double[sizeDoubleWork];
    intWork = new (nothrow) int[sizeIntWork];
    if (matrixWork == 0 || intWork == 0) {
      opserr << "WARNING: Matrix::Matrix() - out of memory creating work area's\n";
      exit(-1);
    }
  }

#ifdef _G3DEBUG
    if (nRows < 0) {
      opserr << "WARNING: Matrix::Matrix(int,int): tried to init matrix ";
      opserr << "with num rows: " << nRows << " <0\n";
      numRows = 0; numCols =0; dataSize =0; data = 0;
    }
    if (nCols < 0) {
      opserr << "WARNING: Matrix::Matrix(int,int): tried to init matrix";
      opserr << "with num cols: " << nCols << " <0\n";
      numRows = 0; numCols =0; dataSize =0; data = 0;
    }
#endif
    dataSize = numRows * numCols;
    data = 0;

    if (dataSize > 0) {
      data = new (nothrow) double[dataSize];
      //data = (double *)malloc(dataSize*sizeof(double));
      if (data == 0) {
	opserr << "WARNING:Matrix::Matrix(int,int): Ran out of memory on init ";
	opserr << "of size " << dataSize << endln;
	numRows = 0; numCols =0; dataSize =0;
      } else {
	// zero the data
	double *dataPtr = data;
	for (int i=0; i<dataSize; i++)
	  *dataPtr++ = 0.0;
      }
    }
}

Matrix::Matrix(double *theData, int row, int col) 
:numRows(row),numCols(col),dataSize(row*col),data(theData),fromFree(1), 
cachedLUFactor(0), cachedPivot(0), isLUFactorized(false), isLUCacheEnabled(false)
{
  // allocate work areas if the first matrix
  if (matrixWork == 0) {
    matrixWork = new (nothrow) double[sizeDoubleWork];
    intWork = new (nothrow) int[sizeIntWork];
    if (matrixWork == 0 || intWork == 0) {
      opserr << "WARNING: Matrix::Matrix() - out of memory creating work area's\n";
      exit(-1);
    }
  }

#ifdef _G3DEBUG
    if (row < 0) {
      opserr << "WARNING: Matrix::Matrix(int,int): tried to init matrix with numRows: ";
      opserr << row << " <0\n";
      numRows = 0; numCols =0; dataSize =0; data = 0;
    }
    if (col < 0) {
      opserr << "WARNING: Matrix::Matrix(int,int): tried to init matrix with numCols: ";
      opserr << col << " <0\n";
      numRows = 0; numCols =0; dataSize =0; data = 0;
    }    
#endif

    // does nothing
}

Matrix::Matrix(const Matrix &other)
:numRows(0), numCols(0), dataSize(0), data(0), fromFree(0), 
cachedLUFactor(0), cachedPivot(0), isLUFactorized(false), isLUCacheEnabled(false)
{
  // allocate work areas if the first matrix
  if (matrixWork == 0) {
    matrixWork = new (nothrow) double[sizeDoubleWork];
    intWork = new (nothrow) int[sizeIntWork];
    if (matrixWork == 0 || intWork == 0) {
      opserr << "WARNING: Matrix::Matrix() - out of memory creating work area's\n";
      exit(-1);
    }
  }

    numRows = other.numRows;
    numCols = other.numCols;
    dataSize = other.dataSize;

    if (dataSize != 0) {
      data = new (nothrow) double[dataSize];
      // data = (double *)malloc(dataSize*sizeof(double));
      if (data == 0) {
	opserr << "WARNING:Matrix::Matrix(Matrix &): ";
	opserr << "Ran out of memory on init of size " << dataSize << endln; 
	numRows = 0; numCols =0; dataSize = 0;
      } else {
	// copy the data
	double *dataPtr = data;
	double *otherDataPtr = other.data;
	for (int i=0; i<dataSize; i++)
	  *dataPtr++ = *otherDataPtr++;
      }
    }
  // Note: this constructor does not copy the LU cache. 
  // Such functionality is provided in Matrix::copyWithCache()
}

// Move ctor
#ifdef USE_CXX11
Matrix::Matrix(Matrix &&other)
:numRows(other.numRows), numCols(other.numCols), dataSize(other.dataSize), data(other.data), fromFree(other.fromFree), 
cachedLUFactor(other.cachedLUFactor), cachedPivot(other.cachedPivot), isLUFactorized(other.isLUFactorized), 
isLUCacheEnabled(other.isLUCacheEnabled)
{
  other.numRows = 0;
  other.numCols = 0;
  other.dataSize = 0;
  other.data = 0;
  other.fromFree = 1;
  other.cachedLUFactor = 0;
  other.cachedPivot = 0;
  other.isLUFactorized = false;
  other.isLUCacheEnabled = false;
}
#endif

//
// DESTRUCTOR
//

Matrix::~Matrix()
{
  if (data != 0 ) {
    if (fromFree == 0 && dataSize > 0){
      delete [] data; 
      data = 0;
    }
  }
  //  if (data != 0) free((void *) data);
  
  // clear LU Factors
  clearLUCache();
}
    

//
// METHODS - Zero, Assemble, Solve
//

int
Matrix::setData(double *theData, int row, int col) 
{
  // delete the old if allocated
  if (data != 0) 
    if (fromFree == 0)
    {
      delete [] data; 
      data = 0;
    }
  
  if (dataSize < row * col || numRows < row){
    // clear LU cache but keep caching enabled
    clearLUCache();
  } else {
    // reset factorization flag, but keep cache data structures
    isLUFactorized = false;
  }

  numRows = row;
  numCols = col;
  dataSize = row*col;
  data = theData;
  fromFree = 1;

#ifdef _G3DEBUG
  if (row < 0) {
    opserr << "WARNING: Matrix::setSize(): tried to init matrix with numRows: ";
    opserr << row << " <0\n";
    numRows = 0; numCols =0; dataSize =0; data = 0;
    return -1;
  }
  if (col < 0) {
    opserr << "WARNING: Matrix::setSize(): tried to init matrix with numCols: ";
    opserr << col << " <0\n";
    numRows = 0; numCols =0; dataSize =0; data = 0;
    return -1;
  }    
#endif
  
  return 0;
}

void
Matrix::Zero(void)
{
  double *dataPtr = data;
  for (int i=0; i<dataSize; i++)
    *dataPtr++ = 0;

  // reset LU factorization flag
  isLUFactorized = false;
}


int
Matrix::resize(int rows, int cols) {

  int newSize = rows*cols;

  if (newSize < 0) {
    opserr << "Matrix::resize) - rows " << rows << " or cols " << cols << " specified <= 0\n";
    return -1;
  }

  else if (newSize > dataSize) {

    // free the old space
    if (data != 0) 
      if (fromFree == 0){
	delete [] data; 
        data = 0;
      }
    //  if (data != 0) free((void *) data);
    
    // clear LU cache but keep caching enabled
    clearLUCache();
    
    fromFree = 0;
    // create new space
    data = new (nothrow) double[newSize];
    // data = (double *)malloc(dataSize*sizeof(double));
    if (data == 0) {
      opserr << "Matrix::resize(" << rows << "," << cols << ") - out of memory\n";
      numRows = 0; numCols =0; dataSize = 0;
      return -2;
    }
    dataSize = newSize;
  }
  
  // check if pivot array size needs to be adjusted
  else if (rows > numRows) {
    clearLUCache();
  } else {
    isLUFactorized = false;
  }

  // just reset the cols and rows - save two memory calls at expense of holding 
  // onto extra memory when dataSize > newSize
  
  numRows = rows;
  numCols = cols;

  return 0;
}





int
Matrix::Assemble(const Matrix &V, const ID &rows, const ID &cols, double fact) 
{
  int pos_Rows, pos_Cols;
  int res = 0;

  for (int i=0; i<cols.Size(); i++) {
    pos_Cols = cols(i);
    for (int j=0; j<rows.Size(); j++) {
      pos_Rows = rows(j);
      
      if ((pos_Cols >= 0) && (pos_Rows >= 0) && (pos_Rows < numRows) &&
	  (pos_Cols < numCols) && (i < V.numCols) && (j < V.numRows))
	(*this)(pos_Rows,pos_Cols) += V(j,i)*fact;
      else {
	opserr << "WARNING: Matrix::Assemble(const Matrix &V, const ID &l): ";
	opserr << " - position (" << pos_Rows << "," << pos_Cols << ") outside bounds \n";
	res = -1;
      }
    }
  }
  // set LU factorization flag 
  isLUFactorized = false;
  return res;
}

#ifdef _WIN32

extern "C" int  DGESV(int *N, int *NRHS, double *A, int *LDA, 
			      int *iPiv, double *B, int *LDB, int *INFO);

extern "C" int  DGETRF(int *M, int *N, double *A, int *LDA, 
			      int *iPiv, int *INFO);

extern "C" int  DGETRS(char *TRANS,
			       int *N, int *NRHS, double *A, int *LDA, 
			       int *iPiv, double *B, int *LDB, int *INFO);

extern "C" int  DGETRI(int *N, double *A, int *LDA, 
			      int *iPiv, double *Work, int *WORKL, int *INFO);
//#endif
#else
extern "C" int dgesv_(int *N, int *NRHS, double *A, int *LDA, int *iPiv, 
		      double *B, int *LDB, int *INFO);

extern "C" int dgetrs_(char *TRANS, int *N, int *NRHS, double *A, int *LDA, 
		       int *iPiv, double *B, int *LDB, int *INFO);		       

extern "C" int dgetrf_(int *M, int *N, double *A, int *LDA, 
		       int *iPiv, int *INFO);

extern "C" int dgetri_(int *N, double *A, int *LDA, 
		       int *iPiv, double *Work, int *WORKL, int *INFO);
extern "C" int dgerfs_(char *TRANS, int *N, int *NRHS, double *A, int *LDA, 
		       double *AF, int *LDAF, int *iPiv, double *B, int *LDB, 
		       double *X, int *LDX, double *FERR, double *BERR, 
		       double *WORK, int *IWORK, int *INFO);

#endif

int
Matrix::Solve(const Vector &b, Vector &x) const
{

    int n = numRows;

#ifdef _G3DEBUG    
    if (numRows != numCols) {
      opserr << "Matrix::Solve(b,x) - the matrix of dimensions " 
	     << numRows << ", " << numCols << " is not square " << endln;
      return -1;
    }

    if (n != x.Size()) {
      opserr << "Matrix::Solve(b,x) - dimension of x, " << numRows << "is not same as matrix " <<  x.Size() << endln;
      return -2;
    }

    if (n != b.Size()) {
      opserr << "Matrix::Solve(b,x) - dimension of x, " << numRows << "is not same as matrix " <<  b.Size() << endln;
      return -2;
    }
#endif
    
    // Set x equal to b
    x = b;

    int nrhs = 1;
    int ldA = n;
    int ldB = n;
    int info;
    double *Xptr = x.theData;

    // Use LU cache if available and factorize if needed
    if (isLUCacheEnabled) {
      info = computeLU();
      if (info != 0) {
        opserr << "WARNING: Matrix::Solve() - LU factorization failed\n";
        return info;
      }
    }

    if (isLUFactorized && cachedLUFactor != 0 && cachedPivot != 0) {
      // Use cached LU factors and pivot information
      double *Aptr = cachedLUFactor;
      int *iPIV = cachedPivot;
      char trans = 'N';

#ifdef _WIN32
      DGETRS(&trans, &n, &nrhs, Aptr, &ldA, iPIV, Xptr, &ldB, &info);
#else
      dgetrs_(&trans, &n, &nrhs, Aptr, &ldA, iPIV, Xptr, &ldB, &info);
#endif
      return -abs(info);
    }
    

    // If not cached, perform LU factorization and solve
    // check work area can hold all the data
    if (dataSize > sizeDoubleWork) {

      if (matrixWork != 0) {
	delete [] matrixWork;
        matrixWork = 0;
      }
      matrixWork = new (nothrow) double[dataSize];
      sizeDoubleWork = dataSize;
      
      if (matrixWork == 0) {
	opserr << "WARNING: Matrix::Solve() - out of memory creating work area's\n";
	sizeDoubleWork = 0;      
	return -3;
      }
    }

    // check work area can hold all the data
    if (n > sizeIntWork) {

      if (intWork != 0) {
	delete [] intWork;
        intWork = 0;
      }
      intWork = new (nothrow) int[n];
      sizeIntWork = n;
      
      if (intWork == 0) {
	opserr << "WARNING: Matrix::Solve() - out of memory creating work area's\n";
	sizeIntWork = 0;      
	return -3;
      }
    }

    
    // copy the data
    int i;
    for (i=0; i<dataSize; i++)
      matrixWork[i] = data[i];

    double *Aptr = matrixWork;
    int *iPIV = intWork;

#ifdef _WIN32

    DGESV(&n,&nrhs,Aptr,&ldA,iPIV,Xptr,&ldB,&info);

#else
    dgesv_(&n,&nrhs,Aptr,&ldA,iPIV,Xptr,&ldB,&info);
#endif

    return -abs(info);
}


int
Matrix::Solve(const Matrix &b, Matrix &x) const
{

    int n = numRows;
    int nrhs = x.numCols;

#ifdef _G3DEBUG    
    if (numRows != numCols) {
      opserr << "Matrix::Solve(B,X) - the matrix of dimensions [" << numRows << " " <<  numCols << "] is not square\n";
      return -1;
    }

    if (n != x.numRows) {
      opserr << "Matrix::Solve(B,X) - #rows of X, " << x.numRows << " is not same as the matrices: " << numRows << endln;
      return -2;
    }

    if (n != b.numRows) {
      opserr << "Matrix::Solve(B,X) - #rows of B, " << b.numRows << " is not same as the matrices: " << numRows << endln;
      return -2;
    }

    if (x.numCols != b.numCols) {
      opserr << "Matrix::Solve(B,X) - #cols of B, " << b.numCols << " , is not same as that of X, b " <<  x.numCols << endln;
      return -3;
    }
#endif

// Set x equal to b
    x = b;

    int ldA = n;
    int ldB = n;
    int info;
    double *Xptr = x.data;

    // Use LU cache if available and factorize if needed
    if (isLUCacheEnabled) {
      info = computeLU();
      if (info != 0) {
        opserr << "WARNING: Matrix::Solve() - LU factorization failed\n";
        return info;
      }
    }

    if (isLUFactorized && cachedLUFactor != 0 && cachedPivot != 0) {
      // Use cached LU factors and pivot information
      double *Aptr = cachedLUFactor;
      int *iPIV = cachedPivot;
      char trans = 'N';

#ifdef _WIN32
      DGETRS(&trans, &n, &nrhs, Aptr, &ldA, iPIV, Xptr, &ldB, &info);
#else
      dgetrs_(&trans, &n, &nrhs, Aptr, &ldA, iPIV, Xptr, &ldB, &info);
#endif
      return -abs(info);
    }

    // If not cached, perform LU factorization and solve
    // check work area can hold all the data
    if (dataSize > sizeDoubleWork) {

      if (matrixWork != 0) {
	delete [] matrixWork;
        matrixWork = 0;
      }
      matrixWork = new (nothrow) double[dataSize];
      sizeDoubleWork = dataSize;
      
      if (matrixWork == 0) {
	opserr << "WARNING: Matrix::Solve() - out of memory creating work area's\n";
	sizeDoubleWork = 0;      
	return -3;
      }
    }

    // check work area can hold all the data
    if (n > sizeIntWork) {

      if (intWork != 0) {
	delete [] intWork;
        intWork = 0;
      }
      intWork = new (nothrow) int[n];
      sizeIntWork = n;
      
      if (intWork == 0) {
	opserr << "WARNING: Matrix::Solve() - out of memory creating work area's\n";
	sizeIntWork = 0;      
	return -3;
      }
    }
    
    // copy the data
    int i;
    for (i=0; i<dataSize; i++)
      matrixWork[i] = data[i];

    double *Aptr = matrixWork;    
    int *iPIV = intWork;
    
	info = -1;

#ifdef _WIN32

    DGESV(&n,&nrhs,Aptr,&ldA,iPIV,Xptr,&ldB,&info);

#else
    dgesv_(&n,&nrhs,Aptr,&ldA,iPIV,Xptr,&ldB,&info);

    /*
    // further correction if required
    double Bptr[n*n];
    for (int i=0; i<n*n; i++) Bptr[i] = b.data[i];
    double *origData = data;
    double Ferr[n];
    double Berr[n];
    double newWork[3*n];
    int newIwork[n];
    
    dgerfs_("N",&n,&n,origData,&ldA,Aptr,&n,iPIV,Bptr,&ldB,Xptr,&ldB,
	    Ferr, Berr, newWork, newIwork, &info);
    */
#endif
    return -abs(info);
}


int
Matrix::Invert(Matrix &theInverse) const
{

    int n = numRows;


#ifdef _G3DEBUG    

    if (numRows != numCols) {
      opserr << "Matrix::Solve(B,X) - the matrix of dimensions [" << numRows << "," << numCols << "] is not square\n";
      return -1;
    }

    if (n != theInverse.numRows) {
      opserr << "Matrix::Solve(B,X) - #rows of X, " << numRows<< ", is not same as matrix " << theInverse.numRows << endln;
      return -2;
    }
#endif

    // check work area can hold all the data
    if (dataSize > sizeDoubleWork) {

      if (matrixWork != 0) {
	delete [] matrixWork;
        matrixWork = 0;
      }
      matrixWork = new (nothrow) double[dataSize];
      sizeDoubleWork = dataSize;
      
      if (matrixWork == 0) {
	opserr << "WARNING: Matrix::Solve() - out of memory creating work area's\n";
	sizeDoubleWork = 0;      
	return -3;
      }
    }

    // check work area can hold all the data
    if (n > sizeIntWork) {

      if (intWork != 0) {
	delete [] intWork;
        intWork = 0;
      }
      intWork = new (nothrow) int[n];
      sizeIntWork = n;
      
      if (intWork == 0) {
	opserr << "WARNING: Matrix::Solve() - out of memory creating work area's\n";
	sizeIntWork = 0;      
	return -3;
      }
    }
    
    // Use LU cache if enabled and already factorized
    if (isLUCacheEnabled) {
      // LU cache is enabled but not factorized, so compute LU factorization
      int info = computeLU();
      if (info != 0) {
          opserr << "WARNING: Matrix::Invert() - LU factorization failed\n";
          return info;
      }
    }

    // If cached LU factors are available, copy them into theInverse
    if (isLUFactorized && cachedLUFactor != 0 && cachedPivot != 0) {
      // Copy cached LU factor into theInverse.data
      for (int i = 0; i < dataSize; i++) {
        theInverse.data[i] = cachedLUFactor[i];
      }

      double *Aptr = theInverse.data;
      int *iPIV = cachedPivot;
      int workSize = sizeDoubleWork;
      double *Wptr = matrixWork;
      int info;

#ifdef _WIN32
      DGETRI(&n, Aptr, &n, iPIV, Wptr, &workSize, &info);
#else
      dgetri_(&n, Aptr, &n, iPIV, Wptr, &workSize, &info);
#endif
      return -abs(info);
    }

    // If no cached LU factorization exists, proceed with regular inversion
    // copy the data
    theInverse = *this;
    
    // for (int i=0; i<dataSize; i++)
    //   matrixWork[i] = data[i];

    int ldA = n;
    int info;
    double *Wptr = matrixWork;
    double *Aptr = theInverse.data;
    int workSize = sizeDoubleWork;
    
    int *iPIV = intWork;
    

#ifdef _WIN32

    DGETRF(&n,&n,Aptr,&ldA,iPIV,&info);

    if (info != 0) 
      return -abs(info);

    DGETRI(&n,Aptr,&ldA,iPIV,Wptr,&workSize,&info);

#else
    dgetrf_(&n,&n,Aptr,&ldA,iPIV,&info);
    if (info != 0) 
      return -abs(info);
    
    dgetri_(&n,Aptr,&ldA,iPIV,Wptr,&workSize,&info);
    
#endif

    return -abs(info);
}

void 
Matrix::activateLUCache() const
{
  isLUCacheEnabled = true;
}

void 
Matrix::deactivateLUCache() const
{
  clearLUCache();
  isLUCacheEnabled = false;
}

Matrix 
Matrix::copyWithCache() const {
  // Use the default copy constructor to copy the matrix data
  Matrix copy(*this);

  // set LU factorization and caching flag
  copy.isLUFactorized = isLUFactorized;
  copy.isLUCacheEnabled = isLUCacheEnabled;

  if (isLUFactorized && cachedLUFactor != 0 && cachedPivot != 0 && 
      dataSize > 0 && numRows > 0) {
    // allocate space for the cached LU factors and pivot
    copy.cachedLUFactor = new (nothrow) double[dataSize];
    copy.cachedPivot = new (nothrow) int[numRows];

    if (copy.cachedLUFactor == 0 || copy.cachedPivot == 0) {
      opserr << "WARNING: Matrix::copyWithCache(): ";
      opserr << "out of memory copying LU factorization data" << endln;
      copy.clearLUCache();
    } else {
      // copy LU factorization and pivot data
      for (int i = 0; i < dataSize; i++) {
        copy.cachedLUFactor[i] = cachedLUFactor[i];
      }
      for (int i = 0; i < numRows; i++) {
        copy.cachedPivot[i] = cachedPivot[i];
      }
    }
  }
  return copy; // RVO avoids copy, or the move constructor will be used
}

int 
Matrix::computeLU() const
{ 
  // check if LU has already been computed
  if (isLUFactorized) {
    return 0;
  }

  // check that LU caching is enabled
  if (!isLUCacheEnabled) {
    opserr << "WARNING: Matrix::computeLU(): ";
    opserr << "LU Caching is not activated. ";
    opserr << "Matrix::activateLUCache() will be called automatically." << endln;
    activateLUCache();
  }

  // allocate space for cache if need
  if (cachedLUFactor == 0 && dataSize > 0) {
    cachedLUFactor = new (nothrow) double[dataSize];
  }
  if (cachedPivot == 0 && numRows > 0) {
    cachedPivot = new (nothrow) int[numRows];
  }
  if (cachedLUFactor == 0 || cachedPivot == 0) {
    opserr << "WARNING:Matrix::computeLU(): ";
    opserr << "Unable to allocate memory for LU factorization" << endln;
    clearLUCache();
    return -1;
  }
  
  // perform LU factorization and cache the result
  for (int i = 0; i < dataSize; i++) {
    cachedLUFactor[i] = data[i];
  }

  int M = numRows;
  int N = numCols;
  int lda = M;
  int info;

#ifdef _WIN32
  DGETRF(&M, &N, cachedLUFactor, &lda, cachedPivot, &info);
#else
  dgetrf_(&M, &N, cachedLUFactor, &lda, cachedPivot, &info);
#endif

  // check if LU factorization succeded and set LU factorization flag
  if (info != 0) {
    clearLUCache();
  } else {
    isLUFactorized = true;
  }

  return -abs(info);
}

void
Matrix::clearLUCache() const 
{
  // early return when matrix is not factorized
  if (!isLUCacheEnabled) {
    return;
  }
  // clear LU factors
  if (cachedLUFactor != 0) {
    delete [] cachedLUFactor;
    cachedLUFactor = 0;
  }
  if (cachedPivot != 0) {
    delete [] cachedPivot;
    cachedPivot = 0;
  }
  // set LU factorization flag
  isLUFactorized = false;
}

int
Matrix::addMatrix(double factThis, const Matrix &other, double factOther)
{
    if (factThis == 1.0 && factOther == 0.0)
      return 0;

#ifdef _G3DEBUG
    if ((other.numRows != numRows) || (other.numCols != numCols)) {
      opserr << "Matrix::addMatrix(): incompatible matrices\n";
      return -1;
    }
#endif

    if (factThis == 1.0) {

      // want: this += other * factOther
      if (factOther == 1.0) {
	double *dataPtr = data;
	double *otherDataPtr = other.data;		    
	for (int i=0; i<dataSize; i++)
	  *dataPtr++ += *otherDataPtr++;
      } else {
	double *dataPtr = data;
	double *otherDataPtr = other.data;		    
	for (int i=0; i<dataSize; i++)
	  *dataPtr++ += *otherDataPtr++ * factOther;
      }
    } 

    else if (factThis == 0.0) {

      // want: this = other * factOther
      if (factOther == 1.0) {
	double *dataPtr = data;
	double *otherDataPtr = other.data;		    
	for (int i=0; i<dataSize; i++)
	  *dataPtr++ = *otherDataPtr++;
      } else {
	double *dataPtr = data;
	double *otherDataPtr = other.data;		    
	for (int i=0; i<dataSize; i++)
	  *dataPtr++ = *otherDataPtr++ * factOther;
      }
    } 

    else {

      // want: this = this * thisFact + other * factOther
      if (factOther == 1.0) {
	double *dataPtr = data;
	double *otherDataPtr = other.data;		    
	for (int i=0; i<dataSize; i++) {
	  double value = *dataPtr * factThis + *otherDataPtr++;
	  *dataPtr++ = value;
	}
      } else {
	double *dataPtr = data;
	double *otherDataPtr = other.data;		    
	for (int i=0; i<dataSize; i++) {
	  double value = *dataPtr * factThis + *otherDataPtr++ * factOther;
	  *dataPtr++ = value;
	}
      }
    } 

    // successful
    return 0;
}


int
Matrix::addMatrixTranspose(double factThis, const Matrix &other, double factOther)
{
    if (factThis == 1.0 && factOther == 0.0)
      return 0;

#ifdef _G3DEBUG
    if ((other.numRows != numCols) || (other.numCols != numRows)) {
      opserr << "Matrix::addMatrixTranspose(): incompatible matrices\n";
      return -1;
    }
#endif

    if (factThis == 1.0) {

      // want: this += other^T * factOther
      if (factOther == 1.0) {
    double *dataPtr = data;
    for (int j=0; j<numCols; j++) {
      for (int i=0; i<numRows; i++)
	    *dataPtr++ += (other.data)[j+i*numCols];
    }
      } else {
	double *dataPtr = data;
    for (int j=0; j<numCols; j++) {
      for (int i=0; i<numRows; i++)
	    *dataPtr++ += (other.data)[j+i*numCols] * factOther;
    }
      }
    } 

    else if (factThis == 0.0) {

      // want: this = other^T * factOther
      if (factOther == 1.0) {
	double *dataPtr = data;
    for (int j=0; j<numCols; j++) {
      for (int i=0; i<numRows; i++)
	    *dataPtr++ = (other.data)[j+i*numCols];
    }
      } else {
	double *dataPtr = data;
    for (int j=0; j<numCols; j++) {
      for (int i=0; i<numRows; i++)
	    *dataPtr++ = (other.data)[j+i*numCols] * factOther;
    }
      }
    } 

    else {

      // want: this = this * thisFact + other^T * factOther
      if (factOther == 1.0) {
	double *dataPtr = data;
    for (int j=0; j<numCols; j++) {
      for (int i=0; i<numRows; i++) {
        double value = *dataPtr * factThis + (other.data)[j+i*numCols];
	    *dataPtr++ = value;
      }
    }
      } else {
	double *dataPtr = data;
    for (int j=0; j<numCols; j++) {
      for (int i=0; i<numRows; i++) {
	    double value = *dataPtr * factThis + (other.data)[j+i*numCols] * factOther;
	    *dataPtr++ = value;
      }
    }
      }
    } 

    // successful
    return 0;
}


int
Matrix::addMatrixProduct(double thisFact, 
			 const Matrix &B, 
			 const Matrix &C, 
			 double otherFact)
{
    if (thisFact == 1.0 && otherFact == 0.0)
      return 0;
#ifdef _G3DEBUG
    if ((B.numRows != numRows) || (C.numCols != numCols) || (B.numCols != C.numRows)) {
      opserr << "Matrix::addMatrixProduct(): incompatible matrices, this\n";
      return -1;
    }
#endif
    // NOTE: looping as per blas3 dgemm_: j,k,i
    if (thisFact == 1.0) {

      // want: this += B * C  otherFact
      int numColB = B.numCols;
      double *ckjPtr  = &(C.data)[0];
      for (int j=0; j<numCols; j++) {
	double *aijPtrA = &data[j*numRows];
	for (int k=0; k<numColB; k++) {
	  double tmp = *ckjPtr++ * otherFact;
	  double *aijPtr = aijPtrA;
	  double *bikPtr = &(B.data)[k*numRows];
	  for (int i=0; i<numRows; i++)
	    *aijPtr++ += *bikPtr++ * tmp;
	}
      }
    }

    else if (thisFact == 0.0) {

      // want: this = B * C  otherFact
      double *dataPtr = data;
      for (int i=0; i<dataSize; i++)
	  *dataPtr++ = 0.0;
      int numColB = B.numCols;
      double *ckjPtr  = &(C.data)[0];
      for (int j=0; j<numCols; j++) {
	double *aijPtrA = &data[j*numRows];
	for (int k=0; k<numColB; k++) {
	  double tmp = *ckjPtr++ * otherFact;
	  double *aijPtr = aijPtrA;
	  double *bikPtr = &(B.data)[k*numRows];
	  for (int i=0; i<numRows; i++)
	    *aijPtr++ += *bikPtr++ * tmp;
	}
      }
    } 

    else {
      // want: this = B * C  otherFact
      double *dataPtr = data;
      for (int i=0; i<dataSize; i++)
	  *dataPtr++ *= thisFact;
      int numColB = B.numCols;
      double *ckjPtr  = &(C.data)[0];
      for (int j=0; j<numCols; j++) {
	double *aijPtrA = &data[j*numRows];
	for (int k=0; k<numColB; k++) {
	  double tmp = *ckjPtr++ * otherFact;
	  double *aijPtr = aijPtrA;
	  double *bikPtr = &(B.data)[k*numRows];
	  for (int i=0; i<numRows; i++)
	    *aijPtr++ += *bikPtr++ * tmp;
	}
      }
    } 

    return 0;
}

int
Matrix::addMatrixTransposeProduct(double thisFact, 
				  const Matrix &B, 
				  const Matrix &C, 
				  double otherFact)
{
  if (thisFact == 1.0 && otherFact == 0.0)
    return 0;

#ifdef _G3DEBUG
  if ((B.numCols != numRows) || (C.numCols != numCols) || (B.numRows != C.numRows)) {
    opserr << "Matrix::addMatrixProduct(): incompatible matrices, this\n";
    return -1;
  }
#endif

  if (thisFact == 1.0) {
    int numMults = C.numRows;
    double *aijPtr = data;
    for (int j=0; j<numCols; j++) {
      for (int i=0; i<numRows; i++) {
	double *bkiPtr  = &(B.data)[i*numMults];
	double *cjkPtr  = &(C.data)[j*numMults];
	double sum = 0.0;
	for (int k=0; k<numMults; k++) {
	  sum += *bkiPtr++ * *cjkPtr++;
	}
	*aijPtr++ += sum * otherFact;
      }
    } 
  } else if (thisFact == 0.0) {
    int numMults = C.numRows;
    double *aijPtr = data;
    for (int j=0; j<numCols; j++) {
      for (int i=0; i<numRows; i++) {
	double *bkiPtr  = &(B.data)[i*numMults];
	double *cjkPtr  = &(C.data)[j*numMults];
	double sum = 0.0;
	for (int k=0; k<numMults; k++) {
	  sum += *bkiPtr++ * *cjkPtr++;
	}
	*aijPtr++ = sum * otherFact;
      }
    } 
  } else {
    int numMults = C.numRows;
    double *aijPtr = data;
    for (int j=0; j<numCols; j++) {
      for (int i=0; i<numRows; i++) {
	double *bkiPtr  = &(B.data)[i*numMults];
	double *cjkPtr  = &(C.data)[j*numMults];
	double sum = 0.0;
	for (int k=0; k<numMults; k++) {
	  sum += *bkiPtr++ * *cjkPtr++;
	}
	*aijPtr = *aijPtr * thisFact + sum * otherFact;
	aijPtr++;
      }
    } 
  }

  return 0;
}


// to perform this += T' * B * T
int
Matrix::addMatrixTripleProduct(double thisFact, 
			       const Matrix &T, 
			       const Matrix &B, 
			       double otherFact)
{
    if (thisFact == 1.0 && otherFact == 0.0)
      return 0;
#ifdef _G3DEBUG
    if ((numCols != numRows) || (B.numCols != B.numRows) || (T.numCols != numRows) ||
	(T.numRows != B.numCols)) {
      opserr << "Matrix::addMatrixTripleProduct() - incompatible matrices\n";
      return -1;
    }
#endif

    // cheack work area can hold the temporary matrix
    int dimB = B.numCols;
    int sizeWork = dimB * numCols;

    if (sizeWork > sizeDoubleWork) {
      this->addMatrix(thisFact, T^B*T, otherFact);
      return 0;
    }

    // zero out the work area
    double *matrixWorkPtr = matrixWork;
    for (int l=0; l<sizeWork; l++)
      *matrixWorkPtr++ = 0.0;
    
    // now form B * T * fact store in matrixWork == A area
    // NOTE: looping as per blas3 dgemm_: j,k,i

    double *tkjPtr  = &(T.data)[0];
    for (int j=0; j<numCols; j++) {
      double *aijPtrA = &matrixWork[j*dimB];
      for (int k=0; k<dimB; k++) {
	double tmp = *tkjPtr++ * otherFact;
	double *aijPtr = aijPtrA;
	double *bikPtr = &(B.data)[k*dimB];
	for (int i=0; i<dimB; i++) 
	  *aijPtr++ += *bikPtr++ * tmp;
      }
    }

    // now form T' * matrixWork
    // NOTE: looping as per blas3 dgemm_: j,i,k
    if (thisFact == 1.0) {
      double *dataPtr = &data[0];
      for (int j=0; j< numCols; j++) {
	double *workkjPtrA = &matrixWork[j*dimB];
	for (int i=0; i<numRows; i++) {
	  double *ckiPtr = &(T.data)[i*dimB];
	  double *workkjPtr = workkjPtrA;
	  double aij = 0.0;
	  for (int k=0; k< dimB; k++)
	    aij += *ckiPtr++ * *workkjPtr++;
	  *dataPtr++ += aij;
	}
      }
    } else if (thisFact == 0.0) {
      double *dataPtr = &data[0];
      for (int j=0; j< numCols; j++) {
	double *workkjPtrA = &matrixWork[j*dimB];
	for (int i=0; i<numRows; i++) {
	  double *ckiPtr = &(T.data)[i*dimB];
	  double *workkjPtr = workkjPtrA;
	  double aij = 0.0;
	  for (int k=0; k< dimB; k++)
	    aij += *ckiPtr++ * *workkjPtr++;
	  *dataPtr++ = aij;
	}
      }

    } else {
      double *dataPtr = &data[0];
      for (int j=0; j< numCols; j++) {
	double *workkjPtrA = &matrixWork[j*dimB];
	for (int i=0; i<numRows; i++) {
	  double *ckiPtr = &(T.data)[i*dimB];
	  double *workkjPtr = workkjPtrA;
	  double aij = 0.0;
	  for (int k=0; k< dimB; k++)
	    aij += *ckiPtr++ * *workkjPtr++;
	  double value = *dataPtr * thisFact + aij;
	  *dataPtr++ = value;
	}
      }
    }

    return 0;
}





// to perform this += At * B * C
int
Matrix::addMatrixTripleProduct(double thisFact, 
			       const Matrix &A, 
			       const Matrix &B,
			       const Matrix &C,
			       double otherFact)
{
    if (thisFact == 1.0 && otherFact == 0.0)
      return 0;
#ifdef _G3DEBUG
    if ((numRows != A.numRows) || (A.numCols != B.numRows) || (B.numCols != C.numRows) ||
	(C.numCols != numCols)) {
      opserr << "Matrix::addMatrixTripleProduct() - incompatible matrices\n";
      return -1;
    }
#endif

    // cheack work area can hold the temporary matrix
    int sizeWork = B.numRows * numCols;

    if (sizeWork > sizeDoubleWork) {
      this->addMatrix(thisFact, A^B*C, otherFact);
      return 0;
    }

    // zero out the work area
    double *matrixWorkPtr = matrixWork;
    for (int l=0; l<sizeWork; l++)
      *matrixWorkPtr++ = 0.0;

    // now form B * C * fact store in matrixWork == A area
    // NOTE: looping as per blas3 dgemm_: j,k,i
    
    int rowsB = B.numRows;
    double *ckjPtr  = &(C.data)[0];
    for (int j=0; j<numCols; j++) {
      double *aijPtrA = &matrixWork[j*rowsB];
      for (int k=0; k<rowsB; k++) {
	double tmp = *ckjPtr++ * otherFact;
	double *aijPtr = aijPtrA;
	double *bikPtr = &(B.data)[k*rowsB];
	for (int i=0; i<rowsB; i++) 
	  *aijPtr++ += *bikPtr++ * tmp;
      }
    }

    // now form A' * matrixWork
    // NOTE: looping as per blas3 dgemm_: j,i,k
    int dimB = rowsB;
    if (thisFact == 1.0) {
      double *dataPtr = &data[0];
      for (int j=0; j< numCols; j++) {
	double *workkjPtrA = &matrixWork[j*dimB];
	for (int i=0; i<numRows; i++) {
	  double *akiPtr = &(A.data)[i*dimB];
	  double *workkjPtr = workkjPtrA;
	  double aij = 0.0;
	  for (int k=0; k< dimB; k++)
	    aij += *akiPtr++ * *workkjPtr++;
	  *dataPtr++ += aij;
	}
      }
    } else if (thisFact == 0.0) {
      double *dataPtr = &data[0];
      for (int j=0; j< numCols; j++) {
	double *workkjPtrA = &matrixWork[j*dimB];
	for (int i=0; i<numRows; i++) {
	  double *akiPtr = &(A.data)[i*dimB];
	  double *workkjPtr = workkjPtrA;
	  double aij = 0.0;
	  for (int k=0; k< dimB; k++)
	    aij += *akiPtr++ * *workkjPtr++;
	  *dataPtr++ = aij;
	}
      }

    } else {
      double *dataPtr = &data[0];
      for (int j=0; j< numCols; j++) {
	double *workkjPtrA = &matrixWork[j*dimB];
	for (int i=0; i<numRows; i++) {
	  double *akiPtr = &(A.data)[i*dimB];
	  double *workkjPtr = workkjPtrA;
	  double aij = 0.0;
	  for (int k=0; k< dimB; k++)
	    aij += *akiPtr++ * *workkjPtr++;
	  double value = *dataPtr * thisFact + aij;
	  *dataPtr++ = value;
	}
      }
    }

    return 0;
}



//
// OVERLOADED OPERATOR () to CONSTRUCT A NEW MATRIX
//

Matrix
Matrix::operator()(const ID &rows, const ID & cols) const
{
    int nRows, nCols;
    nRows = rows.Size();
    nCols = cols.Size();
    Matrix result(nRows,nCols);
    double *dataPtr = result.data;
    for (int i=0; i<nCols; i++)
	for (int j=0; j<nRows; j++)
	    *dataPtr++ = (*this)(rows(j),cols(i));

    return result;
}
		
// Matrix &operator=(const Matrix  &V):
//      the assignment operator, This is assigned to be a copy of V. if sizes
//      are not compatible this.data [] is deleted. The data pointers will not
//      point to the same area in mem after the assignment.
//



Matrix &
Matrix::operator=(const Matrix &other)
{
  // first check we are not trying other = other
  if (this == &other) 
    return *this;

/*
#ifdef _G3DEBUG    
  if ((numCols != other.numCols) || (numRows != other.numRows)) {
    opserr << "Matrix::operator=() - matrix dimensions do not match: [%d %d] != [%d %d]\n",
			    numRows, numCols, other.numRows, other.numCols);
    return *this;
  }
#endif
*/

  if ((numCols != other.numCols) || (numRows != other.numRows)) {
#ifdef _G3DEBUG    
      opserr << "Matrix::operator=() - matrix dimensions do not match\n";
#endif

      if (this->data != 0)
      {
	  delete [] this->data;
          this->data = 0;
      }
      this->clearLUCache();

      int theSize = other.numCols*other.numRows;
      
      data = new (nothrow) double[theSize];
      
      this->dataSize = theSize;
      this->numCols = other.numCols;
      this->numRows = other.numRows;
  }


  // now copy the data
  double *dataPtr = data;
  double *otherDataPtr = other.data;		    
  for (int i=0; i<dataSize; i++)
      *dataPtr++ = *otherDataPtr++;
  
  // Clear LU cache for the current object since we're not copying the cache
  this->clearLUCache();

  return *this;
}


// Move assignment
//
#ifdef USE_CXX11
Matrix &
Matrix::operator=( Matrix &&other)
{
  // first check we are not trying other = other
  if (this == &other) 
    return *this;


  if (this->data != 0 && fromFree == 0){
    delete [] this->data;
    this->data = 0;
  }
  this->clearLUCache();
        
  this->data = other.data;
  this->dataSize = other.numCols*other.numRows;
  this->numCols = other.numCols;
  this->numRows = other.numRows;
  this->fromFree = other.fromFree;
  this->cachedLUFactor = other.cachedLUFactor;
  this->cachedPivot = other.cachedPivot;
  this->isLUFactorized = other.isLUFactorized;
  this->isLUCacheEnabled = other.isLUCacheEnabled;
  other.data = 0;
  other.dataSize = 0;
  other.numCols = 0;
  other.numRows = 0;
  other.fromFree = 1;
  other.cachedLUFactor = 0;
  other.cachedPivot = 0;
  other.isLUFactorized = false;
  other.isLUCacheEnabled = false;

  return *this;
}
#endif


// virtual Matrix &operator+=(double fact);
// virtual Matrix &operator-=(double fact);
// virtual Matrix &operator*=(double fact);
// virtual Matrix &operator/=(double fact); 
//	The above methods all modify the current matrix. If in
//	derived matrices data kept in data and of sizeData no redef necessary.

Matrix &
Matrix::operator+=(double fact)
{
  // check if quick return
  if (fact == 0.0)
    return *this;

  double *dataPtr = data;
  for (int i=0; i<dataSize; i++)
    *dataPtr++ += fact;
  
  return *this;
}




Matrix &
Matrix::operator-=(double fact)
{
  // check if quick return
  if (fact == 0.0)
    return *this;
  
  double *dataPtr = data;
  for (int i=0; i<dataSize; i++)
    *dataPtr++ -= fact;

  return *this;
}


Matrix &
Matrix::operator*=(double fact)
{
  // check if quick return
  if (fact == 1.0)
    return *this;
  
  double *dataPtr = data;
  for (int i=0; i<dataSize; i++)
    *dataPtr++ *= fact;
  
  return *this;
}

Matrix &
Matrix::operator/=(double fact)
{
    // check if quick return
    if (fact == 1.0)
	return *this;

    if (fact != 0.0) {
      double val = 1.0/fact;

      double *dataPtr = data;
      for (int i=0; i<dataSize; i++)
	*dataPtr++ *= val;

      return *this;
    } else {
      // print out the warning message
      opserr << "WARNING:Matrix::operator/= - 0 factor specified all values in Matrix set to ";
      opserr << MATRIX_VERY_LARGE_VALUE << endln;

      double *dataPtr = data;
      for (int i=0; i<dataSize; i++)
	*dataPtr++ = MATRIX_VERY_LARGE_VALUE;
      
      return *this;
    }
}


//    virtual Matrix operator+(double fact);
//    virtual Matrix operator-(double fact);
//    virtual Matrix operator*(double fact);
//    virtual Matrix operator/(double fact);
//	The above methods all return a new full general matrix.

Matrix
Matrix::operator+(double fact) const
{
    Matrix result(*this);
    result += fact;
    return result;
}

Matrix
Matrix::operator-(double fact) const
{
    Matrix result(*this);
    result -= fact;
    return result;
}

Matrix
Matrix::operator*(double fact) const
{
    Matrix result(*this);
    result *= fact;
    return result;
}

Matrix
Matrix::operator/(double fact) const
{
    if (fact == 0.0) {
	opserr << "Matrix::operator/(const double &fact): ERROR divide-by-zero\n";
	exit(0);
    }
    Matrix result(*this);
    result /= fact;
    return result;
}


//
// MATRIX_VECTOR OPERATIONS
//

Vector
Matrix::operator*(const Vector &V) const
{
    Vector result(numRows);
    
    if (V.Size() != numCols) {
	opserr << "Matrix::operator*(Vector): incompatible sizes\n";
	return result;
    } 
    
    double *dataPtr = data;
    for (int i=0; i<numCols; i++)
      for (int j=0; j<numRows; j++)
	result(j) += *dataPtr++ * V(i);

    /*
    opserr << "HELLO: " << V;
    for (int i=0; i<numRows; i++) {
	double sum = 0.0;
	for (int j=0; j<numCols; j++) {
	    sum += (*this)(i,j) * V(j);
	    if (i == 9) opserr << "sum: " << sum << " " << (*this)(i,j)*V(j) << " " << V(j) << 
;
	}
	result(i) += sum;
    }
    opserr << *this;
    opserr << "HELLO result: " << result;    
    */

    return result;
}

Vector
Matrix::operator^(const Vector &V) const
{
    Vector result(numCols);
    
    if (V.Size() != numRows) {
      opserr << "Matrix::operator*(Vector): incompatible sizes\n";
      return result;
    } 

    double *dataPtr = data;
    for (int i=0; i<numCols; i++)
      for (int j=0; j<numRows; j++)
	result(i) += *dataPtr++ * V(j);

    return result;
}


//
// MATRIX - MATRIX OPERATIONS
//
	    

Matrix
Matrix::operator+(const Matrix &M) const
{
    Matrix result(*this);
    result.addMatrix(1.0,M,1.0);    
    return result;
}
	    
Matrix
Matrix::operator-(const Matrix &M) const
{
    Matrix result(*this);
    result.addMatrix(1.0,M,-1.0);    
    return result;
}
	    
    
Matrix
Matrix::operator*(const Matrix &M) const
{
    Matrix result(numRows,M.numCols);
    
    if (numCols != M.numRows || result.numRows != numRows) {
	opserr << "Matrix::operator*(Matrix): incompatible sizes\n";
	return result;
    } 

    result.addMatrixProduct(0.0, *this, M, 1.0);
    
    /****************************************************
    double *resDataPtr = result.data;	    

    int innerDim = numCols;
    int nCols = result.numCols;
    for (int i=0; i<nCols; i++) {
      double *aStartRowDataPtr = data;
      double *bStartColDataPtr = &(M.data[i*innerDim]);
      for (int j=0; j<numRows; j++) {
	double *bDataPtr = bStartColDataPtr;
	double *aDataPtr = aStartRowDataPtr +j;	    
	double sum = 0.0;
	for (int k=0; k<innerDim; k++) {
	  sum += *aDataPtr * *bDataPtr++;
	  aDataPtr += numRows;
	}
	*resDataPtr++ = sum;
      }
    }
    ******************************************************/
    return result;
}



// Matrix operator^(const Matrix &M) const
//	We overload the * operator to perform matrix^t-matrix multiplication.
//	reults = (*this)transposed * M.

Matrix
Matrix::operator^(const Matrix &M) const
{
  Matrix result(numCols,M.numCols);
  
  if (numRows != M.numRows || result.numRows != numCols) {
    opserr << "Matrix::operator*(Matrix): incompatible sizes\n";
    return result;
  } 

    double *resDataPtr = result.data;	    

    int innerDim = numRows;
    int nCols = result.numCols;
    for (int i=0; i<nCols; i++) {
      double *aDataPtr = data;
      double *bStartColDataPtr = &(M.data[i*innerDim]);
      for (int j=0; j<numCols; j++) {
	double *bDataPtr = bStartColDataPtr;
	double sum = 0.0;
	for (int k=0; k<innerDim; k++) {
	  sum += *aDataPtr++ * *bDataPtr++;
	}
	*resDataPtr++ = sum;
      }
    }

    return result;
}
    



Matrix &
Matrix::operator+=(const Matrix &M)
{
#ifdef _G3DEBUG
  if (numRows != M.numRows || numCols != M.numCols) {
    opserr << "Matrix::operator+=(const Matrix &M) - matrices incompatible\n";
    return *this;
  }
#endif

  double *dataPtr = data;
  double *otherData = M.data;
  for (int i=0; i<dataSize; i++)
    *dataPtr++ += *otherData++;
  
  return *this;
}

Matrix &
Matrix::operator-=(const Matrix &M)
{
#ifdef _G3DEBUG
  if (numRows != M.numRows || numCols != M.numCols) {
    opserr << "Matrix::operator-=(const Matrix &M) - matrices incompatible [" << numRows << " " ;
    opserr << numCols << "]" << "[" << M.numRows << "]" << M.numCols << "]\n";

    return *this;
  }
#endif

  double *dataPtr = data;
  double *otherData = M.data;
  for (int i=0; i<dataSize; i++)
    *dataPtr++ -= *otherData++;
  
  return *this;
}


//
// Input/Output Methods
//

void 
Matrix::Output(OPS_Stream &s) const
{
    for (int i=0; i<noRows(); i++) {
	for (int j=0; j<noCols(); j++)
	    s <<  (*this)(i,j) << " ";
	s << endln;
    }
}


/*****************
void 
Matrix::Input(istream &s)
{
    for (int i=0; i<noRows(); i++)
	for (int j=0; j<noCols(); j++)
	    s >> (*this)(i,j);
}	
*****************/

//
// friend stream functions for input and output
//

OPS_Stream &operator<<(OPS_Stream &s, const Matrix &V)
{
    s << endln;
    V.Output(s);
    s << endln;        
    return s;
}

	
/****************	
istream &operator>>(istream &s, Matrix &V)
{
    V.Input(s);
    return s;
}
****************/


int
Matrix::Assemble(const Matrix &V, int init_row, int init_col, double fact) 
{
  int pos_Rows, pos_Cols;
  int res = 0;
  
  int VnumRows = V.numRows;
  int VnumCols = V.numCols;
  
  int final_row = init_row + VnumRows - 1;
  int final_col = init_col + VnumCols - 1;
  
  if ((init_row >= 0) && (final_row < numRows) && (init_col >= 0) && (final_col < numCols))
  {
     for (int i=0; i<VnumCols; i++) 
     {
        pos_Cols = init_col + i;
        for (int j=0; j<VnumRows; j++) 
        {
           pos_Rows = init_row + j;
      
	   (*this)(pos_Rows,pos_Cols) += V(j,i)*fact;
        }
     }
  }  
  else 
  {
     opserr << "WARNING: Matrix::Assemble(const Matrix &V, int init_row, int init_col, double fact): ";
     opserr << "position outside bounds \n";
     res = -1;
  }

  return res;
}


int
Matrix::Assemble(const Vector &V, int init_row, int init_col, double fact) 
{
  int pos_Rows, pos_Cols;
  int res = 0;
  
  int VnumRows = V.sz;
  int VnumCols = 1;
  
  int final_row = init_row + VnumRows - 1;
  int final_col = init_col + VnumCols - 1;
  
  if ((init_row >= 0) && (final_row < numRows) && (init_col >= 0) && (final_col < numCols))
  {
     for (int i=0; i<VnumCols; i++) 
     {
        pos_Cols = init_col + i;
        for (int j=0; j<VnumRows; j++) 
        {
           pos_Rows = init_row + j;
      
	   (*this)(pos_Rows,pos_Cols) += V(j)*fact;
        }
     }
  }  
  else 
  {
     opserr << "WARNING: Matrix::Assemble(const Matrix &V, int init_row, int init_col, double fact): ";
     opserr << "position outside bounds \n";
     res = -1;
  }

  return res;
}


int
Matrix::AssembleTranspose(const Matrix &V, int init_row, int init_col, double fact) 
{
  int pos_Rows, pos_Cols;
  int res = 0;
  
  int VnumRows = V.numRows;
  int VnumCols = V.numCols;
  
  int final_row = init_row + VnumCols - 1;
  int final_col = init_col + VnumRows - 1;
  
  if ((init_row >= 0) && (final_row < numRows) && (init_col >= 0) && (final_col < numCols))
  {
     for (int i=0; i<VnumRows; i++) 
     {
        pos_Cols = init_col + i;
        for (int j=0; j<VnumCols; j++) 
        {
           pos_Rows = init_row + j;
      
	   (*this)(pos_Rows,pos_Cols) += V(i,j)*fact;
        }
     }
  }  
  else 
  {
     opserr << "WARNING: Matrix::AssembleTranspose(const Matrix &V, int init_row, int init_col, double fact): ";
     opserr << "position outside bounds \n";
     res = -1;
  }

  return res;
}


int
Matrix::AssembleTranspose(const Vector &V, int init_row, int init_col, double fact) 
{
  int pos_Rows, pos_Cols;
  int res = 0;
  
  int VnumRows = V.sz;
  int VnumCols = 1;
  
  int final_row = init_row + VnumCols - 1;
  int final_col = init_col + VnumRows - 1;
  
  if ((init_row >= 0) && (final_row < numRows) && (init_col >= 0) && (final_col < numCols))
  {
     for (int i=0; i<VnumRows; i++) 
     {
        pos_Cols = init_col + i;
        for (int j=0; j<VnumCols; j++) 
        {
           pos_Rows = init_row + j;
      
	   (*this)(pos_Rows,pos_Cols) += V(i)*fact;
        }
     }
  }  
  else 
  {
     opserr << "WARNING: Matrix::AssembleTranspose(const Matrix &V, int init_row, int init_col, double fact): ";
     opserr << "position outside bounds \n";
     res = -1;
  }

  return res;
}


int
Matrix::Extract(const Matrix &V, int init_row, int init_col, double fact) 
{
  int pos_Rows, pos_Cols;
  int res = 0;
  
  int VnumRows = V.numRows;
  int VnumCols = V.numCols;
  
  int final_row = init_row + numRows - 1;
  int final_col = init_col + numCols - 1;
  
  if ((init_row >= 0) && (final_row < VnumRows) && (init_col >= 0) && (final_col < VnumCols))
  {
     for (int i=0; i<numCols; i++) 
     {
        pos_Cols = init_col + i;
        for (int j=0; j<numRows; j++) 
        {
           pos_Rows = init_row + j;
      
	   (*this)(j,i) = V(pos_Rows,pos_Cols)*fact;
        }
     }
  }  
  else 
  {
     opserr << "WARNING: Matrix::Extract(const Matrix &V, int init_row, int init_col, double fact): ";
     opserr << "position outside bounds \n";
     res = -1;
  }

  return res;
}


Matrix operator*(double a, const Matrix &V)
{
  return V * a;
}




int
Matrix::Eigen3(const Matrix &M)
{
  //.... compute eigenvalues and vectors for a 3 x 3 symmetric matrix
  //
  //.... INPUTS:
  //        M(3,3) - matrix with initial values (only upper half used)
  //
  //.... OUTPUTS
  //        v(3,3) - matrix of eigenvectors (by column)
  //        d(3)   - eigenvalues associated with columns of v
  //        rot    - number of rotations to diagonalize
  //
  //---------------------------------------------------------------eig3==

  //.... Storage done as follows:
  //
  //       | v(1,1) v(1,2) v(1,3) |     |  d(1)  a(1)  a(3)  |
  //       | v(2,1) v(2,2) v(2,3) |  =  |  a(1)  d(2)  a(2)  |
  //       | v(3,1) v(3,2) v(3,3) |     |  a(3)  a(2)  d(3)  |
  //
  //        Transformations performed on d(i) and a(i) and v(i,j) become
  //        the eigenvectors.  
  //
  //---------------------------------------------------------------eig3==

  int     rot, its, i, j , k ;
  double  g, h, aij, sm, thresh, t, c, s, tau ;

  static Matrix  v(3,3) ;
  static Vector  d(3) ;
  static Vector  a(3) ;
  static Vector  b(3) ; 
  static Vector  z(3) ;

  static const double tol = 1.0e-08 ;

  // set dataPtr 
  double *dataPtr = data;

  // set v = M 
  v = M ;
  
  //.... move array into one-d arrays
  a(0) = v(0,1) ;
  a(1) = v(1,2) ;
  a(2) = v(2,0) ;


  for ( i = 0; i < 3; i++ ) {
    d(i) = v(i,i) ;
    b(i) = v(i,i) ;
    z(i) = 0.0 ;

    for ( j = 0; j < 3; j++ ) 
      v(i,j) = 0.0 ;

    v(i,i) = 1.0 ;

  } //end for i

  rot = 0 ;
  its = 0 ;

  sm = fabs(a(0)) + fabs(a(1)) + fabs(a(2)) ;

  while ( sm > tol ) {
    //.... set convergence test and threshold
    if ( its < 3 ) 
      thresh = 0.011*sm ;
    else
      thresh = 0.0 ;
      
    //.... perform sweeps for rotations
    for ( i = 0; i < 3; i++ ) {

      j = (i+1)%3;
      k = (j+1)%3;

      aij  = a(i) ;

      g    = 100.0 * fabs(aij) ;

      if ( fabs(d(i)) + g != fabs(d(i))  ||
	   fabs(d(j)) + g != fabs(d(j))     ) {

	if ( fabs(aij) > thresh ) {

	  a(i) = 0.0 ; 
	  h    = d(j) - d(i) ; 

	  if( fabs(h)+g == fabs(h) )
	    t = aij / h ;
	  else {
	    //t = 2.0 * sign(h/aij) / ( fabs(h/aij) + sqrt(4.0+(h*h/aij/aij)));
	    double hDIVaij = h/aij;
	    if (hDIVaij > 0.0) 
	      t = 2.0 / (  hDIVaij + sqrt(4.0+(hDIVaij * hDIVaij)));
	    else
	      t = - 2.0 / (-hDIVaij + sqrt(4.0+(hDIVaij * hDIVaij)));
	  }

	  //.... set rotation parameters

	  c    = 1.0 / sqrt(1.0 + t*t) ;
	  s    = t*c ;
	  tau  = s / (1.0 + c) ;

	  //.... rotate diagonal terms

	  h    = t * aij ;
	  z(i) = z(i) - h ;
	  z(j) = z(j) + h ;
	  d(i) = d(i) - h ;
	  d(j) = d(j) + h ;

	  //.... rotate off-diagonal terms

	  h    = a(j) ;
	  g    = a[k] ;
	  a(j) = h + s*(g - h*tau) ;
	  a(k) = g - s*(h + g*tau) ;

	  //.... rotate eigenvectors

	  for ( k = 0; k < 3; k++ ) {
	    g      = v(k,i) ;
	    h      = v(k,j) ;
	    v(k,i) = g - s*(h + g*tau) ;
	    v(k,j) = h + s*(g - h*tau) ;
	  } // end for k

	  rot = rot + 1 ;

	} // end if fabs > thresh 
      } //else
      else 
	a(i) = 0.0 ;

    }  // end for i

    //.... update the diagonal terms
    for ( i = 0; i < 3; i++ ) {
      b(i) = b(i) + z(i) ;
      d(i) = b(i) ;
      z(i) = 0.0 ;
    } // end for i

    its += 1 ;

    sm = fabs(a(0)) + fabs(a(1)) + fabs(a(2)) ;

  } //end while sm
  static Vector  dd(3) ;
  if (d(0)>d(1))
    {
      if (d(0)>d(2))
	{
	  dd(0)=d(0);
	  if (d(1)>d(2))
	    {
	      dd(1)=d(1);
	      dd(2)=d(2);
	    }else
	    {
	      dd(1)=d(2);
	      dd(2)=d(1);
	    }
	}else
	{
	  dd(0)=d(2);
	  dd(1)=d(0);
	  dd(2)=d(1);
	}
    }else
    {
      if (d(1)>d(2))
	{
	  dd(0)=d(1);
	  if (d(0)>d(2))
	    {
	      dd(1)=d(0);
	      dd(2)=d(2);
	    }else
	    {
	      dd(1)=d(2);
	      dd(2)=d(0);
	    }
	}else
	{
	  dd(0)=d(2);
	  dd(1)=d(1);
	  dd(2)=d(0);
	}
    }
  data[0]=dd(2);
  data[4]=dd(1);
  data[8]=dd(0);

  return 0;
}



Vector Matrix::diagonal() const
{
  
  if (numRows != numCols)
  {
    opserr << "Matrix::diagonal() - Matrix is not square numRows = " << numRows << " numCols = " << numCols << " returning truncated diagonal." << endln;
  }

  int size = numRows < numCols ? numRows : numCols;
  Vector diagonal(size);

  for (int i = 0; i < size; ++i)
  {
    diagonal(i) = data[i*numRows + i];
  }

  return diagonal;
}
