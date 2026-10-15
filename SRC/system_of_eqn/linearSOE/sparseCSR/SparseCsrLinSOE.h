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
//
// Description: Host CSR system of equations, the CPU counterpart of CudaBcsrLinSOE (block size 1,
// full storage): CSR from the graph, MatrixStatus, and a direct solve through a CsrMatrix bound to
// the SOE's own arrays (UMFPACK, the profile LDL^T of ProfileSPD, or MUMPS, chosen with -solver). The
// structure and values are exposed to DistributedSparseCsrLinSOE (gather to rank 0) and to
// SparseExplicitAlpha (operators sharing the pattern).
//
//   system SparseCSR <-solver Umfpack|ProfileSPD|Mumps>

#ifndef SparseCsrLinSOE_h
#define SparseCsrLinSOE_h

#include <LinearSOE.h>
#include <Vector.h>
#include <vector>
#include "CsrMatrix.h"

class SparseCsrLinSOE : public LinearSOE
{
public:
    enum class MatrixStatus { UNCHANGED, COEFFICIENTS_CHANGED, STRUCTURE_CHANGED };

    explicit SparseCsrLinSOE(CsrMatrix::Solver solver = CsrMatrix::Solver::Umfpack,
                             double profileMinDiagTol = 1.0e-12);
    ~SparseCsrLinSOE() override;

    int getNumEqn(void) const override;
    int setSize(Graph &theGraph) override;
    int addA(const Matrix &, const ID &, double fact = 1.0) override;
    int addA(const Matrix &) override;
    int addB(const Vector &, const ID &, double fact = 1.0) override;
    int setB(const Vector &, double fact = 1.0) override;
    void zeroA(void) override;
    void zeroB(void) override;
    const Vector &getX(void) override;
    const Vector &getB(void) override;
    double normRHS(void) override;
    void setX(int loc, double value) override;
    void setX(const Vector &x) override;
    int solve(void) override;
    int formAp(const Vector &p, Vector &Ap) override;
    LinearSOE *getCopy(void) const override;

    int sendSelf(int commitTag, Channel &theChannel) override;
    int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker) override;

    MatrixStatus getMatrixStatus(void) const { return m_matrixStatus; }
    void setMatrixStatus(MatrixStatus status) { m_matrixStatus = status; }

    CsrMatrix::Solver getCsrSolver(void) const { return m_solver; }
    /** Set by DistributedSparseCsrLinSOE: distributed = more than one rank; participant = a worker. With
     *  -solver Mumps the factorization and solve of A are then collective (see isCollectiveSolver). */
    void setDistributedSolve(bool distributed, bool participant);
    /** true for distributed MUMPS: every rank calls solve() at the same point. */
    bool isCollectiveSolver(void) const { return m_matA.isCollective(); }
    double getProfileMinDiagTol(void) const { return m_profileMinDiagTol; }

    // Host CSR accessors (layout [rowPtr (n+1) | colIdx (nnz)], as CudaBcsrLinSOE)
    int getNumNonZeroValues(void) const { return static_cast<int>(m_aValues.size()); }
    const int *getRowPtr(void) const;
    const int *getColIndices(void) const;
    double *getAValuesData(void) { return m_aValues.empty() ? nullptr : m_aValues.data(); }
    const double *getAValuesData(void) const { return m_aValues.empty() ? nullptr : m_aValues.data(); }
    Vector &getHostBVector(void) { return m_B; }
    Vector &getHostXVector(void) { return m_X; }

    /** Worker-only (DistributedSparseCsrLinSOE): size X/B without the global CSR. */
    int resizeHostVectors(int numEqn);

private:
    int addAMatrixElement(int globalRow, int globalCol, double value);
    bool isValidGlobalIndex(int index) const { return index >= 0 && index < m_X.Size(); }

    CsrMatrix::Solver m_solver;
    double m_profileMinDiagTol;
    MatrixStatus m_matrixStatus;
    bool m_participant = false;

    std::vector<int> m_csrIndices;
    std::vector<double> m_aValues, m_hostB, m_hostX;
    Vector m_X, m_B;

    CsrMatrix m_matA;  // bound to m_csrIndices / m_aValues; holds the factorization of A
};

#endif
