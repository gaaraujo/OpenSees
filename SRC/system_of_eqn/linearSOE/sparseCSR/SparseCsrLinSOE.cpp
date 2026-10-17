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

#include "SparseCsrLinSOE.h"

#include <Graph.h>
#include <Vertex.h>
#include <Matrix.h>
#include <ID.h>
#include <OPS_Globals.h>
#include <classTags.h>
#include <elementAPI.h>
#include <algorithm>
#include <cstring>

namespace {

// Shared by `system SparseCSR` and `system DistributedSparseCSR`:
//   <-solver Umfpack|ProfileSPD|Mumps> <-minDiagTol tol> <-solverRanks all|root>
// -solverRanks (DistributedSparseCSR with Mumps): all = MUMPS factorizes and solves on every rank, as
// `system Mumps` (default); root = on rank 0 only, as one GPU on rank 0 in CudaMKRAlpha.
bool parseSparseCsrOptions(CsrMatrix::Solver &solver, double &tol, bool &solverOnRoot)
{
    solver = CsrMatrix::Solver::Umfpack;
    tol = 1.0e-12;
    solverOnRoot = false;
    while (OPS_GetNumRemainingInputArgs() > 0) {
        const char *opt = OPS_GetString();
        if (opt == nullptr)
            break;
        if (strcmp(opt, "-solver") == 0 && OPS_GetNumRemainingInputArgs() > 0) {
            const char *name = OPS_GetString();
            if (!CsrMatrix::solverFromString(name, solver)) {
                opserr << "WARNING system SparseCSR - unknown -solver " << (name ? name : "")
                       << "; want Umfpack, ProfileSPD, or Mumps\n";
                return false;
            }
            if (!CsrMatrix::isSolverAvailable(solver)) {
                opserr << "WARNING system SparseCSR - -solver " << name
                       << " is not available in this executable (Mumps requires OpenSeesMP built with MUMPS)\n";
                return false;
            }
        } else if (strcmp(opt, "-minDiagTol") == 0 && OPS_GetNumRemainingInputArgs() > 0) {
            int numData = 1;
            if (OPS_GetDoubleInput(&numData, &tol) != 0) {
                opserr << "WARNING system SparseCSR - invalid -minDiagTol\n";
                return false;
            }
        } else if (strcmp(opt, "-solverRanks") == 0 && OPS_GetNumRemainingInputArgs() > 0) {
            const char *ranks = OPS_GetString();
            if (ranks != nullptr && strcmp(ranks, "root") == 0) {
                solverOnRoot = true;
            } else if (ranks != nullptr && strcmp(ranks, "all") == 0) {
                solverOnRoot = false;
            } else {
                opserr << "WARNING system SparseCSR - -solverRanks " << (ranks ? ranks : "")
                       << "; want all or root\n";
                return false;
            }
        } else {
            opserr << "WARNING system SparseCSR - unknown option " << opt
                   << "; want <-solver Umfpack|ProfileSPD|Mumps> <-minDiagTol tol> <-solverRanks all|root>\n";
            return false;
        }
    }
    return true;
}

} // namespace

void *OPS_SparseCsrLinSOE(void)
{
    CsrMatrix::Solver solver;
    double tol;
    bool solverOnRoot;
    if (!parseSparseCsrOptions(solver, tol, solverOnRoot))
        return nullptr;
    return new SparseCsrLinSOE(solver, tol);  // one process: -solverRanks has no effect
}

bool OPS_ParseSparseCsrOptions(CsrMatrix::Solver &solver, double &tol, bool &solverOnRoot)
{
    return parseSparseCsrOptions(solver, tol, solverOnRoot);
}

SparseCsrLinSOE::SparseCsrLinSOE(CsrMatrix::Solver solver, double profileMinDiagTol)
    : LinearSOE(LinSOE_TAGS_SparseCsrLinSOE),
      m_solver(solver),
      m_profileMinDiagTol(profileMinDiagTol),
      m_matrixStatus(MatrixStatus::STRUCTURE_CHANGED),
      m_matA(solver, profileMinDiagTol)
{
}

SparseCsrLinSOE::~SparseCsrLinSOE() = default;

int SparseCsrLinSOE::getNumEqn(void) const
{
    return m_X.Size();
}

const int *SparseCsrLinSOE::getRowPtr(void) const
{
    return m_csrIndices.empty() ? nullptr : m_csrIndices.data();
}

const int *SparseCsrLinSOE::getColIndices(void) const
{
    return m_csrIndices.empty() ? nullptr : m_csrIndices.data() + m_X.Size() + 1;
}

int SparseCsrLinSOE::setSize(Graph &theGraph)
{
    const int size = theGraph.getNumVertex();
    if (size < 0) {
        opserr << "WARNING SparseCsrLinSOE::setSize() - graph size " << size << " < 0\n";
        return -1;
    }

    // CSR of the graph: each row holds its vertex and the adjacency, sorted (as CudaBcsrLinSOE).
    std::vector<int> rowPtr(static_cast<std::size_t>(size) + 1, 0);
    std::vector<int> cols;
    for (int row = 0; row < size; ++row) {
        Vertex *theVertex = theGraph.getVertexPtr(row);
        if (theVertex == nullptr) {
            opserr << "WARNING SparseCsrLinSOE::setSize() - vertex " << row << " not found in graph\n";
            return -1;
        }
        const ID &theAdjacency = theVertex->getAdjacency();
        ID localColIdx(0, theAdjacency.Size() + 1);
        localColIdx.insert(theVertex->getTag());
        for (int j = 0; j < theAdjacency.Size(); ++j)
            localColIdx.insert(theAdjacency(j));
        for (int j = 0; j < localColIdx.Size(); ++j)
            cols.push_back(localColIdx(j));
        rowPtr[row + 1] = static_cast<int>(cols.size());
    }
    const int nnz = static_cast<int>(cols.size());

    m_csrIndices.resize(static_cast<std::size_t>(size) + 1 + nnz);
    std::copy(rowPtr.begin(), rowPtr.end(), m_csrIndices.begin());
    std::copy(cols.begin(), cols.end(), m_csrIndices.begin() + size + 1);
    m_aValues.assign(static_cast<std::size_t>(nnz), 0.0);
    m_hostB.assign(static_cast<std::size_t>(size), 0.0);
    m_hostX.assign(static_cast<std::size_t>(size), 0.0);
    m_B.setData(m_hostB.empty() ? nullptr : m_hostB.data(), size);
    m_X.setData(m_hostX.empty() ? nullptr : m_hostX.data(), size);

    m_matA.reset();
    m_matrixStatus = MatrixStatus::STRUCTURE_CHANGED;
    return 0;
}

int SparseCsrLinSOE::resizeHostVectors(int numEqn)
{
    if (numEqn < 0)
        return -1;
    m_csrIndices.clear();
    m_aValues.clear();
    m_hostB.assign(static_cast<std::size_t>(numEqn), 0.0);
    m_hostX.assign(static_cast<std::size_t>(numEqn), 0.0);
    m_B.setData(m_hostB.empty() ? nullptr : m_hostB.data(), numEqn);
    m_X.setData(m_hostX.empty() ? nullptr : m_hostX.data(), numEqn);
    m_matA.reset();
    m_matrixStatus = MatrixStatus::STRUCTURE_CHANGED;
    return 0;
}

int SparseCsrLinSOE::addAMatrixElement(int globalRow, int globalCol, double value)
{
    const int n = m_X.Size();
    const int *rowPtr = m_csrIndices.data();
    const int *colIdx = rowPtr + n + 1;
    for (int k = rowPtr[globalRow]; k < rowPtr[globalRow + 1]; ++k) {
        if (colIdx[k] == globalCol) {
            m_aValues[k] += value;
            return 0;
        }
    }
    opserr << "WARNING SparseCsrLinSOE::addA() - no entry for row " << globalRow << ", col " << globalCol << "\n";
    return -1;
}

int SparseCsrLinSOE::addA(const Matrix &m, const ID &id, double fact)
{
    if (fact == 0.0)
        return 0;
    const int idSize = id.Size();
    if (idSize != m.noRows() && idSize != m.noCols()) {
        opserr << "WARNING SparseCsrLinSOE::addA() - Matrix and ID not of similar sizes\n";
        return -1;
    }
    for (int i = 0; i < idSize; ++i) {
        const int globalRow = id(i);
        if (!isValidGlobalIndex(globalRow))
            continue;
        for (int j = 0; j < idSize; ++j) {
            const int globalCol = id(j);
            if (!isValidGlobalIndex(globalCol))
                continue;
            const double v = (fact == 1.0) ? m(i, j) : (fact == -1.0 ? -m(i, j) : fact * m(i, j));
            if (addAMatrixElement(globalRow, globalCol, v) != 0)
                return -1;
        }
    }
    if (m_matrixStatus == MatrixStatus::UNCHANGED)
        m_matrixStatus = MatrixStatus::COEFFICIENTS_CHANGED;
    return 0;
}

int SparseCsrLinSOE::addA(const Matrix &m)
{
    const int n = m.noRows();
    if (n != m.noCols() || n != getNumEqn()) {
        opserr << "SparseCsrLinSOE::addA(Matrix) - matrix size mismatch\n";
        return -1;
    }
    ID all(n);
    for (int i = 0; i < n; ++i)
        all(i) = i;
    return addA(m, all, 1.0);
}

int SparseCsrLinSOE::addB(const Vector &v, const ID &id, double fact)
{
    if (fact == 0.0)
        return 0;
    const int idSize = id.Size();
    if (idSize != v.Size()) {
        opserr << "WARNING SparseCsrLinSOE::addB() - Vector and ID not of similar sizes\n";
        return -1;
    }
    for (int i = 0; i < idSize; ++i) {
        const int globalRow = id(i);
        if (isValidGlobalIndex(globalRow))
            m_B(globalRow) += (fact == 1.0) ? v(i) : (fact == -1.0 ? -v(i) : fact * v(i));
    }
    return 0;
}

int SparseCsrLinSOE::setB(const Vector &v, double fact)
{
    if (fact == 0.0) {
        zeroB();
        return 0;
    }
    const int size = m_B.Size();
    if (size != v.Size()) {
        opserr << "WARNING SparseCsrLinSOE::setB() - Vector size mismatch\n";
        return -1;
    }
    for (int i = 0; i < size; ++i)
        m_hostB[i] = (fact == 1.0) ? v(i) : fact * v(i);
    return 0;
}

void SparseCsrLinSOE::zeroA(void)
{
    std::fill(m_aValues.begin(), m_aValues.end(), 0.0);
    if (m_matrixStatus == MatrixStatus::UNCHANGED)
        m_matrixStatus = MatrixStatus::COEFFICIENTS_CHANGED;
}

void SparseCsrLinSOE::zeroB(void)
{
    std::fill(m_hostB.begin(), m_hostB.end(), 0.0);
}

const Vector &SparseCsrLinSOE::getX(void)
{
    return m_X;
}

const Vector &SparseCsrLinSOE::getB(void)
{
    return m_B;
}

double SparseCsrLinSOE::normRHS(void)
{
    return m_B.Norm();
}

void SparseCsrLinSOE::setX(int loc, double value)
{
    if (isValidGlobalIndex(loc))
        m_X(loc) = value;
}

void SparseCsrLinSOE::setX(const Vector &x)
{
    if (x.Size() == m_X.Size())
        m_X = x;
}

void SparseCsrLinSOE::setDistributedSolve(bool distributed, bool participant)
{
    m_matA.setDistributed(distributed && !m_solverOnRoot);
    m_participant = participant;
}

// MatrixStatus gates the work, as CudaBcsrLinSOE + CudaDirectSparseSolver: a new pattern is analyzed
// and factorized, new values are refactorized, and an unchanged matrix is only solved.
int SparseCsrLinSOE::solve(void)
{
    const int n = m_X.Size();
    if (m_participant) {
        // OpenSeesMP worker: nothing to solve, except joining a distributed MUMPS factorization/solve
        // of rank 0's A (same status-driven sequence of calls as rank 0).
        if (!m_matA.isCollective())
            return 0;
        int rc = 0;
        if (m_matrixStatus == MatrixStatus::STRUCTURE_CHANGED || !m_matA.isStructureBound()) {
            rc = m_matA.bindParticipant(n);
            if (rc == 0)
                rc = m_matA.factorize();
        } else if (m_matrixStatus == MatrixStatus::COEFFICIENTS_CHANGED || !m_matA.isFactored()) {
            rc = m_matA.refactorize();
        }
        if (rc == 0)
            rc = m_matA.solve(nullptr, nullptr, 1);
        return rc == 0 ? 0 : -1;
    }
    if (n == 0 || m_aValues.empty())
        return 0;

    int rc = 0;
    if (m_matrixStatus == MatrixStatus::STRUCTURE_CHANGED || !m_matA.isStructureBound()) {
        if (m_matA.bindStructure(n, getNumNonZeroValues(), getRowPtr(), getColIndices()) != 0 ||
            m_matA.bindValues(m_aValues.data()) != 0)
            return -1;
        rc = m_matA.factorize();
    } else if (m_matrixStatus == MatrixStatus::COEFFICIENTS_CHANGED || !m_matA.isFactored()) {
        rc = m_matA.refactorize();
    }
    if (rc != 0) {
        opserr << "WARNING SparseCsrLinSOE::solve() - factorization failed (" << CsrMatrix::solverName(m_solver) << ")\n";
        return -1;
    }
    if (m_matA.solve(m_hostB.data(), m_hostX.data(), 1) != 0) {
        opserr << "WARNING SparseCsrLinSOE::solve() - solve failed\n";
        return -1;
    }
    m_matrixStatus = MatrixStatus::UNCHANGED;
    return 0;
}

int SparseCsrLinSOE::formAp(const Vector &p, Vector &Ap)
{
    const int n = m_X.Size();
    if (p.Size() != n || Ap.Size() != n) {
        opserr << "WARNING SparseCsrLinSOE::formAp() - vectors must match size " << n << "\n";
        return -1;
    }
    if (n == 0)
        return 0;
    CsrMatrix op;
    if (op.bindStructure(n, getNumNonZeroValues(), getRowPtr(), getColIndices()) != 0 ||
        op.bindValues(m_aValues.data()) != 0)
        return -1;
    return op.spmv(&const_cast<Vector &>(p)(0), &Ap(0));
}

LinearSOE *SparseCsrLinSOE::getCopy(void) const
{
    return new SparseCsrLinSOE(m_solver, m_profileMinDiagTol);
}

int SparseCsrLinSOE::sendSelf(int commitTag, Channel &theChannel)
{
    (void)commitTag;
    (void)theChannel;
    return 0;
}

int SparseCsrLinSOE::recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    (void)commitTag;
    (void)theChannel;
    (void)theBroker;
    return 0;
}
