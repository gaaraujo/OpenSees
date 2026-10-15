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

#include "CsrMatrix.h"

#include <OPS_Globals.h>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "../../../../OTHER/UMFPACK/umfpack.h"

#include "CsrDirectBackend.h"

namespace {

// UMFPACK on the CSC transpose map of the CSR pattern, with the controls of UmfpackGenLinSolver
// (pivot tolerance 1, symmetric strategy). The symbolic analysis is pattern-only and done once.
class UmfpackCsrBackend : public CsrDirectBackend
{
public:
    UmfpackCsrBackend() { umfpack_di_defaults(m_control); }
    ~UmfpackCsrBackend() override { release(); }

    int analyze(int n, int nnz, const int *rowPtr, const int *colIdx) override
    {
        release();
        m_n = n;
        m_Ap.assign(static_cast<std::size_t>(n) + 1, 0);
        m_Ai.assign(static_cast<std::size_t>(nnz), 0);
        m_Ax.assign(static_cast<std::size_t>(nnz), 0.0);
        m_map.assign(static_cast<std::size_t>(nnz), 0);
        for (int k = 0; k < nnz; ++k)
            ++m_Ap[colIdx[k] + 1];
        for (int j = 0; j < n; ++j)
            m_Ap[j + 1] += m_Ap[j];
        std::vector<int> next(m_Ap.begin(), m_Ap.end() - 1);
        for (int r = 0; r < n; ++r) {
            for (int k = rowPtr[r]; k < rowPtr[r + 1]; ++k) {
                const int pos = next[colIdx[k]]++;
                m_Ai[pos] = r;
                m_map[k] = pos;
            }
        }
        m_control[UMFPACK_PIVOT_TOLERANCE] = 1.0;
        m_control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;
        const int status = umfpack_di_symbolic(n, n, m_Ap.data(), m_Ai.data(), nullptr, &m_symbolic, m_control, m_info);
        if (status != UMFPACK_OK) {
            opserr << "WARNING CsrMatrix (Umfpack) - symbolic analysis returns " << status << "\n";
            m_symbolic = nullptr;
            return -1;
        }
        return 0;
    }

    int factorize(const double *values) override
    {
        if (m_symbolic == nullptr)
            return -1;
        const std::size_t nnz = m_map.size();
        for (std::size_t k = 0; k < nnz; ++k)
            m_Ax[m_map[k]] = values[k];
        if (m_numeric != nullptr)
            umfpack_di_free_numeric(&m_numeric);
        const int status = umfpack_di_numeric(m_Ap.data(), m_Ai.data(), m_Ax.data(), m_symbolic, &m_numeric, m_control, m_info);
        if (status != UMFPACK_OK) {
            opserr << "WARNING CsrMatrix (Umfpack) - numeric factorization returns " << status << "\n";
            return -1;
        }
        return 0;
    }

    int solve(const double *b, double *x) override
    {
        if (m_numeric == nullptr)
            return -1;
        const int status = umfpack_di_solve(UMFPACK_A, m_Ap.data(), m_Ai.data(), m_Ax.data(), x, b, m_numeric, m_control, m_info);
        if (status != UMFPACK_OK) {
            opserr << "WARNING CsrMatrix (Umfpack) - solve returns " << status << "\n";
            return -1;
        }
        return 0;
    }

private:
    void release()
    {
        if (m_numeric != nullptr)
            umfpack_di_free_numeric(&m_numeric);
        if (m_symbolic != nullptr)
            umfpack_di_free_symbolic(&m_symbolic);
        m_numeric = nullptr;
        m_symbolic = nullptr;
    }

    int m_n = 0;
    std::vector<int> m_Ap, m_Ai, m_map;  // CSC pattern of A and CSR -> CSC value positions
    std::vector<double> m_Ax;
    void *m_symbolic = nullptr;
    void *m_numeric = nullptr;
    double m_control[UMFPACK_CONTROL];
    double m_info[UMFPACK_INFO];
};

// Profile (skyline) U^T D U factorization of ProfileSPDLinDirectSolver, on the profile of the CSR
// pattern: column j holds A(i, j) for RowTop[j] <= i <= j. Uses the lower triangle of the CSR values
// (the matrix is assumed symmetric, as in ProfileSPD).
class ProfileCsrBackend : public CsrDirectBackend
{
public:
    explicit ProfileCsrBackend(double minDiagTol) : m_minDiagTol(minDiagTol) {}

    int analyze(int n, int nnz, const int *rowPtr, const int *colIdx) override
    {
        m_n = n;
        m_rowTop.resize(static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j)
            m_rowTop[j] = j;
        for (int r = 0; r < n; ++r) {
            for (int k = rowPtr[r]; k < rowPtr[r + 1]; ++k) {
                const int c = colIdx[k];
                const int i = std::min(r, c);
                const int j = std::max(r, c);
                if (i < m_rowTop[j])
                    m_rowTop[j] = i;
            }
        }
        m_colStart.resize(static_cast<std::size_t>(n) + 1);
        m_colStart[0] = 0;
        for (int j = 0; j < n; ++j)
            m_colStart[j + 1] = m_colStart[j] + (j - m_rowTop[j] + 1);
        m_profile.assign(static_cast<std::size_t>(m_colStart[n]), 0.0);
        m_invD.assign(static_cast<std::size_t>(n), 0.0);
        // lower-triangle CSR entries (r, c), c <= r, are A(c, r) in column r of the profile
        m_map.assign(static_cast<std::size_t>(nnz), -1);
        for (int r = 0; r < n; ++r) {
            for (int k = rowPtr[r]; k < rowPtr[r + 1]; ++k) {
                const int c = colIdx[k];
                if (c <= r)
                    m_map[k] = m_colStart[r] + (c - m_rowTop[r]);
            }
        }
        return 0;
    }

    int factorize(const double *values) override
    {
        const int n = m_n;
        std::fill(m_profile.begin(), m_profile.end(), 0.0);
        const std::size_t nnz = m_map.size();
        for (std::size_t k = 0; k < nnz; ++k) {
            if (m_map[k] >= 0)
                m_profile[m_map[k]] += values[k];
        }
        if (n == 0)
            return 0;
        double *A = m_profile.data();
        if (n == 1) {
            if (std::fabs(A[0]) < 1.0e-15) {
                opserr << "WARNING CsrMatrix (ProfileSPD) - singular 1x1 (|aii| < 1e-15)\n";
                return -2;
            }
            m_invD[0] = 1.0 / A[0];
            return 0;
        }
        if (A[0] <= 0.0) {
            opserr << "WARNING CsrMatrix (ProfileSPD) - aii < 0 (i, aii): (0,0)\n";
            return -2;
        }
        m_invD[0] = 1.0 / A[0];
        for (int i = 1; i < n; ++i) {
            const int rowitop = m_rowTop[i];
            double *ajiPtr = top(i);
            for (int j = rowitop; j < i; ++j) {
                double tmp = *ajiPtr;
                const int rowjtop = m_rowTop[j];
                if (rowitop > rowjtop) {
                    const double *akjPtr = top(j) + (rowitop - rowjtop);
                    const double *akiPtr = top(i);
                    for (int k = rowitop; k < j; ++k)
                        tmp -= *akjPtr++ * *akiPtr++;
                } else {
                    const double *akjPtr = top(j);
                    const double *akiPtr = top(i) + (rowjtop - rowitop);
                    for (int k = rowjtop; k < j; ++k)
                        tmp -= *akjPtr++ * *akiPtr++;
                }
                *ajiPtr++ = tmp;
            }
            double aii = A[m_colStart[i] + (i - rowitop)];
            ajiPtr = top(i);
            for (int jj = rowitop; jj < i; ++jj) {
                const double aji = *ajiPtr;
                const double lij = aji * m_invD[jj];
                *ajiPtr++ = lij;
                aii = aii - lij * aji;
            }
            if (aii == 0.0 || std::fabs(aii) <= m_minDiagTol) {
                opserr << "WARNING CsrMatrix (ProfileSPD) - aii < minDiagTol (i, aii): (" << i << ", " << aii << ")\n";
                return -2;
            }
            m_invD[i] = 1.0 / aii;
        }
        return 0;
    }

    int solve(const double *b, double *x) override
    {
        const int n = m_n;
        for (int i = 0; i < n; ++i)
            x[i] = b[i];
        // forward substitution
        for (int i = 1; i < n; ++i) {
            const int rowitop = m_rowTop[i];
            const double *ajiPtr = top(i);
            const double *bjPtr = &x[rowitop];
            double tmp = 0.0;
            for (int j = rowitop; j < i; ++j)
                tmp -= *ajiPtr++ * *bjPtr++;
            x[i] += tmp;
        }
        // divide by the diagonal
        for (int j = 0; j < n; ++j)
            x[j] = m_invD[j] * x[j];
        // back substitution
        for (int k = n - 1; k > 0; --k) {
            const int rowktop = m_rowTop[k];
            const double bk = x[k];
            const double *ajiPtr = top(k);
            for (int j = rowktop; j < k; ++j)
                x[j] -= *ajiPtr++ * bk;
        }
        return 0;
    }

private:
    double *top(int j) { return m_profile.data() + m_colStart[j]; }
    const double *top(int j) const { return m_profile.data() + m_colStart[j]; }

    double m_minDiagTol;
    int m_n = 0;
    std::vector<int> m_rowTop, m_colStart, m_map;
    std::vector<double> m_profile, m_invD;
};

CsrMatrix::BackendFactory s_mumpsFactory = nullptr;

} // namespace

void CsrMatrix::registerMumpsBackend(BackendFactory factory)
{
    s_mumpsFactory = factory;
}

bool CsrMatrix::isSolverAvailable(Solver s)
{
    return s != Solver::Mumps || s_mumpsFactory != nullptr;
}

CsrMatrix::CsrMatrix(Solver solver, double profileMinDiagTol)
    : m_solver(solver), m_profileMinDiagTol(profileMinDiagTol)
{
}

CsrMatrix::~CsrMatrix()
{
    delete m_backend;
}

void CsrMatrix::reset()
{
    delete m_backend;
    m_backend = nullptr;
    m_numRows = 0;
    m_numNnz = 0;
    m_structureBound = false;
    m_analyzed = false;
    m_factored = false;
    m_rowPtr = nullptr;
    m_colIdx = nullptr;
    m_values = nullptr;
    m_ownedValues.clear();
}

int CsrMatrix::bindStructure(int numRows, int numNnz, const int *rowPtr, const int *colIdx)
{
    if (numRows < 0 || numNnz < 0 || (numRows > 0 && (rowPtr == nullptr || colIdx == nullptr))) {
        opserr << "WARNING CsrMatrix::bindStructure - invalid arguments\n";
        return -1;
    }
    const bool hadOwned = !m_ownedValues.empty() && m_values == m_ownedValues.data();
    delete m_backend;
    m_backend = nullptr;
    m_numRows = numRows;
    m_numNnz = numNnz;
    m_rowPtr = rowPtr;
    m_colIdx = colIdx;
    m_structureBound = true;
    m_participant = false;
    m_analyzed = false;
    m_factored = false;
    if (hadOwned) {
        m_ownedValues.assign(static_cast<std::size_t>(numNnz), 0.0);
        m_values = m_ownedValues.data();
    }
    return 0;
}

int CsrMatrix::bindParticipant(int numRows)
{
    if (!isCollective()) {
        opserr << "WARNING CsrMatrix::bindParticipant - only a distributed MUMPS operator has participants\n";
        return -1;
    }
    delete m_backend;
    m_backend = nullptr;
    m_numRows = numRows;
    m_numNnz = 0;
    m_rowPtr = nullptr;
    m_colIdx = nullptr;
    m_values = nullptr;
    m_ownedValues.clear();
    m_structureBound = true;
    m_participant = true;
    m_analyzed = false;
    m_factored = false;
    return 0;
}

int CsrMatrix::bindValues(double *values)
{
    if (!m_structureBound || (m_numNnz > 0 && values == nullptr))
        return -1;
    m_ownedValues.clear();
    m_values = values;
    m_factored = false;
    return 0;
}

int CsrMatrix::copyValues(const double *values)
{
    if (!m_structureBound || (m_numNnz > 0 && values == nullptr))
        return -1;
    if (m_ownedValues.size() != static_cast<std::size_t>(m_numNnz))
        m_ownedValues.assign(static_cast<std::size_t>(m_numNnz), 0.0);
    if (m_numNnz > 0)
        std::memcpy(m_ownedValues.data(), values, static_cast<std::size_t>(m_numNnz) * sizeof(double));
    m_values = m_ownedValues.data();
    m_factored = false;
    return 0;
}

int CsrMatrix::spmv(const double *x, double *y, double alpha, double beta) const
{
    if (!m_structureBound || m_values == nullptr || x == nullptr || y == nullptr)
        return -1;
    for (int r = 0; r < m_numRows; ++r) {
        double sum = 0.0;
        for (int k = m_rowPtr[r]; k < m_rowPtr[r + 1]; ++k)
            sum += m_values[k] * x[m_colIdx[k]];
        y[r] = (beta == 0.0) ? alpha * sum : alpha * sum + beta * y[r];
    }
    return 0;
}

int CsrMatrix::factorize()
{
    if (!m_structureBound || (m_values == nullptr && !m_participant))
        return -1;
    if (m_backend == nullptr) {
        if (m_solver == Solver::ProfileSPD)
            m_backend = new ProfileCsrBackend(m_profileMinDiagTol);
        else if (m_solver == Solver::Mumps)
            m_backend = s_mumpsFactory != nullptr ? s_mumpsFactory(m_distributed) : nullptr;
        else
            m_backend = new UmfpackCsrBackend();
        if (m_backend == nullptr) {
            opserr << "WARNING CsrMatrix - solver " << solverName(m_solver) << " is not available in this executable\n";
            return -1;
        }
        m_analyzed = false;
    }
    if (!m_analyzed) {
        if (m_backend->analyze(m_numRows, m_numNnz, m_rowPtr, m_colIdx) != 0)
            return -1;
        m_analyzed = true;
    }
    return refactorize();
}

int CsrMatrix::refactorize()
{
    if (m_backend == nullptr || !m_analyzed)
        return factorize();
    m_factored = false;
    if (m_backend->factorize(m_values) != 0)
        return -2;
    m_factored = true;
    return 0;
}

int CsrMatrix::solve(const double *rhs, double *solution, int numRhs)
{
    if (!m_factored || numRhs < 1)
        return -1;
    const int n = m_numRows;
    if (m_participant)  // the rank-0 operator owns the right-hand sides
        return m_backend->solveMulti(nullptr, nullptr, n, numRhs) == 0 ? 0 : -2;
    if (rhs == nullptr || solution == nullptr)
        return -1;
    const double *b = rhs;
    if (rhs == solution) {
        m_work.assign(rhs, rhs + static_cast<std::size_t>(numRhs) * n);
        b = m_work.data();
    }
    return m_backend->solveMulti(b, solution, n, numRhs) == 0 ? 0 : -2;
}

const char *CsrMatrix::solverName(Solver s)
{
    switch (s) {
    case Solver::ProfileSPD:
        return "ProfileSPD";
    case Solver::Mumps:
        return "Mumps";
    default:
        return "Umfpack";
    }
}

bool CsrMatrix::solverFromString(const char *s, Solver &out)
{
    if (s == nullptr)
        return false;
    if (strcmp(s, "Umfpack") == 0 || strcmp(s, "UmfPack") == 0 || strcmp(s, "umfpack") == 0 ||
        strcmp(s, "UMFPACK") == 0) {
        out = Solver::Umfpack;
        return true;
    }
    if (strcmp(s, "ProfileSPD") == 0 || strcmp(s, "profileSPD") == 0 || strcmp(s, "Profile") == 0) {
        out = Solver::ProfileSPD;
        return true;
    }
    if (strcmp(s, "Mumps") == 0 || strcmp(s, "MUMPS") == 0 || strcmp(s, "mumps") == 0) {
        out = Solver::Mumps;
        return true;
    }
    return false;
}
