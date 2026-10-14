/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
** ****************************************************************** */

// Written: gaaraujo
// Created: 10/2026
//
// Description: MUMPS backend for CsrMatrix: sequential MUMPS on MPI_COMM_SELF (the process that
// owns the operator, rank 0 under OpenSeesMP), centralized assembled input, unsymmetric, with the
// ordering and workspace defaults of `system Mumps` (ICNTL(7) = 7, ICNTL(14) = 20). Several
// right-hand sides are solved in one call. Compiled only into executables that link MUMPS; the
// interpreter calls OPS_RegisterMumpsCsrBackend() before creating a SparseCSR system.

#include "CsrDirectBackend.h"
#include "CsrMatrix.h"

#include <OPS_Globals.h>
#include <mpi.h>
#include <dmumps_c.h>
#include <cstring>
#include <vector>

#define ICNTL(I) icntl[(I)-1] /* indices as in the MUMPS documentation */

namespace {

class MumpsCsrBackend : public CsrDirectBackend
{
public:
    MumpsCsrBackend()
    {
        std::memset(&m_id, 0, sizeof(m_id));
        m_id.job = -1;
        m_id.par = 1;
        m_id.sym = 0;
        m_id.comm_fortran = (MUMPS_INT)MPI_Comm_c2f(MPI_COMM_SELF);
        dmumps_c(&m_id);
        m_init = (m_id.infog[0] == 0);
        if (!m_init)
            opserr << "WARNING CsrMatrix (Mumps) - initialization returns " << m_id.infog[0] << "\n";
        setControls();
    }

    ~MumpsCsrBackend() override
    {
        if (m_init) {
            m_id.job = -2;
            dmumps_c(&m_id);
        }
    }

    int analyze(int n, int nnz, const int *rowPtr, const int *colIdx) override
    {
        if (!m_init)
            return -1;
        m_n = n;
        m_irn.resize(static_cast<std::size_t>(nnz));
        m_jcn.resize(static_cast<std::size_t>(nnz));
        m_a.assign(static_cast<std::size_t>(nnz), 0.0);
        for (int r = 0; r < n; ++r) {
            for (int k = rowPtr[r]; k < rowPtr[r + 1]; ++k) {
                m_irn[k] = r + 1;  // Fortran indexing
                m_jcn[k] = colIdx[k] + 1;
            }
        }
        m_analyzed = false;  // the analysis runs with the first values (value-based permutations)
        return 0;
    }

    int factorize(const double *values) override
    {
        if (!m_init)
            return -1;
        std::memcpy(m_a.data(), values, m_a.size() * sizeof(double));
        m_id.n = m_n;
        m_id.nnz = static_cast<MUMPS_INT8>(m_a.size());
        m_id.irn = m_irn.data();
        m_id.jcn = m_jcn.data();
        m_id.a = m_a.data();
        if (!m_analyzed) {
            m_id.job = 1;
            dmumps_c(&m_id);
            if (report("analysis") != 0)
                return -1;
            m_analyzed = true;
        }
        m_id.job = 2;
        dmumps_c(&m_id);
        // Grow the workspace and retry, as MumpsSolver does.
        for (int tries = 0; tries < 3 && (m_id.infog[0] == -8 || m_id.infog[0] == -9); ++tries) {
            m_id.ICNTL(14) += 20;
            m_id.job = 2;
            dmumps_c(&m_id);
        }
        return report("factorization");
    }

    int solve(const double *b, double *x) override
    {
        return solveMulti(b, x, m_n, 1);
    }

    int solveMulti(const double *b, double *x, int n, int numRhs) override
    {
        if (!m_init || !m_analyzed)
            return -1;
        // MUMPS overwrites the centralized RHS with the solution.
        if (x != b)
            std::memcpy(x, b, static_cast<std::size_t>(n) * numRhs * sizeof(double));
        m_id.rhs = x;
        m_id.nrhs = numRhs;
        m_id.lrhs = n;
        m_id.job = 3;
        dmumps_c(&m_id);
        return report("solve");
    }

private:
    void setControls()
    {
        m_id.ICNTL(1) = -1;  // no output
        m_id.ICNTL(2) = -1;
        m_id.ICNTL(3) = -1;
        m_id.ICNTL(4) = 0;
        m_id.ICNTL(5) = 0;   // assembled input
        m_id.ICNTL(18) = 0;  // centralized on the host
        m_id.ICNTL(7) = 7;   // automatic ordering
        m_id.ICNTL(14) = 20;
    }

    int report(const char *phase) const
    {
        if (m_id.infog[0] < 0) {
            opserr << "WARNING CsrMatrix (Mumps) - " << phase << " returns INFOG(1) = " << m_id.infog[0]
                   << ", INFOG(2) = " << m_id.infog[1] << "\n";
            return -1;
        }
        return 0;
    }

    DMUMPS_STRUC_C m_id;
    bool m_init = false;
    bool m_analyzed = false;
    int m_n = 0;
    std::vector<MUMPS_INT> m_irn, m_jcn;
    std::vector<double> m_a;
};

CsrDirectBackend *createMumpsCsrBackend()
{
    return new MumpsCsrBackend();
}

} // namespace

void OPS_RegisterMumpsCsrBackend(void)
{
    CsrMatrix::registerMumpsBackend(&createMumpsCsrBackend);
}
