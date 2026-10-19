/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
** ****************************************************************** */

// Written: gaaraujo
// Created: 10/2026
//
// Description: MUMPS backend for CsrMatrix. Distributed (DistributedSparseCSR): MUMPS runs on every
// rank of the OpenSeesMP job, on a duplicate of MPI_COMM_WORLD, so the factorization and the solves
// are parallel as in `system Mumps`; the matrix and the right-hand sides are centralized on rank 0
// (the MUMPS host, which owns the operator), and the other ranks join each call with no data.
// Otherwise (SparseCSR) MUMPS runs on MPI_COMM_SELF. Unsymmetric, with the ordering and workspace
// defaults of `system Mumps` (ICNTL(7) = 7, ICNTL(14) = 20, ICNTL(23) = 0); several right-hand sides in one
// call. OPS_MUMPS_CSR_ICNTL14 / OPS_MUMPS_CSR_ICNTL23 / OPS_MUMPS_CSR_ICNTL24 override the workspace
// relaxation / working memory / null-pivot detection; OPS_MUMPS_CSR_REPORT prints factorizations and solves.
// Compiled only into executables that link MUMPS; the interpreter calls OPS_RegisterMumpsCsrBackend()
// before creating a SparseCSR system.

#include "CsrDirectBackend.h"
#include "CsrMatrix.h"

#include <OPS_Globals.h>
#include <mpi.h>
#include <dmumps_c.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ICNTL(I) icntl[(I)-1] /* indices as in the MUMPS documentation */

namespace {

class MumpsCsrBackend : public CsrDirectBackend
{
public:
    explicit MumpsCsrBackend(bool distributed)
    {
        // Collective over the ranks of the communicator: every rank creates its backend at the same
        // point (CsrMatrix creates it lazily in factorize(), which all ranks call in the same order).
        static int s_count = 0;  // creation order, to tell the operators apart in the diagnostic output
        m_tag = s_count++;
        MPI_Comm_dup(distributed ? MPI_COMM_WORLD : MPI_COMM_SELF, &m_comm);
        MPI_Comm_rank(m_comm, &m_rank);
        std::memset(&m_id, 0, sizeof(m_id));
        m_id.job = -1;
        m_id.par = 1;  // the host (rank 0) takes part in the factorization
        m_id.sym = 0;
        m_id.comm_fortran = (MUMPS_INT)MPI_Comm_c2f(m_comm);
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
        MPI_Comm_free(&m_comm);
    }

    int analyze(int n, int nnz, const int *rowPtr, const int *colIdx) override
    {
        if (!m_init)
            return -1;
        m_n = n;
        if (!isHost())
            return 0;  // centralized input: only the host holds the pattern
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
        if (isHost()) {
            std::memcpy(m_a.data(), values, m_a.size() * sizeof(double));
            m_id.n = m_n;
            m_id.nnz = static_cast<MUMPS_INT8>(m_a.size());
            m_id.irn = m_irn.data();
            m_id.jcn = m_jcn.data();
            m_id.a = m_a.data();
        }
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
        reportMemory();
        reportFactorization();
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
        // MUMPS overwrites the centralized RHS (on the host) with the solution.
        if (isHost()) {
            if (x != b)
                std::memcpy(x, b, static_cast<std::size_t>(n) * numRhs * sizeof(double));
            m_id.rhs = x;
            m_id.nrhs = numRhs;
            m_id.lrhs = n;
        }
        double bNorm = 0.0;
        if (m_report && isHost() && m_solveReports < 12)
            bNorm = maxAbs(x, static_cast<std::size_t>(n) * numRhs, nullptr);
        m_id.job = 3;
        dmumps_c(&m_id);
        if (m_report && isHost() && m_solveReports < 12) {
            ++m_solveReports;
            int nonFinite = 0;
            const double xNorm = maxAbs(x, static_cast<std::size_t>(n) * numRhs, &nonFinite);
            opserr << "CsrMatrix (Mumps) #" << m_tag << " solve " << m_solveReports << ": nrhs " << numRhs
                   << ", max|b| " << bNorm << ", max|x| " << xNorm << ", non-finite x " << nonFinite
                   << ", INFOG(1) " << m_id.infog[0] << "\n";
        }
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
        // Starting workspace relaxation (percent); OPS_MUMPS_CSR_ICNTL14 overrides the default 20. Read on
        // every rank (setControls runs in each rank's constructor): the retry loop in factorize() reads the
        // global INFOG, so all ranks must start from the same ICNTL(14).
        const char *icntl14 = std::getenv("OPS_MUMPS_CSR_ICNTL14");
        m_id.ICNTL(14) = icntl14 ? std::atoi(icntl14) : 20;
        // Working memory per process in MB (0 = MUMPS's own estimate, the default); OPS_MUMPS_CSR_ICNTL23
        // overrides it, read on every rank like ICNTL(14).
        const char *icntl23 = std::getenv("OPS_MUMPS_CSR_ICNTL23");
        m_id.ICNTL(23) = icntl23 ? std::atoi(icntl23) : 0;
        m_reportMemory = icntl23 != nullptr;
        // Null-pivot detection (MUMPS ICNTL(24), default 0 = off); OPS_MUMPS_CSR_ICNTL24 sets it on every rank.
        // With it, or with OPS_MUMPS_CSR_REPORT set, the host prints each factorization (INFOG(1), (2), (28)
        // null pivots, matrix diagonal) and its first solves (max |b|, max |x|, non-finite entries of x).
        const char *icntl24 = std::getenv("OPS_MUMPS_CSR_ICNTL24");
        if (icntl24)
            m_id.ICNTL(24) = std::atoi(icntl24);
        m_report = icntl24 != nullptr || std::getenv("OPS_MUMPS_CSR_REPORT") != nullptr;
    }

    static double maxAbs(const double *v, std::size_t n, int *nonFinite)
    {
        double m = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(v[i])) {
                if (nonFinite)
                    ++*nonFinite;
                continue;
            }
            m = std::max(m, std::fabs(v[i]));
        }
        return m;
    }

    void reportFactorization() const
    {
        if (!m_report || !isHost())
            return;
        double aMax = 0.0, dMin = HUGE_VAL;
        int zeroDiag = 0, nonFinite = 0, nDiag = 0;
        for (std::size_t k = 0; k < m_a.size(); ++k) {
            if (!std::isfinite(m_a[k]))
                ++nonFinite;
            else
                aMax = std::max(aMax, std::fabs(m_a[k]));
            if (m_irn[k] == m_jcn[k]) {
                ++nDiag;
                dMin = std::min(dMin, std::fabs(m_a[k]));
                if (m_a[k] == 0.0)
                    ++zeroDiag;
            }
        }
        opserr << "CsrMatrix (Mumps) #" << m_tag << " factorization: n " << m_n << ", nnz " << m_a.size()
               << ", max|a| " << aMax << ", stored diagonal " << nDiag << " (min |d| " << dMin << ", zero "
               << zeroDiag << "), non-finite values " << nonFinite << "; ICNTL(14) " << m_id.ICNTL(14)
               << ", ICNTL(24) " << m_id.ICNTL(24) << "; INFOG(1) " << m_id.infog[0] << ", INFOG(2) "
               << m_id.infog[1] << ", INFOG(28) (null pivots) " << m_id.infog[27] << ", RINFOG(3) (flops) "
               << m_id.rinfog[2] << "\n";
    }

    // With OPS_MUMPS_CSR_ICNTL23 set, the host prints the settings and the memory the factorization used.
    void reportMemory() const
    {
        if (!m_reportMemory || !isHost())
            return;
        opserr << "CsrMatrix (Mumps) - ICNTL(14) = " << m_id.ICNTL(14) << ", ICNTL(23) = " << m_id.ICNTL(23)
               << " MB; INFOG(1) = " << m_id.infog[0] << "; MB, max per process / total: estimated "
               << m_id.infog[15] << " / " << m_id.infog[16] << " (INFOG(16)/(17)), allocated " << m_id.infog[17]
               << " / " << m_id.infog[18] << " (INFOG(18)/(19)), used " << m_id.infog[20] << " / "
               << m_id.infog[21] << " (INFOG(21)/(22))\n";
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

    bool isHost() const { return m_rank == 0; }

    MPI_Comm m_comm = MPI_COMM_NULL;
    int m_rank = 0;
    DMUMPS_STRUC_C m_id;
    bool m_init = false;
    bool m_analyzed = false;
    bool m_reportMemory = false;
    bool m_report = false;
    int m_tag = 0;
    int m_solveReports = 0;
    int m_n = 0;
    std::vector<MUMPS_INT> m_irn, m_jcn;
    std::vector<double> m_a;
};

CsrDirectBackend *createMumpsCsrBackend(bool distributed)
{
    return new MumpsCsrBackend(distributed);
}

} // namespace

void OPS_RegisterMumpsCsrBackend(void)
{
    CsrMatrix::registerMumpsBackend(&createMumpsCsrBackend);
}
