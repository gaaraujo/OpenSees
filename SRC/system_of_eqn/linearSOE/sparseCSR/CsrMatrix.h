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
// Description: Host CSR matrix operator, the CPU counterpart of CudaCsrMatrix: SpMV plus a direct
// factorization/solve through an interchangeable backend (UMFPACK, the profile LDL^T of
// ProfileSPDLinDirectSolver, or sequential MUMPS where the executable links MUMPS). The structure
// (rowPtr, colIdx) is bound, not copied, so several operators can share one pattern; values are
// either bound or owned (copyValues).

#ifndef CsrMatrix_h
#define CsrMatrix_h

#include <vector>

class CsrDirectBackend;

class CsrMatrix
{
public:
    enum class Solver { Umfpack, ProfileSPD, Mumps };
    using BackendFactory = CsrDirectBackend *(*)();

    /** Called by executables that link MUMPS (OpenSeesMP) before the first `system` command. */
    static void registerMumpsBackend(BackendFactory factory);
    static bool isSolverAvailable(Solver s);

    explicit CsrMatrix(Solver solver = Solver::Umfpack, double profileMinDiagTol = 1.0e-12);
    ~CsrMatrix();

    CsrMatrix(const CsrMatrix &) = delete;
    CsrMatrix &operator=(const CsrMatrix &) = delete;

    /** Drop the structure, the values and the factorization. */
    void reset();

    int bindStructure(int numRows, int numNnz, const int *rowPtr, const int *colIdx);
    int bindValues(double *values);
    int copyValues(const double *values);

    /** y = alpha*A*x + beta*y. */
    int spmv(const double *x, double *y, double alpha = 1.0, double beta = 0.0) const;

    /** Symbolic analysis (first call after bindStructure) and numeric factorization. */
    int factorize();
    /** Numeric factorization only, for new values on the bound structure. */
    int refactorize();
    /** Solve A x = b for numRhs right-hand sides stored one after the other (n each). */
    int solve(const double *rhs, double *solution, int numRhs = 1);

    bool isStructureBound() const { return m_structureBound; }
    bool isFactored() const { return m_factored; }
    int getNumRows() const { return m_numRows; }
    int getNumNnz() const { return m_numNnz; }
    Solver getSolver() const { return m_solver; }
    const double *getValues() const { return m_values; }

    static const char *solverName(Solver s);
    /** Parse "Umfpack" / "ProfileSPD" / "Mumps" (common spellings accepted). */
    static bool solverFromString(const char *s, Solver &out);

private:
    Solver m_solver;
    double m_profileMinDiagTol;

    int m_numRows = 0;
    int m_numNnz = 0;
    bool m_structureBound = false;
    bool m_analyzed = false;
    bool m_factored = false;

    const int *m_rowPtr = nullptr;
    const int *m_colIdx = nullptr;
    double *m_values = nullptr;
    std::vector<double> m_ownedValues;

    CsrDirectBackend *m_backend = nullptr;
    std::vector<double> m_work;  // copy of the right-hand sides when a solve is in place
};

#endif
