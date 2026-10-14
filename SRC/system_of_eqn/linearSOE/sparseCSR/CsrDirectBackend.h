/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
** ****************************************************************** */

// Written: gaaraujo
// Created: 10/2026
//
// Description: direct factorization of one CSR operator for CsrMatrix. UMFPACK and the profile
// LDL^T live in CsrMatrix.cpp; MUMPS (MumpsCsrBackend.cpp) is compiled only into executables that
// link MUMPS and registers itself with CsrMatrix::registerMumpsBackend.

#ifndef CsrDirectBackend_h
#define CsrDirectBackend_h

class CsrDirectBackend
{
public:
    virtual ~CsrDirectBackend() = default;
    /** Pattern-only setup (index conversion, ordering or symbolic analysis where pattern-only). */
    virtual int analyze(int n, int nnz, const int *rowPtr, const int *colIdx) = 0;
    /** Numeric factorization of the values on the analyzed pattern (CSR order). */
    virtual int factorize(const double *values) = 0;
    /** x = A^{-1} b, b and x distinct. */
    virtual int solve(const double *b, double *x) = 0;
    /** numRhs right-hand sides stored one after the other; default: one solve per column. */
    virtual int solveMulti(const double *b, double *x, int n, int numRhs)
    {
        for (int c = 0; c < numRhs; ++c)
            if (solve(b + static_cast<long long>(c) * n, x + static_cast<long long>(c) * n) != 0)
                return -1;
        return 0;
    }
};

#endif
