/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
** ****************************************************************** */

// Written: gaaraujo
// Created: 10/2026
//
// Description: OpenSeesMP gather-to-root system, the CPU counterpart of DistributedCudaBcsrLinSOE:
// rank 0 owns a SparseCsrLinSOE with the global CSR and solves it (UMFPACK, ProfileSPD, or MUMPS); workers
// accumulate their element contributions as (row, col) triplets and a local RHS. At solve() each
// worker sends its RHS with a one-number header (-1 = matrix unchanged, otherwise the number of
// triplets that follow), then the triplets; rank 0 merges, solves, and sends x (and b) back.
//
//   system DistributedSparseCSR <-solver Umfpack|ProfileSPD|Mumps> <-minDiagTol tol> <-solverRanks all|root>
//   (UMFPACK and ProfileSPD solve on rank 0; MUMPS on all ranks, as system Mumps, unless -solverRanks root)

#ifndef DistributedSparseCsrLinSOE_h
#define DistributedSparseCsrLinSOE_h

#include <LinearSOE.h>
#include <SparseAssemblyCache.h>
#include <SparseDofExchange.h>
#include <cstdint>
#include <unordered_map>

class SparseCsrLinSOE;
class Channel;
class FEM_ObjectBroker;
class Matrix;
class ID;
class Vector;

class DistributedSparseCsrLinSOE : public LinearSOE
{
public:
    /** Takes ownership of theSOE. */
    explicit DistributedSparseCsrLinSOE(SparseCsrLinSOE *theSOE);
    ~DistributedSparseCsrLinSOE() override;

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

    int sendSelf(int commitTag, Channel &theChannel) override;
    int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker) override;

    int setProcessID(int processTag);
    int setChannels(int numChannels, Channel **theChannels);
    int getProcessID(void) const { return processID; }

    SparseCsrLinSOE *getSparseCsrLinSOE(void) { return theCsrSOE; }

    /** Gather worker triplets into rank-0 A (no solve); leaves the matrix status unchanged. */
    int ensureMergedA(void);
    /** Collective: rank 0 sends v, workers overwrite v. */
    int broadcastFromRoot(Vector &v);
    /** Collective: fill rank-0 entries of v that are still 0 with non-zero worker values. */
    int gatherOwnedToRoot(Vector &v);
    /** Collective: sum the local RHS of every rank into rank 0's B. */
    int mergeBToRoot(void);

    // Collective: rank 0 sends the entries of the nv vectors at each worker's DOFs (one message per
    // worker); workers overwrite those entries only (an explicit integrator's motion).
    int scatterLocalFromRoot(Vector *const *v, int nv);

    // Set on all ranks before solve(): workers receive X (and B) only at their own DOFs (Linear
    // algorithm, no convergence test reading X). Cleared at the end of solve().
    void setLocalSolution(bool local) { localSolution = local; }

    /** Rank 0 already holds the global RHS in its B (set on every rank before solve()): solve()
     *  skips the RHS merge and the B broadcast (SparseExplicitAlpha). Cleared at the end of solve(). */
    void setGlobalRhsReady(bool ready) { globalRhsReady = ready; }
    bool isGlobalRhsReady(void) const { return globalRhsReady; }

private:
    static uint64_t packRC(int row, int col);
    bool needsMatrixTransfer(void) const;
    void ensureMyBSize(int size);
    void ensureWorkArea(int size);
    int addAWorker(const Matrix &m, const ID &id, double fact);
    int buildMergeIndexMap(void);
    int recvAndMergeTriplets(Channel *theChannel, int worker, int header, bool merge);

    SparseCsrLinSOE *theCsrSOE;

    int processID;
    int numChannels;
    Channel **theChannels;

    double *workArea;
    int sizeWork;

    /** myBsize entries plus one header slot, so solve() sends the RHS and the triplet count in one
     *  message (myVectBHeader views all myBsize+1 entries). */
    double *myB;
    Vector *myVectB;
    Vector *myVectBHeader;
    int myBsize;

    SparseLocalTriplets localTriplets;                // worker: assembled (row, col) contributions
    std::unordered_map<uint64_t, int> mergeIndexMap;  // rank 0: (row, col) -> offset in A values
    SparseRootScatter rootScatter;                    // rank 0: per worker, slot -> offset in A values

    bool globalRhsReady;
    SparseDofExchange dofExchange;  // rank-local DOF lists (setSize): packed RHS gather, local scatters
    bool myBDense;                  // worker: myB set as a whole by setB() since the last zeroB()
    bool localSolution;
    bool lastSolveLocal;            // worker: X outside the own DOFs is stale after a local solve
};

#endif
