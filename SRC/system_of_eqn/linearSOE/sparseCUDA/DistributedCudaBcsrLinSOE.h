/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
** ****************************************************************** */

// Written: gaaraujo
// Created: 07/2026
//
// Description: Gather-to-root distributed wrapper around CudaBcsrLinSOE for
// OpenSeesMP. Workers assemble sparse (row,col,val) triplets and a local RHS;
// rank 0 holds the global CSR/A, merges B and triplets, solves with the
// attached CudaBcsrLinSolver (cuDSS/AmgX/...), and broadcasts X, merged B,
// and status. Workers never invoke the GPU solver.

#ifndef DistributedCudaBcsrLinSOE_h
#define DistributedCudaBcsrLinSOE_h

#include <LinearSOE.h>
#include <Vector.h>

#include <cstdint>
#include <unordered_map>

#ifndef _CUDA
#error "DistributedCudaBcsrLinSOE requires a CUDA build"
#endif

class CudaBcsrLinSOE;
class Channel;
class FEM_ObjectBroker;
class Matrix;
class ID;

class DistributedCudaBcsrLinSOE : public LinearSOE
{
public:
    /** Takes ownership of theCudaSOE (and its solver). */
    explicit DistributedCudaBcsrLinSOE(CudaBcsrLinSOE *theCudaSOE);
    DistributedCudaBcsrLinSOE();
    ~DistributedCudaBcsrLinSOE();

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
    int saveSparseA(OPS_Stream &output, int baseIndex = 0) override;

    int sendSelf(int commitTag, Channel &theChannel) override;
    int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker) override;

    int setProcessID(int processTag);
    int setChannels(int numChannels, Channel **theChannels);
    int getProcessID(void) const { return processID; }

    CudaBcsrLinSOE *getCudaBcsrLinSOE(void) { return theCudaSOE; }

    /** Gather worker triplets into rank-0 host A (no solve). Safe to call repeatedly
     *  after workers' maps have been cleared. Leaves MatrixStatus unchanged so a later
     *  solve still refactorizes. */
    int ensureMergedA(void);

    /** Collective: P0 sends \a v, workers overwrite \a v with the received values. */
    int broadcastFromRoot(Vector &v);

    /** Collective: fill rank-0 entries of \a v that are still 0 with non-zero values
     *  from workers. Used so the GPU predictor on P0 sees the full committed motion
     *  after domainChanged (each rank only seeds its local DOFs into the integrator
     *  vectors). Workers' \a v is unchanged. */
    int gatherOwnedToRoot(Vector &v);

    /** Collective merge of local myB into rank-0 host B (same protocol as getB/solve). */
    int mergeBToRoot(void);

    /** Mark that rank 0 already holds the global RHS on the GPU device.
     *  When set on all ranks before solve(), DistCuDSS skips the host myB gather
     *  and the post-solve B broadcast (CudaMKR-style). Default Newton path leaves
     *  this false. Cleared at the end of solve(), at the start of CudaMKR newStep,
     *  and on formUnbalance failure (all ranks must agree before the next solve). */
    void setGlobalRhsOnDevice(bool ready) { globalRhsOnDevice = ready; }
    bool isGlobalRhsOnDevice(void) const { return globalRhsOnDevice; }

protected:

private:
    static uint64_t packRC(int row, int col);
    bool needsMatrixTransfer(void) const;
    void ensureMyBSize(int size);
    void ensureWorkArea(int size);
    void clearTriplets(void);
    int accumulateTriplet(int row, int col, double value);
    int addAWorker(const Matrix &m, const ID &id, double fact);
    int buildMergeIndexMap(void);
    int mergeTripletsIntoA(const int *rows, const int *cols, const double *vals, int nTrip);
    /** Worker side of ensureMergedA / formAp triplet transfer. */
    int sendPendingTriplets(Channel *theChannel);
    /** Rank-0 side: recv+merge from all workers. */
    int recvAndMergeTriplets(void);

    CudaBcsrLinSOE *theCudaSOE;

    int processID;
    int numChannels;
    Channel **theChannels;

    double *workArea;
    int sizeWork;

    /** myB holds myBsize entries plus one header slot, so solve() sends the RHS and the
     *  worker's triplet count in one message (myVectBHeader views all myBsize+1 entries). */
    double *myB;
    Vector *myVectB;
    Vector *myVectBHeader;
    int myBsize;

    /** Worker: assembled sparse contributions keyed by (row,col). */
    std::unordered_map<uint64_t, double> tripletMap;

    /** Rank 0: (row,col) -> flat offset into host A values; rebuilt in setSize. */
    std::unordered_map<uint64_t, int> mergeIndexMap;

    bool globalRhsOnDevice;
};

#endif
