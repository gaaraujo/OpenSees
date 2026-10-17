/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
** ****************************************************************** */

// Written: gaaraujo
// Created: 07/2026
//
// Description: Gather-to-root distributed wrapper around SparsePython*EigenSOE
// for OpenSeesMP. Workers assemble K and M (row,col,val) triplets; rank 0 holds
// the global matrices, merges triplets, solves with the attached Python eigen
// solver, and broadcasts status plus eigenpairs. Workers never invoke Python.

#ifndef DistributedSparsePythonEigenSOE_h
#define DistributedSparsePythonEigenSOE_h

#include <EigenSOE.h>
#include <Vector.h>
#include <SparseAssemblyCache.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

class Channel;
class FEM_ObjectBroker;
class Matrix;
class ID;

class DistributedSparsePythonEigenSOE : public EigenSOE
{
public:
    /** Takes ownership of thePythonSOE (and its solver). */
    explicit DistributedSparsePythonEigenSOE(EigenSOE *thePythonSOE);
    DistributedSparsePythonEigenSOE();
    ~DistributedSparsePythonEigenSOE();

    int setSize(Graph &theGraph) override;
    int addA(const Matrix &, const ID &, double fact = 1.0) override;
    int addM(const Matrix &, const ID &, double fact = 1.0) override;
    void zeroA(void) override;
    void zeroM(void) override;

    int solve(int numModes, bool generalized, bool findSmallest = true) override;

    const Vector &getEigenvector(int mode) override;
    double getEigenvalue(int mode) override;

    int sendSelf(int commitTag, Channel &theChannel) override;
    int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker) override;

    int setProcessID(int processTag);
    int setChannels(int numChannels, Channel **theChannels);

    EigenSOE *getPythonEigenSOE(void) { return thePythonSOE; }

private:
    static uint64_t packRC(int row, int col);
    bool needsMatrixTransfer(void) const;
    int addMatrixWorker(SparseLocalTriplets &triplets, const Matrix &m, const ID &id, double fact);
    int buildMergeIndexMap(void);
    int resizeWorkerSOE(int numEqn);
    int getInnerNumEqn(void) const;
    int getInnerMatrixStatus(void) const;
    void setInnerMatrixStatus(int status);
    void clearKTriplets(void) { kTriplets.zero(); }
    void clearMTriplets(void) { mTriplets.zero(); }
    void ensureEigenStorage(int numModes, int numEqn);
    int sendTriplets(Channel *theChannel, SparseLocalTriplets &triplets);
    int recvAndMergeTriplets(Channel *theChannel, SparseRootScatter &scatter, int worker,
                             std::vector<double> *values,
                               bool merge);

    EigenSOE *thePythonSOE;

    int processID;
    int numChannels;
    Channel **theChannels;

    int size;

    /** Worker: K and M contributions as flat (row, col, value) triplets. */
    SparseLocalTriplets kTriplets, mTriplets;
    std::unordered_map<uint64_t, int> mergeIndexMap;
    /** Rank 0: per worker, triplet slot -> flat offset into K / M values. */
    SparseRootScatter kScatter, mScatter;

    // Cached eigenpairs (rank 0 copies from inner; workers receive broadcast)
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors; // mode-major: mode * numEqn + eqn
    std::vector<Vector> eigenvectorViews;
    int numStoredModes;
    Vector emptyVector;
};

#endif
