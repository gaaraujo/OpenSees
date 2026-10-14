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
    int accumulateTriplet(std::unordered_map<uint64_t, double> &map, int row, int col, double value);
    int addMatrixWorker(std::unordered_map<uint64_t, double> &map, const Matrix &m, const ID &id, double fact);
    int buildMergeIndexMap(void);
    int mergeTripletsInto(std::vector<double> &values, const int *rows, const int *cols,
                          const double *vals, int nTrip);
    int resizeWorkerSOE(int numEqn);
    int getInnerNumEqn(void) const;
    int getInnerMatrixStatus(void) const;
    void setInnerMatrixStatus(int status);
    void clearKTriplets(void) { kTripletMap.clear(); }
    void clearMTriplets(void) { mTripletMap.clear(); }
    void ensureEigenStorage(int numModes, int numEqn);
    int sendTripletMap(Channel *theChannel, const std::unordered_map<uint64_t, double> &map);
    int recvAndMergeTripletMap(Channel *theChannel, std::vector<double> &values);

    EigenSOE *thePythonSOE;

    int processID;
    int numChannels;
    Channel **theChannels;

    int size;

    std::unordered_map<uint64_t, double> kTripletMap;
    std::unordered_map<uint64_t, double> mTripletMap;
    std::unordered_map<uint64_t, int> mergeIndexMap;

    // Cached eigenpairs (rank 0 copies from inner; workers receive broadcast)
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors; // mode-major: mode * numEqn + eqn
    std::vector<Vector> eigenvectorViews;
    int numStoredModes;
    Vector emptyVector;
};

#endif
