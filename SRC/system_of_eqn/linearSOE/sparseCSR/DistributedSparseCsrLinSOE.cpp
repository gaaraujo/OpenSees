/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
** ****************************************************************** */

// Written: gaaraujo
// Created: 10/2026
//
// DistributedSparseCsrLinSOE: gather-to-root with worker-side triplets (CPU counterpart of
// DistributedCudaBcsrLinSOE, same message protocol).

#include "DistributedSparseCsrLinSOE.h"
#include "SparseCsrLinSOE.h"

#include <Matrix.h>
#include <Graph.h>
#include <Vertex.h>
#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include <ID.h>
#include <Vector.h>
#include <classTags.h>
#include <OPS_Globals.h>

bool OPS_ParseSparseCsrOptions(CsrMatrix::Solver &solver, double &tol, bool &solverOnRoot);

void *OPS_DistributedSparseCsrLinSOE(void)
{
    CsrMatrix::Solver solver;
    double tol;
    bool solverOnRoot;
    if (!OPS_ParseSparseCsrOptions(solver, tol, solverOnRoot))
        return nullptr;
    SparseCsrLinSOE *csr = new SparseCsrLinSOE(solver, tol);
    csr->setSolverOnRoot(solverOnRoot);
    return new DistributedSparseCsrLinSOE(csr);
}

uint64_t
DistributedSparseCsrLinSOE::packRC(int row, int col)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(col));
}

DistributedSparseCsrLinSOE::DistributedSparseCsrLinSOE(SparseCsrLinSOE *theSOE)
    : LinearSOE(LinSOE_TAGS_DistributedSparseCsrLinSOE),
      theCsrSOE(theSOE),
      processID(0), numChannels(0), theChannels(0),
      workArea(0), sizeWork(0),
      myB(0), myVectB(0), myVectBHeader(0), myBsize(0),
      globalRhsReady(false)
{
}

DistributedSparseCsrLinSOE::~DistributedSparseCsrLinSOE()
{
    delete[] theChannels;
    delete[] workArea;
    delete[] myB;
    delete myVectB;
    delete myVectBHeader;
    delete theCsrSOE;
}

bool
DistributedSparseCsrLinSOE::needsMatrixTransfer(void) const
{
    return theCsrSOE == nullptr ||
           theCsrSOE->getMatrixStatus() != SparseCsrLinSOE::MatrixStatus::UNCHANGED;
}

void
DistributedSparseCsrLinSOE::ensureMyBSize(int size)
{
    if (size <= myBsize && myB != 0)
        return;
    delete[] myB;
    delete myVectB;
    delete myVectBHeader;
    // One extra slot: the header that solve() appends to the RHS message.
    myB = new double[size + 1];
    myVectB = new Vector(myB, size);
    myVectBHeader = new Vector(myB, size + 1);
    myBsize = size;
    for (int i = 0; i <= size; ++i)
        myB[i] = 0.0;
}

void
DistributedSparseCsrLinSOE::ensureWorkArea(int size)
{
    if (size <= sizeWork && workArea != 0)
        return;
    delete[] workArea;
    workArea = new double[size];
    sizeWork = size;
}

int
DistributedSparseCsrLinSOE::addAWorker(const Matrix &m, const ID &id, double fact)
{
    if (fact == 0.0)
        return 0;
    const int idSize = id.Size();
    if (idSize != m.noRows() && idSize != m.noCols()) {
        opserr << "WARNING DistributedSparseCsrLinSOE::addA() - Matrix and ID not of similar sizes\n";
        return -1;
    }
    localTriplets.add(m, id, fact, myBsize);
    if (theCsrSOE->getMatrixStatus() == SparseCsrLinSOE::MatrixStatus::UNCHANGED)
        theCsrSOE->setMatrixStatus(SparseCsrLinSOE::MatrixStatus::COEFFICIENTS_CHANGED);
    return 0;
}

int
DistributedSparseCsrLinSOE::buildMergeIndexMap(void)
{
    mergeIndexMap.clear();
    const int n = theCsrSOE->getNumEqn();
    const int *rowPtr = theCsrSOE->getRowPtr();
    const int *colIdx = theCsrSOE->getColIndices();
    if (n > 0 && (rowPtr == nullptr || colIdx == nullptr))
        return -1;
    for (int row = 0; row < n; ++row)
        for (int k = rowPtr[row]; k < rowPtr[row + 1]; ++k)
            mergeIndexMap[packRC(row, colIdx[k])] = k;
    return 0;
}

// Rank 0: receive what one worker announced with header (SparseLocalTriplets: 0 nothing,
// n > 0 full triplets, -(n + 2) values only); merge unless merge is false (drain only).
int
DistributedSparseCsrLinSOE::recvAndMergeTriplets(Channel *theChannel, int worker, int header, bool merge)
{
    auto offsetOf = [this](int row, int col) -> int {
        const auto it = mergeIndexMap.find(packRC(row, col));
        return it == mergeIndexMap.end() ? -1 : it->second;
    };
    return rootScatter.recvAndMerge(*theChannel, worker, header, merge, theCsrSOE->getAValuesData(), offsetOf);
}

int
DistributedSparseCsrLinSOE::getNumEqn(void) const
{
    return theCsrSOE != nullptr ? theCsrSOE->getNumEqn() : 0;
}

int
DistributedSparseCsrLinSOE::setSize(Graph &theGraph)
{
    if (theCsrSOE == nullptr)
        return -1;
    localTriplets.reset();
    rootScatter.reset(numChannels);
    mergeIndexMap.clear();

    if (processID != 0) {
        if (numChannels < 1 || theChannels == 0 || theChannels[0] == 0) {
            opserr << "WARNING DistributedSparseCsrLinSOE::setSize() - worker has no channel\n";
            return -1;
        }
        Channel *theChannel = theChannels[0];
        theGraph.sendSelf(0, *theChannel);
        ID data(2);
        theChannel->recvID(0, 0, data);
        if (data(1) < 0) {
            opserr << "WARNING DistributedSparseCsrLinSOE::setSize() - rank 0 reported failure\n";
            return data(1);
        }
        if (theCsrSOE->resizeHostVectors(data(0)) < 0)
            return -1;
        ensureMyBSize(data(0));
        return 0;
    }

    FEM_ObjectBroker theBroker;
    for (int j = 0; j < numChannels; ++j) {
        Graph theSubGraph;
        theSubGraph.recvSelf(0, *theChannels[j], theBroker);
        theGraph.merge(theSubGraph);
    }
    const int result = theCsrSOE->setSize(theGraph);
    ID data(2);
    data(0) = theCsrSOE->getNumEqn();
    data(1) = result;
    for (int j = 0; j < numChannels; ++j)
        theChannels[j]->sendID(0, 0, data);
    if (result < 0)
        return result;
    if (buildMergeIndexMap() < 0)
        return -1;
    ensureMyBSize(data(0));
    ensureWorkArea(data(0) + 1);
    return 0;
}

int
DistributedSparseCsrLinSOE::addA(const Matrix &m, const ID &id, double fact)
{
    if (processID != 0)
        return addAWorker(m, id, fact);
    return theCsrSOE->addA(m, id, fact);
}

int
DistributedSparseCsrLinSOE::addA(const Matrix &m)
{
    if (processID != 0) {
        const int n = m.noRows();
        ID id(n);
        for (int i = 0; i < n; ++i)
            id(i) = i;
        return addAWorker(m, id, 1.0);
    }
    return theCsrSOE->addA(m);
}

int
DistributedSparseCsrLinSOE::addB(const Vector &v, const ID &id, double fact)
{
    if (fact == 0.0)
        return 0;
    const int idSize = id.Size();
    if (idSize != v.Size()) {
        opserr << "WARNING DistributedSparseCsrLinSOE::addB() - Vector and ID size mismatch\n";
        return -1;
    }
    for (int i = 0; i < idSize; ++i) {
        const int pos = id(i);
        if (pos >= 0 && pos < myBsize)
            myB[pos] += (fact == 1.0) ? v(i) : fact * v(i);
    }
    return 0;
}

int
DistributedSparseCsrLinSOE::setB(const Vector &v, double fact)
{
    if (v.Size() != myBsize) {
        opserr << "WARNING DistributedSparseCsrLinSOE::setB() - incompatible sizes (global " << myBsize
               << ", vector " << v.Size() << ")\n";
        return -1;
    }
    for (int i = 0; i < myBsize; ++i)
        myB[i] = fact * v(i);
    return 0;
}

void
DistributedSparseCsrLinSOE::zeroA(void)
{
    localTriplets.zero();
    if (theCsrSOE != nullptr)
        theCsrSOE->zeroA();
}

void
DistributedSparseCsrLinSOE::zeroB(void)
{
    for (int i = 0; i < myBsize; ++i)
        myB[i] = 0.0;
}

const Vector &
DistributedSparseCsrLinSOE::getX(void)
{
    return theCsrSOE->getX();
}

const Vector &
DistributedSparseCsrLinSOE::getB(void)
{
    Vector &vectB = theCsrSOE->getHostBVector();
    if (processID != 0) {
        theChannels[0]->sendVector(0, 0, *myVectB);
        theChannels[0]->recvVector(0, 0, vectB);
        return vectB;
    }
    vectB = *myVectB;
    ensureWorkArea(myBsize + 1);
    Vector remoteB(workArea, myBsize);
    for (int j = 0; j < numChannels; ++j) {
        theChannels[j]->recvVector(0, 0, remoteB);
        vectB += remoteB;
    }
    for (int j = 0; j < numChannels; ++j)
        theChannels[j]->sendVector(0, 0, vectB);
    return vectB;
}

double
DistributedSparseCsrLinSOE::normRHS(void)
{
    return this->getB().Norm();
}

void
DistributedSparseCsrLinSOE::setX(int loc, double value)
{
    theCsrSOE->setX(loc, value);
}

void
DistributedSparseCsrLinSOE::setX(const Vector &x)
{
    theCsrSOE->setX(x);
}

int
DistributedSparseCsrLinSOE::solve(void)
{
    ID result(1);
    const bool sendA = needsMatrixTransfer();
    // SparseExplicitAlpha may already hold the transformed global RHS on rank 0 (setGlobalRhsReady
    // on every rank), so the RHS merge and the B broadcast are skipped.
    const bool skipBMerge = globalRhsReady;
    Vector &vectX = theCsrSOE->getHostXVector();
    Vector &vectB = theCsrSOE->getHostBVector();

    if (processID != 0) {
        Channel *theChannel = theChannels[0];
        // Triplet count of SparseLocalTriplets::header(): n >= 0 full triplets, or -(n + 2) values
        // only once rank 0 holds this rank's pattern.
        const int nTrip = sendA ? localTriplets.header() : 0;
        if (!skipBMerge) {
            // Header after the RHS: -1 if the matrix is unchanged on this rank, else the number of
            // triplets that follow. Rank 0 reads it instead of its own status.
            myB[myBsize] = sendA ? static_cast<double>(nTrip) : -1.0;
            theChannel->sendVector(0, 0, *myVectBHeader);
        } else if (sendA) {
            ID nTripID(1);
            nTripID(0) = nTrip;
            theChannel->sendID(0, 0, nTripID);
        }
        if (sendA && localTriplets.send(*theChannel) < 0)
            return -1;
        // Distributed MUMPS: join rank 0's factorization and solve of A.
        if (theCsrSOE->isCollectiveSolver())
            theCsrSOE->solve();

        theChannel->recvVector(0, 0, vectX);
        if (!skipBMerge)
            theChannel->recvVector(0, 0, vectB);
        theChannel->recvID(0, 0, result);
        if (result(0) == 0)
            theCsrSOE->setMatrixStatus(SparseCsrLinSOE::MatrixStatus::UNCHANGED);
        globalRhsReady = false;
        return result(0);
    }

    result(0) = 0;
    if (!skipBMerge)
        vectB = *myVectB;

    ensureWorkArea(myBsize + 1);
    Vector remoteBHeader(workArea, myBsize + 1);
    Vector remoteB(workArea, myBsize);
    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        int nTrip = 0;
        bool workerSendsA = sendA;
        if (!skipBMerge) {
            theChannel->recvVector(0, 0, remoteBHeader);
            vectB += remoteB;
            const int header = static_cast<int>(workArea[myBsize]);
            workerSendsA = header != -1;
            nTrip = workerSendsA ? header : 0;
        } else if (sendA) {
            ID nTripID(1);
            theChannel->recvID(0, 0, nTripID);
            nTrip = nTripID(0);
        }
        if (workerSendsA != sendA || nTrip == -1) {
            opserr << "WARNING DistributedSparseCsrLinSOE::solve() - matrix status of rank " << j + 1
                   << " differs from rank 0\n";
            result(0) = -1;
        }
        if (recvAndMergeTriplets(theChannel, j, nTrip, result(0) == 0) < 0)
            result(0) = -1;
    }

    if (theCsrSOE->isCollectiveSolver()) {
        // The workers are already inside the collective call: always enter it.
        const int rc = theCsrSOE->solve();
        if (result(0) == 0)
            result(0) = rc;
    } else if (result(0) == 0) {
        result(0) = theCsrSOE->solve();
    }

    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        theChannel->sendVector(0, 0, vectX);
        if (!skipBMerge)
            theChannel->sendVector(0, 0, vectB);
        theChannel->sendID(0, 0, result);
    }
    globalRhsReady = false;
    return result(0);
}

int
DistributedSparseCsrLinSOE::formAp(const Vector &p, Vector &Ap)
{
    if (numChannels <= 0 || theChannels == 0)
        return theCsrSOE->formAp(p, Ap);
    if (ensureMergedA() < 0)
        return -1;
    if (processID != 0)
        return theChannels[0]->recvVector(0, 0, Ap) < 0 ? -1 : 0;
    const int rc = theCsrSOE->formAp(p, Ap);
    if (rc != 0)
        Ap.Zero();
    for (int j = 0; j < numChannels; ++j)
        if (theChannels[j]->sendVector(0, 0, Ap) < 0)
            return -1;
    return rc;
}

int
DistributedSparseCsrLinSOE::ensureMergedA(void)
{
    if (numChannels <= 0 || theChannels == 0)
        return 0;
    if (!needsMatrixTransfer())
        return 0;
    if (processID != 0) {
        ID nTripID(1);
        nTripID(0) = localTriplets.header();
        if (theChannels[0]->sendID(0, 0, nTripID) < 0 || localTriplets.send(*theChannels[0]) < 0)
            return -1;
        localTriplets.clearValues();  // rank 0 now holds these contributions
        return 0;
    }
    int status = 0;
    for (int j = 0; j < numChannels; ++j) {
        ID nTripID(1);
        if (theChannels[j]->recvID(0, 0, nTripID) < 0 || nTripID(0) == -1) {
            status = -1;
            continue;
        }
        if (recvAndMergeTriplets(theChannels[j], j, nTripID(0), status == 0) < 0)
            status = -1;
    }
    return status;
}

int
DistributedSparseCsrLinSOE::broadcastFromRoot(Vector &v)
{
    if (numChannels <= 0 || theChannels == 0)
        return 0;
    if (processID != 0)
        return theChannels[0]->recvVector(0, 0, v);
    for (int j = 0; j < numChannels; ++j)
        if (theChannels[j]->sendVector(0, 0, v) < 0)
            return -1;
    return 0;
}

int
DistributedSparseCsrLinSOE::gatherOwnedToRoot(Vector &v)
{
    if (numChannels <= 0 || theChannels == 0)
        return 0;
    if (processID != 0)
        return theChannels[0]->sendVector(0, 0, v);
    ensureWorkArea(v.Size());
    Vector remote(workArea, v.Size());
    for (int j = 0; j < numChannels; ++j) {
        if (theChannels[j]->recvVector(0, 0, remote) < 0)
            return -1;
        // Each rank seeds only its own DOFs after domainChanged; fill rank 0's holes.
        for (int i = 0; i < v.Size(); ++i)
            if (v(i) == 0.0 && remote(i) != 0.0)
                v(i) = remote(i);
    }
    return 0;
}

int
DistributedSparseCsrLinSOE::mergeBToRoot(void)
{
    Vector &vectB = theCsrSOE->getHostBVector();
    if (numChannels <= 0 || theChannels == 0) {
        vectB = *myVectB;
        return 0;
    }
    if (processID != 0)
        return theChannels[0]->sendVector(0, 0, *myVectB) < 0 ? -1 : 0;
    vectB = *myVectB;
    ensureWorkArea(myBsize + 1);
    Vector remoteB(workArea, myBsize);
    for (int j = 0; j < numChannels; ++j) {
        if (theChannels[j]->recvVector(0, 0, remoteB) < 0)
            return -1;
        vectB += remoteB;
    }
    return 0;
}

int
DistributedSparseCsrLinSOE::sendSelf(int commitTag, Channel &theChannel)
{
    int sendID = processID;
    if (processID == 0) {
        bool found = false;
        for (int i = 0; i < numChannels; ++i)
            if (theChannels[i] == &theChannel) {
                sendID = i + 1;
                found = true;
            }
        if (!found) {
            Channel **next = new Channel *[numChannels + 1];
            for (int i = 0; i < numChannels; ++i)
                next[i] = theChannels[i];
            next[numChannels] = &theChannel;
            delete[] theChannels;
            theChannels = next;
            numChannels++;
            sendID = numChannels;
        }
    }
    ID idData(1);
    idData(0) = sendID;
    return theChannel.sendID(0, commitTag, idData) < 0 ? -1 : 0;
}

int
DistributedSparseCsrLinSOE::recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    (void)theBroker;
    ID idData(1);
    if (theChannel.recvID(0, commitTag, idData) < 0)
        return -1;
    processID = idData(0);
    numChannels = 1;
    delete[] theChannels;
    theChannels = new Channel *[1];
    theChannels[0] = &theChannel;
    return 0;
}

int
DistributedSparseCsrLinSOE::setProcessID(int dTag)
{
    processID = dTag;
    theCsrSOE->setDistributedSolve(numChannels > 0, processID != 0);
    return 0;
}

int
DistributedSparseCsrLinSOE::setChannels(int nChannels, Channel **theC)
{
    delete[] theChannels;
    theChannels = 0;
    numChannels = 0;
    if (nChannels > 0 && theC != 0) {
        numChannels = nChannels;
        theChannels = new Channel *[numChannels];
        for (int i = 0; i < numChannels; ++i)
            theChannels[i] = theC[i];
    }
    theCsrSOE->setDistributedSolve(numChannels > 0, processID != 0);
    return 0;
}
