/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
** ****************************************************************** */

// Written: gaaraujo
// Created: 07/2026
//
// DistributedSparsePythonLinSOE: gather-to-root with worker-side triplets.

#include "DistributedSparsePythonLinSOE.h"
#include "SparsePythonCompressedLinSOE.h"
#include "SparsePythonCOOLinSOE.h"
#include "SparsePythonCommon.h"

#include <Matrix.h>
#include <Graph.h>
#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include <ID.h>
#include <classTags.h>
#include <OPS_Globals.h>

#include <cstring>
#include <vector>

uint64_t
DistributedSparsePythonLinSOE::packRC(int row, int col)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(col));
}

DistributedSparsePythonLinSOE::DistributedSparsePythonLinSOE(LinearSOE *theSOE)
    : LinearSOE(LinSOE_TAGS_DistributedSparsePythonLinSOE),
      thePythonSOE(theSOE),
      processID(0), numChannels(0), theChannels(0),
      workArea(0), sizeWork(0),
      myB(0), myVectB(0), myBsize(0)
{
    if (thePythonSOE == nullptr) {
        opserr << "FATAL DistributedSparsePythonLinSOE - null SparsePython LinSOE\n";
    }
}

DistributedSparsePythonLinSOE::DistributedSparsePythonLinSOE()
    : LinearSOE(LinSOE_TAGS_DistributedSparsePythonLinSOE),
      thePythonSOE(0),
      processID(0), numChannels(0), theChannels(0),
      workArea(0), sizeWork(0),
      myB(0), myVectB(0), myBsize(0)
{
}

DistributedSparsePythonLinSOE::~DistributedSparsePythonLinSOE()
{
    if (theChannels != 0)
        delete[] theChannels;

    if (workArea != 0)
        delete[] workArea;

    if (myB != 0)
        delete[] myB;

    if (myVectB != 0)
        delete myVectB;

    delete thePythonSOE;
}

int
DistributedSparsePythonLinSOE::getInnerMatrixStatus(void) const
{
    if (auto *c = dynamic_cast<SparsePythonCompressedLinSOE *>(thePythonSOE))
        return static_cast<int>(c->getMatrixStatus());
    if (auto *c = dynamic_cast<SparsePythonCOOLinSOE *>(thePythonSOE))
        return static_cast<int>(c->getMatrixStatus());
    return static_cast<int>(SparsePythonMatrixStatus::STRUCTURE_CHANGED);
}

void
DistributedSparsePythonLinSOE::setInnerMatrixStatus(int status)
{
    const auto s = static_cast<SparsePythonMatrixStatus>(status);
    if (auto *c = dynamic_cast<SparsePythonCompressedLinSOE *>(thePythonSOE))
        c->setMatrixStatus(s);
    else if (auto *c = dynamic_cast<SparsePythonCOOLinSOE *>(thePythonSOE))
        c->setMatrixStatus(s);
}

bool
DistributedSparsePythonLinSOE::needsMatrixTransfer(void) const
{
    if (thePythonSOE == nullptr)
        return true;
    return getInnerMatrixStatus() != static_cast<int>(SparsePythonMatrixStatus::UNCHANGED);
}

void
DistributedSparsePythonLinSOE::ensureMyBSize(int size)
{
    if (size <= myBsize && myB != 0 && myVectB != 0)
        return;

    if (myB != 0)
        delete[] myB;
    if (myVectB != 0)
        delete myVectB;

    myB = new double[size];
    myVectB = new Vector(myB, size);
    myBsize = size;
    for (int i = 0; i < size; ++i)
        myB[i] = 0.0;
}

void
DistributedSparsePythonLinSOE::ensureWorkArea(int size)
{
    if (size <= sizeWork && workArea != 0)
        return;

    if (workArea != 0)
        delete[] workArea;
    workArea = new double[size];
    sizeWork = size;
}

void
DistributedSparsePythonLinSOE::clearTriplets(void)
{
    tripletMap.clear();
}

int
DistributedSparsePythonLinSOE::accumulateTriplet(int row, int col, double value)
{
    if (row < 0 || col < 0 || row >= myBsize || col >= myBsize)
        return 0;

    tripletMap[packRC(row, col)] += value;

    if (auto *c = dynamic_cast<SparsePythonCompressedLinSOE *>(thePythonSOE)) {
        if (c->getMatrixStatus() == SparsePythonMatrixStatus::UNCHANGED)
            setInnerMatrixStatus(static_cast<int>(SparsePythonMatrixStatus::COEFFICIENTS_CHANGED));
    } else if (auto *c = dynamic_cast<SparsePythonCOOLinSOE *>(thePythonSOE)) {
        if (c->getMatrixStatus() == SparsePythonMatrixStatus::UNCHANGED)
            setInnerMatrixStatus(static_cast<int>(SparsePythonMatrixStatus::COEFFICIENTS_CHANGED));
    }
    return 0;
}

int
DistributedSparsePythonLinSOE::addAWorker(const Matrix &m, const ID &id, double fact)
{
    if (fact == 0.0)
        return 0;

    const int idSize = id.Size();
    if (idSize != m.noRows() || idSize != m.noCols()) {
        opserr << "WARNING DistributedSparsePythonLinSOE::addA() - "
               << "Matrix and ID not of similar sizes\n";
        return -1;
    }

    if (fact == 1.0) {
        for (int i = 0; i < idSize; ++i) {
            const int globalRow = id(i);
            if (globalRow < 0 || globalRow >= myBsize)
                continue;
            for (int j = 0; j < idSize; ++j) {
                const int globalCol = id(j);
                if (globalCol < 0 || globalCol >= myBsize)
                    continue;
                if (accumulateTriplet(globalRow, globalCol, m(i, j)) != 0)
                    return -1;
            }
        }
    } else if (fact == -1.0) {
        for (int i = 0; i < idSize; ++i) {
            const int globalRow = id(i);
            if (globalRow < 0 || globalRow >= myBsize)
                continue;
            for (int j = 0; j < idSize; ++j) {
                const int globalCol = id(j);
                if (globalCol < 0 || globalCol >= myBsize)
                    continue;
                if (accumulateTriplet(globalRow, globalCol, -m(i, j)) != 0)
                    return -1;
            }
        }
    } else {
        for (int i = 0; i < idSize; ++i) {
            const int globalRow = id(i);
            if (globalRow < 0 || globalRow >= myBsize)
                continue;
            for (int j = 0; j < idSize; ++j) {
                const int globalCol = id(j);
                if (globalCol < 0 || globalCol >= myBsize)
                    continue;
                if (accumulateTriplet(globalRow, globalCol, fact * m(i, j)) != 0)
                    return -1;
            }
        }
    }
    return 0;
}

int
DistributedSparsePythonLinSOE::buildMergeIndexMap(void)
{
    mergeIndexMap.clear();
    if (thePythonSOE == nullptr)
        return -1;

    if (auto *c = dynamic_cast<SparsePythonCompressedLinSOE *>(thePythonSOE)) {
        const auto &indexPtr = c->getIndexPtr();
        const auto &indices = c->getIndices();
        const int numEqn = c->getNumEqn();
        if (indexPtr.size() != static_cast<std::size_t>(numEqn) + 1u)
            return -1;

        if (c->getStorageScheme() == SparsePythonStorageScheme::CSR) {
            for (int row = 0; row < numEqn; ++row) {
                for (int k = indexPtr[static_cast<std::size_t>(row)];
                     k < indexPtr[static_cast<std::size_t>(row + 1)]; ++k) {
                    mergeIndexMap[packRC(row, indices[static_cast<std::size_t>(k)])] = k;
                }
            }
        } else { // CSC
            for (int col = 0; col < numEqn; ++col) {
                for (int k = indexPtr[static_cast<std::size_t>(col)];
                     k < indexPtr[static_cast<std::size_t>(col + 1)]; ++k) {
                    mergeIndexMap[packRC(indices[static_cast<std::size_t>(k)], col)] = k;
                }
            }
        }
        return 0;
    }

    if (auto *c = dynamic_cast<SparsePythonCOOLinSOE *>(thePythonSOE)) {
        const auto &rows = c->getRowIndices();
        const auto &cols = c->getColIndices();
        const std::size_t nnz = rows.size();
        if (cols.size() != nnz)
            return -1;
        for (std::size_t k = 0; k < nnz; ++k) {
            mergeIndexMap[packRC(rows[k], cols[k])] = static_cast<int>(k);
        }
        return 0;
    }

    return -1;
}

int
DistributedSparsePythonLinSOE::mergeTripletsIntoA(const int *rows, const int *cols,
                                                  const double *vals, int nTrip)
{
    if (nTrip <= 0)
        return 0;
    if (thePythonSOE == nullptr || rows == nullptr || cols == nullptr || vals == nullptr)
        return -1;

    double *A = nullptr;
    if (auto *c = dynamic_cast<SparsePythonCompressedLinSOE *>(thePythonSOE))
        A = c->getValues().data();
    else if (auto *c = dynamic_cast<SparsePythonCOOLinSOE *>(thePythonSOE))
        A = c->getValues().data();

    if (A == nullptr) {
        opserr << "WARNING DistributedSparsePythonLinSOE::mergeTripletsIntoA() - no A values\n";
        return -1;
    }

    for (int i = 0; i < nTrip; ++i) {
        const auto it = mergeIndexMap.find(packRC(rows[i], cols[i]));
        if (it == mergeIndexMap.end()) {
            opserr << "WARNING DistributedSparsePythonLinSOE::mergeTripletsIntoA() - "
                   << "no matrix entry for (" << rows[i] << ", " << cols[i] << ")\n";
            return -1;
        }
        A[it->second] += vals[i];
    }
    return 0;
}

int
DistributedSparsePythonLinSOE::resizeWorkerSOE(int numEqn)
{
    if (auto *c = dynamic_cast<SparsePythonCompressedLinSOE *>(thePythonSOE))
        return c->resizeVectors(numEqn);
    if (auto *c = dynamic_cast<SparsePythonCOOLinSOE *>(thePythonSOE))
        return c->resizeVectors(numEqn);
    return -1;
}

int
DistributedSparsePythonLinSOE::getNumEqn(void) const
{
    return thePythonSOE != nullptr ? thePythonSOE->getNumEqn() : 0;
}

int
DistributedSparsePythonLinSOE::setSize(Graph &theGraph)
{
    if (thePythonSOE == nullptr) {
        opserr << "WARNING DistributedSparsePythonLinSOE::setSize() - no Python LinSOE\n";
        return -1;
    }

    clearTriplets();
    mergeIndexMap.clear();

    int result = 0;

    // Worker: send local graph, receive metadata only
    if (processID != 0) {
        if (numChannels < 1 || theChannels == 0 || theChannels[0] == 0) {
            opserr << "WARNING DistributedSparsePythonLinSOE::setSize() - worker has no channel\n";
            return -1;
        }

        Channel *theChannel = theChannels[0];
        theGraph.sendSelf(0, *theChannel);

        static ID data(2);
        theChannel->recvID(0, 0, data);
        const int numEqn = data(0);
        result = data(1);

        if (result < 0) {
            opserr << "WARNING DistributedSparsePythonLinSOE::setSize() - rank 0 reported failure\n";
            return result;
        }

        result = resizeWorkerSOE(numEqn);
        if (result < 0)
            return result;

        ensureMyBSize(numEqn);
        return 0;
    }

    // Rank 0: merge remote graphs, build structure, send metadata to workers
    FEM_ObjectBroker theBroker;
    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        Graph theSubGraph;
        theSubGraph.recvSelf(0, *theChannel, theBroker);
        theGraph.merge(theSubGraph);
    }

    result = thePythonSOE->setSize(theGraph);

    const int numEqn = thePythonSOE->getNumEqn();

    static ID data(2);
    data(0) = numEqn;
    data(1) = result;

    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        theChannel->sendID(0, 0, data);
    }

    if (result < 0)
        return result;

    if (buildMergeIndexMap() < 0) {
        opserr << "WARNING DistributedSparsePythonLinSOE::setSize() - failed to build merge index map\n";
        return -1;
    }

    ensureMyBSize(numEqn);
    ensureWorkArea(numEqn);
    return 0;
}

int
DistributedSparsePythonLinSOE::addA(const Matrix &m, const ID &id, double fact)
{
    if (thePythonSOE == nullptr)
        return -1;
    if (processID != 0)
        return addAWorker(m, id, fact);
    return thePythonSOE->addA(m, id, fact);
}

int
DistributedSparsePythonLinSOE::addB(const Vector &v, const ID &id, double fact)
{
    if (fact == 0.0)
        return 0;

    const int idSize = id.Size();
    if (idSize != v.Size()) {
        opserr << "WARNING DistributedSparsePythonLinSOE::addB() - Vector and ID size mismatch\n";
        return -1;
    }

    const int size = myBsize;
    if (fact == 1.0) {
        for (int i = 0; i < idSize; ++i) {
            const int pos = id(i);
            if (pos >= 0 && pos < size)
                myB[pos] += v(i);
        }
    } else if (fact == -1.0) {
        for (int i = 0; i < idSize; ++i) {
            const int pos = id(i);
            if (pos >= 0 && pos < size)
                myB[pos] -= v(i);
        }
    } else {
        for (int i = 0; i < idSize; ++i) {
            const int pos = id(i);
            if (pos >= 0 && pos < size)
                myB[pos] += v(i) * fact;
        }
    }
    return 0;
}

int
DistributedSparsePythonLinSOE::setB(const Vector &v, double fact)
{
    if (fact == 0.0)
        return 0;

    if (v.Size() != myBsize) {
        opserr << "WARNING DistributedSparsePythonLinSOE::setB() - incompatible sizes "
               << "SOE(global)=" << myBsize << " vector=" << v.Size()
               << " (expected global eqn count; check ParallelNumberer setNumEqn "
               << "and ArpackSOE setProcessID for OpenSeesMP)\n";
        return -1;
    }

    if (fact == 1.0) {
        for (int i = 0; i < myBsize; ++i)
            myB[i] = v(i);
    } else if (fact == -1.0) {
        for (int i = 0; i < myBsize; ++i)
            myB[i] = -v(i);
    } else {
        for (int i = 0; i < myBsize; ++i)
            myB[i] = v(i) * fact;
    }
    return 0;
}

void
DistributedSparsePythonLinSOE::zeroA(void)
{
    clearTriplets();
    if (thePythonSOE != nullptr)
        thePythonSOE->zeroA();
}

void
DistributedSparsePythonLinSOE::zeroB(void)
{
    for (int i = 0; i < myBsize; ++i)
        myB[i] = 0.0;
}

const Vector &
DistributedSparsePythonLinSOE::getX(void)
{
    return thePythonSOE->getX();
}

const Vector &
DistributedSparsePythonLinSOE::getB(void)
{
    Vector &vectB = const_cast<Vector &>(thePythonSOE->getB());

    if (processID != 0) {
        Channel *theChannel = theChannels[0];
        theChannel->sendVector(0, 0, *myVectB);
        theChannel->recvVector(0, 0, vectB);
        return vectB;
    }

    vectB = *myVectB;
    ensureWorkArea(myBsize);
    Vector remoteB(workArea, myBsize);
    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        theChannel->recvVector(0, 0, remoteB);
        vectB += remoteB;
    }
    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        theChannel->sendVector(0, 0, vectB);
    }
    return vectB;
}

double
DistributedSparsePythonLinSOE::normRHS(void)
{
    return this->getB().Norm();
}

void
DistributedSparsePythonLinSOE::setX(int loc, double value)
{
    if (thePythonSOE != nullptr)
        thePythonSOE->setX(loc, value);
}

void
DistributedSparsePythonLinSOE::setX(const Vector &x)
{
    if (thePythonSOE != nullptr)
        thePythonSOE->setX(x);
}

int
DistributedSparsePythonLinSOE::solve(void)
{
    static ID result(1);

    if (thePythonSOE == nullptr) {
        opserr << "WARNING DistributedSparsePythonLinSOE::solve() - no Python LinSOE\n";
        return -1;
    }

    const bool sendA = needsMatrixTransfer();
    Vector &vectX = const_cast<Vector &>(thePythonSOE->getX());
    Vector &vectB = const_cast<Vector &>(thePythonSOE->getB());

    if (processID != 0) {
        Channel *theChannel = theChannels[0];
        theChannel->sendVector(0, 0, *myVectB);

        if (sendA) {
            const int nTrip = static_cast<int>(tripletMap.size());
            ID nTripID(1);
            nTripID(0) = nTrip;
            theChannel->sendID(0, 0, nTripID);

            if (nTrip > 0) {
                ID rows(nTrip);
                ID cols(nTrip);
                Vector vals(nTrip);
                int k = 0;
                for (const auto &entry : tripletMap) {
                    rows(k) = static_cast<int>(entry.first >> 32);
                    cols(k) = static_cast<int>(entry.first & 0xffffffffu);
                    vals(k) = entry.second;
                    ++k;
                }
                theChannel->sendID(0, 0, rows);
                theChannel->sendID(0, 0, cols);
                theChannel->sendVector(0, 0, vals);
            }
        }

        theChannel->recvVector(0, 0, vectX);
        theChannel->recvVector(0, 0, vectB);
        theChannel->recvID(0, 0, result);

        if (result(0) == 0)
            setInnerMatrixStatus(static_cast<int>(SparsePythonMatrixStatus::UNCHANGED));

        return result(0);
    }

    // Rank 0: merge RHS and (if needed) triplets, then Python solve
    vectB = *myVectB;
    result(0) = 0;

    ensureWorkArea(myBsize);
    Vector remoteB(myBsize);

    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        theChannel->recvVector(0, 0, remoteB);
        vectB += remoteB;

        if (sendA) {
            ID nTripID(1);
            theChannel->recvID(0, 0, nTripID);
            const int nTrip = nTripID(0);
            if (nTrip < 0) {
                opserr << "WARNING DistributedSparsePythonLinSOE::solve() - invalid nTrip\n";
                result(0) = -1;
                continue;
            }
            if (nTrip > 0) {
                ID rows(nTrip);
                ID cols(nTrip);
                Vector vals(nTrip);
                theChannel->recvID(0, 0, rows);
                theChannel->recvID(0, 0, cols);
                theChannel->recvVector(0, 0, vals);
                if (result(0) == 0 &&
                    mergeTripletsIntoA(&rows(0), &cols(0), &vals(0), nTrip) < 0) {
                    result(0) = -1;
                }
            }
        }
    }

    if (result(0) == 0)
        result(0) = thePythonSOE->solve();

    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        theChannel->sendVector(0, 0, vectX);
        theChannel->sendVector(0, 0, vectB);
        theChannel->sendID(0, 0, result);
    }

    return result(0);
}

int
DistributedSparsePythonLinSOE::sendSelf(int commitTag, Channel &theChannel)
{
    int sendID = 0;

    if (processID == 0) {
        bool found = false;
        for (int i = 0; i < numChannels; ++i) {
            if (theChannels[i] == &theChannel) {
                sendID = i + 1;
                found = true;
            }
        }

        if (!found) {
            int nextNumChannels = numChannels + 1;
            Channel **nextChannels = new Channel *[nextNumChannels];
            for (int i = 0; i < numChannels; ++i)
                nextChannels[i] = theChannels[i];
            nextChannels[numChannels] = &theChannel;

            if (theChannels != 0)
                delete[] theChannels;
            theChannels = nextChannels;
            numChannels = nextNumChannels;
            sendID = numChannels;
        }
    } else {
        sendID = processID;
    }

    ID idData(1);
    idData(0) = sendID;
    int res = theChannel.sendID(0, commitTag, idData);
    if (res < 0) {
        opserr << "WARNING DistributedSparsePythonLinSOE::sendSelf() - failed to send data\n";
        return -1;
    }
    return 0;
}

int
DistributedSparsePythonLinSOE::recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    ID idData(1);
    int res = theChannel.recvID(0, commitTag, idData);
    if (res < 0) {
        opserr << "WARNING DistributedSparsePythonLinSOE::recvSelf() - failed to recv data\n";
        return -1;
    }
    processID = idData(0);

    numChannels = 1;
    if (theChannels != 0)
        delete[] theChannels;
    theChannels = new Channel *[1];
    theChannels[0] = &theChannel;

    (void)theBroker;
    return 0;
}

int
DistributedSparsePythonLinSOE::setProcessID(int dTag)
{
    processID = dTag;
    return 0;
}

int
DistributedSparsePythonLinSOE::setChannels(int nChannels, Channel **theC)
{
    numChannels = nChannels;

    if (theChannels != 0)
        delete[] theChannels;

    if (numChannels > 0 && theC != 0) {
        theChannels = new Channel *[numChannels];
        for (int i = 0; i < numChannels; ++i)
            theChannels[i] = theC[i];
    } else {
        theChannels = 0;
        numChannels = 0;
    }
    return 0;
}
