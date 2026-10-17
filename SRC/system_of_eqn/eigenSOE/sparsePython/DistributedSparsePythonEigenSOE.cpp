/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
** ****************************************************************** */

// Written: gaaraujo
// Created: 07/2026
//
// DistributedSparsePythonEigenSOE: gather-to-root with worker-side K/M triplets.

#include "DistributedSparsePythonEigenSOE.h"
#include "SparsePythonCompressedEigenSOE.h"
#include "SparsePythonCOOEigenSOE.h"
#include "SparsePythonEigenCommon.h"

#include <Matrix.h>
#include <Graph.h>
#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include <ID.h>
#include <classTags.h>
#include <OPS_Globals.h>

#include <vector>

uint64_t
DistributedSparsePythonEigenSOE::packRC(int row, int col)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(col));
}

DistributedSparsePythonEigenSOE::DistributedSparsePythonEigenSOE(EigenSOE *theSOE)
    : EigenSOE(EigenSOE_TAGS_DistributedSparsePythonEigenSOE),
      thePythonSOE(theSOE),
      processID(0), numChannels(0), theChannels(0),
      size(0), numStoredModes(0), emptyVector(1)
{
    if (thePythonSOE == nullptr) {
        opserr << "FATAL DistributedSparsePythonEigenSOE - null SparsePython EigenSOE\n";
    }
}

DistributedSparsePythonEigenSOE::DistributedSparsePythonEigenSOE()
    : EigenSOE(EigenSOE_TAGS_DistributedSparsePythonEigenSOE),
      thePythonSOE(0),
      processID(0), numChannels(0), theChannels(0),
      size(0), numStoredModes(0), emptyVector(1)
{
}

DistributedSparsePythonEigenSOE::~DistributedSparsePythonEigenSOE()
{
    if (theChannels != 0)
        delete[] theChannels;

    // Owned EigenSOE deletes its solver; avoid double-delete via EigenSOE base.
    delete thePythonSOE;
    thePythonSOE = nullptr;
}

int
DistributedSparsePythonEigenSOE::getInnerNumEqn(void) const
{
    if (auto *c = dynamic_cast<SparsePythonCompressedEigenSOE *>(thePythonSOE))
        return c->getNumEqn();
    if (auto *c = dynamic_cast<SparsePythonCOOEigenSOE *>(thePythonSOE))
        return c->getNumEqn();
    return size;
}

int
DistributedSparsePythonEigenSOE::getInnerMatrixStatus(void) const
{
    if (auto *c = dynamic_cast<SparsePythonCompressedEigenSOE *>(thePythonSOE))
        return static_cast<int>(c->getMatrixStatus());
    if (auto *c = dynamic_cast<SparsePythonCOOEigenSOE *>(thePythonSOE))
        return static_cast<int>(c->getMatrixStatus());
    return static_cast<int>(SparsePythonEigenMatrixStatus::STRUCTURE_CHANGED);
}

void
DistributedSparsePythonEigenSOE::setInnerMatrixStatus(int status)
{
    const auto s = static_cast<SparsePythonEigenMatrixStatus>(status);
    if (auto *c = dynamic_cast<SparsePythonCompressedEigenSOE *>(thePythonSOE))
        c->setMatrixStatus(s);
    else if (auto *c = dynamic_cast<SparsePythonCOOEigenSOE *>(thePythonSOE))
        c->setMatrixStatus(s);
}

bool
DistributedSparsePythonEigenSOE::needsMatrixTransfer(void) const
{
    if (thePythonSOE == nullptr)
        return true;
    return getInnerMatrixStatus() != static_cast<int>(SparsePythonEigenMatrixStatus::UNCHANGED);
}

int
DistributedSparsePythonEigenSOE::addMatrixWorker(SparseLocalTriplets &triplets,
                                                 const Matrix &m, const ID &id, double fact)
{
    if (fact == 0.0)
        return 0;

    const int idSize = id.Size();
    if (idSize != m.noRows() || idSize != m.noCols()) {
        opserr << "WARNING DistributedSparsePythonEigenSOE::addA/addM() - "
               << "Matrix and ID not of similar sizes\n";
        return -1;
    }

    if (triplets.add(m, id, fact, size) &&
        getInnerMatrixStatus() == static_cast<int>(SparsePythonEigenMatrixStatus::UNCHANGED))
        setInnerMatrixStatus(static_cast<int>(SparsePythonEigenMatrixStatus::COEFFICIENTS_CHANGED));
    return 0;
}

int
DistributedSparsePythonEigenSOE::buildMergeIndexMap(void)
{
    mergeIndexMap.clear();
    if (thePythonSOE == nullptr)
        return -1;

    if (auto *c = dynamic_cast<SparsePythonCompressedEigenSOE *>(thePythonSOE)) {
        const auto &indexPtr = c->getIndexPtr();
        const auto &indices = c->getIndices();
        const int numEqn = c->getNumEqn();
        if (indexPtr.size() != static_cast<std::size_t>(numEqn) + 1u)
            return -1;

        if (c->getStorageScheme() == SparsePythonEigenStorageScheme::CSR) {
            for (int row = 0; row < numEqn; ++row) {
                for (int k = indexPtr[static_cast<std::size_t>(row)];
                     k < indexPtr[static_cast<std::size_t>(row + 1)]; ++k) {
                    mergeIndexMap[packRC(row, indices[static_cast<std::size_t>(k)])] = k;
                }
            }
        } else {
            for (int col = 0; col < numEqn; ++col) {
                for (int k = indexPtr[static_cast<std::size_t>(col)];
                     k < indexPtr[static_cast<std::size_t>(col + 1)]; ++k) {
                    mergeIndexMap[packRC(indices[static_cast<std::size_t>(k)], col)] = k;
                }
            }
        }
        return 0;
    }

    if (auto *c = dynamic_cast<SparsePythonCOOEigenSOE *>(thePythonSOE)) {
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
DistributedSparsePythonEigenSOE::resizeWorkerSOE(int numEqn)
{
    if (auto *c = dynamic_cast<SparsePythonCompressedEigenSOE *>(thePythonSOE))
        return c->resizeVectors(numEqn);
    if (auto *c = dynamic_cast<SparsePythonCOOEigenSOE *>(thePythonSOE))
        return c->resizeVectors(numEqn);
    return -1;
}

void
DistributedSparsePythonEigenSOE::ensureEigenStorage(int numModes, int numEqn)
{
    eigenvalues.assign(static_cast<std::size_t>(numModes), 0.0);
    eigenvectors.assign(static_cast<std::size_t>(numModes) * static_cast<std::size_t>(numEqn), 0.0);
    eigenvectorViews.clear();
    eigenvectorViews.reserve(static_cast<std::size_t>(numModes));
    for (int mode = 0; mode < numModes; ++mode) {
        double *modeData = eigenvectors.data() + mode * numEqn;
        eigenvectorViews.emplace_back(Vector(modeData, numEqn));
    }
    numStoredModes = numModes;
}

int
DistributedSparsePythonEigenSOE::sendTriplets(Channel *theChannel, SparseLocalTriplets &triplets)
{
    // Count of SparseLocalTriplets::header(): n >= 0 full triplets, or -(n + 2)
    // values only once rank 0 holds this rank's pattern.
    ID nTripID(1);
    nTripID(0) = triplets.header();
    if (theChannel->sendID(0, 0, nTripID) < 0)
        return -1;
    return triplets.send(*theChannel);
}

int
DistributedSparsePythonEigenSOE::recvAndMergeTriplets(Channel *theChannel, SparseRootScatter &scatter,
                                                      int worker, std::vector<double> *values,
                                                      bool merge)
{
    ID nTripID(1);
    if (theChannel->recvID(0, 0, nTripID) < 0)
        return -1;
    const int header = nTripID(0);
    if (header == -1)
        return -1;

    const int numValues = values != nullptr ? static_cast<int>(values->size()) : 0;
    auto offsetOf = [&](int row, int col) -> int {
        const auto it = mergeIndexMap.find(packRC(row, col));
        if (it == mergeIndexMap.end() || it->second >= numValues)
            return -1;
        return it->second;
    };
    return scatter.recvAndMerge(*theChannel, worker, header, merge && values != nullptr,
                                values != nullptr ? values->data() : nullptr, offsetOf);
}

int
DistributedSparsePythonEigenSOE::setSize(Graph &theGraph)
{
    if (thePythonSOE == nullptr) {
        opserr << "WARNING DistributedSparsePythonEigenSOE::setSize() - no Python EigenSOE\n";
        return -1;
    }

    kTriplets.reset();
    mTriplets.reset();
    mergeIndexMap.clear();
    kScatter.reset(numChannels);
    mScatter.reset(numChannels);
    numStoredModes = 0;

    int result = 0;

    if (processID != 0) {
        if (numChannels < 1 || theChannels == 0 || theChannels[0] == 0) {
            opserr << "WARNING DistributedSparsePythonEigenSOE::setSize() - worker has no channel\n";
            return -1;
        }

        Channel *theChannel = theChannels[0];
        theGraph.sendSelf(0, *theChannel);

        static ID data(2);
        theChannel->recvID(0, 0, data);
        size = data(0);
        result = data(1);

        if (result < 0) {
            opserr << "WARNING DistributedSparsePythonEigenSOE::setSize() - rank 0 reported failure\n";
            return result;
        }

        return resizeWorkerSOE(size);
    }

    FEM_ObjectBroker theBroker;
    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        Graph theSubGraph;
        theSubGraph.recvSelf(0, *theChannel, theBroker);
        theGraph.merge(theSubGraph);
    }

    result = thePythonSOE->setSize(theGraph);
    size = getInnerNumEqn();

    static ID data(2);
    data(0) = size;
    data(1) = result;

    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        theChannel->sendID(0, 0, data);
    }

    if (result < 0)
        return result;

    if (buildMergeIndexMap() < 0) {
        opserr << "WARNING DistributedSparsePythonEigenSOE::setSize() - failed to build merge index map\n";
        return -1;
    }

    return 0;
}

int
DistributedSparsePythonEigenSOE::addA(const Matrix &m, const ID &id, double fact)
{
    if (thePythonSOE == nullptr)
        return -1;
    if (processID != 0)
        return addMatrixWorker(kTriplets, m, id, fact);
    return thePythonSOE->addA(m, id, fact);
}

int
DistributedSparsePythonEigenSOE::addM(const Matrix &m, const ID &id, double fact)
{
    if (thePythonSOE == nullptr)
        return -1;
    if (processID != 0)
        return addMatrixWorker(mTriplets, m, id, fact);
    return thePythonSOE->addM(m, id, fact);
}

void
DistributedSparsePythonEigenSOE::zeroA(void)
{
    clearKTriplets();
    if (thePythonSOE != nullptr)
        thePythonSOE->zeroA();
}

void
DistributedSparsePythonEigenSOE::zeroM(void)
{
    clearMTriplets();
    if (thePythonSOE != nullptr)
        thePythonSOE->zeroM();
}

int
DistributedSparsePythonEigenSOE::solve(int numModes, bool generalized, bool findSmallest)
{
    static ID result(1);

    if (thePythonSOE == nullptr) {
        opserr << "WARNING DistributedSparsePythonEigenSOE::solve() - no Python EigenSOE\n";
        return -1;
    }

    const bool sendMats = needsMatrixTransfer();

    if (processID != 0) {
        Channel *theChannel = theChannels[0];

        // Header: -1 if K and M are unchanged on this rank, 1 if their triplets
        // follow. Rank 0 reads it instead of its own status, so the triplet
        // messages cannot go unmatched.
        ID header(1);
        header(0) = sendMats ? 1 : -1;
        theChannel->sendID(0, 0, header);
        if (sendMats) {
            sendTriplets(theChannel, kTriplets);
            sendTriplets(theChannel, mTriplets);
        }

        theChannel->recvID(0, 0, result);
        if (result(0) < 0)
            return result(0);

        ID meta(2);
        theChannel->recvID(0, 0, meta);
        const int nModes = meta(0);
        const int nEqn = meta(1);
        ensureEigenStorage(nModes, nEqn);

        Vector evals(eigenvalues.data(), nModes);
        Vector evecs(eigenvectors.data(), nModes * nEqn);
        theChannel->recvVector(0, 0, evals);
        theChannel->recvVector(0, 0, evecs);

        if (result(0) == 0)
            setInnerMatrixStatus(static_cast<int>(SparsePythonEigenMatrixStatus::UNCHANGED));

        return result(0);
    }

    result(0) = 0;

    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        ID header(1);
        theChannel->recvID(0, 0, header);
        const bool workerSendsMats = header(0) > 0;
        if (workerSendsMats != sendMats) {
            // Fail on every rank instead of solving a partial K or M; the
            // worker's triplets are still received below.
            opserr << "WARNING DistributedSparsePythonEigenSOE::solve() - matrix status of rank "
                   << j + 1 << " differs from rank 0\n";
            result(0) = -1;
        }
        if (workerSendsMats) {
            std::vector<double> *kVals = nullptr;
            std::vector<double> *mVals = nullptr;
            if (auto *c = dynamic_cast<SparsePythonCompressedEigenSOE *>(thePythonSOE)) {
                kVals = &c->getKValues();
                mVals = &c->getMValues();
            } else if (auto *c = dynamic_cast<SparsePythonCOOEigenSOE *>(thePythonSOE)) {
                kVals = &c->getKValues();
                mVals = &c->getMValues();
            }
            if (kVals == nullptr || mVals == nullptr)
                result(0) = -1;
            const bool merge = result(0) == 0;
            if (recvAndMergeTriplets(theChannel, kScatter, j, kVals, merge) < 0)
                result(0) = -1;
            if (recvAndMergeTriplets(theChannel, mScatter, j, mVals, merge) < 0)
                result(0) = -1;
        }
    }

    if (result(0) == 0)
        result(0) = thePythonSOE->solve(numModes, generalized, findSmallest);

    const int nEqn = getInnerNumEqn();
    ensureEigenStorage(numModes, nEqn);

    if (result(0) == 0) {
        for (int mode = 1; mode <= numModes; ++mode) {
            eigenvalues[static_cast<std::size_t>(mode - 1)] = thePythonSOE->getEigenvalue(mode);
            const Vector &ev = thePythonSOE->getEigenvector(mode);
            for (int i = 0; i < nEqn; ++i)
                eigenvectors[static_cast<std::size_t>((mode - 1) * nEqn + i)] = ev(i);
        }
        setInnerMatrixStatus(static_cast<int>(SparsePythonEigenMatrixStatus::UNCHANGED));
    }

    ID meta(2);
    meta(0) = numModes;
    meta(1) = nEqn;
    Vector evals(eigenvalues.data(), numModes);
    Vector evecs(eigenvectors.data(), numModes * nEqn);

    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        theChannel->sendID(0, 0, result);
        if (result(0) < 0)
            continue;
        theChannel->sendID(0, 0, meta);
        theChannel->sendVector(0, 0, evals);
        theChannel->sendVector(0, 0, evecs);
    }

    return result(0);
}

const Vector &
DistributedSparsePythonEigenSOE::getEigenvector(int mode)
{
    if (mode < 1 || mode > numStoredModes)
        return emptyVector;
    return eigenvectorViews[static_cast<std::size_t>(mode - 1)];
}

double
DistributedSparsePythonEigenSOE::getEigenvalue(int mode)
{
    if (mode < 1 || mode > numStoredModes)
        return 0.0;
    return eigenvalues[static_cast<std::size_t>(mode - 1)];
}

int
DistributedSparsePythonEigenSOE::sendSelf(int commitTag, Channel &theChannel)
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
        opserr << "WARNING DistributedSparsePythonEigenSOE::sendSelf() - failed to send data\n";
        return -1;
    }
    return 0;
}

int
DistributedSparsePythonEigenSOE::recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    ID idData(1);
    int res = theChannel.recvID(0, commitTag, idData);
    if (res < 0) {
        opserr << "WARNING DistributedSparsePythonEigenSOE::recvSelf() - failed to recv data\n";
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
DistributedSparsePythonEigenSOE::setProcessID(int dTag)
{
    processID = dTag;
    return 0;
}

int
DistributedSparsePythonEigenSOE::setChannels(int nChannels, Channel **theC)
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
