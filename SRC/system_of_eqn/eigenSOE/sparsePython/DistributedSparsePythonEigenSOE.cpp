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
DistributedSparsePythonEigenSOE::accumulateTriplet(std::unordered_map<uint64_t, double> &map,
                                                   int row, int col, double value)
{
    if (row < 0 || col < 0 || row >= size || col >= size)
        return 0;

    map[packRC(row, col)] += value;
    if (getInnerMatrixStatus() == static_cast<int>(SparsePythonEigenMatrixStatus::UNCHANGED))
        setInnerMatrixStatus(static_cast<int>(SparsePythonEigenMatrixStatus::COEFFICIENTS_CHANGED));
    return 0;
}

int
DistributedSparsePythonEigenSOE::addMatrixWorker(std::unordered_map<uint64_t, double> &map,
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

    for (int i = 0; i < idSize; ++i) {
        const int globalRow = id(i);
        if (globalRow < 0 || globalRow >= size)
            continue;
        for (int j = 0; j < idSize; ++j) {
            const int globalCol = id(j);
            if (globalCol < 0 || globalCol >= size)
                continue;
            if (accumulateTriplet(map, globalRow, globalCol, fact * m(i, j)) != 0)
                return -1;
        }
    }
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
DistributedSparsePythonEigenSOE::mergeTripletsInto(std::vector<double> &values,
                                                   const int *rows, const int *cols,
                                                   const double *vals, int nTrip)
{
    if (nTrip <= 0)
        return 0;
    if (rows == nullptr || cols == nullptr || vals == nullptr)
        return -1;

    for (int i = 0; i < nTrip; ++i) {
        const auto it = mergeIndexMap.find(packRC(rows[i], cols[i]));
        if (it == mergeIndexMap.end()) {
            opserr << "WARNING DistributedSparsePythonEigenSOE::mergeTripletsInto() - "
                   << "no matrix entry for (" << rows[i] << ", " << cols[i] << ")\n";
            return -1;
        }
        if (static_cast<std::size_t>(it->second) >= values.size())
            return -1;
        values[static_cast<std::size_t>(it->second)] += vals[i];
    }
    return 0;
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
DistributedSparsePythonEigenSOE::sendTripletMap(Channel *theChannel,
                                                const std::unordered_map<uint64_t, double> &map)
{
    const int nTrip = static_cast<int>(map.size());
    ID nTripID(1);
    nTripID(0) = nTrip;
    theChannel->sendID(0, 0, nTripID);

    if (nTrip > 0) {
        ID rows(nTrip);
        ID cols(nTrip);
        Vector vals(nTrip);
        int k = 0;
        for (const auto &entry : map) {
            rows(k) = static_cast<int>(entry.first >> 32);
            cols(k) = static_cast<int>(entry.first & 0xffffffffu);
            vals(k) = entry.second;
            ++k;
        }
        theChannel->sendID(0, 0, rows);
        theChannel->sendID(0, 0, cols);
        theChannel->sendVector(0, 0, vals);
    }
    return 0;
}

int
DistributedSparsePythonEigenSOE::recvAndMergeTripletMap(Channel *theChannel,
                                                        std::vector<double> &values)
{
    ID nTripID(1);
    theChannel->recvID(0, 0, nTripID);
    const int nTrip = nTripID(0);
    if (nTrip < 0)
        return -1;
    if (nTrip == 0)
        return 0;

    ID rows(nTrip);
    ID cols(nTrip);
    Vector vals(nTrip);
    theChannel->recvID(0, 0, rows);
    theChannel->recvID(0, 0, cols);
    theChannel->recvVector(0, 0, vals);
    return mergeTripletsInto(values, &rows(0), &cols(0), &vals(0), nTrip);
}

int
DistributedSparsePythonEigenSOE::setSize(Graph &theGraph)
{
    if (thePythonSOE == nullptr) {
        opserr << "WARNING DistributedSparsePythonEigenSOE::setSize() - no Python EigenSOE\n";
        return -1;
    }

    clearKTriplets();
    clearMTriplets();
    mergeIndexMap.clear();
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
        return addMatrixWorker(kTripletMap, m, id, fact);
    return thePythonSOE->addA(m, id, fact);
}

int
DistributedSparsePythonEigenSOE::addM(const Matrix &m, const ID &id, double fact)
{
    if (thePythonSOE == nullptr)
        return -1;
    if (processID != 0)
        return addMatrixWorker(mTripletMap, m, id, fact);
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

        if (sendMats) {
            sendTripletMap(theChannel, kTripletMap);
            sendTripletMap(theChannel, mTripletMap);
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
        if (sendMats) {
            std::vector<double> *kVals = nullptr;
            std::vector<double> *mVals = nullptr;
            if (auto *c = dynamic_cast<SparsePythonCompressedEigenSOE *>(thePythonSOE)) {
                kVals = &c->getKValues();
                mVals = &c->getMValues();
            } else if (auto *c = dynamic_cast<SparsePythonCOOEigenSOE *>(thePythonSOE)) {
                kVals = &c->getKValues();
                mVals = &c->getMValues();
            }
            if (kVals == nullptr || mVals == nullptr) {
                result(0) = -1;
                continue;
            }
            if (recvAndMergeTripletMap(theChannel, *kVals) < 0)
                result(0) = -1;
            if (recvAndMergeTripletMap(theChannel, *mVals) < 0)
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
