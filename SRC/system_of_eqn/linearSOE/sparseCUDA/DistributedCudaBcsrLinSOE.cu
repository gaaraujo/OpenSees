/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
** ****************************************************************** */

// Written: gaaraujo
// Created: 07/2026
//
// DistributedCudaBcsrLinSOE: gather-to-root with worker-side triplets.

#include <DistributedCudaBcsrLinSOE.h>
#include <CudaBcsrLinSOE.h>
#include <CudaBcsrLinSolver.h>
#include <Matrix.h>
#include <Graph.h>
#include <Vertex.h>
#include <VertexIter.h>
#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include <ID.h>
#include <classTags.h>
#include <OPS_Globals.h>

#include <algorithm>
#include <cstring>
#include <vector>

uint64_t
DistributedCudaBcsrLinSOE::packRC(int row, int col)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(col));
}

DistributedCudaBcsrLinSOE::DistributedCudaBcsrLinSOE(CudaBcsrLinSOE *theSOE)
    : LinearSOE(LinSOE_TAGS_DistributedCudaBcsrLinSOE),
      theCudaSOE(theSOE),
      processID(0), numChannels(0), theChannels(0),
      workArea(0), sizeWork(0),
      myB(0), myVectB(0), myVectBHeader(0), myBsize(0),
      globalRhsOnDevice(false)
{
    if (theCudaSOE == nullptr) {
        opserr << "FATAL DistributedCudaBcsrLinSOE - null CudaBcsrLinSOE\n";
    }
}

DistributedCudaBcsrLinSOE::DistributedCudaBcsrLinSOE()
    : LinearSOE(LinSOE_TAGS_DistributedCudaBcsrLinSOE),
      theCudaSOE(0),
      processID(0), numChannels(0), theChannels(0),
      workArea(0), sizeWork(0),
      myB(0), myVectB(0), myVectBHeader(0), myBsize(0),
      globalRhsOnDevice(false)
{
}

DistributedCudaBcsrLinSOE::~DistributedCudaBcsrLinSOE()
{
    if (theChannels != 0)
        delete[] theChannels;

    if (workArea != 0)
        delete[] workArea;

    if (myB != 0)
        delete[] myB;

    if (myVectB != 0)
        delete myVectB;

    if (myVectBHeader != 0)
        delete myVectBHeader;

    delete theCudaSOE;
}

bool
DistributedCudaBcsrLinSOE::needsMatrixTransfer(void) const
{
    if (theCudaSOE == nullptr)
        return true;
    return theCudaSOE->getMatrixStatus() != CudaBcsrLinSOE::MatrixStatus::UNCHANGED;
}

void
DistributedCudaBcsrLinSOE::ensureMyBSize(int size)
{
    if (size <= myBsize && myB != 0 && myVectB != 0)
        return;

    if (myB != 0)
        delete[] myB;
    if (myVectB != 0)
        delete myVectB;
    if (myVectBHeader != 0)
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
DistributedCudaBcsrLinSOE::ensureWorkArea(int size)
{
    if (size <= sizeWork && workArea != 0)
        return;

    if (workArea != 0)
        delete[] workArea;
    workArea = new double[size];
    sizeWork = size;
}

void
DistributedCudaBcsrLinSOE::clearTriplets(void)
{
    tripletMap.clear();
}

int
DistributedCudaBcsrLinSOE::accumulateTriplet(int row, int col, double value)
{
    if (row < 0 || col < 0 || row >= myBsize || col >= myBsize)
        return 0;

    if (theCudaSOE != nullptr &&
        theCudaSOE->getMatrixStorageMode() == CudaBcsrLinSOE::MatrixStorageMode::SYMMETRIC_LOWER &&
        row < col) {
        return 0;
    }

    tripletMap[packRC(row, col)] += value;

    if (theCudaSOE != nullptr &&
        theCudaSOE->getMatrixStatus() == CudaBcsrLinSOE::MatrixStatus::UNCHANGED) {
        theCudaSOE->setMatrixStatus(CudaBcsrLinSOE::MatrixStatus::COEFFICIENTS_CHANGED);
    }
    return 0;
}

int
DistributedCudaBcsrLinSOE::addAWorker(const Matrix &m, const ID &id, double fact)
{
    if (fact == 0.0)
        return 0;

    const int idSize = id.Size();
    if (idSize != m.noRows() && idSize != m.noCols()) {
        opserr << "WARNING DistributedCudaBcsrLinSOE::addA() - "
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
DistributedCudaBcsrLinSOE::buildMergeIndexMap(void)
{
    mergeIndexMap.clear();
    if (theCudaSOE == nullptr)
        return -1;

    const int *csr = theCudaSOE->getHostCsrIndicesData();
    if (csr == nullptr)
        return -1;

    const int blockSize = theCudaSOE->getBlockSize();
    const int numEqn = theCudaSOE->getNumEqn();

    if (blockSize <= 1) {
        const int *rowPtr = csr;
        const int *colIdx = csr + numEqn + 1;
        for (int row = 0; row < numEqn; ++row) {
            for (int k = rowPtr[row]; k < rowPtr[row + 1]; ++k) {
                mergeIndexMap[packRC(row, colIdx[k])] = k;
            }
        }
        return 0;
    }

    const int numBlockRows = theCudaSOE->getNumRowBlocks();
    const int *rowPtr = csr;
    const int *colIdx = csr + numBlockRows + 1;
    const int bs2 = blockSize * blockSize;
    for (int blockRow = 0; blockRow < numBlockRows; ++blockRow) {
        for (int k = rowPtr[blockRow]; k < rowPtr[blockRow + 1]; ++k) {
            const int blockCol = colIdx[k];
            const int blockOffset = k * bs2;
            for (int lr = 0; lr < blockSize; ++lr) {
                const int globalRow = blockRow * blockSize + lr;
                if (globalRow >= numEqn)
                    continue;
                for (int lc = 0; lc < blockSize; ++lc) {
                    const int globalCol = blockCol * blockSize + lc;
                    if (globalCol >= numEqn)
                        continue;
                    mergeIndexMap[packRC(globalRow, globalCol)] =
                        blockOffset + lr * blockSize + lc;
                }
            }
        }
    }
    return 0;
}

int
DistributedCudaBcsrLinSOE::mergeTripletsIntoA(const int *rows, const int *cols,
                                              const double *vals, int nTrip)
{
    if (nTrip <= 0)
        return 0;
    if (theCudaSOE == nullptr || rows == nullptr || cols == nullptr || vals == nullptr)
        return -1;

    double *A = theCudaSOE->getHostAValuesData();
    if (A == nullptr) {
        opserr << "WARNING DistributedCudaBcsrLinSOE::mergeTripletsIntoA() - no host A\n";
        return -1;
    }

    for (int i = 0; i < nTrip; ++i) {
        const auto it = mergeIndexMap.find(packRC(rows[i], cols[i]));
        if (it == mergeIndexMap.end()) {
            opserr << "WARNING DistributedCudaBcsrLinSOE::mergeTripletsIntoA() - "
                   << "no CSR entry for (" << rows[i] << ", " << cols[i] << ")\n";
            return -1;
        }
        A[it->second] += vals[i];
    }
    theCudaSOE->setAValuesPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
    return 0;
}

int
DistributedCudaBcsrLinSOE::getNumEqn(void) const
{
    return theCudaSOE != nullptr ? theCudaSOE->getNumEqn() : 0;
}

int
DistributedCudaBcsrLinSOE::setSize(Graph &theGraph)
{
    if (theCudaSOE == nullptr) {
        opserr << "WARNING DistributedCudaBcsrLinSOE::setSize() - no CudaBcsrLinSOE\n";
        return -1;
    }

    clearTriplets();
    mergeIndexMap.clear();

    int result = 0;

    // Worker: send local graph, receive metadata only (no global CSR)
    if (processID != 0) {
        if (numChannels < 1 || theChannels == 0 || theChannels[0] == 0) {
            opserr << "WARNING DistributedCudaBcsrLinSOE::setSize() - worker has no channel\n";
            return -1;
        }

        Channel *theChannel = theChannels[0];
        theGraph.sendSelf(0, *theChannel);

        static ID data(4);
        theChannel->recvID(0, 0, data);
        const int numEqn = data(0);
        const int blockSize = data(1);
        const int paddedSize = data(2);
        result = data(3);

        if (result < 0) {
            opserr << "WARNING DistributedCudaBcsrLinSOE::setSize() - rank 0 reported failure\n";
            return result;
        }

        result = theCudaSOE->resizeHostVectors(numEqn, blockSize, paddedSize);
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

    result = theCudaSOE->setSize(theGraph);

    const int numEqn = theCudaSOE->getNumEqn();
    const int blockSize = theCudaSOE->getBlockSize();
    const int paddedSize = theCudaSOE->getPaddedVectorSize();

    static ID data(4);
    data(0) = numEqn;
    data(1) = blockSize;
    data(2) = paddedSize;
    data(3) = result;

    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        theChannel->sendID(0, 0, data);
    }

    if (result < 0)
        return result;

    if (buildMergeIndexMap() < 0) {
        opserr << "WARNING DistributedCudaBcsrLinSOE::setSize() - failed to build merge index map\n";
        return -1;
    }

    ensureMyBSize(numEqn);
    ensureWorkArea(numEqn);
    return 0;
}

int
DistributedCudaBcsrLinSOE::addA(const Matrix &m, const ID &id, double fact)
{
    if (theCudaSOE == nullptr)
        return -1;
    if (processID != 0)
        return addAWorker(m, id, fact);
    return theCudaSOE->addA(m, id, fact);
}

int
DistributedCudaBcsrLinSOE::addA(const Matrix &m)
{
    if (theCudaSOE == nullptr)
        return -1;
    if (processID != 0) {
        const int n = m.noRows();
        if (n != m.noCols()) {
            opserr << "WARNING DistributedCudaBcsrLinSOE::addA() - matrix not square\n";
            return -1;
        }
        ID id(n);
        for (int i = 0; i < n; ++i)
            id(i) = i;
        return addAWorker(m, id, 1.0);
    }
    return theCudaSOE->addA(m);
}

int
DistributedCudaBcsrLinSOE::addB(const Vector &v, const ID &id, double fact)
{
    if (fact == 0.0)
        return 0;

    const int idSize = id.Size();
    if (idSize != v.Size()) {
        opserr << "WARNING DistributedCudaBcsrLinSOE::addB() - Vector and ID size mismatch\n";
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
DistributedCudaBcsrLinSOE::setB(const Vector &v, double fact)
{
    if (fact == 0.0)
        return 0;

    if (v.Size() != myBsize) {
        // Gather-to-root SOEs keep a *global* RHS after graph merge. Callers that
        // pass AnalysisModel/Arpack local size (processID unset, ParallelNumberer
        // without setNumEqn) hit this path — see ArpackSOE::setProcessID wiring.
        opserr << "WARNING DistributedCudaBcsrLinSOE::setB() - incompatible sizes "
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
DistributedCudaBcsrLinSOE::zeroA(void)
{
    clearTriplets();
    if (theCudaSOE != nullptr)
        theCudaSOE->zeroA();
}

void
DistributedCudaBcsrLinSOE::zeroB(void)
{
    for (int i = 0; i < myBsize; ++i)
        myB[i] = 0.0;
}

const Vector &
DistributedCudaBcsrLinSOE::getX(void)
{
    return theCudaSOE->getX();
}

const Vector &
DistributedCudaBcsrLinSOE::getB(void)
{
    Vector &vectB = theCudaSOE->getHostBVector();

    if (processID != 0) {
        Channel *theChannel = theChannels[0];
        theChannel->sendVector(0, 0, *myVectB);
        theChannel->recvVector(0, 0, vectB);
        theCudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
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
    theCudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
    return vectB;
}

double
DistributedCudaBcsrLinSOE::normRHS(void)
{
    return this->getB().Norm();
}

void
DistributedCudaBcsrLinSOE::setX(int loc, double value)
{
    if (theCudaSOE != nullptr)
        theCudaSOE->setX(loc, value);
}

void
DistributedCudaBcsrLinSOE::setX(const Vector &x)
{
    if (theCudaSOE != nullptr)
        theCudaSOE->setX(x);
}

int
DistributedCudaBcsrLinSOE::solve(void)
{
    static ID result(1);

    if (theCudaSOE == nullptr) {
        opserr << "WARNING DistributedCudaBcsrLinSOE::solve() - no CudaBcsrLinSOE\n";
        return -1;
    }

    const bool sendA = needsMatrixTransfer();
    // CudaMKR (etc.) may already hold the transformed global RHS on the GPU.
    // All ranks must agree (setGlobalRhsOnDevice) so B send/recv stay matched.
    const bool skipBMerge = globalRhsOnDevice;
    Vector &vectX = theCudaSOE->getHostXVector();
    Vector &vectB = theCudaSOE->getHostBVector();

    if (processID != 0) {
        Channel *theChannel = theChannels[0];
        const int nTrip = sendA ? static_cast<int>(tripletMap.size()) : 0;
        if (!skipBMerge) {
            // Header after the RHS: -1 if the matrix is unchanged on this rank, else the
            // number of triplets that follow. Rank 0 reads it instead of its own status,
            // so the triplet messages cannot go unmatched.
            myB[myBsize] = sendA ? static_cast<double>(nTrip) : -1.0;
            theChannel->sendVector(0, 0, *myVectBHeader);
        } else if (sendA) {
            // RHS already on the device (CudaMKR): no RHS message to carry the header.
            ID nTripID(1);
            nTripID(0) = nTrip;
            theChannel->sendID(0, 0, nTripID);
        }

        if (sendA) {
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
        if (!skipBMerge)
            theChannel->recvVector(0, 0, vectB);
        theChannel->recvID(0, 0, result);

        theCudaSOE->setXPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
        if (!skipBMerge)
            theCudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
        if (result(0) == 0)
            theCudaSOE->setMatrixStatus(CudaBcsrLinSOE::MatrixStatus::UNCHANGED);

        globalRhsOnDevice = false;
        return result(0);
    }

    // Rank 0: merge RHS (unless already on device) and (if needed) triplets, then GPU solve
    result(0) = 0;
    if (!skipBMerge) {
        vectB = *myVectB;
        theCudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
    }

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
            const double header = workArea[myBsize];
            workerSendsA = header >= 0.0;
            nTrip = workerSendsA ? static_cast<int>(header) : 0;
        } else if (sendA) {
            ID nTripID(1);
            theChannel->recvID(0, 0, nTripID);
            nTrip = nTripID(0);
            if (nTrip < 0) {
                opserr << "WARNING DistributedCudaBcsrLinSOE::solve() - invalid nTrip\n";
                result(0) = -1;
                continue;
            }
        }

        if (workerSendsA != sendA) {
            // A rank skipped (or repeated) its assembly: fail on every rank instead of
            // merging a partial matrix. The triplets are still received below.
            opserr << "WARNING DistributedCudaBcsrLinSOE::solve() - matrix status of rank "
                   << j + 1 << " differs from rank 0\n";
            result(0) = -1;
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

    if (result(0) == 0)
        result(0) = theCudaSOE->solve();

    // Force host X current before channel broadcast (integrators may leave xSyncMode=false).
    theCudaSOE->syncXToHost();
    if (!skipBMerge)
        theCudaSOE->syncBToHost();

    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        theChannel->sendVector(0, 0, vectX);
        if (!skipBMerge)
            theChannel->sendVector(0, 0, vectB);
        theChannel->sendID(0, 0, result);
    }

    globalRhsOnDevice = false;

    return result(0);
}

int
DistributedCudaBcsrLinSOE::formAp(const Vector &p, Vector &Ap)
{
    if (theCudaSOE == nullptr) {
        opserr << "WARNING DistributedCudaBcsrLinSOE::formAp() - no CudaBcsrLinSOE\n";
        return -1;
    }

    const int size = theCudaSOE->getNumEqn();
    if (size != p.Size() || size != Ap.Size()) {
        opserr << "WARNING DistributedCudaBcsrLinSOE::formAp() - vectors must match global size "
               << size << "\n";
        return -1;
    }
    if (size == 0) {
        Ap.Zero();
        return 0;
    }

    // Serial / single-process: local SpMV only.
    if (numChannels <= 0 || theChannels == 0)
        return theCudaSOE->formAp(p, Ap);

    // Same gate as solve(): skip triplet gather when MatrixStatus is UNCHANGED.
    if (ensureMergedA() < 0) {
        opserr << "WARNING DistributedCudaBcsrLinSOE::formAp() - ensureMergedA failed\n";
        return -1;
    }

    if (processID != 0) {
        if (theChannels[0] == 0) {
            opserr << "WARNING DistributedCudaBcsrLinSOE::formAp() - worker has no channel\n";
            return -1;
        }
        if (theChannels[0]->recvVector(0, 0, Ap) < 0) {
            opserr << "WARNING DistributedCudaBcsrLinSOE::formAp() - recv Ap failed\n";
            return -1;
        }
        return 0;
    }

    const int rc = theCudaSOE->formAp(p, Ap);
    if (rc != 0)
        Ap.Zero();

    for (int j = 0; j < numChannels; ++j) {
        if (theChannels[j]->sendVector(0, 0, Ap) < 0) {
            opserr << "WARNING DistributedCudaBcsrLinSOE::formAp() - send Ap failed\n";
            return -1;
        }
    }
    return rc;
}

LinearSOE *
DistributedCudaBcsrLinSOE::getCopy(void) const
{
    if (theCudaSOE == nullptr)
        return nullptr;

    LinearSOE *cudaCopy = theCudaSOE->getCopy();
    if (cudaCopy == nullptr)
        return nullptr;

    CudaBcsrLinSOE *cudaSOE = dynamic_cast<CudaBcsrLinSOE *>(cudaCopy);
    if (cudaSOE == nullptr) {
        delete cudaCopy;
        return nullptr;
    }

    DistributedCudaBcsrLinSOE *out = new DistributedCudaBcsrLinSOE(cudaSOE);
    out->setProcessID(processID);
    out->setChannels(numChannels, theChannels);
    return out;
}

int
DistributedCudaBcsrLinSOE::saveSparseA(OPS_Stream &output, int baseIndex)
{
    if (theCudaSOE == nullptr)
        return -1;
    return theCudaSOE->saveSparseA(output, baseIndex);
}

int
DistributedCudaBcsrLinSOE::sendSelf(int commitTag, Channel &theChannel)
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
        opserr << "WARNING DistributedCudaBcsrLinSOE::sendSelf() - failed to send data\n";
        return -1;
    }
    return 0;
}

int
DistributedCudaBcsrLinSOE::recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    ID idData(1);
    int res = theChannel.recvID(0, commitTag, idData);
    if (res < 0) {
        opserr << "WARNING DistributedCudaBcsrLinSOE::recvSelf() - failed to recv data\n";
        return -1;
    }
    processID = idData(0);

    numChannels = 1;
    if (theChannels != 0)
        delete[] theChannels;
    theChannels = new Channel *[1];
    theChannels[0] = &theChannel;

    // Broker path does not reconstruct the CUDA SOE/solver; OpenSeesMP
    // constructs DistributedCudaBcsrLinSOE via the Distributed* commands.
    (void)theBroker;
    return 0;
}

int
DistributedCudaBcsrLinSOE::setProcessID(int dTag)
{
    processID = dTag;
    // Workers never own the global CSR / GPU operators.
    if (theCudaSOE != nullptr && processID != 0)
        theCudaSOE->setCudaDeviceEnabled(false);
    return 0;
}

int
DistributedCudaBcsrLinSOE::setChannels(int nChannels, Channel **theC)
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

int
DistributedCudaBcsrLinSOE::sendPendingTriplets(Channel *theChannel)
{
    if (theChannel == 0)
        return -1;

    const int nTrip = static_cast<int>(tripletMap.size());
    ID nTripID(1);
    nTripID(0) = nTrip;
    if (theChannel->sendID(0, 0, nTripID) < 0)
        return -1;

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
        if (theChannel->sendID(0, 0, rows) < 0 ||
            theChannel->sendID(0, 0, cols) < 0 ||
            theChannel->sendVector(0, 0, vals) < 0) {
            return -1;
        }
    }
    clearTriplets();
    return 0;
}

int
DistributedCudaBcsrLinSOE::recvAndMergeTriplets(void)
{
    int status = 0;
    for (int j = 0; j < numChannels; ++j) {
        Channel *theChannel = theChannels[j];
        ID nTripID(1);
        if (theChannel->recvID(0, 0, nTripID) < 0) {
            status = -1;
            continue;
        }
        const int nTrip = nTripID(0);
        if (nTrip < 0) {
            status = -1;
            continue;
        }
        if (nTrip > 0) {
            ID rows(nTrip);
            ID cols(nTrip);
            Vector vals(nTrip);
            if (theChannel->recvID(0, 0, rows) < 0 ||
                theChannel->recvID(0, 0, cols) < 0 ||
                theChannel->recvVector(0, 0, vals) < 0) {
                status = -1;
                continue;
            }
            if (status == 0 &&
                mergeTripletsIntoA(&rows(0), &cols(0), &vals(0), nTrip) < 0) {
                status = -1;
            }
        }
    }
    return status;
}

int
DistributedCudaBcsrLinSOE::ensureMergedA(void)
{
    if (theCudaSOE == nullptr)
        return -1;
    if (numChannels <= 0 || theChannels == 0)
        return 0;
    if (!needsMatrixTransfer())
        return 0;

    if (processID != 0) {
        if (theChannels[0] == 0)
            return -1;
        return sendPendingTriplets(theChannels[0]);
    }
    return recvAndMergeTriplets();
}

int
DistributedCudaBcsrLinSOE::broadcastFromRoot(Vector &v)
{
    if (numChannels <= 0 || theChannels == 0)
        return 0;

    if (processID != 0) {
        if (theChannels[0] == 0)
            return -1;
        return theChannels[0]->recvVector(0, 0, v);
    }

    for (int j = 0; j < numChannels; ++j) {
        if (theChannels[j]->sendVector(0, 0, v) < 0)
            return -1;
    }
    return 0;
}

int
DistributedCudaBcsrLinSOE::gatherOwnedToRoot(Vector &v)
{
    if (numChannels <= 0 || theChannels == 0)
        return 0;

    if (processID != 0) {
        if (theChannels[0] == 0)
            return -1;
        return theChannels[0]->sendVector(0, 0, v);
    }

    ensureWorkArea(v.Size());
    Vector remote(workArea, v.Size());
    for (int j = 0; j < numChannels; ++j) {
        if (theChannels[j]->recvVector(0, 0, remote) < 0)
            return -1;
        // Each rank seeds only its AnalysisModel DOFs after domainChanged; other
        // entries stay 0. Fill P0 holes from workers so the GPU predictor sees the
        // full global motion before broadcastFromRoot.
        for (int i = 0; i < v.Size(); ++i) {
            if (v(i) == 0.0 && remote(i) != 0.0)
                v(i) = remote(i);
        }
    }
    return 0;
}

int
DistributedCudaBcsrLinSOE::mergeBToRoot(void)
{
    if (theCudaSOE == nullptr)
        return -1;

    Vector &vectB = theCudaSOE->getHostBVector();

    if (numChannels <= 0 || theChannels == 0) {
        if (myVectB != 0)
            vectB = *myVectB;
        theCudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
        return 0;
    }

    if (processID != 0) {
        if (theChannels[0] == 0)
            return -1;
        if (theChannels[0]->sendVector(0, 0, *myVectB) < 0)
            return -1;
        // Workers keep their myB; callers that need a one-shot global RHS should zeroB after.
        return 0;
    }

    vectB = *myVectB;
    ensureWorkArea(myBsize);
    Vector remoteB(workArea, myBsize);
    for (int j = 0; j < numChannels; ++j) {
        if (theChannels[j]->recvVector(0, 0, remoteB) < 0)
            return -1;
        vectB += remoteB;
    }
    theCudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
    return 0;
}
