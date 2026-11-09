#include "SparsePythonCompressedEigenSOE.h"

#include "SparsePythonCompressedEigenSolver.h"

#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include <Graph.h>
#include <ID.h>
#include <Matrix.h>
#include <OPS_Globals.h>
#include <Vertex.h>
#include <VertexIter.h>

#include <algorithm>
#include <cmath>

SparsePythonCompressedEigenSOE::SparsePythonCompressedEigenSOE(
    SparsePythonCompressedEigenSolver &theSolver,
    StorageScheme scheme)
    : EigenSOE(theSolver, EigenSOE_TAGS_SparsePythonCompressedEigenSOE),
      matrixStatus(MatrixStatus::STRUCTURE_CHANGED),
      storageScheme(scheme)
{
    if (storageScheme == StorageScheme::COO) {
        opserr << "WARNING: SparsePythonCompressedEigenSOE does not support COO; defaulting to CSR" << endln;
        storageScheme = StorageScheme::CSR;
    }
    theSolver.setEigenSOE(*this);
}

SparsePythonCompressedEigenSOE::SparsePythonCompressedEigenSOE(StorageScheme scheme)
    : EigenSOE(EigenSOE_TAGS_SparsePythonCompressedEigenSOE),
      matrixStatus(MatrixStatus::STRUCTURE_CHANGED),
      storageScheme(scheme)
{
    if (storageScheme == StorageScheme::COO) {
        opserr << "WARNING: SparsePythonCompressedEigenSOE does not support COO; defaulting to CSR" << endln;
        storageScheme = StorageScheme::CSR;
    }
}

SparsePythonCompressedEigenSOE::~SparsePythonCompressedEigenSOE() = default;

int
SparsePythonCompressedEigenSOE::getNumEqn(void) const
{
    return static_cast<int>(indexPtr.size() > 0 ? indexPtr.size() - 1 : 0);
}

void
SparsePythonCompressedEigenSOE::buildCSRPattern(Graph &theGraph)
{
    kCache.clear();
    mCache.clear();
    const int size = theGraph.getNumVertex();
    const std::size_t nnz = static_cast<std::size_t>(size) + 2 * theGraph.getNumEdge();

    // Build shared sparsity pattern
    indexPtr.assign(static_cast<std::size_t>(size) + 1u, 0);
    indices.resize(nnz);
    kValues.assign(nnz, 0.0);
    mValues.assign(nnz, 0.0);

    int last = 0;
    indexPtr[0] = 0;
    Vertex *theVertex = nullptr;
    for (int row = 0; row < size; ++row) {
        theVertex = theGraph.getVertexPtr(row);
        if (theVertex == nullptr) {
            opserr << "WARNING: SparsePythonCompressedEigenSOE::buildCSRPattern - "
                   << "vertex " << row << " not found in graph, "
                   << "size set to 0" << endln;
            indexPtr.assign(1u, 0);
            indices.clear();
            kValues.clear();
            mValues.clear();
            return;
        }

        const ID &theAdjacency = theVertex->getAdjacency();
        ID columns(0, theAdjacency.Size() + 1);

        columns.insert(theVertex->getTag());
        for (int i = 0; i < theAdjacency.Size(); ++i) {
            columns.insert(theAdjacency(i));
        }

        for (int i = 0; i < columns.Size(); ++i) {
            indices[static_cast<std::size_t>(last)] = columns(i);
            last++;
        }

        indexPtr[static_cast<std::size_t>(row + 1)] = last;
    }
}

void
SparsePythonCompressedEigenSOE::buildCSCPattern(Graph &theGraph)
{
    kCache.clear();
    mCache.clear();
    const int size = theGraph.getNumVertex();
    const std::size_t nnz = static_cast<std::size_t>(size) + 2 * theGraph.getNumEdge();

    // Build shared sparsity pattern
    indexPtr.assign(static_cast<std::size_t>(size) + 1u, 0);
    indices.resize(nnz);
    kValues.assign(nnz, 0.0);
    mValues.assign(nnz, 0.0);

    int last = 0;
    indexPtr[0] = 0;
    Vertex *theVertex = nullptr;
    for (int col = 0; col < size; ++col) {
        theVertex = theGraph.getVertexPtr(col);
        if (theVertex == nullptr) {
            opserr << "WARNING: SparsePythonCompressedEigenSOE::buildCSCPattern - "
                   << "vertex " << col << " not found in graph, "
                   << "size set to 0" << endln;
            indexPtr.assign(1u, 0);
            indices.clear();
            kValues.clear();
            mValues.clear();
            return;
        }

        const ID &theAdjacency = theVertex->getAdjacency();
        ID rows(0, theAdjacency.Size() + 1);

        rows.insert(theVertex->getTag());
        for (int i = 0; i < theAdjacency.Size(); ++i) {
            rows.insert(theAdjacency(i));
        }

        for (int i = 0; i < rows.Size(); ++i) {
            indices[static_cast<std::size_t>(last)] = rows(i);
            last++;
        }

        indexPtr[static_cast<std::size_t>(col + 1)] = last;
    }
}

int
SparsePythonCompressedEigenSOE::setSize(Graph &theGraph)
{
    const int size = theGraph.getNumVertex();
    if (size < 0) {
        opserr << "WARNING: SparsePythonCompressedEigenSOE::setSize - "
               << "size of SOE < 0" << endln;
        return -1;
    }

    if (storageScheme == StorageScheme::CSR) {
        buildCSRPattern(theGraph);
    } else if (storageScheme == StorageScheme::CSC) {
        buildCSCPattern(theGraph);
    } else {
        opserr << "WARNING: SparsePythonCompressedEigenSOE::setSize - unknown storage scheme\n";
        return -1;
    }

    EigenSolver *theSolver = this->getSolver();
    if (theSolver != nullptr) {
        const int solverOK = theSolver->setSize();
        if (solverOK < 0) {
            opserr << "WARNING: SparsePythonCompressedEigenSOE::setSize -"
                   << "solver failed setSize()" << endln;
            return solverOK;
        }
    }

    matrixStatus = MatrixStatus::STRUCTURE_CHANGED;

    return 0;
}

SparsePythonCompressedEigenSOE::MatrixStatus
SparsePythonCompressedEigenSOE::getMatrixStatus() const
{
    return matrixStatus;
}

int
SparsePythonCompressedEigenSOE::resizeVectors(int size)
{
    if (size < 0) {
        opserr << "WARNING: SparsePythonCompressedEigenSOE::resizeVectors - size < 0\n";
        return -1;
    }

    // Keep getNumEqn() == size via indexPtr length; no values/pattern on workers.
    indexPtr.assign(static_cast<std::size_t>(size) + 1u, 0);
    kCache.clear();
    mCache.clear();
    indices.clear();
    kValues.clear();
    mValues.clear();
    matrixStatus = MatrixStatus::STRUCTURE_CHANGED;
    return 0;
}

void
SparsePythonCompressedEigenSOE::updateMatrixStatus()
{
    if (matrixStatus != MatrixStatus::STRUCTURE_CHANGED) {
        matrixStatus = MatrixStatus::COEFFICIENTS_CHANGED;
    }
}

int
SparsePythonCompressedEigenSOE::addToMatrix(SparseSlotCache &cache, std::vector<double> &values,
                                            const Matrix &m, const ID &id, double fact,
                                            const char *caller)
{
    if (fact == 0.0) {
        return 0;
    }

    const int idSize = id.Size();
    if (idSize != m.noRows() || idSize != m.noCols()) {
        opserr << "SparsePythonCompressedEigenSOE::" << caller << " - Matrix and ID not of similar sizes\n";
        return -1;
    }

    const int size = getNumEqn();
    const bool csr = (storageScheme == StorageScheme::CSR);

    // Outer index: row for CSR, column for CSC. Each compressed slice is sorted,
    // so entries are located by binary search the first time; later assemblies
    // replay the cached slots.
    auto find = [&](int a, int b) -> int {
        const int outer = id(a);
        const int inner = id(b);
        if (outer < 0 || outer >= size || inner < 0 || inner >= size) {
            return SparseSlotCache::SKIP;
        }
        auto beginIt = indices.begin() + indexPtr[static_cast<std::size_t>(outer)];
        auto endIt = indices.begin() + indexPtr[static_cast<std::size_t>(outer + 1)];
        auto it = std::lower_bound(beginIt, endIt, inner);
        if (it != endIt && *it == inner) {
            return static_cast<int>(it - indices.begin());
        }
        return SparseSlotCache::SKIP;
    };
    cache.setCapacity(8 * values.size() + (std::size_t(1) << 20));
    addWithCache(cache, m, id, fact, !csr, find, [&]() { return values.data(); });

    updateMatrixStatus();
    return 0;
}

int
SparsePythonCompressedEigenSOE::addA(const Matrix &m, const ID &id, double fact)
{
    return addToMatrix(kCache, kValues, m, id, fact, "addA");
}

int
SparsePythonCompressedEigenSOE::addM(const Matrix &m, const ID &id, double fact)
{
    return addToMatrix(mCache, mValues, m, id, fact, "addM");
}

void
SparsePythonCompressedEigenSOE::zeroA(void)
{
    kCache.rewind();
    std::fill(kValues.begin(), kValues.end(), 0.0);
    updateMatrixStatus();
}

void
SparsePythonCompressedEigenSOE::zeroM(void)
{
    mCache.rewind();
    std::fill(mValues.begin(), mValues.end(), 0.0);
    updateMatrixStatus();
}

int
SparsePythonCompressedEigenSOE::setPythonSolver(SparsePythonCompressedEigenSolver &newSolver)
{
    return setSolver(newSolver);
}

int
SparsePythonCompressedEigenSOE::sendSelf(int commitTag, Channel &theChannel)
{
    // Not implemented yet
    return 0;
}

int
SparsePythonCompressedEigenSOE::recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    // Not implemented yet
    return 0;
}

