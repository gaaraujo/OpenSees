#include "SparsePythonCompressedLinSOE.h"

#include "SparsePythonCompressedLinSolver.h"

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

SparsePythonCompressedLinSOE::SparsePythonCompressedLinSOE(
    SparsePythonCompressedLinSolver &theSolver,
    StorageScheme scheme,
    MatrixType type)
    : LinearSOE(theSolver, LinSOE_TAGS_SparsePythonCompressedLinSOE),
      matrixStatus(MatrixStatus::STRUCTURE_CHANGED),
      storageScheme(scheme),
      theMatrixType(type)
{
    if (storageScheme == StorageScheme::COO) {
        opserr << "WARNING: SparsePythonCompressedLinSOE does not support COO; defaulting to CSR" << endln;
        storageScheme = StorageScheme::CSR;
    }
    theSolver.setLinearSOE(*this);
    updateVectorViews();
}

SparsePythonCompressedLinSOE::SparsePythonCompressedLinSOE(StorageScheme scheme,
                                                           MatrixType type)
    : LinearSOE(LinSOE_TAGS_SparsePythonCompressedLinSOE),
      matrixStatus(MatrixStatus::STRUCTURE_CHANGED),
      storageScheme(scheme),
      theMatrixType(type)
{
    if (storageScheme == StorageScheme::COO) {
        opserr << "WARNING: SparsePythonCompressedLinSOE does not support COO; defaulting to CSR" << endln;
        storageScheme = StorageScheme::CSR;
    }
    updateVectorViews();
}

SparsePythonCompressedLinSOE::~SparsePythonCompressedLinSOE() = default;

int
SparsePythonCompressedLinSOE::getNumEqn(void) const
{
    return static_cast<int>(sol.size());
}

void
SparsePythonCompressedLinSOE::buildCSRPattern(Graph &theGraph)
{
    const int size = theGraph.getNumVertex();
    const std::size_t nnz = static_cast<std::size_t>(size) + 2 * theGraph.getNumEdge();
    const bool lower = (theMatrixType == MatrixType::SYMMETRIC_LOWER);

    indexPtr.assign(static_cast<std::size_t>(size) + 1u, 0);
    indices.resize(nnz);
    values.assign(nnz, 0.0);
    rhs.assign(static_cast<std::size_t>(size), 0.0);
    sol.assign(static_cast<std::size_t>(size), 0.0);

    int last = 0;
    indexPtr[0] = 0;
    Vertex *theVertex = nullptr;
    for (int row = 0; row < size; ++row) {
        theVertex = theGraph.getVertexPtr(row);
        if (theVertex == nullptr) {
            opserr << "WARNING: SparsePythonCompressedLinSOE::buildCSRPattern - "
                   << "vertex " << row << " not found in graph, "
                   << "size set to 0" << endln;
            indexPtr.assign(1u, 0);
            indices.clear();
            values.clear();
            rhs.clear();
            sol.clear();
            updateVectorViews();
            return;
        }

        const ID &theAdjacency = theVertex->getAdjacency();
        ID columns(0, theAdjacency.Size() + 1);

        columns.insert(theVertex->getTag());
        for (int i = 0; i < theAdjacency.Size(); ++i) {
            columns.insert(theAdjacency(i));
        }

        for (int i = 0; i < columns.Size(); ++i) {
            if (lower && columns(i) > row) {
                continue;
            }
            indices[static_cast<std::size_t>(last++)] = columns(i);
        }

        indexPtr[static_cast<std::size_t>(row + 1)] = last;
    }

    if (static_cast<std::size_t>(last) < nnz) {
        indices.resize(static_cast<std::size_t>(last));
        values.resize(static_cast<std::size_t>(last));
    }
}

void
SparsePythonCompressedLinSOE::buildCSCPattern(Graph &theGraph)
{
    const int size = theGraph.getNumVertex();
    const std::size_t nnz = static_cast<std::size_t>(size) + 2 * theGraph.getNumEdge();
    const bool lower = (theMatrixType == MatrixType::SYMMETRIC_LOWER);

    indexPtr.assign(static_cast<std::size_t>(size) + 1u, 0);
    indices.resize(nnz);
    values.assign(nnz, 0.0);
    rhs.assign(static_cast<std::size_t>(size), 0.0);
    sol.assign(static_cast<std::size_t>(size), 0.0);

    int last = 0;
    indexPtr[0] = 0;
    Vertex *theVertex = nullptr;
    for (int col = 0; col < size; ++col) {
        theVertex = theGraph.getVertexPtr(col);
        if (theVertex == nullptr) {
            opserr << "WARNING: SparsePythonCompressedLinSOE::buildCSCPattern - "
                   << "vertex " << col << " not found in graph, "
                   << "size set to 0" << endln;
            indexPtr.assign(1u, 0);
            indices.clear();
            values.clear();
            rhs.clear();
            sol.clear();
            updateVectorViews();
            return;
        }

        const ID &theAdjacency = theVertex->getAdjacency();
        ID rows(0, theAdjacency.Size() + 1);

        rows.insert(theVertex->getTag());
        for (int i = 0; i < theAdjacency.Size(); ++i) {
            rows.insert(theAdjacency(i));
        }

        for (int i = 0; i < rows.Size(); ++i) {
            if (lower && rows(i) < col) {
                continue;
            }
            indices[static_cast<std::size_t>(last++)] = rows(i);
        }

        indexPtr[static_cast<std::size_t>(col + 1)] = last;
    }

    if (static_cast<std::size_t>(last) < nnz) {
        indices.resize(static_cast<std::size_t>(last));
        values.resize(static_cast<std::size_t>(last));
    }
}

int
SparsePythonCompressedLinSOE::setSize(Graph &theGraph)
{
    const int size = theGraph.getNumVertex();
    if (size < 0) {
        opserr << "WARNING: SparsePythonCompressedLinSOE::setSize - "
               << "size of SOE < 0" << endln;
        return -1;
    }

    if (storageScheme == StorageScheme::CSR) {
        buildCSRPattern(theGraph);
    } else if (storageScheme == StorageScheme::CSC) {
        buildCSCPattern(theGraph);
    } else {
        opserr << "WARNING: SparsePythonCompressedLinSOE::setSize - unknown storage scheme\n";
        return -1;
    }

    updateVectorViews();

    LinearSOESolver *theSolver = this->getSolver();
    if (theSolver != nullptr) {
        const int solverOK = theSolver->setSize();
        if (solverOK < 0) {
            opserr << "WARNING: SparsePythonCompressedLinSOE::setSize -"
                   << "solver failed setSize()" << endln;
            return solverOK;
        }
    }

    matrixStatus = MatrixStatus::STRUCTURE_CHANGED;

    return 0;
}

SparsePythonCompressedLinSOE::MatrixStatus
SparsePythonCompressedLinSOE::getMatrixStatus() const
{
    return matrixStatus;
}

int
SparsePythonCompressedLinSOE::resizeVectors(int size)
{
    if (size < 0) {
        opserr << "WARNING: SparsePythonCompressedLinSOE::resizeVectors - size < 0\n";
        return -1;
    }

    indexPtr.clear();
    indices.clear();
    values.clear();
    rhs.assign(static_cast<std::size_t>(size), 0.0);
    sol.assign(static_cast<std::size_t>(size), 0.0);
    updateVectorViews();
    matrixStatus = MatrixStatus::STRUCTURE_CHANGED;
    return 0;
}

int
SparsePythonCompressedLinSOE::addA(const Matrix &m, const ID &id, double fact)
{
    if (fact == 0.0) {
        return 0;
    }

    const int idSize = id.Size();
    if (idSize != m.noRows() || idSize != m.noCols()) {
        opserr << "SparsePythonCompressedLinSOE::addA - Matrix and ID not of similar sizes\n";
        return -1;
    }

    const int size = getNumEqn();
    const bool csr = (storageScheme == StorageScheme::CSR);
    const bool lower = (theMatrixType == MatrixType::SYMMETRIC_LOWER);

    // Outer index: row for CSR, column for CSC. Each compressed slice is sorted,
    // so entries are located by binary search.
    for (int a = 0; a < idSize; ++a) {
        const int outer = id(a);
        if (outer < 0 || outer >= size) {
            continue;
        }

        auto beginIt = indices.begin() + indexPtr[static_cast<std::size_t>(outer)];
        auto endIt = indices.begin() + indexPtr[static_cast<std::size_t>(outer + 1)];

        for (int b = 0; b < idSize; ++b) {
            const int inner = id(b);
            if (inner < 0 || inner >= size) {
                continue;
            }
            // SYMMETRIC_LOWER keeps row >= column only.
            if (lower && (csr ? inner > outer : inner < outer)) {
                continue;
            }

            auto it = std::lower_bound(beginIt, endIt, inner);
            if (it != endIt && *it == inner) {
                const std::size_t offset = static_cast<std::size_t>(it - indices.begin());
                values[offset] += fact * (csr ? m(a, b) : m(b, a));
            }
        }
    }

    if (matrixStatus != MatrixStatus::STRUCTURE_CHANGED) {
        matrixStatus = MatrixStatus::COEFFICIENTS_CHANGED;
    }
    return 0;
}

int
SparsePythonCompressedLinSOE::addB(const Vector &v, const ID &id, double fact)
{
    if (fact == 0.0) {
        return 0;
    }

    const int idSize = id.Size();
    if (idSize != v.Size()) {
        opserr << "SparsePythonCompressedLinSOE::addB - Vector and ID not of similar sizes\n";
        return -1;
    }

    const int size = getNumEqn();

    if (fact == 1.0) {
        for (int i = 0; i < idSize; ++i) {
            const int pos = id(i);
            if (pos < 0 || pos >= size) {
                continue;
            }
            rhs[static_cast<std::size_t>(pos)] += v(i);
        }
    } else if (fact == -1.0) {
        for (int i = 0; i < idSize; ++i) {
            const int pos = id(i);
            if (pos < 0 || pos >= size) {
                continue;
            }
            rhs[static_cast<std::size_t>(pos)] -= v(i);
        }
    } else {
        for (int i = 0; i < idSize; ++i) {
            const int pos = id(i);
            if (pos < 0 || pos >= size) {
                continue;
            }
            rhs[static_cast<std::size_t>(pos)] += v(i) * fact;
        }
    }
    return 0;
}

int
SparsePythonCompressedLinSOE::setB(const Vector &v, double fact)
{
    if (fact == 0.0) {
        std::fill(rhs.begin(), rhs.end(), 0.0);
        bVector.Zero();
        return 0;
    }

    if (v.Size() != getNumEqn()) {
        opserr << "SparsePythonCompressedLinSOE::setB - incompatible sizes "
               << getNumEqn() << " and " << v.Size() << endln;
        return -1;
    }

    const int size = getNumEqn();
    if (fact == 1.0) {
        for (int i = 0; i < size; ++i) {
            rhs[static_cast<std::size_t>(i)] = v(i);
        }
    } else if (fact == -1.0) {
        for (int i = 0; i < size; ++i) {
            rhs[static_cast<std::size_t>(i)] = -v(i);
        }
    } else {
        for (int i = 0; i < size; ++i) {
            rhs[static_cast<std::size_t>(i)] = v(i) * fact;
        }
    }
    return 0;
}

void
SparsePythonCompressedLinSOE::zeroA(void)
{
    std::fill(values.begin(), values.end(), 0.0);
    if (matrixStatus != MatrixStatus::STRUCTURE_CHANGED) {
        matrixStatus = MatrixStatus::COEFFICIENTS_CHANGED;
    }
}

void
SparsePythonCompressedLinSOE::zeroB(void)
{
    std::fill(rhs.begin(), rhs.end(), 0.0);
}

const Vector &
SparsePythonCompressedLinSOE::getX(void)
{
    return xVector;
}

const Vector &
SparsePythonCompressedLinSOE::getB(void)
{
    return bVector;
}

double
SparsePythonCompressedLinSOE::normRHS(void)
{
    double norm = 0.0;
    for (double value : rhs) {
        norm += value * value;
    }
    return std::sqrt(norm);
}

void
SparsePythonCompressedLinSOE::setX(int loc, double value)
{
    if (loc >= 0 && loc < getNumEqn()) {
        sol[static_cast<std::size_t>(loc)] = value;
    }
}

void
SparsePythonCompressedLinSOE::setX(const Vector &X)
{
    if (X.Size() != getNumEqn()) {
        return;
    }

    const int size = getNumEqn();
    for (int i = 0; i < size; ++i) {
        sol[static_cast<std::size_t>(i)] = X(i);
    }
}

int
SparsePythonCompressedLinSOE::sendSelf(int, Channel &)
{
    return 0;
}

int
SparsePythonCompressedLinSOE::recvSelf(int, Channel &, FEM_ObjectBroker &)
{
    return 0;
}

int
SparsePythonCompressedLinSOE::setPythonSolver(SparsePythonCompressedLinSolver &newSolver)
{
    newSolver.setLinearSOE(*this);
    if (getNumEqn() != 0) {
        const int solverOK = newSolver.setSize();
        if (solverOK < 0) {
            opserr << "WARNING: SparsePythonCompressedLinSOE::setPythonSolver - "
                   << "the new solver could not setSize(), keeping the old solver\n";
            return -1;
        }
    }
    return this->LinearSOE::setSolver(newSolver);
}

void
SparsePythonCompressedLinSOE::updateVectorViews()
{
    if (sol.empty()) {
        xVector.resize(0);
    } else {
        xVector.setData(sol.data(), static_cast<int>(sol.size()));
    }

    if (rhs.empty()) {
        bVector.resize(0);
    } else {
        bVector.setData(rhs.data(), static_cast<int>(rhs.size()));
    }
}

int
SparsePythonCompressedLinSOE::solve(void)
{
    int status = this->LinearSOE::solve();
    if (status < 0) {
        return status;
    }
    matrixStatus = MatrixStatus::UNCHANGED;
    return status;
}


