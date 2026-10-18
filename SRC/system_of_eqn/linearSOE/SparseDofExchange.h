/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
** ****************************************************************** */

// Written: gaaraujo
// Created: 10/2026
//
// Description: rank-local DOF lists for the gather-to-root SOEs
// (DistributedCudaBcsrLinSOE, DistributedSparseCsrLinSOE), so that vectors
// exchanged between rank 0 and a worker carry only the entries of the
// equations that the worker's analysis model holds (the vertices of the DOF
// graph it sends in setSize: owned plus interface DOFs) instead of the whole
// global vector.
//
//   RHS gather (worker -> rank 0): the worker's RHS is zero outside its DOFs
//   (addB only writes the IDs of its own FE_Elements and DOF_Groups), so rank 0
//   adds the packed values at those positions, worker by worker in the same
//   order as with full-length messages: the merged RHS is the same. A worker
//   whose RHS was set as a whole (setB) flags it and sends the full vector in a
//   second message, which rank 0 adds as before.
//
//   Local scatter (rank 0 -> worker): the values of several vectors at the
//   worker's DOFs, in one message. Entries outside the worker's DOFs are left
//   untouched on the worker, so this is only for vectors that the worker uses
//   through its own DOF IDs (the motion of an explicit integrator, or a solution
//   that no convergence test reads).
//
// OPS_SOE_FULL_VECTORS=1 in the environment (same on every rank) makes every
// rank's list hold all equations, i.e. the full-length messages of before, to
// bisect a regression.

#ifndef SparseDofExchange_h
#define SparseDofExchange_h

#include <Channel.h>
#include <Graph.h>
#include <Vector.h>
#include <Vertex.h>
#include <VertexIter.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>

class SparseDofExchange
{
public:
    void reset(int numWorkers)
    {
        local.clear();
        workers.assign(numWorkers > 0 ? static_cast<std::size_t>(numWorkers) : 0, std::vector<int>());
    }

    /** Worker: its own DOFs, from the DOF graph it sends to rank 0 in setSize. */
    void setLocal(Graph &theGraph, int numEqn) { local = vertexTags(theGraph, numEqn); }

    /** Rank 0: the DOFs of worker j (0-based channel index), from the graph it received. */
    void setWorker(int j, Graph &theGraph, int numEqn)
    {
        if (j >= 0 && j < static_cast<int>(workers.size()))
            workers[j] = vertexTags(theGraph, numEqn);
    }

    /** After setSize, on every rank: with OPS_SOE_FULL_VECTORS=1, every list becomes all numEqn
     *  equations (full-length messages, as before the packed exchange). */
    void applyFullVectorsOption(int numEqn)
    {
        const char *e = std::getenv("OPS_SOE_FULL_VECTORS");
        if (e == nullptr || *e == '\0' || *e == '0')
            return;
        std::vector<int> all(static_cast<std::size_t>(numEqn > 0 ? numEqn : 0));
        for (int i = 0; i < numEqn; ++i)
            all[i] = i;
        local = all;
        for (std::vector<int> &w : workers)
            w = all;
    }

    /** Worker -> rank 0: the RHS b (n entries) packed at the local DOFs, a dense flag, then nExtra
     *  trailing values (headers) in one message; if dense, the full b follows in a second message. */
    int sendRhs(Channel &theChannel, const double *b, int n, bool dense, const double *extra, int nExtra)
    {
        const int m = static_cast<int>(local.size());
        buf.resize(static_cast<std::size_t>(m) + 1 + nExtra);
        for (int k = 0; k < m; ++k)
            buf[k] = b[local[k]];
        buf[m] = dense ? 1.0 : 0.0;
        for (int k = 0; k < nExtra; ++k)
            buf[m + 1 + k] = extra[k];
        Vector msg(buf.data(), m + 1 + nExtra);
        if (theChannel.sendVector(0, 0, msg) < 0)
            return -1;
        if (dense) {
            Vector full(const_cast<double *>(b), n);
            if (theChannel.sendVector(0, 0, full) < 0)
                return -1;
        }
        return 0;
    }

    /** Rank 0: add worker j's RHS into b (n entries); the trailing values go to extra. */
    int recvAddRhs(Channel &theChannel, int j, double *b, int n, double *extra, int nExtra)
    {
        const std::vector<int> &dofs = workers[j];
        const int m = static_cast<int>(dofs.size());
        buf.resize(static_cast<std::size_t>(m) + 1 + nExtra);
        Vector msg(buf.data(), m + 1 + nExtra);
        if (theChannel.recvVector(0, 0, msg) < 0)
            return -1;
        for (int k = 0; k < nExtra; ++k)
            extra[k] = buf[m + 1 + k];
        if (buf[m] != 0.0) {
            full.resize(static_cast<std::size_t>(n));
            Vector fullMsg(full.data(), n);
            if (theChannel.recvVector(0, 0, fullMsg) < 0)
                return -1;
            for (int i = 0; i < n; ++i)
                b[i] += full[i];
            return 0;
        }
        for (int k = 0; k < m; ++k)
            b[dofs[k]] += buf[k];
        return 0;
    }

    /** Rank 0 -> worker j: the entries of nv vectors at its DOFs, then nExtra trailing values. */
    int sendLocal(Channel &theChannel, int j, const double *const *v, int nv, const double *extra, int nExtra)
    {
        const std::vector<int> &dofs = workers[j];
        const int m = static_cast<int>(dofs.size());
        const int size = std::max(1, nv * m + nExtra);
        buf.resize(static_cast<std::size_t>(size));
        for (int a = 0; a < nv; ++a) {
            const double *va = v[a];
            double *out = buf.data() + static_cast<std::size_t>(a) * m;
            for (int k = 0; k < m; ++k)
                out[k] = va[dofs[k]];
        }
        for (int k = 0; k < nExtra; ++k)
            buf[nv * m + k] = extra[k];
        Vector msg(buf.data(), size);
        return theChannel.sendVector(0, 0, msg);
    }

    /** Worker: receive what sendLocal sent into its DOFs of the nv vectors; trailing values to extra. */
    int recvLocal(Channel &theChannel, double *const *v, int nv, double *extra, int nExtra)
    {
        const int m = static_cast<int>(local.size());
        const int size = std::max(1, nv * m + nExtra);
        buf.resize(static_cast<std::size_t>(size));
        Vector msg(buf.data(), size);
        if (theChannel.recvVector(0, 0, msg) < 0)
            return -1;
        for (int a = 0; a < nv; ++a) {
            double *va = v[a];
            const double *in = buf.data() + static_cast<std::size_t>(a) * m;
            for (int k = 0; k < m; ++k)
                va[local[k]] = in[k];
        }
        for (int k = 0; k < nExtra; ++k)
            extra[k] = buf[nv * m + k];
        return 0;
    }

private:
    static std::vector<int> vertexTags(Graph &theGraph, int numEqn)
    {
        std::vector<int> tags;
        tags.reserve(static_cast<std::size_t>(theGraph.getNumVertex()));
        VertexIter &it = theGraph.getVertices();
        Vertex *v;
        while ((v = it()) != 0) {
            const int t = v->getTag();
            if (t >= 0 && t < numEqn)
                tags.push_back(t);
        }
        std::sort(tags.begin(), tags.end());
        tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
        return tags;
    }

    std::vector<int> local;                 // worker: own DOFs, sorted
    std::vector<std::vector<int>> workers;  // rank 0: DOFs of each worker
    std::vector<double> buf;                // message buffer
    std::vector<double> full;               // rank 0: a dense worker RHS
};

#endif
