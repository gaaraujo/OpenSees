/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
** ****************************************************************** */

// Written: gaaraujo
// Created: 10/2026
//
// Description: helpers that let sparse LinearSOEs with a fixed pattern
// assemble without searching for every entry of every element matrix.
//
//   SparseSlotCache    records, in call order, the value slot of every entry
//                      that addA() touches, and replays the slots on later
//                      assemblies. Each call is guarded by its ID: a call whose
//                      ID differs from the recorded one drops the rest of the
//                      record and is searched (and recorded) again. zeroA()
//                      rewinds the cursor; a structure change clears the cache.
//   SparseLocalTriplets worker side of a gather-to-root SOE: rank-local
//                      (row, col, value) triplets in flat arrays, with slots
//                      cached as above. After the pattern has reached rank 0,
//                      only the values are sent.
//   SparseRootScatter  rank-0 side: one slot -> value-offset array per worker.
//
// Results are identical to a search-based assembly: every value is added to
// the same slot in the same order.

#ifndef SparseAssemblyCache_h
#define SparseAssemblyCache_h

#include <Matrix.h>
#include <ID.h>
#include <Vector.h>
#include <Channel.h>
#include <OPS_Globals.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

class SparseSlotCache
{
public:
    // Slot values returned by the search functor of addWithCache().
    enum : int {
        SKIP = -1,    // entry not stored (out of range, other triangle)
        MISSING = -2  // entry missing from the pattern
    };

    void clear(void)
    {
        ids.clear();
        slots.clear();
        idStart.assign(1, 0);
        slotStart.assign(1, 0);
        cursor = 0;
    }

    void rewind(void) { cursor = 0; }

    // Upper bound on the number of cached slots (guards against callers that
    // keep adding without ever calling zeroA()).
    void setCapacity(std::size_t maxSlots) { capacity = maxSlots; }

    std::size_t numSlots(void) const { return slots.size(); }

    // Slots of the call at the cursor if its ID matches, else nullptr (and the
    // record from the cursor on is dropped).
    const int *lookup(const ID &id)
    {
        const std::size_t numCalls = idStart.size() - 1;
        if (cursor >= numCalls)
            return nullptr;
        const int n = id.Size();
        const std::size_t i0 = idStart[cursor];
        bool same = (idStart[cursor + 1] - i0 == static_cast<std::size_t>(n));
        for (int k = 0; same && k < n; ++k)
            same = (ids[i0 + k] == id(k));
        if (same)
            return slots.data() + slotStart[cursor++];
        truncate();
        return nullptr;
    }

    // Writable slots for a new call at the cursor, or nullptr if the cache is full.
    int *beginRecord(const ID &id)
    {
        truncate();
        const int n = id.Size();
        const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
        if (slots.size() + nn > capacity)
            return nullptr;
        for (int k = 0; k < n; ++k)
            ids.push_back(id(k));
        slots.resize(slots.size() + nn);
        idStart.push_back(ids.size());
        slotStart.push_back(slots.size());
        return slots.data() + slotStart[cursor++];
    }

    // Undo the last beginRecord() (the call failed part way).
    void dropLast(void)
    {
        if (cursor == 0 || cursor != idStart.size() - 1)
            return;
        --cursor;
        truncate();
    }

private:
    void truncate(void)
    {
        if (cursor + 1 >= idStart.size())
            return;
        ids.resize(idStart[cursor]);
        slots.resize(slotStart[cursor]);
        idStart.resize(cursor + 1);
        slotStart.resize(cursor + 1);
    }

    std::vector<int> ids, slots;
    std::vector<std::size_t> idStart = std::vector<std::size_t>(1, 0);
    std::vector<std::size_t> slotStart = std::vector<std::size_t>(1, 0);
    std::size_t cursor = 0;
    std::size_t capacity = static_cast<std::size_t>(-1);
};

// Adds fact * m (or its transpose) over id into the values returned by values(),
// using the cached slots when the call matches the record. find(i, j) returns the
// value slot of entry (id(i), id(j)), SparseSlotCache::SKIP or SparseSlotCache::MISSING;
// values() is re-read after each find() so find() may grow the value array.
// Returns 0, or -1 (with failedI/failedJ set) on the first MISSING entry.
template <class Find, class Values>
int addWithCache(SparseSlotCache &cache, const Matrix &m, const ID &id, double fact,
                 bool transpose, Find find, Values values, int *failedI = nullptr, int *failedJ = nullptr)
{
    const int n = id.Size();
    if (const int *s = cache.lookup(id)) {
        double *v = values();
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j, ++s)
                if (*s >= 0)
                    v[*s] += fact * (transpose ? m(j, i) : m(i, j));
        return 0;
    }

    int *rec = cache.beginRecord(id);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const int slot = find(i, j);
            if (rec != nullptr)
                *rec++ = slot;
            if (slot >= 0) {
                values()[slot] += fact * (transpose ? m(j, i) : m(i, j));
            } else if (slot == SparseSlotCache::MISSING) {
                if (rec != nullptr)
                    cache.dropLast();
                if (failedI != nullptr)
                    *failedI = i;
                if (failedJ != nullptr)
                    *failedJ = j;
                return -1;
            }
        }
    }
    return 0;
}

class SparseLocalTriplets
{
public:
    void reset(void)
    {
        index.clear();
        rows.clear();
        cols.clear();
        vals.clear();
        cache.clear();
        numSent = 0;
        pending = false;
    }

    // zeroA(): keep the pattern and the slots, zero the values.
    void zero(void)
    {
        clearValues();
        cache.rewind();
    }

    // After a transfer that hands the values over to rank 0: nothing pending.
    void clearValues(void)
    {
        std::fill(vals.begin(), vals.end(), 0.0);
        pending = false;
    }

    // Accumulate fact * m over id, keeping (row, col) with 0 <= row, col < size
    // (and row >= col when lowerOnly). Returns true if any entry was kept.
    bool add(const Matrix &m, const ID &id, double fact, int size, bool lowerOnly = false)
    {
        bool kept = false;
        for (int i = 0; i < id.Size() && !kept; ++i) {
            const int row = id(i);
            if (row < 0 || row >= size)
                continue;
            for (int j = 0; j < id.Size(); ++j) {
                const int col = id(j);
                if (col >= 0 && col < size && !(lowerOnly && row < col)) {
                    kept = true;
                    break;
                }
            }
        }
        if (!kept)
            return false;

        cache.setCapacity(8 * vals.size() + (std::size_t(1) << 20));
        auto find = [&](int i, int j) -> int {
            const int row = id(i);
            const int col = id(j);
            if (row < 0 || row >= size || col < 0 || col >= size || (lowerOnly && row < col))
                return SparseSlotCache::SKIP;
            const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) |
                                 static_cast<uint64_t>(static_cast<uint32_t>(col));
            const auto it = index.emplace(key, static_cast<int>(vals.size()));
            if (it.second) {
                rows.push_back(row);
                cols.push_back(col);
                vals.push_back(0.0);
            }
            return it.first->second;
        };
        addWithCache(cache, m, id, fact, false, find, [&]() { return vals.data(); });
        pending = true;
        return true;
    }

    int size(void) const { return static_cast<int>(vals.size()); }

    // Count announced to rank 0 for the triplets that send() will transfer:
    // n >= 0 for n full triplets (rows, cols, values), -(n + 2) for n values only.
    int header(void) const
    {
        if (!pending || vals.empty())
            return 0;
        return (numSent == size()) ? -(size() + 2) : size();
    }

    // Sends what header() announced; afterwards rank 0 knows the whole pattern.
    int send(Channel &theChannel)
    {
        const int h = header();
        if (h == 0)
            return 0;
        const int n = size();
        Vector v(vals.data(), n);
        if (h > 0) {
            ID r(rows.data(), n, false);
            ID c(cols.data(), n, false);
            if (theChannel.sendID(0, 0, r) < 0 || theChannel.sendID(0, 0, c) < 0)
                return -1;
        }
        if (theChannel.sendVector(0, 0, v) < 0)
            return -1;
        numSent = n;
        return 0;
    }

    bool pending = false;

private:
    std::unordered_map<uint64_t, int> index;  // (row, col) -> slot, used while recording
    std::vector<int> rows, cols;
    std::vector<double> vals;
    SparseSlotCache cache;
    int numSent = 0;
};

class SparseRootScatter
{
public:
    void reset(int numWorkers) { offsets.assign(static_cast<std::size_t>(numWorkers), std::vector<int>()); }

    // Receives what a worker announced with SparseLocalTriplets::header() and, when
    // merge is true, adds the values into A. offsetOf(row, col) returns the value offset
    // of (row, col) in A or -1. Messages are always drained, so channels stay matched;
    // a full send always rebuilds the worker's scatter, so later value-only sends match.
    template <class OffsetOf>
    int recvAndMerge(Channel &theChannel, int worker, int header, bool merge, double *A, OffsetOf offsetOf)
    {
        if (header == 0 || header == -1)
            return 0;
        if (worker < 0)
            return -1;
        if (worker >= static_cast<int>(offsets.size()))
            offsets.resize(static_cast<std::size_t>(worker) + 1);
        std::vector<int> &off = offsets[static_cast<std::size_t>(worker)];
        const bool full = header > 0;
        const int n = full ? header : -header - 2;
        bool known = true;
        if (full) {
            ID r(n);
            ID c(n);
            if (theChannel.recvID(0, 0, r) < 0 || theChannel.recvID(0, 0, c) < 0)
                return -1;
            off.resize(static_cast<std::size_t>(n));
            for (int k = 0; k < n; ++k) {
                off[k] = offsetOf(r(k), c(k));
                if (off[k] < 0 && known) {
                    if (merge)
                        opserr << "WARNING SparseRootScatter - no matrix entry for (" << r(k) << ", "
                               << c(k) << ")\n";
                    known = false;
                }
            }
            if (!known)
                off.clear();
        } else if (static_cast<int>(off.size()) != n) {
            if (merge)
                opserr << "WARNING SparseRootScatter - worker " << worker + 1 << " sent " << n
                       << " values for a pattern of " << static_cast<int>(off.size()) << "\n";
            known = false;
        }
        vals.resize(static_cast<std::size_t>(n));
        Vector v(vals.data(), n);
        if (theChannel.recvVector(0, 0, v) < 0)
            return -1;
        if (!merge)
            return 0;
        if (!known || A == nullptr)
            return -1;
        for (int k = 0; k < n; ++k)
            A[off[k]] += vals[k];
        return 0;
    }

private:
    std::vector<std::vector<int>> offsets;
    std::vector<double> vals;
};

#endif
