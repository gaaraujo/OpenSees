// PhaseTimers.h -- opt-in per-step phase timers for the MKR-alpha integrators (CudaExplicitAlpha,
// SparseExplicitAlpha). Benchmark build only (bench-ch3/RTHS-twin2t, 2026-10-15); not for upstream.
//
// Enabled only when both environment variables are set:
//   OPS_PHASE_TIMERS  output prefix (each rank writes <prefix>.r<rank>.csv)
//   OPS_PHASE_N       number of committed steps to record; the file is written at the N-th commit
// Otherwise every mark is one branch on a cached flag. No I/O and no synchronization are added during
// the run; timestamps use std::chrono::steady_clock on the calling rank.
//
// Phases per step (seconds), from the marks below:
//   predict   newStep entry -> before updateDomain (GPU/CPU predictor, motion broadcast)
//   statedet  updateDomain in newStep (element and material state determination at the alpha level)
//   assemble  TransientIntegrator::formUnbalance (resisting forces into the local RHS)
//   merge     RHS merge to rank 0 (root: wait for the slowest rank + transfer; workers: send)
//   rhs       rest of formUnbalance (root: M solve + alpha product; workers: wait for the root's status)
//   solve     formUnbalance exit -> update entry (the linear solve and the solution broadcast)
//   update    update() (kinematics; no element update with -noSecondPass)
//   commit    commit() (commitDomain)
//   total     newStep entry -> commit exit
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace ops_phase {

enum Mark {
    StepBegin, PredictDone, StateDetDone, UnbalBegin, AssembleDone, MergeDone, UnbalDone,
    UpdateBegin, UpdateDone, CommitBegin, CommitDone, NMarks
};

struct Timers {
    bool on = false;
    long n = 0, count = 0;
    int rank = 0;
    bool written = false;
    std::string prefix;
    double t[NMarks] = {0.0};
    bool seen[NMarks] = {false};
    std::vector<float> rows;  // 9 values per step

    Timers()
    {
        const char *p = std::getenv("OPS_PHASE_TIMERS");
        const char *k = std::getenv("OPS_PHASE_N");
        if (p != nullptr && *p != '\0' && k != nullptr) {
            prefix = p;
            n = std::atol(k);
            on = n > 0;
            if (on)
                rows.reserve(static_cast<std::size_t>(n) * 9);
        }
    }

    static double now()
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    double span(Mark a, Mark b) const { return (seen[a] && seen[b]) ? t[b] - t[a] : 0.0; }

    void mark(Mark m)
    {
        if (!on || written)
            return;
        t[m] = now();
        if (m == StepBegin)
            for (int i = 1; i < NMarks; ++i)
                seen[i] = false;
        seen[m] = true;
        if (m == CommitDone)
            endStep();
    }

    void endStep()
    {
        if (!seen[StepBegin])
            return;
        const Mark afterAssemble = seen[MergeDone] ? MergeDone : AssembleDone;
        const float v[9] = {
            static_cast<float>(span(StepBegin, PredictDone)),
            static_cast<float>(span(PredictDone, StateDetDone)),
            static_cast<float>(span(UnbalBegin, AssembleDone)),
            static_cast<float>(seen[MergeDone] ? span(AssembleDone, MergeDone) : 0.0),
            static_cast<float>(span(afterAssemble, UnbalDone)),
            static_cast<float>(span(UnbalDone, UpdateBegin)),
            static_cast<float>(span(UpdateBegin, UpdateDone)),
            static_cast<float>(span(CommitBegin, CommitDone)),
            static_cast<float>(span(StepBegin, CommitDone))};
        rows.insert(rows.end(), v, v + 9);
        seen[StepBegin] = false;
        if (++count == n)
            write();
    }

    void write()
    {
        written = true;
        const std::string f = prefix + ".r" + std::to_string(rank) + ".csv";
        FILE *fp = std::fopen(f.c_str(), "w");
        if (fp == nullptr)
            return;
        std::fprintf(fp, "predict,statedet,assemble,merge,rhs,solve,update,commit,total\n");
        for (std::size_t i = 0; i + 8 < rows.size(); i += 9)
            std::fprintf(fp, "%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e\n", rows[i], rows[i + 1],
                         rows[i + 2], rows[i + 3], rows[i + 4], rows[i + 5], rows[i + 6], rows[i + 7],
                         rows[i + 8]);
        std::fclose(fp);
    }
};

inline Timers &get()
{
    static Timers instance;
    return instance;
}
inline void mark(Mark m) { get().mark(m); }
inline void setRank(int r) { get().rank = r; }

}  // namespace ops_phase
