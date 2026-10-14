"""Offline linear-solver benchmark on exported OpenSees tangents (A.npz + b.npy).

Each solver is called exactly as OpenSees' PythonSparse system calls it
(memoryview buffers, CSR, lower triangle when the solver asks for it), in a
Newton-like sequence:

1. STRUCTURE_CHANGED (symbolic analysis / setup + factorization + solve),
2. ``--refactor`` calls with COEFFICIENTS_CHANGED (numeric refactorization +
   solve; the values are perturbed by 1e-9 relative so nothing can be cached),
3. ``--resolve`` calls with UNCHANGED (solve only, new right-hand side).

Recorded per call: wall time (the package copies x back to the host, so GPU
work is synchronized), success, linear iterations (Krylov), and the true
relative residual ||b - A x|| / ||b|| computed with the full matrix in double.

Solvers: every PythonSparse configuration of bench_conventional.py (package
and hand-written; CPU and GPU; direct, Krylov, hybrid, float32) plus
smoothed-aggregation AMG (pyamg) as a CG/GMRES preconditioner. Krylov
configurations are run at each ``--rtol``.

    python bench_matrices.py osu_matrices/*.npz --out matrices.jsonl
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_conventional as bc  # noqa: E402


class HandAMG:
    """scipy CG (SPD) or GMRES preconditioned by pyamg smoothed aggregation."""

    def __init__(self, krylov="cg", rtol=1e-8, maxiter=2000):
        import pyamg

        self.pyamg, self.krylov, self.rtol, self.maxiter = pyamg, krylov, rtol, maxiter
        self.ml = None
        self.iterations = None

    def solve(self, **kw):
        ptr, ind, val, n = bc._csr(kw)
        A = sp.csr_matrix((val.copy(), ind.copy(), ptr.copy()), shape=(n, n))
        if kw["matrix_status"] != "UNCHANGED" or self.ml is None:
            self.ml = self.pyamg.smoothed_aggregation_solver(A, max_coarse=500)
            self.A = A
        M = self.ml.aspreconditioner(cycle="V")
        b = np.frombuffer(kw["rhs"], np.float64, n)
        count = [0]

        def cb(*_a):
            count[0] += 1
        solver = spla.cg if self.krylov == "cg" else spla.gmres
        extra = {} if self.krylov == "cg" else {"restart": 50,
                                                 "maxiter": max(1, self.maxiter // 50)}
        if self.krylov == "cg":
            extra["maxiter"] = self.maxiter
        x, info = solver(self.A, b, rtol=self.rtol, M=M, callback=cb, **extra)
        self.iterations = count[0]
        np.frombuffer(kw["x"], np.float64, n)[:] = x
        return 0 if info == 0 else -1


def configs(rtol, symmetric):
    bc.ITER_RTOL = rtol
    out = {}
    for name, make in bc.configurations().items():
        if name.startswith("compiled"):
            continue
        if not symmetric and ("spd" in name or "symmetric" in name or " cg " in name + " "):
            continue
        out[name] = make
    out["hand pyamg-SA cg" if symmetric else "hand pyamg-SA gmres"] = (
        lambda: ("PythonSparse", {"solver": HandAMG("cg" if symmetric else "gmres", rtol),
                                  "scheme": "CSR"}))
    if symmetric:
        out["hand pyamg-SA gmres"] = lambda: ("PythonSparse", {
            "solver": HandAMG("gmres", rtol), "scheme": "CSR"})
    return out


def buffers(A, lower):
    M = sp.tril(A, format="csr") if lower else A
    M.sort_indices()
    return {"index_ptr": np.ascontiguousarray(M.indptr, np.int32),
            "indices": np.ascontiguousarray(M.indices, np.int32),
            "values": np.ascontiguousarray(M.data, np.float64)}


def call(solver, buf, b, x, status, lower):
    kw = {"num_eqn": b.shape[0], "nnz": buf["values"].shape[0],
          "values": memoryview(buf["values"]), "indices": memoryview(buf["indices"]),
          "index_ptr": memoryview(buf["index_ptr"]), "rhs": memoryview(b),
          "x": memoryview(x), "matrix_status": status, "storage_scheme": "CSR"}
    if lower:
        kw["matrix_type"] = "SYMMETRIC_LOWER"
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = solver.solve(**kw)
        except Exception as exc:  # noqa: BLE001
            return time.perf_counter() - t, -1, str(exc)[:200]
    return time.perf_counter() - t, r, None


def iterations(solver):
    if hasattr(solver, "iterations") and solver.iterations is not None:
        return solver.iterations
    st = getattr(solver, "stats", None)
    return getattr(st, "last_num_iterations", None) if st is not None else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("matrices", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rtol", nargs="+", type=float, default=[1e-6, 1e-8, 1e-10])
    ap.add_argument("--refactor", type=int, default=5)
    ap.add_argument("--resolve", type=int, default=5)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=None)
    ap.add_argument("--max-call-seconds", type=float, default=120.0,
                    help="stop a solver's sequence after a call slower than this")
    args = ap.parse_args()
    out = open(args.out, "a", encoding="utf-8")
    out.write(json.dumps({"meta": {"platform": platform.platform(), "rtol": args.rtol,
                                   "refactor": args.refactor, "resolve": args.resolve}}) + "\n")
    rng = np.random.default_rng(0)
    for path in args.matrices:
        A = sp.load_npz(path).tocsr()
        A.sum_duplicates()
        A.sort_indices()
        bpath = str(path).replace("_A.npz", "_b.npy")
        b0 = np.load(bpath) if Path(bpath).exists() else A @ np.ones(A.shape[0])
        n = A.shape[0]
        asym = abs(A - A.T).max() / abs(A).max()
        symmetric = asym < 1e-10
        nb = np.linalg.norm(b0)
        print(f"== {Path(path).name}: n={n} nnz={A.nnz} asym={asym:.1e}", flush=True)
        for rtol in args.rtol:
            for name, make in configs(rtol, symmetric).items():
                inexact = bc.is_inexact(name) or "pyamg" in name
                if rtol != args.rtol[0] and not inexact:
                    continue      # direct solvers do not depend on rtol
                if args.only and not any(s in name for s in args.only):
                    continue
                if args.skip and any(s in name for s in args.skip):
                    continue
                try:
                    _sys, cfg = make()
                except Exception as exc:  # noqa: BLE001
                    print(f"   {name:28s} unavailable: {exc}", flush=True)
                    continue
                solver = cfg["solver"]
                lower = cfg.get("matrix_type") == "SYMMETRIC_LOWER"
                buf = buffers(A, lower)
                seq = (["STRUCTURE_CHANGED"] + ["COEFFICIENTS_CHANGED"] * args.refactor
                       + ["UNCHANGED"] * args.resolve)
                calls = []
                for k, status in enumerate(seq):
                    if status == "COEFFICIENTS_CHANGED":
                        buf["values"] *= (1.0 + 1e-9 * rng.standard_normal())
                    b = b0 * (1.0 + 0.1 * rng.standard_normal()) if k else b0.copy()
                    x = np.zeros(n)
                    dt, r, err = call(solver, buf, b, x, status, lower)
                    Afull = sp.csr_matrix((buf["values"], buf["indices"], buf["index_ptr"]),
                                          shape=(n, n))
                    if lower:
                        Afull = Afull + Afull.T - sp.diags(Afull.diagonal())
                    res = float(np.linalg.norm(b - Afull @ x) / np.linalg.norm(b))
                    calls.append({"status": status, "seconds": dt, "ok": r, "relres": res,
                                  "iterations": iterations(solver), "error": err})
                    if dt > args.max_call_seconds or err:
                        break
                rec = {"matrix": Path(path).name, "n": n, "nnz": int(A.nnz), "asym": asym,
                       "config": name, "rtol": rtol if inexact else None, "calls": calls}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                fac = [c["seconds"] for c in calls if c["status"] == "COEFFICIENTS_CHANGED"]
                sol = [c["seconds"] for c in calls if c["status"] == "UNCHANGED"]
                its = [c["iterations"] for c in calls if c["iterations"] is not None]
                bad = sum(1 for c in calls if c["ok"] != 0)
                print(f"   {name:28s} rtol={rtol if inexact else '-':<6} "
                      f"setup {1e3 * calls[0]['seconds']:9.1f} ms  "
                      f"refactor+solve {1e3 * np.median(fac) if fac else np.nan:9.1f} ms  "
                      f"solve {1e3 * np.median(sol) if sol else np.nan:8.1f} ms  "
                      f"relres max {max(c['relres'] for c in calls):.1e}  "
                      f"lin.it {max(its) if its else '-'}  failed {bad}/{len(calls)}",
                      flush=True)
                del solver
        _ = nb
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
