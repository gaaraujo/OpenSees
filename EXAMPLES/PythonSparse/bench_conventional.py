"""Conventional-analysis benchmark: compiled systems vs PythonSparse solvers.

For each model / size / algorithm (bench_models.py) this times one transient
analysis per configuration and writes one JSON line per run:

* compiled OpenSeesPy systems: UmfPack, Mumps, SparseSYM (symmetric models),
* hand-written PythonSparse solvers (what a careful user would write):
  SciPy SuperLU, scikit-umfpack, raw nvmath cuDSS,
* openseespy-solvers: spsolve, umfpack, pardiso (+symmetric), nvmath
  (+spd), hybrid(nvmath),
* compiled ``system CuDSS`` (full and spd) through the Tcl twin of the model,
  run with an OpenSees.exe built with cuDSS (``--tcl-exe``).

Recorded per run: total analysis seconds, Newton iterations per step, number
of equations; for package solvers in a separate instrumented run, the split of
solver time into Python overhead, value/RHS transfer, factorization and solve.

Usage::

    python bench_conventional.py --out results.jsonl [--sizes 2k 12k 54k]
        [--models plastic sliding] [--repeat 3] [--tcl-exe path/to/OpenSees.exe]

Set OPENSEESPY_PATH to use a locally built opensees module.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

if os.environ.get("OPENSEESPY_PATH"):
    sys.path.insert(0, os.environ["OPENSEESPY_PATH"])
    import opensees as ops  # noqa: E402
else:
    import openseespy.opensees as ops  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_models as bm  # noqa: E402

ALGORITHMS = {"Newton": ("Newton",), "ModifiedNewton": ("ModifiedNewton",)}


# ---------------------------------------------------------------------------
# Hand-written PythonSparse solvers (baselines for the package)
# ---------------------------------------------------------------------------


def _csr(kw):
    n, nnz = kw["num_eqn"], kw["nnz"]
    return (np.frombuffer(kw["index_ptr"], np.int32, n + 1),
            np.frombuffer(kw["indices"], np.int32, nnz),
            np.frombuffer(kw["values"], np.float64, nnz), n)


class HandSuperLU:
    """CSC buffers straight into scipy splu; refactor on any change."""

    def __init__(self):
        self.lu = None

    def solve(self, **kw):
        ptr, ind, val, n = _csr(kw)
        if kw["matrix_status"] != "UNCHANGED" or self.lu is None:
            self.lu = spla.splu(sp.csc_matrix((val, ind, ptr), shape=(n, n)))
        np.frombuffer(kw["x"], np.float64, n)[:] = self.lu.solve(
            np.frombuffer(kw["rhs"], np.float64, n))
        return 0


class HandUmfpack:
    """CSC buffers into scikit-umfpack; symbolic once, numeric on change."""

    def __init__(self):
        import scikits.umfpack as um

        self.um = um
        self.ctx = um.UmfpackContext("di")
        self.A = None

    def solve(self, **kw):
        ptr, ind, val, n = _csr(kw)
        st = kw["matrix_status"]
        if st == "STRUCTURE_CHANGED" or self.A is None:
            self.A = sp.csc_matrix((val.copy(), ind.copy(), ptr.copy()), shape=(n, n))
            self.ctx.symbolic(self.A)
            self.ctx.numeric(self.A)
        elif st == "COEFFICIENTS_CHANGED":
            self.A.data[:] = val
            self.ctx.numeric(self.A)
        x = self.ctx.solve(self.um.UMFPACK_A, self.A, np.frombuffer(kw["rhs"], np.float64, n))
        np.frombuffer(kw["x"], np.float64, n)[:] = x
        return 0


class HandCuDSS:
    """Raw nvmath DirectSolver on cupy CSR; values updated in place."""

    def __init__(self):
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
        import nvmath

        self.cp, self.cpsp, self.nvmath = cp, cpsp, nvmath
        self.solver = None

    def solve(self, **kw):
        cp = self.cp
        ptr, ind, val, n = _csr(kw)
        rhs = np.frombuffer(kw["rhs"], np.float64, n)
        st = kw["matrix_status"]
        if st == "STRUCTURE_CHANGED" or self.solver is None:
            if self.solver is not None:
                self.solver.free()
            self.A = self.cpsp.csr_matrix((cp.asarray(val), cp.asarray(ind), cp.asarray(ptr)),
                                          shape=(n, n))
            self.b = cp.asarray(rhs)
            self.solver = self.nvmath.sparse.advanced.DirectSolver(self.A, self.b)
            self.solver.plan()
            self.solver.factorize()
        else:
            if st == "COEFFICIENTS_CHANGED":
                self.A.data[:] = cp.asarray(val)
                self.solver.factorize()
            self.b[:] = cp.asarray(rhs)
        x = self.solver.solve()
        np.frombuffer(kw["x"], np.float64, n)[:] = cp.asnumpy(x)
        return 0


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


def configurations(symmetric: bool):
    cfgs = [("compiled UmfPack", lambda: ("UmfPack",)),
            ("compiled Mumps", lambda: ("Mumps",))]
    if symmetric:
        cfgs.append(("compiled SparseSYM", lambda: ("SparseSYM",)))
    cfgs.append(("hand SuperLU", lambda: ("PythonSparse", {"solver": HandSuperLU(),
                                                            "scheme": "CSC"})))
    try:
        HandUmfpack()
        cfgs.append(("hand umfpack", lambda: ("PythonSparse", {"solver": HandUmfpack(),
                                                                "scheme": "CSC"})))
    except Exception:  # noqa: BLE001
        pass
    gpu = False
    try:
        import cupy as cp

        gpu = cp.cuda.runtime.getDeviceCount() > 0
        import nvmath  # noqa: F401
    except Exception:  # noqa: BLE001
        gpu = False
    if gpu:
        cfgs.append(("hand cuDSS", lambda: ("PythonSparse", {"solver": HandCuDSS(),
                                                              "scheme": "CSR"})))
    try:
        from openseespy_solvers.scipy import spsolve, umfpack

        cfgs.append(("pkg spsolve", lambda: ("PythonSparse", spsolve().to_openseespy())))
        try:
            umfpack()
            cfgs.append(("pkg umfpack", lambda: ("PythonSparse", umfpack().to_openseespy())))
        except ImportError:
            pass
        try:
            from openseespy_solvers.scipy import pardiso

            pardiso()
            cfgs.append(("pkg pardiso", lambda: ("PythonSparse", pardiso().to_openseespy())))
            if symmetric:
                cfgs.append(("pkg pardiso symmetric", lambda: (
                    "PythonSparse", pardiso(matrix_type="symmetric").to_openseespy())))
        except Exception:  # noqa: BLE001
            pass
        if gpu:
            from openseespy_solvers import hybrid
            from openseespy_solvers.nvmath import direct_solver

            cfgs.append(("pkg cuDSS", lambda: ("PythonSparse", direct_solver().to_openseespy())))
            if symmetric:
                cfgs.append(("pkg cuDSS spd", lambda: (
                    "PythonSparse", direct_solver(matrix_type="spd").to_openseespy())))
            cfgs.append(("pkg hybrid(cuDSS)", lambda: (
                "PythonSparse", hybrid(direct_solver(), rtol=1e-10).to_openseespy())))
    except ImportError:
        pass
    return cfgs


# ---------------------------------------------------------------------------
# Instrumentation of package solvers (separate run; adds synchronizations)
# ---------------------------------------------------------------------------


class Timers:
    def __init__(self):
        self.t = {}

    def add(self, key, dt):
        self.t[key] = self.t.get(key, 0.0) + dt


def instrument(solver, timers: Timers):
    """Wrap a package LinearSolver's stages with synchronized timers."""
    try:
        import cupy as cp

        sync = cp.cuda.get_current_stream().synchronize if solver._on_device else (lambda: None)
    except Exception:  # noqa: BLE001
        sync = lambda: None  # noqa: E731

    def wrap(name, key):
        fn = getattr(solver, name)

        def inner(*a, **k):
            sync()
            t = time.perf_counter()
            r = fn(*a, **k)
            sync()
            timers.add(key, time.perf_counter() - t)
            return r
        setattr(solver, name, inner)

    wrap("_current_matrix", "values to backend")
    wrap("_rhs_to_compute", "rhs to backend")
    wrap("_solve_system", "factorize + solve")
    wrap("_write_opensees_buffer", "x to OpenSees")
    outer = solver.solve

    def solve(**kw):
        t = time.perf_counter()
        r = outer(**kw)
        timers.add("solve() total", time.perf_counter() - t)
        timers.add("calls", 1)
        timers.add(f"calls {kw['matrix_status']}", 1)
        return r
    solver.solve = solve
    return solver


# ---------------------------------------------------------------------------


def run_python(model, size, alg_name, make):
    bm.build(ops, model, size)
    args = make()
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ok, iters = bm.analysis(ops, args, ALGORITHMS[alg_name])
    except Exception as exc:  # noqa: BLE001  (e.g. system not in this build)
        ops.wipe()
        return {"ok": -998, "seconds": float("nan"), "iters": [], "neq": -1,
                "error": f"{type(exc).__name__}: {exc}"}
    elapsed = time.perf_counter() - t0
    neq = ops.systemSize()
    ops.wipe()
    return {"ok": ok, "seconds": elapsed, "iters": iters, "neq": neq}


def run_tcl(exe, model, size, alg_name, system_line):
    script = bm.tcl(model, size) + bm.tcl_analysis(system_line, alg_name)
    with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        out = subprocess.run([exe, path], capture_output=True, text=True, timeout=3600)
    finally:
        os.unlink(path)
    for line in (out.stdout + out.stderr).splitlines():
        if line.startswith("RESULT"):
            fields = dict(item.split("=", 1) for item in line.split()[1:4] + [line.split()[-1]])
            iters_txt = line.split("iters=", 1)[1].rsplit(" neq=", 1)[0]
            return {"ok": int(fields["ok"]), "seconds": float(fields["seconds"]),
                    "iters": [int(v) for v in iters_txt.split()], "neq": int(fields["neq"])}
    return {"ok": -999, "seconds": float("nan"), "iters": [], "neq": -1,
            "error": (out.stdout + out.stderr)[-400:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", nargs="+", default=["2k", "12k", "54k"])
    ap.add_argument("--models", nargs="+", default=["plastic", "sliding"])
    ap.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tcl-exe", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="substring filter on config names")
    ap.add_argument("--max-seconds", type=float, default=600.0,
                    help="skip larger sizes for a config after a run exceeds this")
    args = ap.parse_args()

    meta = {"platform": platform.platform(), "python": platform.python_version(),
            "opensees": getattr(ops, "__file__", "?"), "processor": platform.processor()}
    try:
        import openseespy_solvers

        meta["openseespy_solvers"] = openseespy_solvers.__version__
    except ImportError:
        pass
    out = open(args.out, "a", encoding="utf-8")
    out.write(json.dumps({"meta": meta}) + "\n")
    slow = set()

    for model in args.models:
        symmetric = model == "plastic"
        algs = args.algorithms if model == "plastic" else ["Newton"]
        for size in args.sizes:
            for alg_name in algs:
                cfgs = configurations(symmetric)
                if args.tcl_exe:
                    cfgs.append(("compiled CuDSS (Tcl)", "system CuDSS"))
                    if symmetric:
                        cfgs.append(("compiled CuDSS spd (Tcl)", "system CuDSS -matrixType spd"))
                for name, make in cfgs:
                    if args.only and not any(s in name for s in args.only):
                        continue
                    if name in slow:
                        continue
                    runs = []
                    for _ in range(args.repeat):
                        if isinstance(make, str):
                            r = run_tcl(args.tcl_exe, model, size, alg_name, make)
                        else:
                            r = run_python(model, size, alg_name, make)
                        runs.append(r)
                        if r["ok"] != 0 or r["seconds"] > args.max_seconds:
                            break
                    rec = {"model": model, "size": size, "algorithm": alg_name, "config": name,
                           "neq": runs[0]["neq"], "ok": runs[0]["ok"],
                           "iters": runs[0]["iters"],
                           "seconds": [r["seconds"] for r in runs],
                           "median_seconds": statistics.median(r["seconds"] for r in runs)}
                    if rec["ok"] == -998:
                        print(f"{model:8s} {size:4s} {alg_name:15s} {name:28s} unavailable: "
                              f"{runs[0]['error']}", flush=True)
                        out.write(json.dumps(rec) + "\n")
                        continue
                    if "error" in runs[0]:
                        rec["error"] = runs[0]["error"]
                    # Stage split for package solvers (one extra, instrumented run).
                    if name.startswith("pkg") and runs[0]["ok"] == 0:
                        timers = Timers()

                        def make_instrumented(make=make):
                            system, cfg = make()
                            cfg = dict(cfg)
                            cfg["solver"] = instrument(cfg["solver"], timers)
                            return system, cfg
                        run_python(model, size, alg_name, make_instrumented)
                        rec["stages"] = timers.t
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(f"{model:8s} {size:4s} {alg_name:15s} {name:28s} neq={rec['neq']:6d} "
                          f"ok={rec['ok']} t={rec['median_seconds']:8.3f}s iters={sum(rec['iters'])}",
                          flush=True)
                    if rec["median_seconds"] > args.max_seconds:
                        slow.add(name)
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
