"""Conventional-analysis benchmark on the bench_site.py soil column.

Compiled systems vs PythonSparse solvers (hand-written and openseespy-solvers)
on the laminar soil column under the Lysmer-base input (elastic or PIMY clay):
elastic gravity, plastic gravity (PIMY), then a transient window that starts
at rest at the velocity zero crossing before the peak of the record.

Configurations:

* compiled OpenSeesPy systems: UmfPack, Mumps, SparseSYM,
* hand-written PythonSparse solvers (what a careful user would write):
  SciPy SuperLU, scikit-umfpack, raw nvmath cuDSS,
* openseespy-solvers: spsolve, umfpack, pardiso (+symmetric), cuDSS (+spd),
  hybrid(cuDSS),
* compiled ``system CuDSS`` (full and spd) through the Tcl twin of the model,
  run with an OpenSees.exe built with cuDSS (``--tcl-exe``).

Algorithms: Newton; KrylovNewton (default: tangent formed once per step);
KrylovNewton on the initial tangent; ModifiedNewton -initial.

Recorded per run and phase: ok flag, Newton iterations per step, wall time of
every ``analyze(1, dt)`` call; for PythonSparse configurations the solver time
inside each step (for package solvers split into value/RHS transfer,
factorize + solve, and copy back; GPU stages are synchronized). Each run is a
separate subprocess with a hard timeout (``--max-seconds``), so one slow or
crashing configuration does not stop the schedule.

Usage::

    python bench_conventional.py --motion velocityHistory.out --out results.jsonl
        [--sizes 6k 27k 54k] [--materials pimy] [--algorithms Newton KrylovNewton]
        [--only cuDSS] [--steps 200] [--tcl-exe path/to/OpenSees.exe]

Set OPENSEESPY_PATH to use a locally built opensees module.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
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
import bench_site as bs  # noqa: E402

ALGORITHMS = {
    "Newton": ("Newton",),
    "KrylovNewton": ("KrylovNewton",),
    "KrylovNewton-initial": ("KrylovNewton", "-iterate", "initial", "-increment", "initial"),
    "ModifiedNewton-initial": ("ModifiedNewton", "-initial"),
}

# name: (refine, plan); neq ~ 3 ((plan - 1)^2 + 1) * 25 * refine after constraints
ITER_RTOL = 1e-8   # relative residual of the iterative linear solvers (--iter-rtol)
ITER_MAXITER = 2000

SIZES = {"6k": (1, 10), "27k": (1, 20), "54k": (2, 20), "126k": (2, 30), "189k": (3, 30),
         "457k": (4, 40)}


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


def _gpu():
    try:
        import cupy as cp
        import nvmath  # noqa: F401

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


def configurations():
    """{name: factory returning system args}; the column tangent is symmetric."""
    cfgs = {"compiled UmfPack": lambda: ("UmfPack",),
            "compiled Mumps": lambda: ("Mumps",),
            "compiled SparseSYM": lambda: ("SparseSYM",),
            "hand SuperLU": lambda: ("PythonSparse", {"solver": HandSuperLU(), "scheme": "CSC"})}
    try:
        HandUmfpack()
        cfgs["hand umfpack"] = lambda: ("PythonSparse", {"solver": HandUmfpack(),
                                                         "scheme": "CSC"})
    except Exception:  # noqa: BLE001
        pass
    gpu = _gpu()
    if gpu:
        cfgs["hand cuDSS"] = lambda: ("PythonSparse", {"solver": HandCuDSS(), "scheme": "CSR"})
    try:
        from openseespy_solvers.scipy import spsolve, umfpack

        cfgs["pkg spsolve"] = lambda: ("PythonSparse", spsolve().to_openseespy())
        try:
            umfpack()
            cfgs["pkg umfpack"] = lambda: ("PythonSparse", umfpack().to_openseespy())
        except ImportError:
            pass
        try:
            from openseespy_solvers.scipy import pardiso

            pardiso()
            cfgs["pkg pardiso"] = lambda: ("PythonSparse", pardiso().to_openseespy())
            cfgs["pkg pardiso symmetric"] = lambda: (
                "PythonSparse", pardiso(matrix_type="symmetric").to_openseespy())
        except Exception:  # noqa: BLE001
            pass
        if gpu:
            from openseespy_solvers import hybrid
            from openseespy_solvers.nvmath import direct_solver

            cfgs["pkg cuDSS"] = lambda: ("PythonSparse", direct_solver().to_openseespy())
            cfgs["pkg cuDSS spd"] = lambda: (
                "PythonSparse", direct_solver(matrix_type="spd").to_openseespy())
            cfgs["pkg hybrid(cuDSS)"] = lambda: (
                "PythonSparse", hybrid(direct_solver(), rtol=ITER_RTOL, maxiter=ITER_MAXITER,
                                       record_stats=True).to_openseespy())
            # single precision: factorization in float32, Newton sees the rounding
            cfgs["pkg cuDSS float32"] = lambda: (
                "PythonSparse", direct_solver(dtype=np.float32).to_openseespy())
            cfgs["pkg cuDSS spd float32"] = lambda: (
                "PythonSparse", direct_solver(matrix_type="spd", dtype=np.float32).to_openseespy())
            from openseespy_solvers import cupy as ocp
            from openseespy_solvers.cupy import precond as cprec

            for pname, M in (("jacobi", cprec.jacobi), ("ilu0", cprec.ilu)):
                cfgs[f"pkg cupy cg {pname}"] = (lambda M=M: ("PythonSparse", ocp.cg(
                    rtol=ITER_RTOL, maxiter=ITER_MAXITER, M=M, record_stats=True).to_openseespy()))
            # ILU is not a symmetric preconditioner: pair it with GMRES
            cfgs["pkg cupy gmres ilu0"] = lambda: ("PythonSparse", ocp.gmres(
                rtol=ITER_RTOL, restart=50, maxiter=ITER_MAXITER // 50, M=cprec.ilu,
                record_stats=True).to_openseespy())
            cfgs["pkg cupy cg jacobi float32"] = lambda: ("PythonSparse", ocp.cg(
                rtol=ITER_RTOL, maxiter=ITER_MAXITER, M=cprec.jacobi, dtype=np.float32,
                record_stats=True).to_openseespy())
        from openseespy_solvers import scipy as osc
        from openseespy_solvers.scipy import precond as sprec

        for pname, M in (("jacobi", sprec.jacobi), ("ilu", sprec.ilu)):
            cfgs[f"pkg scipy cg {pname}"] = (lambda M=M: ("PythonSparse", osc.cg(
                rtol=ITER_RTOL, maxiter=ITER_MAXITER, M=M, record_stats=True).to_openseespy()))
        cfgs["pkg scipy gmres ilu"] = lambda: ("PythonSparse", osc.gmres(
            rtol=ITER_RTOL, restart=50, maxiter=ITER_MAXITER // 50, M=sprec.ilu,
            record_stats=True).to_openseespy())
        cfgs["pkg hybrid(umfpack)"] = lambda: ("PythonSparse", hybrid(
            umfpack(), rtol=ITER_RTOL, maxiter=ITER_MAXITER, record_stats=True).to_openseespy())
    except ImportError:
        pass
    return cfgs


def _relres(kw):
    """True relative residual ||b - A x|| / ||b|| from the OpenSees buffers (CSR/CSC)."""
    ptr, ind, val, n = _csr(kw)
    cls = sp.csr_matrix if kw["storage_scheme"] == "CSR" else sp.csc_matrix
    A = cls((val, ind, ptr), shape=(n, n))
    if kw.get("matrix_type") == "SYMMETRIC_LOWER":   # only one triangle is stored
        A = A + A.T - sp.diags(A.diagonal())
    b = np.frombuffer(kw["rhs"], np.float64, n)
    x = np.frombuffer(kw["x"], np.float64, n)
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(b - A @ x) / nb) if nb > 0 else 0.0


TCL_CONFIGS = {"compiled CuDSS (Tcl)": "system CuDSS",
               "compiled CuDSS spd (Tcl)": "system CuDSS -matrixType spd",
               # dFFI: float factorization; -irNSteps adds iterative refinement in double
               "compiled CuDSS dFFI (Tcl)": "system CuDSS -precision dFFI",
               "compiled CuDSS spd dFFI (Tcl)": "system CuDSS -matrixType spd -precision dFFI",
               "compiled CuDSS spd dFFI ir2 (Tcl)":
                   "system CuDSS -matrixType spd -precision dFFI -irNSteps 2"}


# ---------------------------------------------------------------------------
# Per-step solver timing
# ---------------------------------------------------------------------------


class StepTimers:
    """Cumulative solver timers, differenced after every analyze() call."""

    def __init__(self):
        self.t = {}
        self.last = {}
        self.steps = {}   # phase -> list of {key: seconds in that step}

    def add(self, key, dt):
        self.t[key] = self.t.get(key, 0.0) + dt

    def probe(self, phase):
        row = {k: (v if k.endswith(" max") else v - self.last.get(k, 0.0))
               for k, v in self.t.items()}
        self.steps.setdefault(phase, []).append(row)
        for k in [k for k in self.t if k.endswith(" max")]:
            self.t[k] = 0.0     # per-step maximum
        self.last = dict(self.t)

    def per_phase(self):
        out = {}
        for phase, rows in self.steps.items():
            keys = sorted({k for r in rows for k in r})
            out[phase] = {k: [r.get(k, 0.0) for r in rows] for k in keys}
        return out


def instrument(solver, timers: StepTimers, stages: bool, relres: bool = False):
    """Time solve() (and, for package solvers, its stages) with synchronization."""
    sync = lambda: None  # noqa: E731
    if stages:
        try:
            import cupy as cp

            if solver._on_device:
                sync = cp.cuda.get_current_stream().synchronize
        except Exception:  # noqa: BLE001
            pass

        def wrap(name, key):
            fn = getattr(solver, name, None)
            if fn is None:
                return

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
        timers.add("solver total", time.perf_counter() - t)
        timers.add("calls", 1)
        if kw["matrix_status"] != "UNCHANGED":
            timers.add("calls with new matrix", 1)
        stats = getattr(solver, "stats", None)
        if stats is not None and getattr(stats, "num_solves", 0):
            # iterative / hybrid / reduced precision: record linear iterations,
            # failures and the true residual (outside "solver total")
            if stats.last_num_iterations is not None:
                timers.add("linear iterations", stats.last_num_iterations)
            if stats.last_info not in (0, None) or r != 0:
                timers.add("linear failures", 1)
            timers.add("factorizations", 0)
            timers.t["factorizations"] = stats.num_factorizations
        if relres:
            res = _relres(kw)
            timers.t["relres max"] = max(timers.t.get("relres max", 0.0), res)
        return r
    solver.solve = solve
    return solver


# ---------------------------------------------------------------------------
# Single runs
# ---------------------------------------------------------------------------


def is_inexact(config):
    return any(s in config for s in ("cg", "hybrid", "float32", "AmgX"))


def run_python(material, size, alg_name, config, motion, steps):
    refine, plan = SIZES[size]
    make = configurations()[config]
    t_build = time.perf_counter()
    bs.build(ops, refine, material, plan=plan)
    t_build = time.perf_counter() - t_build
    args = make()
    timers = StepTimers()
    if args[0] == "PythonSparse":
        cfg = dict(args[1])
        cfg["solver"] = instrument(cfg["solver"], timers, stages=config.startswith("pkg"),
                                   relres=is_inexact(config))
        args = (args[0], cfg)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = bs.analysis(ops, args, ALGORITHMS[alg_name], material, motion=motion,
                              dyn_steps=steps, step_probe=timers.probe)
    except Exception as exc:  # noqa: BLE001  (e.g. system not in this build)
        ops.wipe()
        return {"ok": -998, "error": f"{type(exc).__name__}: {exc}"}
    neq = ops.systemSize()
    ops.wipe()
    solver_steps = timers.per_phase()
    phases = {}
    for name, (ok, iters, seconds, step_seconds) in out.items():
        phases[name] = {"ok": ok, "iters": iters, "seconds": seconds,
                        "step_seconds": step_seconds}
        if name in solver_steps:
            phases[name]["solver"] = solver_steps[name]
    ok = 0 if all(p["ok"] == 0 for p in phases.values()) and "transient" in phases else -1
    return {"ok": ok, "neq": neq, "build_seconds": t_build, "phases": phases}


def run_tcl(exe, material, size, alg_name, system_line, motion, steps, timeout):
    refine, plan = SIZES[size]
    script = bs.tcl(refine, material, plan) + bs.tcl_analysis(
        system_line, " ".join(ALGORITHMS[alg_name]), material, motion=motion, dyn_steps=steps)
    with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        res = subprocess.run([exe, path], capture_output=True, text=True, timeout=timeout)
        text = res.stdout + res.stderr
    except subprocess.TimeoutExpired:
        return {"ok": -997, "error": f"timeout after {timeout} s"}
    finally:
        os.unlink(path)
    phases, neq = {}, -1
    for line in text.splitlines():
        if not line.startswith("RESULT"):
            continue
        parts = line.split()
        fields = dict(item.split("=", 1) for item in parts[2:])
        neq = int(fields["neq"])
        iters = [int(v) for v in fields.get("iters", "").split(",") if v]
        times = [float(v) for v in fields.get("times", "").split(",") if v]
        phases[parts[1]] = {"ok": int(fields["ok"]), "iters": iters,
                            "seconds": float(fields["seconds"]), "step_seconds": times}
    if not phases:
        return {"ok": -999, "error": text[-600:]}
    ok = 0 if all(p["ok"] == 0 for p in phases.values()) and "transient" in phases else -1
    return {"ok": ok, "neq": neq, "phases": phases}


def summary(rec):
    if rec["ok"] in (-997, -998, -999):
        return rec.get("error", "")[:160]
    parts = [f"neq={rec['neq']}"]
    for name, p in rec["phases"].items():
        st = np.array(p["step_seconds"]) if p["step_seconds"] else np.array([np.nan])
        parts.append(f"{name}: ok={p['ok']} it={sum(p['iters'])} t={p['seconds']:.2f}s "
                     f"step med={1e3 * np.median(st[1:] if len(st) > 1 else st):.1f}ms "
                     f"max={1e3 * st.max():.1f}ms")
    return "  ".join(parts)


# ---------------------------------------------------------------------------


def main() -> int:
    global ITER_RTOL
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True, help="velocity record (m/s)")
    ap.add_argument("--motion-dt", type=float, default=0.005)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sizes", nargs="+", default=["6k", "27k", "54k"], choices=list(SIZES))
    ap.add_argument("--materials", nargs="+", default=["pimy"], choices=["elastic", "pimy"])
    ap.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS),
                    choices=list(ALGORITHMS))
    ap.add_argument("--steps", type=int, default=200, help="transient steps")
    ap.add_argument("--tcl-exe", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="substring filter on config names")
    ap.add_argument("--skip", nargs="*", default=None, help="substring filter to exclude")
    ap.add_argument("--max-seconds", type=float, default=1800.0,
                    help="hard timeout per run; a config that times out skips larger sizes")
    ap.add_argument("--iter-rtol", type=float, default=ITER_RTOL,
                    help="relative residual tolerance of the iterative linear solvers")
    ap.add_argument("--list", action="store_true", help="print the configurations and exit")
    ap.add_argument("--single", nargs=4, metavar=("MATERIAL", "SIZE", "ALGORITHM", "CONFIG"),
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    motion = bs.Motion.from_file(args.motion, args.motion_dt)
    ITER_RTOL = args.iter_rtol

    if args.single:
        material, size, alg, config = args.single
        rec = run_python(material, size, alg, config, motion, args.steps)
        print("JSON " + json.dumps(rec), flush=True)
        return 0

    names = list(configurations()) + (list(TCL_CONFIGS) if args.tcl_exe else [])
    names = [n for n in names if (not args.only or any(s in n for s in args.only))
             and not (args.skip and any(s in n for s in args.skip))]
    if args.list:
        print("\n".join(names))
        return 0
    if not args.out:
        ap.error("--out is required")

    meta = {"platform": platform.platform(), "python": platform.python_version(),
            "opensees": getattr(ops, "__file__", "?"), "processor": platform.processor(),
            "motion": motion.name, "start_time": motion.strong_start(), "steps": args.steps,
            "dyn_test": bs.DYN_TEST, "gravity_test": bs.GRAVITY_TEST, "dyn_dt": bs.DYN_DT,
            "tcl_exe": args.tcl_exe}
    try:
        import openseespy_solvers

        meta["openseespy_solvers"] = openseespy_solvers.__version__
    except ImportError:
        pass
    out = open(args.out, "a", encoding="utf-8")
    out.write(json.dumps({"meta": meta}) + "\n")
    skipped = set()   # (material, alg, config) that timed out at a smaller size

    for material in args.materials:
        for size in args.sizes:
            for alg in args.algorithms:
                for name in names:
                    key = (material, alg, name)
                    if key in skipped:
                        continue
                    t0 = time.perf_counter()
                    if name in TCL_CONFIGS:
                        rec = run_tcl(args.tcl_exe, material, size, alg, TCL_CONFIGS[name],
                                      motion, args.steps, args.max_seconds)
                    else:
                        cmd = [sys.executable, str(Path(__file__).resolve()), "--motion",
                               args.motion, "--motion-dt", str(args.motion_dt), "--steps",
                               str(args.steps), "--iter-rtol", str(args.iter_rtol),
                               "--single", material, size, alg, name]
                        try:
                            res = subprocess.run(cmd, capture_output=True, text=True,
                                                 timeout=args.max_seconds)
                            line = [ln for ln in res.stdout.splitlines()
                                    if ln.startswith("JSON ")]
                            rec = (json.loads(line[-1][5:]) if line else
                                   {"ok": -999, "error": (res.stdout + res.stderr)[-600:]})
                        except subprocess.TimeoutExpired:
                            rec = {"ok": -997, "error": f"timeout after {args.max_seconds} s"}
                    if rec["ok"] == -997:
                        skipped.add(key)
                    rec.update({"material": material, "size": size, "algorithm": alg,
                                "config": name, "wall_seconds": time.perf_counter() - t0,
                                "iter_rtol": args.iter_rtol if is_inexact(name) else None})
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(f"{material:7s} {size:5s} {alg:22s} {name:26s} ok={rec['ok']:4d} "
                          f"{summary(rec)}", flush=True)
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
