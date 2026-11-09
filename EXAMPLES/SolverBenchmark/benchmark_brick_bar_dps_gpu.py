#!/usr/bin/env python3
"""
Brick-bar static benchmark (openseespy-solvers docs style) extended with:
  - serial PythonSparse backends (scipy.spsolve, scipy.umfpack, nvmath.direct_solver)
  - DistributedPythonSparse with the same backends (MPI)
  - Mumps (MPI)

Use the ``openseespy-solvers`` conda env (Python 3.12 + CuPy/nvmath) and a local
MPI OpenSeesPy built against that interpreter, e.g.:

  export PYTHONPATH=/path/to/OpenSees/build-mp-py312/Release
  # serial
  python EXAMPLES/SolverBenchmark/benchmark_brick_bar_dps_gpu.py --mesh-factors 2,3
  # MPI
  mpirun -np 2|4 python ... --mesh-factors 2,3 --distributed

See https://openseespy-solvers.readthedocs.io/en/stable/examples/
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (
    REPO_ROOT / "build-mp-py312" / "Release",
    REPO_ROOT / "build-mp" / "Release",
    REPO_ROOT / "build" / "Release",
):
    if (candidate / "opensees.so").exists() or (candidate / "OpenSeesPy.so").exists():
        sys.path.insert(0, str(candidate))
        break

import opensees as ops  # noqa: E402

BAR_LENGTH = 10.0
BAR_HEIGHT = 2.0
BAR_THICKNESS = 1.0
ELASTIC_MODULUS = 29_000.0
POISSON_RATIO = 0.3
STEEL_DENSITY = 0.284e-3 / 386.4
YIELD_STRESS = 50.0
NUM_STEPS = 10


def mesh_counts(factor: float):
    mesh_size = BAR_THICKNESS / factor
    nx = max(1, int(math.ceil(BAR_LENGTH / mesh_size)))
    ny = max(1, int(math.ceil(BAR_THICKNESS / mesh_size)))
    nz = max(1, int(math.ceil(BAR_HEIGHT / mesh_size)))
    return nx, ny, nz


def build_model(nx, ny, nz):
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    ops.nDMaterial("ElasticIsotropic", 1, ELASTIC_MODULUS, POISSON_RATIO, STEEL_DENSITY)
    ops.block3D(
        nx, ny, nz, 1, 1, "stdBrick", 1,
        1, 0.0, -BAR_THICKNESS / 2.0, -BAR_HEIGHT / 2.0,
        2, BAR_LENGTH, -BAR_THICKNESS / 2.0, -BAR_HEIGHT / 2.0,
        3, BAR_LENGTH, BAR_THICKNESS / 2.0, -BAR_HEIGHT / 2.0,
        4, 0.0, BAR_THICKNESS / 2.0, -BAR_HEIGHT / 2.0,
        5, 0.0, -BAR_THICKNESS / 2.0, BAR_HEIGHT / 2.0,
        6, BAR_LENGTH, -BAR_THICKNESS / 2.0, BAR_HEIGHT / 2.0,
        7, BAR_LENGTH, BAR_THICKNESS / 2.0, BAR_HEIGHT / 2.0,
        8, 0.0, BAR_THICKNESS / 2.0, BAR_HEIGHT / 2.0,
    )
    ops.fixX(0.0, 1, 1, 1)


def apply_load(n_far_global: int | None = None):
    total = 1.25 * YIELD_STRESS * (BAR_THICKNESS * BAR_HEIGHT**2) / (6 * BAR_LENGTH)
    ops.timeSeries("Trig", 1, 0.0, 6.0, 4.0, "-factor", 1.0)
    ops.pattern("Plain", 1, 1)
    far = [
        n for n in ops.getNodeTags()
        if math.isclose(ops.nodeCoord(n, 1), BAR_LENGTH, abs_tol=1e-9)
    ]
    denom = n_far_global if n_far_global and n_far_global > 0 else max(len(far), 1)
    if far:
        load = total / denom
        for n in far:
            ops.load(n, 0.0, 0.0, -load)


def tip_disp():
    tip = max(
        (
            n for n in ops.getNodeTags()
            if math.isclose(ops.nodeCoord(n, 1), BAR_LENGTH, abs_tol=1e-9)
        ),
        key=lambda n: ops.nodeCoord(n, 3),
        default=None,
    )
    return ops.nodeDisp(tip) if tip is not None else None


def python_backends():
    """Serial/distributed PythonSparse backends from openseespy-solvers."""
    from openseespy_solvers.scipy import spsolve, umfpack

    backends = [("scipy.spsolve", spsolve())]
    try:
        backends.append(("scipy.umfpack", umfpack()))
    except Exception:
        pass
    try:
        import cupy as cp

        if int(cp.cuda.runtime.getDeviceCount()) > 0:
            from openseespy_solvers.nvmath import direct_solver

            backends.append(("nvmath.direct_solver", direct_solver()))
            try:
                from openseespy_solvers.cupy import spsolve as cupy_spsolve

                backends.append(("cupy.spsolve", cupy_spsolve()))
            except Exception:
                pass
    except Exception:
        pass
    return backends


def configure(system_kind: str, backend_name: str | None, solver_obj, pid: int, distributed: bool):
    if system_kind == "Mumps":
        ops.system("Mumps")
        ops.numberer("ParallelPlain" if distributed else "RCM")
    elif system_kind == "DistributedPythonSparse":
        cfg = solver_obj.to_openseespy() if pid == 0 else {
            "solver": None,
            "scheme": "CSR",
            "writable": "none",
        }
        ops.system("DistributedPythonSparse", cfg)
        ops.numberer("ParallelPlain")
    elif system_kind == "PythonSparse":
        ops.system("PythonSparse", solver_obj.to_openseespy())
        ops.numberer("RCM")
    else:
        ops.system(system_kind)
        ops.numberer("ParallelPlain" if distributed else "RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0 / NUM_STEPS)
    ops.test("NormUnbalance", 1.0e-7, 50, 0)
    ops.algorithm("ModifiedNewton", "-FactorOnce")
    ops.analysis("Static")


def run_case(system_kind, backend_name, solver_obj, factor, pid, np_, distributed):
    nx, ny, nz = mesh_counts(factor)
    build_model(nx, ny, nz)
    n_far = sum(
        1 for n in ops.getNodeTags()
        if math.isclose(ops.nodeCoord(n, 1), BAR_LENGTH, abs_tol=1e-9)
    )
    if distributed and np_ > 1:
        ops.partition()
    apply_load(n_far)
    configure(system_kind, backend_name, solver_obj, pid, distributed)
    t0 = time.perf_counter()
    status = ops.analyze(NUM_STEPS)
    seconds = time.perf_counter() - t0
    neq = ops.systemSize() if status == 0 else -1
    disp = tip_disp() if status == 0 else None
    label = system_kind if backend_name is None else f"{system_kind} ({backend_name})"
    if distributed:
        label = f"{label}[np={np_}]"
    return {
        "label": label,
        "factor": factor,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "neq": neq,
        "status": status,
        "seconds": seconds,
        "disp": disp,
        "n_ele": len(ops.getEleTags()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-factors", default="2,3")
    ap.add_argument(
        "--distributed",
        action="store_true",
        help="run DistributedPythonSparse + Mumps (requires mpirun np>=2)",
    )
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument(
        "--backends",
        default="all",
        help="comma list: spsolve,umfpack,nvmath,cupy or 'all'",
    )
    args = ap.parse_args()

    pid = ops.getPID()
    np_ = ops.getNP()
    distributed = args.distributed or np_ > 1
    if distributed and np_ < 2:
        if pid == 0:
            print("Need mpirun -np >= 2 for --distributed", file=sys.stderr)
        sys.exit(1)
    if np_ > 1:
        ops.start()

    factors = [float(x) for x in args.mesh_factors.split(",") if x.strip()]
    backends = python_backends()
    if args.backends != "all":
        want = {x.strip() for x in args.backends.split(",")}
        backends = [
            (n, s) for n, s in backends
            if any(w in n for w in want)
        ]

    cases = []
    if not distributed:
        for name, sol in backends:
            cases.append(("PythonSparse", name, sol))
        for native in ("BandGeneral", "SuperLU", "UmfPack", "Mumps"):
            cases.append((native, None, None))
    else:
        for name, sol in backends:
            cases.append(("DistributedPythonSparse", name, sol))
        cases.append(("Mumps", None, None))

    rows = []
    if pid == 0:
        print(f"ops={getattr(ops, '__file__', ops)}")
        print(f"np={np_} distributed={distributed} factors={factors}")
        print(f"backends={[n for n,_ in backends]}")
        print(f"{'Mesh':>6} {'Eqns':>8} {'Solver':<45} {'Status':>6} {'Time(s)':>10}")
        print("-" * 80)

    for factor in factors:
        for system_kind, backend_name, solver_obj in cases:
            # Native "Mumps" serial may not exist without parallel; skip if not distributed
            if system_kind == "Mumps" and not distributed and np_ == 1:
                # Try anyway — OpenSeesPy MPI build often still registers Mumps
                pass
            try:
                res = run_case(
                    system_kind, backend_name, solver_obj, factor, pid, np_, distributed
                )
            except Exception as exc:
                if pid == 0:
                    label = system_kind if backend_name is None else f"{system_kind} ({backend_name})"
                    print(f"{factor:>6} {'?':>8} {label:<45} {'ERR':>6} {exc}")
                continue
            rows.append(res)
            if pid == 0:
                print(
                    f"{res['factor']:>6} {res['neq']:>8} {res['label']:<45} "
                    f"{res['status']:>6} {res['seconds']:>10.3f}",
                    flush=True,
                )
            elif res["disp"] is not None:
                print(f"[pid={pid}] tip={res['disp']}", flush=True)

    if pid == 0 and args.csv is not None:
        write_header = not args.csv.exists()
        with args.csv.open("a", newline="") as fh:
            w = csv.writer(fh)
            if write_header:
                w.writerow([
                    "label", "np", "factor", "nx", "ny", "nz", "neq",
                    "status", "time_seconds", "n_ele",
                ])
            for r in rows:
                w.writerow([
                    r["label"], np_, r["factor"], r["nx"], r["ny"], r["nz"], r["neq"],
                    r["status"], f"{r['seconds']:.6f}", r["n_ele"],
                ])


if __name__ == "__main__":
    main()
