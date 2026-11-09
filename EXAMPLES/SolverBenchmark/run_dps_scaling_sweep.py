#!/usr/bin/env python3
"""
Agreed linear SolverBenchmark matrix (PR #1676 style, distributed + GPU).

Python backends via openseespy-solvers >= 0.2.0 (no SciPy spsolve):
  - cupy-cg  : CuPy CG via Py/DPS (CSR, RCM)
  - umfpack  : scikit-umfpack via Py/DPS (CSC, di/auto, Plain)
  - nvmath   : nvmath/cuDSS via Py/DPS (CSR, Plain)

Native references:
  - umfpack-native : OpenSees UmfPack (np=1, Plain)
  - bandspd        : OpenSees BandSPD (np=1, RCM)
  - mumps          : OpenSees Mumps (np>1, ParallelPlain)

Requires editable/install of openseespy-solvers 0.2.0+ in the active env.
Do not start full sweeps until that package is installed (tag v0.2.0 / PyPI).

Usage:
  python3 EXAMPLES/SolverBenchmark/run_dps_scaling_sweep.py \\
    --out figures/dps_linear.csv

  # resume after interrupt / OOM
  python3 ... --out figures/dps_linear.csv --append
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dps_common import (
    AGREED_LINEAR_FACTORS,
    AGREED_NPS,
    NATIVE_SKIP_LIMITS,
    apply_far_face_load,
    build_solid_bar,
    cantilever_load,
    estimate_free_dofs,
    far_corner_disp,
    import_opensees,
    lin_numberer,
    make_python_lin_solver,
    mesh_counts,
    python_lin_scheme,
)

ops = import_opensees()

CSV_HEADER = (
    "solver",
    "backend",
    "np",
    "mesh_factor",
    "est_free_dofs",
    "num_equations",
    "num_elements_local",
    "num_nodes_local",
    "status",
    "time_seconds",
    "displacement_x",
    "displacement_y",
    "displacement_z",
)

PYTHON_BACKENDS = ("cupy-cg", "umfpack", "nvmath")
NATIVE_SERIAL = ("umfpack-native", "bandspd")
NATIVE_MPI = ("mumps",)


def solver_label(backend: str, np_: int) -> str:
    pretty = {
        "cupy-cg": "CuPyCG",
        "umfpack": "scikit-UMFPACK",
        "nvmath": "nvmath/cuDSS",
        "umfpack-native": "UmfPack",
        "bandspd": "BandSPD",
        "mumps": "Mumps",
    }.get(backend, backend)
    return f"{pretty} (np={np_})"


def configure_backend(backend: str, pid: int) -> None:
    ops.constraints("Plain")
    np_ = ops.getNP()
    if backend in PYTHON_BACKENDS:
        ops.numberer(lin_numberer(backend, np_))
        name = "PythonSparse" if np_ == 1 else "DistributedPythonSparse"
        solver = make_python_lin_solver(backend) if pid == 0 or np_ == 1 else None
        ops.system(
            name,
            {
                "solver": solver,
                "scheme": python_lin_scheme(backend),
                "writable": "none",
            },
        )
    elif backend == "umfpack-native":
        ops.numberer(lin_numberer(backend, np_))
        ops.system("UmfPack")
    elif backend == "bandspd":
        ops.numberer(lin_numberer(backend, np_))
        ops.system("BandSPD")
    elif backend == "mumps":
        ops.numberer(lin_numberer(backend, np_))
        ops.system("Mumps")
    else:
        raise SystemExit(
            f"unknown backend {backend} "
            "(use cupy-cg, umfpack, nvmath, umfpack-native, bandspd, mumps)"
        )

    ops.integrator("LoadControl", 0.2)
    ops.test("NormUnbalance", 1.0e-8, 25, 0)
    ops.algorithm("ModifiedNewton", "-FactorOnce")
    ops.analysis("Static")


def run_case(backend: str, mesh_factor: float, pid: int, np_: int, num_steps: int = 5):
    _, (nx, ny, nz) = mesh_counts(mesh_factor)
    est = estimate_free_dofs(nx, ny, nz)
    build_solid_bar(ops, nx, ny, nz)
    n_far = sum(
        1
        for n in ops.getNodeTags()
        if math.isclose(ops.nodeCoord(n, 1), 10.0, abs_tol=1e-9)
    )
    if backend in NATIVE_SERIAL and np_ > 1:
        raise RuntimeError(f"{backend} only for np=1")
    if backend == "mumps" and np_ == 1:
        raise RuntimeError("mumps only for np>1 in this matrix")

    distributed = backend in (*PYTHON_BACKENDS, "mumps") and np_ > 1
    if distributed:
        ops.partition()
    apply_far_face_load(ops, cantilever_load(), n_far_global=n_far)
    configure_backend(backend, pid)

    t0 = time.perf_counter()
    status = ops.analyze(num_steps)
    seconds = time.perf_counter() - t0
    neq = int(ops.systemSize()) if status == 0 else -1
    tip = far_corner_disp(ops) if status == 0 else None
    return {
        "est": est,
        "neq": neq,
        "nele": len(ops.getEleTags()),
        "nnodes": len(ops.getNodeTags()),
        "status": status,
        "seconds": seconds,
        "tip": tip,
    }


def auto_backends(np_: int) -> list[str]:
    if np_ == 1:
        return list(PYTHON_BACKENDS) + list(NATIVE_SERIAL)
    return list(PYTHON_BACKENDS) + list(NATIVE_MPI)


def _append_rows(out: Path, rows: list) -> None:
    """Flush rows immediately so OOM/kill mid-sweep does not lose prior timings."""
    if out is None or not rows:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(CSV_HEADER)
        w.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())


def _existing_keys(out: Path) -> set[tuple]:
    if out is None or not out.exists():
        return set()
    keys: set[tuple] = set()
    with out.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                keys.add((row["backend"], int(row["np"]), float(row["mesh_factor"])))
            except (KeyError, ValueError):
                continue
    return keys


def worker_main(args) -> int:
    pid = ops.getPID()
    np_ = ops.getNP()
    if np_ > 1:
        ops.start()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if backends == ["auto"]:
        backends = auto_backends(np_)

    factors = [float(x) for x in args.mesh_factors.split(",") if x.strip()]
    existing = _existing_keys(args.out) if args.skip_existing else set()
    if pid == 0 and existing:
        print(f"[skip-existing] {len(existing)} prior rows in {args.out}", flush=True)

    for factor in factors:
        for backend in backends:
            key = (backend, np_, float(factor))
            if key in existing:
                if pid == 0:
                    print(
                        f"\n>>> np={np_} backend={backend} factor={factor} SKIP (existing)",
                        flush=True,
                    )
                continue

            skip_at = NATIVE_SKIP_LIMITS.get(backend)
            if skip_at is not None and factor >= skip_at:
                if pid == 0:
                    print(
                        f"\n>>> np={np_} backend={backend} factor={factor} SKIP "
                        f"(>= {skip_at})",
                        flush=True,
                    )
                    _append_rows(
                        args.out,
                        [[
                            solver_label(backend, np_), backend, np_, factor,
                            estimate_free_dofs(*mesh_counts(factor)[1]),
                            -1, -1, -1, -999, float("nan"), None, None, None,
                        ]],
                    )
                continue

            if pid == 0:
                print(f"\n>>> np={np_} backend={backend} factor={factor}", flush=True)
            try:
                result = run_case(backend, factor, pid, np_)
            except Exception as exc:
                if pid == 0:
                    print(f"FAIL {backend}: {exc}", file=sys.stderr, flush=True)
                result = {
                    "est": estimate_free_dofs(*mesh_counts(factor)[1]),
                    "neq": -1,
                    "nele": -1,
                    "nnodes": -1,
                    "status": -1,
                    "seconds": float("nan"),
                    "tip": None,
                }
            if result["tip"] is not None:
                print(f"[pid={pid}] tip={result['tip']}", flush=True)
            if pid == 0:
                tip = result["tip"] or (None, None, None)
                print(
                    f"<<< {backend} status={result['status']} "
                    f"neq={result['neq']} ~dofs={result['est']} "
                    f"time={result['seconds']:.4f}s",
                    flush=True,
                )
                _append_rows(
                    args.out,
                    [[
                        solver_label(backend, np_),
                        backend,
                        np_,
                        factor,
                        result["est"],
                        result["neq"],
                        result["nele"],
                        result["nnodes"],
                        result["status"],
                        result["seconds"],
                        tip[0],
                        tip[1],
                        tip[2],
                    ]],
                )

    return 0


def find_mpirun() -> str:
    for candidate in (
        os.environ.get("MPI_RUN"),
        "/opt/intel/oneapi/mpi/latest/bin/mpirun",
        "/opt/intel/oneapi/mpi/2021.16/bin/mpirun",
        "mpirun",
    ):
        if not candidate:
            continue
        if candidate == "mpirun" or Path(candidate).exists():
            return candidate
    return "mpirun"


def launch_sweep(args) -> int:
    script = Path(__file__).resolve()
    out = args.out if args.out.is_absolute() else (Path.cwd() / args.out).resolve()
    args.out = out
    if out.exists() and not args.append:
        out.unlink()
    nps = [int(x) for x in args.nps.split(",") if x.strip()]
    mpirun = find_mpirun()
    for np_ in nps:
        cmd = [
            mpirun, "-np", str(np_),
            sys.executable, str(script),
            "--worker", "--out", str(out),
            "--mesh-factors", args.mesh_factors,
            "--backends", args.backends,
        ]
        if args.append or args.skip_existing:
            cmd.append("--skip-existing")
        print("=" * 60, flush=True)
        print(" ".join(cmd), flush=True)
        rc = subprocess.call(cmd, env=os.environ.copy())
        if rc != 0:
            print(f"mpirun np={np_} failed rc={rc}", file=sys.stderr)
            continue
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--mesh-factors",
        default=",".join(str(int(f) if f == int(f) else f) for f in AGREED_LINEAR_FACTORS),
    )
    ap.add_argument("--backends", default="auto")
    ap.add_argument(
        "--nps",
        default=",".join(str(n) for n in AGREED_NPS),
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="do not truncate CSV; implies --skip-existing",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip backend/np/factor already in CSV",
    )
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.append:
        args.skip_existing = True
    if not args.out.is_absolute():
        args.out = (Path.cwd() / args.out).resolve()
    if args.worker:
        raise SystemExit(worker_main(args))
    raise SystemExit(launch_sweep(args))


if __name__ == "__main__":
    main()
