#!/usr/bin/env python3
"""
Agreed eigen SolverBenchmark matrix (distributed + genBandArpack).

  - scipy-eigsh : SciPy eigsh via PythonSparse / DistributedPythonSparse
  - genBandArpack : native OpenSees (np=1 only in this matrix)

Linear SOE for assembly uses openseespy-solvers >= 0.2.0 umfpack (CSC).
No fullGenLapack.

Usage:
  python3 EXAMPLES/SolverBenchmark/run_dps_eigen_sweep.py \\
    --out figures/dps_eigen.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dps_common import (
    AGREED_EIGEN_FACTORS,
    AGREED_NPS,
    build_solid_bar,
    eigen_cfg,
    estimate_free_dofs,
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
    "num_modes",
    "status",
    "time_seconds",
    "eigenvalue_1",
    "eigenvalue_5",
)


def solver_label(backend: str, np_: int) -> str:
    pretty = {
        "scipy-eigsh": "SciPyEigsh",
        "genBandArpack": "genBandArpack",
    }.get(backend, backend)
    return f"{pretty} (np={np_})"


def run_case(backend: str, mesh_factor: float, pid: int, np_: int, num_modes: int):
    _, (nx, ny, nz) = mesh_counts(mesh_factor)
    est = estimate_free_dofs(nx, ny, nz)
    build_solid_bar(ops, nx, ny, nz)

    if backend == "genBandArpack":
        if np_ > 1:
            raise RuntimeError("genBandArpack only for np=1 in this matrix")
        ops.constraints("Plain")
        ops.numberer("RCM")  # band eigen path
        ops.system("BandSPD")
        ops.analysis("Static")
        t0 = time.perf_counter()
        lam = ops.eigen("genBandArpack", num_modes)
        seconds = time.perf_counter() - t0
    elif backend == "scipy-eigsh":
        distributed = np_ > 1
        if distributed:
            ops.partition()
        name = "DistributedPythonSparse" if distributed else "PythonSparse"
        # eigsh is iterative / shift-invert; Plain is fine (not a band solver)
        numberer = lin_numberer("umfpack", np_)
        lin = make_python_lin_solver("umfpack") if (not distributed or pid == 0) else None
        ops.constraints("Plain")
        ops.numberer(numberer)
        ops.system(
            name,
            {
                "solver": lin,
                "scheme": python_lin_scheme("umfpack"),
                "writable": "none",
            },
        )
        ops.analysis("Static")
        t0 = time.perf_counter()
        if distributed:
            lam = ops.eigen("DistributedPythonSparse", num_modes, eigen_cfg(True, pid))
        else:
            lam = ops.eigen("PythonSparse", num_modes, eigen_cfg(False, pid))
        seconds = time.perf_counter() - t0
    else:
        raise SystemExit(f"unknown backend {backend}")

    status = 0 if lam is not None and len(lam) >= 1 else -1
    neq = -1
    try:
        neq = int(ops.systemSize())
    except Exception:
        pass
    return {
        "est": est,
        "neq": neq if neq > 0 else est,
        "nele": len(ops.getEleTags()),
        "nnodes": len(ops.getNodeTags()),
        "status": status,
        "seconds": seconds,
        "lam": lam,
    }


def auto_backends(np_: int) -> list[str]:
    if np_ == 1:
        return ["scipy-eigsh", "genBandArpack"]
    return ["scipy-eigsh"]


def _append_rows(out: Path, rows: list) -> None:
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
            if pid == 0:
                print(f"\n>>> np={np_} backend={backend} factor={factor}", flush=True)
            try:
                result = run_case(backend, factor, pid, np_, args.modes)
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
                    "lam": None,
                }
            if pid == 0:
                lam = result["lam"] or []
                print(
                    f"<<< {backend} status={result['status']} "
                    f"~dofs={result['est']} time={result['seconds']:.4f}s "
                    f"lam0={lam[0] if lam else None}",
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
                        args.modes,
                        result["status"],
                        result["seconds"],
                        lam[0] if len(lam) > 0 else "",
                        lam[4] if len(lam) > 4 else "",
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
            "--modes", str(args.modes),
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
        default=",".join(str(int(f) if f == int(f) else f) for f in AGREED_EIGEN_FACTORS),
    )
    ap.add_argument("--backends", default="auto")
    ap.add_argument("--nps", default=",".join(str(n) for n in AGREED_NPS))
    ap.add_argument("--modes", type=int, default=5)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
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
