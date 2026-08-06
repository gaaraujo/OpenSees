#!/usr/bin/env python3
"""
Eigen benchmark for DistributedPythonSparse vs serial PythonSparse.

Same Scott cantilever brick as benchmark_python_sparse_eigen.py; SciPy CSR
generalized eigen on rank 0 only when distributed.

Usage:
  python3 EXAMPLES/SolverBenchmark/benchmark_distributed_python_sparse_eigen.py out.csv
  mpirun -np 2 python3 EXAMPLES/SolverBenchmark/benchmark_distributed_python_sparse_eigen.py out.csv
  mpirun -np 4 python3 ... out.csv --mesh-factors 2,4 --modes 5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dps_common import (
    build_solid_bar,
    eigen_cfg,
    import_opensees,
    mesh_counts,
    system_cfg,
)

ops = import_opensees()

CSV_HEADER = (
    "solver",
    "np",
    "mesh_factor",
    "mesh_c",
    "num_elements",
    "num_nodes",
    "num_modes",
    "status",
    "time_seconds",
    "eigenvalue_1",
    "eigenvalue_5",
)

DEFAULT_FACTORS = [2.0, 4.0, 6.0]


def run_one(mesh_factor: float, distributed: bool, pid: int, num_modes: int):
    mesh_size, (nx, ny, nz) = mesh_counts(mesh_factor)
    build_solid_bar(ops, nx, ny, nz)
    if distributed and ops.getNP() > 1:
        ops.partition()

    name, cfg, numberer = system_cfg(ops, distributed, pid)
    ops.constraints("Plain")
    ops.numberer(numberer)
    ops.system(name, cfg)
    ops.analysis("Static")

    t0 = time.perf_counter()
    if distributed:
        lam = ops.eigen("DistributedPythonSparse", num_modes, eigen_cfg(True, pid))
    else:
        lam = ops.eigen("PythonSparse", num_modes, eigen_cfg(False, pid))
    seconds = time.perf_counter() - t0
    status = 0 if lam is not None and len(lam) >= 1 else -1
    return {
        "mesh_size": mesh_size,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "nele": len(ops.getEleTags()),
        "nnodes": len(ops.getNodeTags()),
        "status": status,
        "seconds": seconds,
        "lam": lam,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--mesh-factors", default=",".join(str(f) for f in DEFAULT_FACTORS))
    ap.add_argument("--modes", type=int, default=5)
    args = ap.parse_args()

    pid = ops.getPID()
    np_ = ops.getNP()
    if np_ > 1:
        ops.start()

    distributed = np_ > 1
    solver_name = (
        f"DistributedPythonSparse(np={np_})" if distributed else "PythonSparse(serial)"
    )
    factors = [float(x) for x in args.mesh_factors.split(",") if x.strip()]

    if pid == 0:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        print(f"=== {solver_name} eigen benchmark ===")
        print(f"modes={args.modes} factors={factors}\n")

    rows = []
    for factor in factors:
        result = run_one(factor, distributed, pid, args.modes)
        if pid == 0:
            status_str = "✓" if result["status"] == 0 else "✗"
            lam0 = result["lam"][0] if result["lam"] else None
            print(
                f"  factor={factor:.1f} {status_str} {result['seconds']:.4f}s "
                f"lam0={lam0}",
                flush=True,
            )
            lam = result["lam"] or []
            rows.append([
                solver_name,
                np_,
                factor,
                result["mesh_size"],
                result["nele"],
                result["nnodes"],
                args.modes,
                result["status"],
                f"{result['seconds']:.6f}",
                lam[0] if len(lam) > 0 else "",
                lam[4] if len(lam) > 4 else "",
            ])
        if result["status"] != 0 and pid == 0:
            raise SystemExit(2)

    if pid == 0:
        write_header = not args.csv.exists()
        with args.csv.open("a", newline="") as fh:
            w = csv.writer(fh)
            if write_header:
                w.writerow(CSV_HEADER)
            w.writerows(rows)


if __name__ == "__main__":
    main()
