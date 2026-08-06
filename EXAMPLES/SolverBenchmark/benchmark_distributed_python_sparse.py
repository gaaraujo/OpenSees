#!/usr/bin/env python3
"""
Benchmark DistributedPythonSparse vs serial PythonSparse on the Scott cantilever
brick (same geometry as benchmark_python_sparse.py).

Uses inline SciPy CSR solvers (no CuPy / openseespy-solvers required).

Usage (MPI OpenSeesPy on PYTHONPATH / build-mp/Release):

  # Serial baseline (np=1) — PythonSparse + RCM
  python3 EXAMPLES/SolverBenchmark/benchmark_distributed_python_sparse.py out.csv

  # Partitioned gather-to-root — DistributedPythonSparse + ParallelPlain
  mpirun -np 2 python3 EXAMPLES/SolverBenchmark/benchmark_distributed_python_sparse.py out.csv
  mpirun -np 4 python3 ... out.csv --mesh-factors 12,14,16

Default mesh factors start at 12 (~117k free DOFs) so timings are past the
toy-size regime. Override with --mesh-factors only if you know the DOF count.

Note: solve is gather-to-root on rank 0. Expect enablement / overhead vs serial
PythonSparse, not strong scaling of the factorization.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dps_common import (
    DEFAULT_BENCHMARK_FACTORS,
    MIN_BENCHMARK_DOFS,
    apply_far_face_load,
    build_solid_bar,
    cantilever_load,
    estimate_free_dofs,
    far_corner_disp,
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
    "est_free_dofs",
    "status",
    "time_seconds",
    "displacement_x",
    "displacement_y",
    "displacement_z",
)


def run_one(mesh_factor: float, distributed: bool, pid: int, num_steps: int = 5):
    mesh_size, (nx, ny, nz) = mesh_counts(mesh_factor)
    est_dofs = estimate_free_dofs(nx, ny, nz)
    build_solid_bar(ops, nx, ny, nz)
    n_far = sum(
        1
        for n in ops.getNodeTags()
        if abs(ops.nodeCoord(n, 1) - 10.0) < 1e-9
    )
    if distributed and ops.getNP() > 1:
        ops.partition()
    apply_far_face_load(ops, cantilever_load(), n_far_global=n_far)

    name, cfg, numberer = system_cfg(ops, distributed, pid)
    ops.constraints("Plain")
    ops.numberer(numberer)
    ops.system(name, cfg)
    ops.integrator("LoadControl", 1.0 / num_steps)
    ops.test("NormUnbalance", 1.0e-8, 25, 0)
    ops.algorithm("ModifiedNewton", "-FactorOnce")
    ops.analysis("Static")

    t0 = time.perf_counter()
    status = ops.analyze(num_steps)
    seconds = time.perf_counter() - t0

    tip = far_corner_disp(ops) if status == 0 else None
    return {
        "mesh_size": mesh_size,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "est_dofs": est_dofs,
        "nele": len(ops.getEleTags()),
        "nnodes": len(ops.getNodeTags()),
        "status": status,
        "seconds": seconds,
        "tip": tip,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="output CSV path")
    ap.add_argument(
        "--mesh-factors",
        default=",".join(str(f) for f in DEFAULT_BENCHMARK_FACTORS),
        help="comma-separated mesh refinement factors (default starts ~117k DOFs)",
    )
    ap.add_argument(
        "--allow-small",
        action="store_true",
        help=f"allow meshes with fewer than {MIN_BENCHMARK_DOFS} estimated free DOFs",
    )
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
        max_dofs = max(
            estimate_free_dofs(*mesh_counts(f)[1]) for f in factors
        )
        if max_dofs < MIN_BENCHMARK_DOFS and not args.allow_small:
            print(
                f"ERROR: largest mesh has ~{max_dofs} free DOFs; "
                f"need >= {MIN_BENCHMARK_DOFS} (use factor>=12 or --allow-small)",
                file=sys.stderr,
            )
            raise SystemExit(1)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        print(f"=== {solver_name} brick benchmark ===")
        print(f"mesh factors: {factors} (max ~{max_dofs} free DOFs)")
        print(f"CSV: {args.csv}\n")

    rows = []
    for factor in factors:
        result = run_one(factor, distributed, pid)
        if result["tip"] is not None:
            print(
                f"[pid={pid}] factor={factor} tip={result['tip']}",
                flush=True,
            )
        if pid == 0:
            status_str = "✓" if result["status"] == 0 else "✗"
            print(
                f"  factor={factor:.1f} mesh=({result['nx']},{result['ny']},{result['nz']}) "
                f"~dofs={result['est_dofs']} ele={result['nele']} "
                f"{status_str} {result['seconds']:.4f}s",
                flush=True,
            )
            tip = result["tip"] or (None, None, None)
            rows.append([
                solver_name,
                np_,
                factor,
                result["mesh_size"],
                result["nele"],
                result["nnodes"],
                result["est_dofs"],
                result["status"],
                f"{result['seconds']:.6f}",
                tip[0],
                tip[1],
                tip[2],
            ])
        if result["status"] != 0 and pid == 0:
            print("FAIL: analyze returned non-zero", file=sys.stderr)
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
