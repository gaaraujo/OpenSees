#!/usr/bin/env python3
"""
Benchmark DistributedPythonSparse vs serial PythonSparse on the cantilever
brick mesh from the openseespy-solvers docs / SolverBenchmark suite.

Usage (from repo root, MPI OpenSeesPy build on PYTHONPATH):

  # Serial baseline (np=1): uses PythonSparse
  python3 EXAMPLES/SolverBenchmark/benchmark_distributed_python_sparse.py --mesh-factor 2

  # Partitioned DistributedPythonSparse
  mpirun -np 2 python3 EXAMPLES/SolverBenchmark/benchmark_distributed_python_sparse.py --mesh-factor 2
  mpirun -np 4 python3 ... --mesh-factor 2
  mpirun -np 8 python3 ... --mesh-factor 2

With --mesh-factor matching the docs examples (2, 4, …), timings are comparable
in spirit to https://openseespy-solvers.readthedocs.io/en/stable/examples/

Note: DistributedPythonSparse is gather-to-root — rank 0 still factors the full
system. Expect communication overhead vs serial PythonSparse, not strong-scaling
of the factorization itself.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_CANDIDATES = (
    REPO_ROOT / "build-mp" / "Release",
    REPO_ROOT / "build" / "Release",
    REPO_ROOT / "build",
)
for candidate in BUILD_CANDIDATES:
    if (candidate / "opensees.so").exists() or (candidate / "OpenSeesPy.so").exists():
        sys.path.insert(0, str(candidate))
        break

import opensees as ops  # noqa: E402

try:
    from openseespy_solvers.scipy import spsolve, eigsh
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "openseespy-solvers is required (pip install openseespy-solvers)"
    ) from exc

BAR_LENGTH = 10.0
BAR_HEIGHT = 2.0
BAR_THICKNESS = 1.0
ELASTIC_MODULUS = 29_000.0
POISSON_RATIO = 0.3
STEEL_DENSITY = 0.284e-3 / 386.4
NUM_STEPS = 10
NUM_MODES = 5
TOTAL_LOAD = 1.0


def mesh_counts(factor: float):
    mesh_size = BAR_THICKNESS / factor
    nx = max(1, int(math.ceil(BAR_LENGTH / mesh_size)))
    ny = max(1, int(math.ceil(BAR_THICKNESS / mesh_size)))
    nz = max(1, int(math.ceil(BAR_HEIGHT / mesh_size)))
    return mesh_size, nx, ny, nz


def build_model(nx: int, ny: int, nz: int) -> None:
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


def tip_disp():
    tip = max(
        (
            n
            for n in ops.getNodeTags()
            if math.isclose(ops.nodeCoord(n, 1), BAR_LENGTH, abs_tol=1e-9)
        ),
        key=lambda n: ops.nodeCoord(n, 3),
        default=None,
    )
    return ops.nodeDisp(tip) if tip is not None else None


def configure_linear(distributed: bool, pid: int) -> None:
    cfg = spsolve().to_openseespy()
    if distributed:
        if pid != 0:
            cfg = {"solver": None, "scheme": cfg.get("scheme", "CSR"), "writable": "none"}
        ops.system("DistributedPythonSparse", cfg)
        ops.numberer("ParallelPlain")
    else:
        ops.system("PythonSparse", cfg)
        ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0 / NUM_STEPS)
    ops.test("NormUnbalance", 1.0e-8, 25, 0)
    ops.algorithm("ModifiedNewton", "-FactorOnce")
    ops.analysis("Static")


def apply_load(n_far_global: int) -> None:
    """Load free-face nodes owned by this rank (after partition)."""
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    far = [
        n for n in ops.getNodeTags()
        if math.isclose(ops.nodeCoord(n, 1), BAR_LENGTH, abs_tol=1e-9)
    ]
    if far and n_far_global > 0:
        f = TOTAL_LOAD / n_far_global
        for n in far:
            ops.load(n, 0.0, 0.0, -f)


def run_linear(distributed: bool, pid: int, nx, ny, nz):
    build_model(nx, ny, nz)
    n_far_global = sum(
        1
        for n in ops.getNodeTags()
        if math.isclose(ops.nodeCoord(n, 1), BAR_LENGTH, abs_tol=1e-9)
    )
    if distributed and ops.getNP() > 1:
        ops.partition()
    apply_load(n_far_global)
    configure_linear(distributed, pid)
    t0 = time.perf_counter()
    status = ops.analyze(NUM_STEPS)
    seconds = time.perf_counter() - t0
    disp = tip_disp() if status == 0 else None
    return status, seconds, disp, len(ops.getEleTags()), len(ops.getNodeTags())


def run_eigen(distributed: bool, pid: int, nx, ny, nz):
    build_model(nx, ny, nz)
    if distributed and ops.getNP() > 1:
        ops.partition()
    configure_linear(distributed, pid)
    eig_cfg = eigsh().to_openseespy()
    t0 = time.perf_counter()
    if distributed:
        if pid != 0:
            eig_cfg = {"solver": None, "scheme": eig_cfg.get("scheme", "CSR")}
        lam = ops.eigen("DistributedPythonSparse", NUM_MODES, eig_cfg)
    else:
        lam = ops.eigen("PythonSparse", NUM_MODES, eig_cfg)
    seconds = time.perf_counter() - t0
    status = 0 if lam is not None and len(lam) >= 1 else -1
    return status, seconds, lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-factor", type=float, default=2.0)
    ap.add_argument("--eigen", action="store_true", help="also time eigen")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    pid = ops.getPID()
    np_ = ops.getNP()
    if np_ > 1:
        ops.start()

    distributed = np_ > 1
    _, nx, ny, nz = mesh_counts(args.mesh_factor)
    solver_name = (
        f"DistributedPythonSparse(np={np_})" if distributed else "PythonSparse(serial)"
    )

    status, seconds, disp, n_ele, n_node = run_linear(distributed, pid, nx, ny, nz)
    # Tip may live off rank 0; any rank with a tip value may print it.
    if disp is not None:
        print(
            f"[linear][pid={pid}] tip={disp}",
            flush=True,
        )
    if pid == 0:
        print(
            f"[linear] {solver_name} mesh_factor={args.mesh_factor} "
            f"mesh=({nx},{ny},{nz}) ele={n_ele} nodes={n_node} "
            f"status={status} time={seconds:.6f}s",
            flush=True,
        )

    eigen_seconds = None
    eigen_status = None
    if args.eigen:
        eigen_status, eigen_seconds, lam = run_eigen(distributed, pid, nx, ny, nz)
        if pid == 0:
            print(
                f"[eigen]  {solver_name} modes={NUM_MODES} "
                f"status={eigen_status} time={eigen_seconds:.6f}s "
                f"lam0={lam[0] if lam else None}",
                flush=True,
            )

    if pid == 0 and args.csv is not None:
        write_header = not args.csv.exists()
        with args.csv.open("a", newline="") as fh:
            w = csv.writer(fh)
            if write_header:
                w.writerow([
                    "kind", "solver", "np", "mesh_factor", "nx", "ny", "nz",
                    "num_elements", "num_nodes", "status", "time_seconds",
                    "disp_x", "disp_y", "disp_z",
                ])
            dx = dy = dz = None
            if disp is not None:
                dx, dy, dz = disp[0], disp[1], disp[2]
            w.writerow([
                "linear", solver_name, np_, args.mesh_factor, nx, ny, nz,
                n_ele, n_node, status, f"{seconds:.6f}", dx, dy, dz,
            ])
            if args.eigen:
                w.writerow([
                    "eigen", solver_name, np_, args.mesh_factor, nx, ny, nz,
                    n_ele, n_node, eigen_status, f"{eigen_seconds:.6f}",
                    "", "", "",
                ])

    if status != 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
