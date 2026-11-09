#!/usr/bin/env python3
"""
Parallel Tri31 example (OpenSeesPy docs) adapted for DistributedPythonSparse
and mesh-size refinement.

Original:
  https://openseespydoc.readthedocs.io/en/latest/src/paralleltri31.html

Usage:
  mpirun -np 4 python3 EXAMPLES/SolverBenchmark/parallel_tri31_dps.py
  mpirun -np 4 python3 ... --system Mumps --mesh-size 0.05
  mpirun -np 4 python3 ... --system DistributedPythonSparse --mesh-size 0.1,0.05,0.025
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (
    REPO_ROOT / "build-mp" / "Release",
    REPO_ROOT / "build" / "Release",
    REPO_ROOT / "build",
):
    if (candidate / "opensees.so").exists() or (candidate / "OpenSeesPy.so").exists():
        sys.path.insert(0, str(candidate))
        break

import opensees as ops  # noqa: E402


def parse_mesh_sizes(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def run_once(system: str, meshsize: float, pid: int, np_: int) -> dict:
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    L = 48.0
    H = 4.0
    Lp = L / np_
    ndf = 2

    ops.node(pid, Lp * pid, 0.0)
    ops.node(pid + 1, Lp * (pid + 1), 0.0)
    ops.node(np_ + pid + 2, Lp * (pid + 1), H)
    ops.node(np_ + pid + 1, Lp * pid, H)

    sid = 1
    ops.setStartNodeTag(2 * np_ + 2 + pid * int(H / meshsize + 10))
    ops.mesh("line", 3, 2, pid, np_ + pid + 1, sid, ndf, meshsize)
    ops.setStartNodeTag(2 * np_ + 2 + (pid + 1) * int(H / meshsize + 10))
    ops.mesh("line", 4, 2, pid + 1, np_ + pid + 2, sid, ndf, meshsize)

    ops.setStartNodeTag(
        int(2 * L / meshsize + (np_ + 1) * H / meshsize * 2)
        + pid * int(H * L / meshsize**2 * 2)
    )
    ops.mesh("line", 1, 2, pid, pid + 1, sid, ndf, meshsize)
    ops.mesh("line", 2, 2, np_ + pid + 1, np_ + pid + 2, sid, ndf, meshsize)

    ops.nDMaterial("ElasticIsotropic", 1, 3000.0, 0.3)
    eleArgs = ["tri31", 1.0, "PlaneStress", 1]
    ops.mesh("quad", 5, 4, 1, 4, 2, 3, sid, ndf, meshsize, *eleArgs)

    if pid == 0:
        ops.fix(pid, 1, 1)
        ops.fix(np_ + pid + 1, 1, 1)
    if pid == np_ - 1:
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(np_ + pid + 2, 0.0, -1.0)

    ops.constraints("Transformation")
    ops.numberer("ParallelPlain")

    if system == "Mumps":
        ops.system("Mumps")
    elif system == "DistributedPythonSparse":
        try:
            from openseespy_solvers.scipy import spsolve

            cfg = spsolve().to_openseespy() if pid == 0 else {
                "solver": None,
                "scheme": "CSR",
                "writable": "none",
            }
        except ImportError:
            # Minimal scipy fallback without openseespy-solvers
            import numpy as np
            from scipy import sparse
            from scipy.sparse.linalg import spsolve as sp_spsolve

            class _Sp:
                def __init__(self):
                    self.A = None

                def solve(self, **kw):
                    n = kw["num_eqn"]
                    nnz = kw["nnz"]
                    indptr = np.frombuffer(kw["index_ptr"], dtype=np.int32, count=n + 1)
                    idx = np.frombuffer(kw["indices"], dtype=np.int32, count=nnz)
                    vals = np.frombuffer(kw["values"], dtype=np.float64, count=nnz)
                    b = np.frombuffer(kw["rhs"], dtype=np.float64, count=n)
                    x = np.frombuffer(kw["x"], dtype=np.float64, count=n)
                    if kw["matrix_status"] != "UNCHANGED" or self.A is None:
                        self.A = sparse.csr_matrix(
                            (vals.copy(), idx.copy(), indptr.copy()), shape=(n, n)
                        )
                    x[:] = sp_spsolve(self.A, b)
                    return 0

            cfg = (
                {"solver": _Sp(), "scheme": "CSR", "writable": "none"}
                if pid == 0
                else {"solver": None, "scheme": "CSR", "writable": "none"}
            )
        ops.system("DistributedPythonSparse", cfg)
    else:
        raise ValueError(f"unknown system {system}")

    ops.test("NormDispIncr", 1e-6, 25, 0)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")

    n_ele = len(ops.getEleTags())
    n_node = len(ops.getNodeTags())

    ops.stop()
    ops.start()
    t0 = time.perf_counter()
    status = ops.analyze(1)
    seconds = time.perf_counter() - t0
    ops.stop()

    tip = None
    tip_tag = None
    if pid == np_ - 1:
        tip_tag = pid + 1
        tip = ops.nodeDisp(tip_tag)

    return {
        "status": status,
        "seconds": seconds,
        "n_ele": n_ele,
        "n_node": n_node,
        "tip_tag": tip_tag,
        "tip": tip,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--system",
        choices=("DistributedPythonSparse", "Mumps"),
        default="DistributedPythonSparse",
    )
    ap.add_argument(
        "--mesh-size",
        default="0.1,0.05,0.025",
        help="comma-separated mesh sizes (smaller = finer)",
    )
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    pid = ops.getPID()
    np_ = ops.getNP()
    if np_ < 2:
        if pid == 0:
            print("Need at least 2 MPI ranks", file=sys.stderr)
        sys.exit(1)

    ops.start()
    sizes = parse_mesh_sizes(args.mesh_size)

    rows = []
    for meshsize in sizes:
        if pid == 0:
            print(
                f"=== system={args.system} np={np_} meshsize={meshsize} ===",
                flush=True,
            )
        res = run_once(args.system, meshsize, pid, np_)
        if pid == 0:
            print(
                f"status={res['status']} time={res['seconds']:.6f}s "
                f"local_ele={res['n_ele']} local_nodes={res['n_node']}",
                flush=True,
            )
        if pid == np_ - 1 and res["tip"] is not None:
            print(f"Node {res['tip_tag']} {res['tip']}", flush=True)
        rows.append((meshsize, res))

    if pid == 0 and args.csv is not None:
        write_header = not args.csv.exists()
        with args.csv.open("a", newline="") as fh:
            w = csv.writer(fh)
            if write_header:
                w.writerow([
                    "system", "np", "meshsize", "status", "time_seconds",
                    "local_ele", "local_nodes",
                ])
            for meshsize, res in rows:
                w.writerow([
                    args.system, np_, meshsize, res["status"],
                    f"{res['seconds']:.6f}", res["n_ele"], res["n_node"],
                ])

    if any(r["status"] != 0 for _, r in rows):
        sys.exit(2)


if __name__ == "__main__":
    main()
