#!/usr/bin/env python3
"""
Correctness checks for DistributedPythonSparse (mirrors serial PythonSparse paths).

Usage:
  mpirun -np 2 python3 EXAMPLES/SolverBenchmark/test_distributed_python_sparse.py
  mpirun -np 4 python3 EXAMPLES/SolverBenchmark/test_distributed_python_sparse.py

Checks:
  1) Hand-partitioned 2D truss (linear + eigen) — same as the original smoke.
  2) Scott cantilever brick (mesh_factor=2): tip displacement matches the serial
     PythonSparse reference to relative tolerance (gather-to-root correctness).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dps_common import (
    apply_far_face_load,
    build_solid_bar,
    cantilever_load,
    eigen_cfg,
    far_corner_disp,
    import_opensees,
    mesh_counts,
    system_cfg,
)

ops = import_opensees()

# Serial PythonSparse tip on mesh_factor=2 (same load / Newton settings as below)
REF_TIP = (0.010351113527020863, 1.974276956407577e-05, -0.07053295742609549)
TIP_RTOL = 1.0e-6
TIP_ATOL = 1.0e-10


def check_truss(pid: int) -> None:
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.uniaxialMaterial("Elastic", 1, 3000.0)

    if pid == 0:
        ops.node(1, 0.0, 0.0)
        ops.node(4, 72.0, 96.0)
        ops.fix(1, 1, 1)
        ops.mass(4, 100.0, 100.0)
        ops.element("Truss", 1, 1, 4, 10.0, 1)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(4, 100.0, -50.0)
    else:
        ops.node(2, 144.0, 0.0)
        ops.node(3, 168.0, 0.0)
        ops.node(4, 72.0, 96.0)
        ops.fix(2, 1, 1)
        ops.fix(3, 1, 1)
        ops.mass(4, 100.0, 100.0)
        ops.element("Truss", 2, 2, 4, 5.0, 1)
        ops.element("Truss", 3, 3, 4, 5.0, 1)

    name, cfg, numberer = system_cfg(ops, True, pid)
    ops.constraints("Transformation")
    ops.numberer(numberer)
    ops.system(name, cfg)
    ops.test("NormDispIncr", 1e-8, 10, 0)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")

    ok = ops.analyze(1)
    disp = ops.nodeDisp(4)
    if pid == 0:
        print(f"[truss] analyze={ok} node4 disp={disp}")
        if ok != 0:
            raise SystemExit(2)
        if abs(disp[0]) < 1e-12 and abs(disp[1]) < 1e-12:
            raise SystemExit("truss: zero displacement")

    # Eigen on a fresh copy of the same partitioned topology
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.uniaxialMaterial("Elastic", 1, 3000.0)
    if pid == 0:
        ops.node(1, 0.0, 0.0)
        ops.node(4, 72.0, 96.0)
        ops.fix(1, 1, 1)
        ops.mass(4, 100.0, 100.0)
        ops.element("Truss", 1, 1, 4, 10.0, 1)
    else:
        ops.node(2, 144.0, 0.0)
        ops.node(3, 168.0, 0.0)
        ops.node(4, 72.0, 96.0)
        ops.fix(2, 1, 1)
        ops.fix(3, 1, 1)
        ops.mass(4, 100.0, 100.0)
        ops.element("Truss", 2, 2, 4, 5.0, 1)
        ops.element("Truss", 3, 3, 4, 5.0, 1)

    name, cfg, numberer = system_cfg(ops, True, pid)
    ops.constraints("Transformation")
    ops.numberer(numberer)
    ops.system(name, cfg)
    ops.analysis("Static")
    eigs = ops.eigen("DistributedPythonSparse", 1, eigen_cfg(True, pid))
    if pid == 0:
        print(f"[truss] eigenvalues={eigs}")
        if eigs is None or len(eigs) < 1:
            raise SystemExit("truss: no eigenvalues")


def check_brick(pid: int, np_: int) -> None:
    _, (nx, ny, nz) = mesh_counts(2.0)
    build_solid_bar(ops, nx, ny, nz)
    n_far = sum(
        1
        for n in ops.getNodeTags()
        if math.isclose(ops.nodeCoord(n, 1), 10.0, abs_tol=1e-9)
    )
    if np_ > 1:
        ops.partition()
    apply_far_face_load(ops, cantilever_load(), n_far_global=n_far)

    name, cfg, numberer = system_cfg(ops, True, pid)
    ops.constraints("Plain")
    ops.numberer(numberer)
    ops.system(name, cfg)
    ops.integrator("LoadControl", 0.2)
    ops.test("NormUnbalance", 1.0e-8, 25, 0)
    ops.algorithm("ModifiedNewton", "-FactorOnce")
    ops.analysis("Static")

    ok = ops.analyze(5)
    if ok != 0:
        raise SystemExit(f"[brick][pid={pid}] analyze failed status={ok}")

    tip = far_corner_disp(ops)
    if tip is not None:
        print(f"[brick][pid={pid}] tip={tip}", flush=True)
        for a, b in zip(tip, REF_TIP):
            if abs(a - b) > TIP_ATOL + TIP_RTOL * abs(b):
                raise SystemExit(
                    f"[brick][pid={pid}] tip mismatch: got {tip} expected {REF_TIP}"
                )
        print(
            f"[brick][pid={pid}] tip matches serial PythonSparse reference",
            flush=True,
        )
    if pid == 0:
        print(f"[brick] analyze ok mesh=({nx},{ny},{nz})", flush=True)


def main() -> None:
    pid = ops.getPID()
    np_ = ops.getNP()
    if np_ < 2:
        if pid == 0:
            print("Need at least 2 MPI ranks", file=sys.stderr)
        raise SystemExit(1)

    ops.start()
    check_truss(pid)
    check_brick(pid, np_)
    if pid == 0:
        print("PASS")


if __name__ == "__main__":
    main()
