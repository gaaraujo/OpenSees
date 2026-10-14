#!/usr/bin/env python3
"""Smoke test: DistributedPythonSparse on a 2-rank partitioned truss."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, eigsh

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD = REPO_ROOT / "build-mp" / "Release"
sys.path.insert(0, str(BUILD))
import opensees as ops  # noqa: E402


class ScipyLinSolver:
    def __init__(self):
        self.A = None

    def solve(self, **kwargs):
        index_ptr = kwargs["index_ptr"]
        indices = kwargs["indices"]
        values = kwargs["values"]
        rhs = kwargs["rhs"]
        x = kwargs["x"]
        num_eqn = kwargs["num_eqn"]
        nnz = kwargs["nnz"]
        matrix_status = kwargs["matrix_status"]

        indptr = np.frombuffer(index_ptr, dtype=np.int32, count=num_eqn + 1)
        idx = np.frombuffer(indices, dtype=np.int32, count=nnz)
        vals = np.frombuffer(values, dtype=np.float64, count=nnz)
        b = np.frombuffer(rhs, dtype=np.float64, count=num_eqn)
        x_view = np.frombuffer(x, dtype=np.float64, count=num_eqn)

        if matrix_status != "UNCHANGED" or self.A is None:
            self.A = sparse.csr_matrix(
                (vals.copy(), idx.copy(), indptr.copy()), shape=(num_eqn, num_eqn)
            )

        x_view[:] = spsolve(self.A, b)
        return 0


class ScipyEigenSolver:
    def solve(self, **kwargs):
        index_ptr = kwargs["index_ptr"]
        indices = kwargs["indices"]
        k_values = kwargs["k_values"]
        m_values = kwargs["m_values"]
        eigenvalues = kwargs["eigenvalues"]
        eigenvectors = kwargs["eigenvectors"]
        num_eqn = kwargs["num_eqn"]
        nnz = kwargs["nnz"]
        num_modes = min(int(kwargs["num_modes"]), max(num_eqn - 1, 1))

        indptr = np.frombuffer(index_ptr, dtype=np.int32, count=num_eqn + 1)
        idx = np.frombuffer(indices, dtype=np.int32, count=nnz)
        kvals = np.frombuffer(k_values, dtype=np.float64, count=nnz)
        mvals = np.frombuffer(m_values, dtype=np.float64, count=nnz)

        K = sparse.csr_matrix((kvals.copy(), idx.copy(), indptr.copy()), shape=(num_eqn, num_eqn))
        M = sparse.csr_matrix((mvals.copy(), idx.copy(), indptr.copy()), shape=(num_eqn, num_eqn))

        if num_eqn <= 3:
            from scipy.linalg import eigh

            evals, evecs = eigh(K.toarray(), M.toarray())
            evals = evals[:num_modes]
            evecs = evecs[:, :num_modes]
        else:
            evals, evecs = eigsh(K, k=num_modes, M=M, which="SM", sigma=0.0)

        evals_buf = np.frombuffer(eigenvalues, dtype=np.float64, count=num_modes)
        evecs_buf = np.frombuffer(eigenvectors, dtype=np.float64, count=num_modes * num_eqn)
        evals_buf[:] = evals[:num_modes]
        # mode-major layout
        for mode in range(num_modes):
            evecs_buf[mode * num_eqn : (mode + 1) * num_eqn] = evecs[:, mode]
        return None


def build_model(pid: int, solver):
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

    ops.constraints("Transformation")
    ops.numberer("ParallelPlain")
    ops.system(
        "DistributedPythonSparse",
        {"solver": solver, "scheme": "CSR", "writable": "none"},
    )
    ops.test("NormDispIncr", 1e-8, 10, 0)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")


def main():
    pid = ops.getPID()
    np_ = ops.getNP()
    if np_ < 2:
        if pid == 0:
            print("Need at least 2 MPI ranks", file=sys.stderr)
        sys.exit(1)

    ops.start()

    # --- Linear solve ---
    lin = ScipyLinSolver() if pid == 0 else None
    build_model(pid, lin)
    ok = ops.analyze(1)
    disp = ops.nodeDisp(4)
    if pid == 0:
        print(f"linear analyze={ok} node4 disp={disp}")
        if ok != 0:
            sys.exit(2)
        if abs(disp[0]) < 1e-12 and abs(disp[1]) < 1e-12:
            print("ERROR: zero displacement", file=sys.stderr)
            sys.exit(3)

    # --- Eigen solve ---
    lin2 = ScipyLinSolver() if pid == 0 else None
    build_model(pid, lin2)
    eigen_solver = ScipyEigenSolver() if pid == 0 else None
    eigs = ops.eigen(
        "DistributedPythonSparse",
        1,
        {"solver": eigen_solver, "scheme": "CSR"},
    )
    if pid == 0:
        print(f"eigenvalues={eigs}")
        if eigs is None or len(eigs) < 1:
            print("ERROR: no eigenvalues", file=sys.stderr)
            sys.exit(4)
        print("SMOKE OK")


if __name__ == "__main__":
    main()
