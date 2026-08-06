"""Shared helpers for DistributedPythonSparse smoke / correctness / benchmarks."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve

# Scott cantilever brick (same geometry as benchmark_python_sparse.py)
BAR_LENGTH = 10.0
BAR_HEIGHT = 2.0
BAR_THICKNESS = 1.0
ELASTIC_MODULUS = 29_000.0
POISSON_RATIO = 0.3
STEEL_DENSITY = 0.284e-3 / 386.4
YIELD_STRESS = 50.0


def import_opensees():
    """Prefer a local OpenSeesPy build (MPI or serial), else installed package."""
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (
        repo_root / "build-mp" / "Release",
        repo_root / "build" / "Release",
        repo_root / "build",
    ):
        if (candidate / "opensees.so").exists() or (candidate / "OpenSeesPy.so").exists():
            sys.path.insert(0, str(candidate))
            break
    import opensees as ops  # noqa: WPS433

    return ops


class ScipyLinSolver:
    """CSR SciPy direct solver for PythonSparse / DistributedPythonSparse."""

    def __init__(self):
        self.A = None

    def solve(self, **kwargs):
        num_eqn = kwargs["num_eqn"]
        nnz = kwargs["nnz"]
        indptr = np.frombuffer(kwargs["index_ptr"], dtype=np.int32, count=num_eqn + 1)
        idx = np.frombuffer(kwargs["indices"], dtype=np.int32, count=nnz)
        vals = np.frombuffer(kwargs["values"], dtype=np.float64, count=nnz)
        b = np.frombuffer(kwargs["rhs"], dtype=np.float64, count=num_eqn)
        x = np.frombuffer(kwargs["x"], dtype=np.float64, count=num_eqn)
        if kwargs["matrix_status"] != "UNCHANGED" or self.A is None:
            self.A = sparse.csr_matrix(
                (vals.copy(), idx.copy(), indptr.copy()), shape=(num_eqn, num_eqn)
            )
        x[:] = spsolve(self.A, b)
        return 0


class ScipyEigenSolver:
    """Generalized eigen solver matching the serial PythonSparse eigen benchmark."""

    def __init__(self):
        self._k = None
        self._m = None

    def solve(self, **kwargs):
        num_eqn = kwargs["num_eqn"]
        nnz = kwargs["nnz"]
        num_modes = int(kwargs["num_modes"])
        indptr = np.frombuffer(kwargs["index_ptr"], dtype=np.int32, count=num_eqn + 1)
        idx = np.frombuffer(kwargs["indices"], dtype=np.int32, count=nnz)
        kvals = np.frombuffer(kwargs["k_values"], dtype=np.float64, count=nnz)
        mvals = np.frombuffer(kwargs["m_values"], dtype=np.float64, count=nnz)
        status = kwargs.get("matrix_status", "STRUCTURE_CHANGED")

        if status == "STRUCTURE_CHANGED" or self._k is None:
            self._k = sparse.csr_matrix(
                (kvals.copy(), idx.copy(), indptr.copy()), shape=(num_eqn, num_eqn)
            )
            self._m = sparse.csr_matrix(
                (mvals.copy(), idx.copy(), indptr.copy()), shape=(num_eqn, num_eqn)
            )
        elif status == "COEFFICIENTS_CHANGED":
            self._k.data[:] = kvals
            self._m.data[:] = mvals

        k = min(num_modes, max(num_eqn - 1, 1))
        if num_eqn <= 3:
            from scipy.linalg import eigh

            evals, evecs = eigh(self._k.toarray(), self._m.toarray())
            evals = evals[:k]
            evecs = evecs[:, :k]
        else:
            evals, evecs = eigsh(self._k, k=k, M=self._m, which="SM", sigma=0.0)

        evals_buf = np.frombuffer(kwargs["eigenvalues"], dtype=np.float64, count=num_modes)
        evecs_buf = np.frombuffer(
            kwargs["eigenvectors"], dtype=np.float64, count=num_modes * num_eqn
        )
        evals_buf[:k] = evals[:k]
        for mode in range(k):
            evecs_buf[mode * num_eqn : (mode + 1) * num_eqn] = evecs[:, mode]
        return None


def mesh_counts(factor: float) -> Tuple[float, Tuple[int, int, int]]:
    mesh_size = BAR_THICKNESS / factor

    def count(dim: float) -> int:
        return max(1, int(math.ceil(dim / mesh_size)))

    return mesh_size, (count(BAR_LENGTH), count(BAR_THICKNESS), count(BAR_HEIGHT))


def estimate_free_dofs(nx: int, ny: int, nz: int) -> int:
    """Approximate free DOFs after fixX(0) on the Scott brick (3 dof/node)."""
    nodes = (nx + 1) * (ny + 1) * (nz + 1)
    fixed = (ny + 1) * (nz + 1)
    return 3 * (nodes - fixed)


# Performance benchmarks should use meshes with at least this many free DOFs.
# factor=12 → ~117k; factor=10 is only ~69k.
MIN_BENCHMARK_DOFS = 100_000
DEFAULT_BENCHMARK_FACTORS = [12.0, 14.0, 16.0]

def build_solid_bar(ops, nx: int, ny: int, nz: int) -> None:
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


def cantilever_load() -> float:
    return 1.25 * YIELD_STRESS * (BAR_THICKNESS * BAR_HEIGHT**2) / (6 * BAR_LENGTH)


def apply_far_face_load(ops, total_load: float, n_far_global: Optional[int] = None) -> int:
    """Load free-face nodes; n_far_global is pre-partition count when partitioned."""
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    far = [
        n for n in ops.getNodeTags()
        if math.isclose(ops.nodeCoord(n, 1), BAR_LENGTH, abs_tol=1e-9)
    ]
    n_ref = n_far_global if n_far_global is not None else len(far)
    if far and n_ref > 0:
        f = total_load / n_ref
        for n in far:
            ops.load(n, 0.0, 0.0, -f)
    return len(far)


def far_corner_disp(ops) -> Optional[Tuple[float, float, float]]:
    target = [BAR_LENGTH, BAR_THICKNESS / 2.0, BAR_HEIGHT / 2.0]
    for node in ops.getNodeTags():
        xyz = [ops.nodeCoord(node, 1), ops.nodeCoord(node, 2), ops.nodeCoord(node, 3)]
        if np.isclose(xyz, target, atol=1e-9).all():
            return (
                ops.nodeDisp(node, 1),
                ops.nodeDisp(node, 2),
                ops.nodeDisp(node, 3),
            )
    return None


def system_cfg(ops, distributed: bool, pid: int):
    """Return (system_name, dict, numberer) for scipy CSR."""
    if distributed:
        cfg = {
            "solver": ScipyLinSolver() if pid == 0 else None,
            "scheme": "CSR",
            "writable": "none",
        }
        return "DistributedPythonSparse", cfg, "ParallelPlain"
    cfg = {"solver": ScipyLinSolver(), "scheme": "CSR", "writable": "none"}
    return "PythonSparse", cfg, "RCM"


def eigen_cfg(distributed: bool, pid: int):
    if distributed:
        return {
            "solver": ScipyEigenSolver() if pid == 0 else None,
            "scheme": "CSR",
        }
    return {"solver": ScipyEigenSolver(), "scheme": "CSR"}
