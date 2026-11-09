"""Shared helpers for DistributedPythonSparse smoke / correctness / benchmarks."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import eigsh

# Scott cantilever brick (same geometry as benchmark_python_sparse.py)
BAR_LENGTH = 10.0
BAR_HEIGHT = 2.0
BAR_THICKNESS = 1.0
ELASTIC_MODULUS = 29_000.0
POISSON_RATIO = 0.3
STEEL_DENSITY = 0.284e-3 / 386.4
YIELD_STRESS = 50.0

# optional scikit-umfpack
try:
    import scikits.umfpack as _umfpack
except ImportError:  # pragma: no cover
    _umfpack = None


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


class SciPyUmfpackSolver:
    """
    Fallback UMFPACK solver if openseespy-solvers < 0.2.0 is unavailable.

    Prefer ``openseespy_solvers.scipy.umfpack(scheme='CSC', index_dtype='auto')``.
    """

    def __init__(self):
        if _umfpack is None:
            raise ImportError(
                "scikits.umfpack is required (pip/conda install scikit-umfpack)"
            )
        self.umf = _umfpack.UmfpackContext("di")
        self.A = None

    def _wrap_csc(self, num_eqn: int, nnz: int, index_ptr, indices, values):
        # OpenSees buffers are int32 / float64. Own contiguous copies for the
        # UMFPACK C API, but stay on int32 ('di') — do not widen to int64.
        indptr = np.array(
            np.frombuffer(index_ptr, dtype=np.int32, count=num_eqn + 1),
            dtype=np.int32,
            order="C",
            copy=True,
        )
        idx = np.array(
            np.frombuffer(indices, dtype=np.int32, count=nnz),
            dtype=np.int32,
            order="C",
            copy=True,
        )
        vals = np.array(
            np.frombuffer(values, dtype=np.float64, count=nnz),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        return sp.csc_matrix((vals, idx, indptr), shape=(num_eqn, num_eqn))

    def solve(self, **kwargs):
        num_eqn = int(kwargs["num_eqn"])
        nnz = int(kwargs["nnz"])
        matrix_status = kwargs["matrix_status"]
        rhs = np.frombuffer(kwargs["rhs"], dtype=np.float64, count=num_eqn)
        x = np.frombuffer(kwargs["x"], dtype=np.float64, count=num_eqn)

        if num_eqn == 0:
            return 0
        if num_eqn == 1:
            vals = np.frombuffer(kwargs["values"], dtype=np.float64, count=nnz)
            if nnz == 1 and vals[0] != 0.0:
                x[0] = rhs[0] / vals[0]
                return 0
            return -1

        if matrix_status == "STRUCTURE_CHANGED" or self.A is None:
            self.A = self._wrap_csc(
                num_eqn, nnz, kwargs["index_ptr"], kwargs["indices"], kwargs["values"]
            )
            self.umf.symbolic(self.A)
            self.umf.numeric(self.A)
        elif matrix_status == "COEFFICIENTS_CHANGED":
            vals = np.frombuffer(kwargs["values"], dtype=np.float64, count=nnz)
            self.A.data[:] = np.ascontiguousarray(vals)
            self.umf.numeric(self.A)
        elif getattr(self.umf, "_numeric", None) is None:
            self.A = self._wrap_csc(
                num_eqn, nnz, kwargs["index_ptr"], kwargs["indices"], kwargs["values"]
            )
            self.umf.symbolic(self.A)
            self.umf.numeric(self.A)

        # One write into OpenSees x buffer (UMFPACK returns a new array).
        x[:] = self.umf.solve(_umfpack.UMFPACK_A, self.A, np.ascontiguousarray(rhs))
        return 0


# Back-compat alias
SciPyUmfpackSolver64 = SciPyUmfpackSolver


class CuPyCGSolver:
    """CuPy conjugate gradient — same interface as PR #1676 benchmark_python_sparse."""

    def __init__(self, rtol: float = 1.0e-7, atol: float = 1.0e-12, maxiter=None):
        import cupy as cp
        import cupyx.scipy.sparse.linalg  # noqa: F401

        self._cp = cp
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.A = None

    def solve(self, **kwargs):
        cp = self._cp
        import cupyx.scipy.sparse.linalg as cpsla

        num_eqn = kwargs["num_eqn"]
        nnz = kwargs["nnz"]
        matrix_status = kwargs["matrix_status"]
        indptr = np.frombuffer(kwargs["index_ptr"], dtype=np.int32, count=num_eqn + 1)
        idx = np.frombuffer(kwargs["indices"], dtype=np.int32, count=nnz)
        vals = np.frombuffer(kwargs["values"], dtype=np.float64, count=nnz)

        if matrix_status == "STRUCTURE_CHANGED" or self.A is None:
            self.A = cp.sparse.csr_matrix(
                (cp.asarray(vals), cp.asarray(idx), cp.asarray(indptr)),
                shape=(num_eqn, num_eqn),
            )
        elif matrix_status == "COEFFICIENTS_CHANGED":
            self.A.data[:] = cp.asarray(vals)

        rhs_gpu = cp.asarray(np.frombuffer(kwargs["rhs"], dtype=np.float64, count=num_eqn))
        x_gpu, info = cpsla.cg(
            self.A, rhs_gpu, tol=self.rtol, atol=self.atol, maxiter=self.maxiter
        )
        np.frombuffer(kwargs["x"], dtype=np.float64, count=num_eqn)[:] = cp.asnumpy(x_gpu)
        return -int(info)


def python_lin_scheme(backend: str) -> str:
    """OpenSees sparse scheme for a Python linear backend (CSC for UMFPACK)."""
    backend = backend.lower().replace("_", "-")
    if backend in ("umfpack", "scikit-umfpack", "scikitumfpack"):
        return "CSC"
    return "CSR"


def lin_numberer(backend: str, np_: int = 1) -> str:
    """
    Numberer for linear backends.

    RCM only for BandSPD / CuPy CG (bandwidth / iterative locality).
    Direct sparse solvers use Plain (serial) or ParallelPlain (MPI).
    """
    backend = backend.lower().replace("_", "-")
    if backend in ("cupy-cg", "cupycg", "cg", "bandspd"):
        if np_ > 1:
            return "ParallelPlain"
        return "RCM"
    # Direct sparse: umfpack, nvmath, umfpack-native, mumps, …
    if np_ > 1:
        return "ParallelPlain"
    return "Plain"


def make_python_lin_solver(backend: str) -> Any:
    """
    Build a PythonSparse-compatible linear solver (openseespy-solvers >= 0.2.0).

    backend:
      - 'umfpack' / 'scikit-umfpack' : scipy.umfpack (CSC, index_dtype=auto→di)
      - 'nvmath' / 'cudss'           : nvmath.direct_solver (CSR / cuDSS)
      - 'cupy-cg' / 'cupycg'         : cupy.cg

    Stats/residual SpMV stay off (record_stats=False). Local lean fallbacks are
    used only if the package import fails.
    """
    backend = backend.lower().replace("_", "-")
    if backend in ("umfpack", "scikit-umfpack", "scikitumfpack"):
        try:
            from openseespy_solvers.scipy import umfpack

            return umfpack(
                scheme="CSC",
                writable="none",
                index_dtype="auto",
                record_stats=False,
            ).to_openseespy()["solver"]
        except Exception:
            return SciPyUmfpackSolver()

    if backend in ("nvmath", "cudss", "nvmath-cudss"):
        from openseespy_solvers.nvmath import direct_solver

        return direct_solver(
            scheme="CSR", writable="none", record_stats=False
        ).to_openseespy()["solver"]

    if backend in ("cupy-cg", "cupycg", "cg"):
        try:
            from openseespy_solvers.cupy import cg

            return cg(
                scheme="CSR",
                writable="none",
                rtol=1.0e-7,
                atol=1.0e-12,
                record_stats=False,
            ).to_openseespy()["solver"]
        except Exception:
            return CuPyCGSolver(rtol=1.0e-7, atol=1.0e-12)

    raise ValueError(
        f"unsupported python linear backend '{backend}' "
        "(use 'umfpack', 'nvmath', or 'cupy-cg'; SciPy spsolve is intentionally excluded)"
    )


class ScipyEigenSolver:
    """Generalized eigen solver matching benchmark_python_sparse_eigen.py (PR #1676)."""

    def __init__(self, maxiter=None, tol: float = 0.0):
        self.maxiter = maxiter
        self.tol = tol
        self._k = None
        self._m = None

    def solve(self, **kwargs):
        num_eqn = kwargs["num_eqn"]
        nnz = kwargs["nnz"]
        num_modes = int(kwargs["num_modes"])
        find_smallest = bool(kwargs.get("find_smallest", True))
        indptr = np.frombuffer(kwargs["index_ptr"], dtype=np.int32, count=num_eqn + 1)
        idx = np.frombuffer(kwargs["indices"], dtype=np.int32, count=nnz)
        kvals = np.frombuffer(kwargs["k_values"], dtype=np.float64, count=nnz)
        mvals = np.frombuffer(kwargs["m_values"], dtype=np.float64, count=nnz)
        status = kwargs.get("matrix_status", "STRUCTURE_CHANGED")

        if status == "STRUCTURE_CHANGED" or self._k is None:
            self._k = sp.csr_matrix(
                (kvals.copy(), idx.copy(), indptr.copy()), shape=(num_eqn, num_eqn)
            )
            self._m = sp.csr_matrix(
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
            eigsh_kwargs = {
                "k": k,
                "M": self._m,
                "which": "LM",
            }
            if find_smallest:
                eigsh_kwargs["sigma"] = 0.0
            if self.maxiter is not None:
                eigsh_kwargs["maxiter"] = self.maxiter
            if self.tol > 0.0:
                eigsh_kwargs["tol"] = self.tol
            evals, evecs = eigsh(self._k, **eigsh_kwargs)

        evals_buf = np.frombuffer(kwargs["eigenvalues"], dtype=np.float64, count=num_modes)
        evecs_buf = np.frombuffer(
            kwargs["eigenvectors"], dtype=np.float64, count=num_modes * num_eqn
        )
        evals_buf[:k] = evals[:k]
        # mode-major layout (same as PR #1676)
        evecs_buf[: k * num_eqn] = evecs[:, :k].T.flatten()
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


# Large-mesh gate used by older DPS-only scripts (factor=12 → ~117k).
MIN_BENCHMARK_DOFS = 100_000
DEFAULT_BENCHMARK_FACTORS = [12.0, 14.0, 16.0]

# Agreed PR #1676-style matrices (distributed + GPU backends).
AGREED_LINEAR_FACTORS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
AGREED_EIGEN_FACTORS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
AGREED_NPS = [1, 2, 4, 8]

# Native / heavy-direct skip thresholds (same spirit as benchmark_python_sparse.py).
# scikit-umfpack at factor=16 (~269k) has OOM'd this machine; skip like native UmfPack.
NATIVE_SKIP_LIMITS = {
    "umfpack-native": 14.0,
    "umfpack": 16.0,
    "bandspd": 20.0,
}


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


def system_cfg(ops, distributed: bool, pid: int, python_backend: str = "umfpack"):
    """Return (system_name, dict, numberer) for a PythonSparse-family solver."""
    np_ = 2 if distributed else 1
    solver = make_python_lin_solver(python_backend) if (not distributed or pid == 0) else None
    cfg = {
        "solver": solver,
        "scheme": python_lin_scheme(python_backend),
        "writable": "none",
    }
    numberer = lin_numberer(python_backend, np_)
    if distributed:
        return "DistributedPythonSparse", cfg, numberer
    return "PythonSparse", cfg, numberer


def eigen_cfg(distributed: bool, pid: int):
    if distributed:
        return {
            "solver": ScipyEigenSolver() if pid == 0 else None,
            "scheme": "CSR",
        }
    return {"solver": ScipyEigenSolver(), "scheme": "CSR"}
