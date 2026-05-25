"""
Example GPU-backed solvers for the ``PythonSparse`` system command using CuPy.

This module mirrors :mod:`example_scipy_sparse_solver`, but executes on CUDA
via CuPy. OpenSees hands us CPU buffers; we copy them to the device, solve
there, and copy the solution back to host memory.

Two solver types are provided:

* ``CuPySparseDirectSolver`` caches a sparse factorization via
  :func:`cupyx.scipy.sparse.linalg.factorized`.
* ``CuPySparseCGSolver`` runs the conjugate-gradient iteration with optional
  Jacobi (diagonal) preconditioning (assumes symmetric matrices, works directly with CSR format).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

import cupy as cp
import cupyx.scipy.sparse as cpsparse
import cupyx.scipy.sparse.linalg as cpsparse_linalg


def _wrap_csr_views(
    *,
    index_ptr,
    indices,
    values,
    rhs,
    x,
    num_eqn: int,
    nnz: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert OpenSees memoryviews into NumPy arrays without copying."""

    indptr = np.frombuffer(index_ptr, dtype=np.int32, count=num_eqn + 1)
    idx = np.frombuffer(indices, dtype=np.int32, count=nnz)
    data = np.frombuffer(values, dtype=np.float64, count=nnz)
    rhs_vec = np.frombuffer(rhs, dtype=np.float64, count=num_eqn)
    x_vec = np.frombuffer(x, dtype=np.float64, count=num_eqn)
    return indptr, idx, data, rhs_vec, x_vec


def _matrix_from_scheme(
    *,
    storage_scheme: str,
    indptr_gpu: cp.ndarray,
    indices_gpu: cp.ndarray,
    data_gpu: cp.ndarray,
    num_eqn: int,
) -> cpsparse.spmatrix:
    """Create a compressed sparse matrix from the provided GPU buffers."""

    if storage_scheme == "CSR":
        return cpsparse.csr_matrix((data_gpu, indices_gpu, indptr_gpu), shape=(num_eqn, num_eqn))
    if storage_scheme == "CSC":
        return cpsparse.csc_matrix((data_gpu, indices_gpu, indptr_gpu), shape=(num_eqn, num_eqn))
    raise ValueError(f"Unsupported storage scheme {storage_scheme}; expected 'CSR' or 'CSC'.")


def _build_diagonal_preconditioner(matrix: cpsparse.csr_matrix) -> Optional[cpsparse_linalg.LinearOperator]:
    """Build a Jacobi (diagonal) preconditioner."""
    diag = matrix.diagonal()
    if diag.size == 0:
        return None

    inv_diag = cp.ones_like(diag)
    mask = diag != 0.0
    inv_diag[mask] = 1.0 / diag[mask]

    def apply(vec):
        return inv_diag * vec

    return cpsparse_linalg.LinearOperator(matrix.shape, matvec=apply)


@dataclass
class CuPySparseDirectSolver:
    """
    GPU-based solver that uses CuPy's SciPy-compatible sparse routines.

    The solver keeps track of the sparse structure (storage scheme, index_ptr, indices)
    and rebuilds the factorization when the structure changes or OpenSees indicates the
    matrix coefficients were modified.
    """

    _factor: Optional[Callable[[cp.ndarray], cp.ndarray]] = None

    def solve(
        self,
        *,
        index_ptr,
        indices,
        values,
        rhs,
        x,
        num_eqn,
        nnz,
        matrix_status,
        storage_scheme,
    ):
        """
        Solve ``Ax = b`` in place using the supplied compressed-sparse buffers.

        Parameters mirror the keyword arguments OpenSees passes to ``solve``.
        Returning 0 signals success; returning a non-zero integer signals failure.
        """
        indptr, idx, data, rhs_vec, x_vec = _wrap_csr_views(
            index_ptr=index_ptr,
            indices=indices,
            values=values,
            rhs=rhs,
            x=x,
            num_eqn=num_eqn,
            nnz=nnz,
        )

        needs_refactor = self._factor is None or matrix_status != "UNCHANGED"

        if needs_refactor:
            indptr_gpu = cp.asarray(indptr)
            indices_gpu = cp.asarray(idx)
            data_gpu = cp.asarray(data)

            try:
                matrix_gpu = _matrix_from_scheme(
                    storage_scheme=storage_scheme,
                    indptr_gpu=indptr_gpu,
                    indices_gpu=indices_gpu,
                    data_gpu=data_gpu,
                    num_eqn=num_eqn,
                )
                self._factor = cpsparse_linalg.factorized(matrix_gpu)
            except Exception:
                return -1

        rhs_gpu = cp.asarray(rhs_vec)
        sol_gpu = self._factor(rhs_gpu)

        # Copy the solution back to the host buffer expected by OpenSees.
        x_vec[:] = cp.asnumpy(sol_gpu)
        return 0


@dataclass
class CuPySparseCGSolver:
    """
    Conjugate-gradient solver with optional preconditioning on the GPU.

    The solver keeps track of the sparse structure (storage scheme, index_ptr, indices)
    and rebuilds the system matrix / preconditioner when the structure changes or
    OpenSees indicates the matrix coefficients were modified.

    Assumes the matrix is symmetric (CG is designed for symmetric positive definite matrices).
    Works directly with CSR format to avoid format conversions.

    Preconditioning options:
    - ``preconditioner_type=None``: No preconditioning (plain CG)
    - ``preconditioner_type="jacobi"``: Jacobi/diagonal preconditioner (default)
    """

    rtol: Optional[float] = None
    atol: Optional[float] = None
    maxiter: Optional[int] = None
    preconditioner_type: Optional[str] = "jacobi"  # None disables preconditioning

    _matrix: Optional[cpsparse.csr_matrix] = None  # CSR format for CG solver
    _preconditioner: Optional[cpsparse_linalg.LinearOperator] = None
    _rhs_gpu: Optional[cp.ndarray] = None
    _x0_gpu: Optional[cp.ndarray] = None

    def solve(
        self,
        *,
        index_ptr,
        indices,
        values,
        rhs,
        x,
        num_eqn,
        nnz,
        matrix_status,
        storage_scheme,
    ):
        """
        Solve ``Ax = b`` in place using the supplied compressed-sparse buffers.

        Parameters mirror the keyword arguments OpenSees passes to ``solve``.
        Returning 0 signals success; returning a non-zero integer signals failure.
        """
        if storage_scheme not in ("CSR", "CSC"):
            raise ValueError(f"Unsupported storage scheme {storage_scheme}; expected 'CSR' or 'CSC'.")

        indptr, idx, data, rhs_vec, x_vec = _wrap_csr_views(
            index_ptr=index_ptr,
            indices=indices,
            values=values,
            rhs=rhs,
            x=x,
            num_eqn=num_eqn,
            nnz=nnz,
        )

        rebuild_matrix = self._matrix is None or matrix_status == "STRUCTURE_CHANGED"

        if rebuild_matrix:
            indptr_gpu = cp.asarray(indptr)
            indices_gpu = cp.asarray(idx)
            data_gpu = cp.asarray(data)

            self._matrix = _matrix_from_scheme(
                storage_scheme="CSR", # CSR, symmetric matrix, so CSR(A) = CSC(A.T)
                indptr_gpu=indptr_gpu,
                indices_gpu=indices_gpu,
                data_gpu=data_gpu,
                num_eqn=num_eqn,
            )
            self._preconditioner = None
        elif matrix_status == "COEFFICIENTS_CHANGED":
            assert self._matrix is not None
            self._matrix.data.set(data)
            self._preconditioner = None
        else:
            assert self._matrix is not None

        if self._matrix is None:
            return -1

        requested_prec = (
            self.preconditioner_type.lower() if self.preconditioner_type is not None else None
        )
        if requested_prec is None:
            self._preconditioner = None
        else:
            if requested_prec not in ("jacobi", "diagonal"):
                raise ValueError(
                    f"Unknown preconditioner type '{self.preconditioner_type}'. "
                    "Only 'jacobi' (or 'diagonal') is supported."
                )
            if self._preconditioner is None:
                self._preconditioner = _build_diagonal_preconditioner(self._matrix)

        if self._rhs_gpu is None or self._rhs_gpu.size != num_eqn:
            self._rhs_gpu = cp.empty(num_eqn, dtype=cp.float64)
            self._x0_gpu = cp.empty(num_eqn, dtype=cp.float64)
        assert self._rhs_gpu is not None and self._x0_gpu is not None
        self._rhs_gpu.set(rhs_vec)
        self._x0_gpu.fill(0.0)
        # self._x0_gpu.set(x_vec)  # use the host-provided initial guess instead of zeros

        cg_kwargs = {
            "x0": self._x0_gpu,
            "maxiter": self.maxiter,
            "M": self._preconditioner,
        }

        cg_kwargs["tol"] = self.rtol if self.rtol is not None else 1.0e-8
        if self.atol is not None:
            cg_kwargs["atol"] = self.atol

        solution_gpu, info = cpsparse_linalg.cg(self._matrix, self._rhs_gpu, **cg_kwargs)

        if info < 0:
            return int(info)

        x_vec[:] = cp.asnumpy(solution_gpu)
        return int(info)




if __name__ == "__main__":  # simple self-test using a 2x2 system
    print("Direct solver demo")
    direct_solver = CuPySparseDirectSolver()
    cg_solver = CuPySparseCGSolver()

    # Build a tiny positive-definite system and verify both solve paths.
    indptr = np.array([0, 2, 4], dtype=np.int32)
    indices = np.array([0, 1, 0, 1], dtype=np.int32)
    data = np.array([4.0, 1.0, 2.0, 3.0], dtype=np.float64)
    rhs = np.array([1.0, 2.0], dtype=np.float64)
    x = np.zeros_like(rhs)

    status = direct_solver.solve(
        index_ptr=memoryview(indptr),
        indices=memoryview(indices),
        values=memoryview(data),
        rhs=memoryview(rhs),
        x=memoryview(x),
        num_eqn=2,
        nnz=4,
        matrix_status="STRUCTURE_CHANGED",
        storage_scheme="CSR",
    )
    print(f"status={status}, solution={x}")

    print("Conjugate-gradient solver demo")
    x_cg = np.zeros_like(rhs)
    status_cg = cg_solver.solve(
        index_ptr=memoryview(indptr),
        indices=memoryview(indices),
        values=memoryview(data),
        rhs=memoryview(rhs),
        x=memoryview(x_cg),
        num_eqn=2,
        nnz=4,
        matrix_status="STRUCTURE_CHANGED",
        storage_scheme="CSR",
    )
    print(f"cg status={status_cg}, solution={x_cg}")

    # Example usage from OpenSeesPy:
    # --------------------------------
    # import openseespy.opensees as ops
    # from solvers.cupy_sparse_solver import (
    #     CuPySparseDirectSolver,
    #     CuPySparseCGSolver,
    # )
    #
    # # Direct solver
    # direct_solver = CuPySparseDirectSolver()
    # ops.system("PythonSparse", {"solver": direct_solver, "scheme": "CSC"})
    #
    # # CG solver (assumes symmetric matrix, works with CSR)
    # # Without preconditioning (plain CG)
    # cg_solver_none = CuPySparseCGSolver(preconditioner_type=None)
    # ops.system("PythonSparse", {"solver": cg_solver_none, "scheme": "CSR"})
    #
    # # CG solver with Jacobi preconditioner (default)
    # cg_solver_jacobi = CuPySparseCGSolver(preconditioner_type="jacobi")
    # ops.system("PythonSparse", {"solver": cg_solver_jacobi, "scheme": "CSR"})
    #