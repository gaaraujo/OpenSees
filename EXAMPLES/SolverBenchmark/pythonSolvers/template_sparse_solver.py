"""
Template PythonSparse solver demonstrating how to bridge OpenSees to a custom
Python linear solver.

Copy this file, rename the class, and fill in the sections marked "TODO" to plug
in your favorite sparse linear algebra backend.  The template shows how to:

    * Receive the sparse system data from OpenSees via memoryviews.
    * Read the CSR data (row pointer, column indices, values) that OpenSees exposes.
    * Cache matrix structure to avoid rebuilding factorizations unnecessarily.
    * Write the solution back into OpenSees-provided buffers.

All data is exchanged zero-copy using Python's buffer protocol.  The only time
you need to copy is when your solver requires ownership of the matrix arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Helper utilities (feel free to adapt or replace in your own solver)
# ---------------------------------------------------------------------------


def _require_csr(storage_scheme: str) -> None:
    """
    This template assumes the solver is registered with `scheme='CSR'`.
    Adjust this check (or add conversions) if you want to accept CSC/COO.
    """
    if storage_scheme != "CSR":
        raise ValueError(
            "Template solver expects CSR storage. "
            "Register your solver with scheme='CSR' or add conversion logic."
        )


@dataclass
class CSRViews:
    """Zero-copy view of the CSR buffers OpenSees provides."""

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray

    @classmethod
    def from_buffers(
        cls, *, index_ptr, indices, values, num_eqn: int, nnz: int
    ) -> "CSRViews":
        return cls(
            indptr=np.frombuffer(index_ptr, dtype=np.int32, count=num_eqn + 1),
            indices=np.frombuffer(indices, dtype=np.int32, count=nnz),
            data=np.frombuffer(values, dtype=np.float64, count=nnz),
        )


# ---------------------------------------------------------------------------
# Template solver implementation
# ---------------------------------------------------------------------------

@dataclass
class TemplateSparseDirectSolver:
    """
    Minimal example of a direct solver.

    Replace the sections marked TODO with your backend's factorization calls.
    """

    # Example configuration knobs (add/remove as needed):
    reordering: str = "rcm"  # TODO: hook into backend-specific options

    # Internal caches for reuse between analyses:
    _factor: Optional[object] = None  # replace "object" with your factor type

    def solve(
        self,
        *,
        index_ptr,
        indices,
        values,
        rhs,
        x,
        storage_scheme: str,
        matrix_status: str,
        num_eqn: int,
        nnz: int,
        row=None, # only present for storage_scheme == "COO"
        col=None, # only present for storage_scheme == "COO"
    ) -> int:
        """
        Entry point called by OpenSees for each solve.

        Parameters mirror the C++ `Python*LinSolver::callPythonSolver` calls.
        """

        structure_changed = matrix_status == "STRUCTURE_CHANGED" or self._factor is None

        _require_csr(storage_scheme)
        csr = CSRViews.from_buffers(
            index_ptr=index_ptr,
            indices=indices,
            values=values,
            num_eqn=num_eqn,
            nnz=nnz,
        )

        if structure_changed:
            # TODO: replace this with your backend's "build matrix" + factorization.
            # For example:
            #     self._matrix = backend_matrix_from_csr(csr.indptr, csr.indices, csr.data, num_eqn)
            #     self._factor = self._matrix.factorize(reordering=self.reordering)
            #
            # Keep references on self so the same factorization can be reused
            # when matrix_status == "UNCHANGED" (structure + coefficients identical).
            self._factor = ("placeholder-factor",)  # DELETE ME
        elif matrix_status != "UNCHANGED":
            # Pattern unchanged, but coefficients updated -> refresh numeric values.
            # TODO: update your backend matrix in-place here (no need to redo symbolic
            # factorization). For example:
            #     self._matrix.update_csr_values(csr.indptr, csr.indices, csr.data)
            #     self._factor.refactor(self._matrix)
            pass  # DELETE ME

        # Step 3: Wrap RHS and solution vectors as NumPy arrays (zero-copy).
        rhs_vec = np.frombuffer(rhs, dtype=np.float64, count=num_eqn)
        sol_vec = np.frombuffer(x, dtype=np.float64, count=num_eqn)

        # Step 4: Solve.
        try:
            if self._factor is None:
                raise RuntimeError("factorization was not built")

            # TODO: call your backend's solve routine here.  Example:
            #     sol_vec[:] = self._factor.solve(rhs_vec)
            sol_vec[:] = rhs_vec  # placeholder: identity solve, DELETE ME
        except Exception as exc:
            print(f"TemplateSparseDirectSolver: solve failed: {exc}")
            return -1

        return 0


@dataclass
class TemplateSparseCGSolver:
    """
    Minimal example of an iterative solver (Conjugate Gradient).

    The pattern mirrors the direct solver, but instead of factorizations we
    configure an iterative method with optional preconditioning.
    """

    rtol: float = 1.0e-6
    atol: float = 0.0
    maxiter: int = 10_000
    use_preconditioner: bool = False

    _matrix_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def solve(
        self,
        *,
        index_ptr,
        indices,
        values,
        rhs,
        x,
        storage_scheme: str,
        matrix_status: str,
        num_eqn: int,
        nnz: int,
        row=None, # only present for storage_scheme == "COO"
        col=None, # only present for storage_scheme == "COO"
    ) -> int:
        _require_csr(storage_scheme)

        structure_changed = matrix_status == "STRUCTURE_CHANGED" or self._matrix_cache is None
        if structure_changed or matrix_status != "UNCHANGED" or self._matrix_cache is None:
            csr = CSRViews.from_buffers(
                index_ptr=index_ptr,
                indices=indices,
                values=values,
                num_eqn=num_eqn,
                nnz=nnz,
            )
            self._matrix_cache = (
                csr.indptr.copy(),
                csr.indices.copy(),
                csr.data.copy(),
            )
        assert self._matrix_cache is not None
        indptr, idx, data = self._matrix_cache
        rhs_vec = np.frombuffer(rhs, dtype=np.float64, count=num_eqn)
        sol_vec = np.frombuffer(x, dtype=np.float64, count=num_eqn)

        # TODO: hook this up to your iterative solver of choice.
        # Example with SciPy (pseudo-code):
        #
        #     A = scipy.sparse.csr_matrix((data, idx, indptr), shape=(num_eqn, num_eqn))
        #     M = build_preconditioner(A) if self.use_preconditioner else None
        #     sol_vec[:], info = scipy.sparse.linalg.cg(
        #         A,
        #         rhs_vec,
        #         x0=sol_vec,
        #         rtol=self.rtol,
        #         atol=self.atol,
        #         maxiter=self.maxiter,
        #         M=M,
        #     )
        #
        # Below we simply copy the RHS as a placeholder.
        sol_vec[:] = rhs_vec

        # TODO: return -1 if your iterative solver reports non-convergence.
        return 0


if __name__ == "__main__":  # simple smoke test
    # Demonstrate how OpenSees invokes the solver by emulating a 2x2 system.
    indptr = (0, 2, 4)
    indices = (0, 1, 0, 1)
    values = (4.0, 1.0, 1.0, 3.0)
    rhs = (1.0, 2.0)
    sol = (0.0, 0.0)

    mv_indptr = memoryview(np.asarray(indptr, dtype=np.int32))
    mv_indices = memoryview(np.asarray(indices, dtype=np.int32))
    mv_values = memoryview(np.asarray(values, dtype=np.float64))
    mv_rhs = memoryview(np.asarray(rhs, dtype=np.float64))
    mv_sol = memoryview(np.asarray(sol, dtype=np.float64))

    direct = TemplateSparseDirectSolver()
    cg_solver = TemplateSparseCGSolver()

    kwargs = dict(
        index_ptr=mv_indptr,
        indices=mv_indices,
        values=mv_values,
        rhs=mv_rhs,
        x=mv_sol,
        storage_scheme="CSR",
        matrix_status="UNCHANGED",
        num_eqn=2,
        nnz=4,
    )

    print("Direct solver result:", direct.solve(**kwargs))
    print("CG solver result:", cg_solver.solve(**kwargs))

