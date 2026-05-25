"""
Example direct solver for the ``PythonSparse`` system command using nvmath.

This module provides a nvmath-python direct solver that can execute on CPU or GPU
depending on the sparse module provided. The numpy module is automatically inferred
from the sparse module. OpenSees hands us CPU buffers; we can either use them
directly (CPU execution) or copy them to GPU (GPU execution).

The solver uses the stateful ``DirectSolver`` API which separates planning,
factorization, and solve phases, allowing efficient reuse of factorizations when
the matrix structure remains constant.

Usage:
    # GPU execution (default if CuPy available)
    import cupyx.scipy.sparse as cpsparse
    solver = NvMathSparseDirectSolver(sp_module=cpsparse)
    
    # CPU execution
    import scipy.sparse
    solver = NvMathSparseDirectSolver(sp_module=scipy.sparse)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import scipy.sparse
import nvmath


def _wrap_csr_views(
    *,
    index_ptr,
    indices,
    values,
    rhs,
    x,
    num_eqn: int,
    nnz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert OpenSees memoryviews into NumPy arrays without copying."""

    indptr = np.frombuffer(index_ptr, dtype=np.int32, count=num_eqn + 1)
    idx = np.frombuffer(indices, dtype=np.int32, count=nnz)
    data = np.frombuffer(values, dtype=np.float64, count=nnz)
    rhs_vec = np.frombuffer(rhs, dtype=np.float64, count=num_eqn)
    x_vec = np.frombuffer(x, dtype=np.float64, count=num_eqn)
    return indptr, idx, data, rhs_vec, x_vec


def _find_multithreading_lib() -> Optional[str]:
    """
    Find the path to the multithreading library for nvmath.
    
    For conda installations, checks conda environment lib directory.
    For pip installations, checks nvidia.cu12 package location.
    Returns None if not found.
    """
    # Library name variants to try
    lib_names = [
        "libcudss_mtlayer_gomp.so.0",
        "libcudss_mtlayer_gomp.so",
    ]
    
    # Try conda environment first
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for lib_name in lib_names:
            lib_path = Path(conda_prefix) / "lib" / lib_name
            if lib_path.exists():
                return str(lib_path)
    
    # Try finding via nvidia.cu12 package (for pip installations)
    try:
        import importlib.resources
        try:
            # Python 3.9+
            files = importlib.resources.files("nvidia.cu12")
            for lib_name in lib_names:
                lib_path = files / "lib" / lib_name
                if lib_path.exists():
                    return str(lib_path)
        except AttributeError:
            # Python < 3.9
            import nvidia.cu12
            cu12_path = Path(nvidia.cu12.__file__).parent.parent
            for lib_name in lib_names:
                lib_path = cu12_path / "lib" / lib_name
                if lib_path.exists():
                    return str(lib_path)
    except (ImportError, AttributeError):
        pass
    
    # Try system library paths as fallback
    system_lib_paths = [
        Path("/usr/local/lib"),
        Path("/usr/lib"),
        Path("/lib"),
    ]
    for sys_path in system_lib_paths:
        for lib_name in lib_names:
            lib_path = sys_path / lib_name
            if lib_path.exists():
                return str(lib_path)
    
    return None


def _matrix_from_scheme(
    *,
    storage_scheme: str,
    indptr: Any,  # Can be np.ndarray or cp.ndarray depending on backend
    indices: Any,  # Can be np.ndarray or cp.ndarray depending on backend
    data: Any,  # Can be np.ndarray or cp.ndarray depending on backend
    num_eqn: int,
    sp_module: Any, # either scipy.sparse or cupyx.scipy
) -> Any:
    """Create a compressed sparse matrix from the provided buffers using the specified sparse module.
    
    The arrays must match the sparse module type:
    - CuPy sparse matrices require CuPy arrays
    - SciPy sparse matrices require NumPy arrays
    """

    if storage_scheme == "CSR":
        return sp_module.csr_matrix((data, indices, indptr), shape=(num_eqn, num_eqn), copy=False)
    if storage_scheme == "CSC":
        matrix = sp_module.csc_matrix((data, indices, indptr), shape=(num_eqn, num_eqn), copy=False)
        # nvmath expects CSR format for the LHS
        return matrix.tocsr() if hasattr(matrix, 'tocsr') else matrix
    raise ValueError(f"Unsupported storage scheme {storage_scheme}; expected 'CSR' or 'CSC'.")


@dataclass
class NvMathSparseDirectSolver:
    """
    Direct solver using nvmath's advanced DirectSolver stateful API with planning.

    This solver uses the stateful DirectSolver object which allows caching and reusing
    the factorization across multiple solves. It separates planning, factorization, and
    solve phases, which is more efficient when the matrix structure remains constant.

    The solver can execute on CPU or GPU depending on the sparse module provided.
    The numpy module is automatically inferred from the sparse module.
    - GPU execution: sp_module=cpsparse (default if CuPy available) -> np_module=cp
    - CPU execution: sp_module=scipy.sparse -> np_module=np

    Execution policy:
    - execution=None: Default execution mode (GPU for CuPy arrays, CPU for NumPy arrays)
    - execution=ExecutionHybrid(): Hybrid CPU/GPU execution (can benefit from multithreading on CPU side)
    - execution=ExecutionCUDA(): Force GPU execution (if using CuPy arrays)

    The solver automatically configures multithreading support if the library is available.
    """

    sp_module: Any = field(default_factory=lambda: cpsparse if HAS_CUPY else scipy.sparse)
    plan_algorithm: Any = field(default=None)  # Optional: DirectSolverAlgType
    execution: Optional[Any] = field(default=None)  # Optional: ExecutionHybrid(), ExecutionCUDA(), or None
    np_module: Any = field(default=None, init=False)  # Inferred from sp_module

    _matrix: Optional[Any] = None
    _solver: Optional[Any] = None  # DirectSolver stateful object
    _options: Optional[dict] = None  # DirectSolverOptions
    _is_gpu: bool = field(default=False, init=False)
    _is_planned: bool = field(default=False, init=False)
    _is_factorized: bool = field(default=False, init=False)

    def __post_init__(self):
        """Infer np_module from sp_module and detect if GPU execution is being used."""
        # Infer np_module from sp_module by checking if it's CuPy sparse
        sp_module_name = getattr(self.sp_module, '__name__', '')
        is_cupy_sparse = 'cupy' in sp_module_name.lower() or 'cupyx' in sp_module_name.lower()
        
        if is_cupy_sparse and HAS_CUPY:
            self.np_module = cp
            self._is_gpu = True
        else:
            # SciPy sparse or CuPy not available -> use NumPy
            self.np_module = np
            if is_cupy_sparse and not HAS_CUPY:
                # Fall back to SciPy if CuPy requested but not available
                self.sp_module = scipy.sparse
            self._is_gpu = False

    def _initialize_options(self) -> None:
        """Initialize DirectSolverOptions with multithreading support."""
        if self._options is not None:
            return  # Already initialized

        try:
            # Try to find and configure multithreading library
            multithreading_lib = _find_multithreading_lib()
            if multithreading_lib:
                self._options = nvmath.sparse.advanced.DirectSolverOptions(
                    multithreading_lib=multithreading_lib,
                )
            else:
                # No options (will work, just slower)
                self._options = None
        except Exception as exc:
            # If initialization fails, continue without options
            print(f"NvMathSparseDirectSolver: Warning - failed to initialize options: {exc}")
            self._options = None


    def cleanup(self) -> None:
        """Clean up the DirectSolver object. Called automatically when solver is destroyed."""
        if self._solver is not None:
            try:
                self._solver.__exit__(None, None, None)
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self._solver = None
                self._is_planned = False
                self._is_factorized = False

    def __del__(self):
        """Clean up solver on destruction."""
        self.cleanup()

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

        # Check if we need to rebuild the matrix or replan
        structure_changed = (
            self._matrix is None 
            or matrix_status == "STRUCTURE_CHANGED"
        )
        coefficients_changed = (
            matrix_status == "COEFFICIENTS_CHANGED"
        )

        # Build or update the matrix
        try:
            if structure_changed:
                # Structure changed - need to build new matrix
                if self._is_gpu:
                    indptr_array = self.np_module.asarray(indptr)
                    indices_array = self.np_module.asarray(idx)
                    data_array = self.np_module.asarray(data)
                else:
                    indptr_array = indptr
                    indices_array = idx
                    data_array = data

                # Build the sparse matrix
                matrix = _matrix_from_scheme(
                    storage_scheme=storage_scheme,
                    indptr=indptr_array,
                    indices=indices_array,
                    data=data_array,
                    num_eqn=num_eqn,
                    sp_module=self.sp_module,
                )

                # Ensure CSR format (nvmath expects CSR)
                if not isinstance(matrix, self.sp_module.csr_matrix):
                    matrix = matrix.tocsr()

                self._matrix = matrix
            elif coefficients_changed:
                # Coefficients changed - update matrix values in-place
                assert self._matrix is not None
                if self._is_gpu:
                    data_array = self.np_module.asarray(data)
                    self._matrix.data.set(data_array)
                else:
                    self._matrix.data[:] = data
            # For UNCHANGED, keep existing matrix
        except Exception as exc:
            print(f"NvMathSparseDirectSolver: matrix build failed: {exc}")
            return -1

        if self._matrix is None:
            return -1

        # Prepare RHS array (always new from OpenSees)
        if self._is_gpu:
            rhs_array = self.np_module.asarray(rhs_vec)
        else:
            rhs_array = rhs_vec

        try:
            # Initialize options on first call (includes multithreading library configuration)
            if self._options is None:
                self._initialize_options()

            # Create solver on first call
            if self._solver is None:
                # Create new DirectSolver with options and execution policy
                solver_kwargs = {}
                if self._options is not None:
                    solver_kwargs["options"] = self._options
                if self.execution is not None:
                    solver_kwargs["execution"] = self.execution
                
                if solver_kwargs:
                    self._solver = nvmath.sparse.advanced.DirectSolver(
                        self._matrix, rhs_array, **solver_kwargs
                    )
                else:
                    self._solver = nvmath.sparse.advanced.DirectSolver(self._matrix, rhs_array)
                
                # Enter context manager
                self._solver.__enter__()
                self._is_planned = False
                self._is_factorized = False
            else:
                if structure_changed:
                    # Matrix object replaced out-of-place; reset A and b
                    self._solver.reset_operands(a=self._matrix, b=rhs_array)
                    self._is_planned = False
                    self._is_factorized = False
                else:
                    # Structure reused; only RHS changes
                    self._solver.reset_operands(b=rhs_array)
                    if coefficients_changed:
                        # Numerical values updated in-place; need new factorization
                        self._is_factorized = False

            # Configure plan algorithm if specified and we need to plan
            if self.plan_algorithm is not None and not self._is_planned:
                p = self._solver.plan_config
                p.algorithm = self.plan_algorithm

            # Plan if needed (required after creating solver or when structure changed)
            if not self._is_planned:
                plan_info = self._solver.plan()
                self._is_planned = True

            # Factorize if needed (required after creating solver, planning, or coefficient changes)
            if not self._is_factorized:
                fac_info = self._solver.factorize()
                self._is_factorized = True

            # Solve the system
            sol = self._solver.solve()

            # Synchronize CUDA stream only for GPU execution
            if self._is_gpu:
                self.np_module.cuda.get_current_stream().synchronize()
                x_vec[:] = self.np_module.asnumpy(sol)
            else:
                # No synchronization needed for CPU execution - it always blocks
                x_vec[:] = sol

        except Exception as exc:
            print(f"NvMathSparseDirectSolver: solve failed: {exc}")
            return -1

        return 0


if __name__ == "__main__":  # simple self-test using a 2x2 system
    print("NvMath direct solver demo (GPU)")
    if HAS_CUPY:
        direct_solver_gpu = NvMathSparseDirectSolver(sp_module=cpsparse)
    else:
        direct_solver_gpu = NvMathSparseDirectSolver()  # Will default to CPU

    # Build a tiny positive-definite system and verify the solve path.
    indptr = np.array([0, 2, 4], dtype=np.int32)
    indices = np.array([0, 1, 0, 1], dtype=np.int32)
    data = np.array([4.0, 1.0, 2.0, 3.0], dtype=np.float64)
    rhs = np.array([1.0, 2.0], dtype=np.float64)
    x = np.zeros_like(rhs)

    if HAS_CUPY:
        status = direct_solver_gpu.solve(
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

    print("\nNvMath direct solver demo (CPU)")
    direct_solver_cpu = NvMathSparseDirectSolver(sp_module=scipy.sparse)
    x_cpu = np.zeros_like(rhs)
    status_cpu = direct_solver_cpu.solve(
        index_ptr=memoryview(indptr),
        indices=memoryview(indices),
        values=memoryview(data),
        rhs=memoryview(rhs),
        x=memoryview(x_cpu),
        num_eqn=2,
        nnz=4,
        matrix_status="STRUCTURE_CHANGED",
        storage_scheme="CSR",
    )
    print(f"status={status_cpu}, solution={x_cpu}")

    # Example usage from OpenSeesPy:
    # --------------------------------
    # import openseespy.opensees as ops
    # from solvers.nvmath_sparse_solver import NvMathSparseDirectSolver
    # import cupyx.scipy.sparse as cpsparse
    # import scipy.sparse
    #
    # # GPU-accelerated direct solver using nvmath
    # direct_solver_gpu = NvMathSparseDirectSolver(sp_module=cpsparse)
    # ops.system("PythonSparse", {"solver": direct_solver_gpu, "scheme": "CSR"})
    #
    # # CPU-based direct solver using nvmath
    # direct_solver_cpu = NvMathSparseDirectSolver(sp_module=scipy.sparse)
    # ops.system("PythonSparse", {"solver": direct_solver_cpu, "scheme": "CSR"})
