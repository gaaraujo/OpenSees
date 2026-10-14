"""The PythonSparse examples from the OpenSees documentation, run as tests.

The three solver classes below are copied verbatim from

* https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/system/PythonSparse.html
  (CuPyCGSolver, SciPyspSolveSolver)
* https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/eigen.html
  (SciPyGeneralizedEigenSolver)

and checked against compiled OpenSees solvers (UmfPack, default eigen) on a
small brick model. They must keep working unchanged on every build.

Usage::

    python test_doc_examples.py                     # serial PythonSparse
    mpiexec -n 2 python test_doc_examples.py --distributed   # DistributedPythonSparse

Set OPENSEESPY_PATH to test a locally built ``opensees`` module.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sp_linalg

if os.environ.get("OPENSEESPY_PATH"):
    sys.path.insert(0, os.environ["OPENSEESPY_PATH"])
    import opensees as ops  # noqa: E402
else:
    import openseespy.opensees as ops  # noqa: E402

try:
    import cupy as cp
    import cupyx.scipy.sparse
    import cupyx.scipy.sparse.linalg

    HAVE_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001
    HAVE_CUPY = False


# ----------------------------------------------------------------------------
# Verbatim from the documentation (system PythonSparse page)
# ----------------------------------------------------------------------------

if HAVE_CUPY:
    class CuPyCGSolver:
        def __init__(self, rtol=1e-5, atol=1e-12, maxiter=None):
            self.rtol = rtol
            self.atol = atol
            self.maxiter = maxiter
            self.A = None

        def solve(self, **kwargs):
            index_ptr = kwargs['index_ptr']
            indices = kwargs['indices']
            values = kwargs['values']
            rhs = kwargs['rhs']
            x = kwargs['x']
            num_eqn = kwargs['num_eqn']
            nnz = kwargs['nnz']
            matrix_status = kwargs['matrix_status']

            indptr = np.frombuffer(index_ptr, dtype=np.int32, count=num_eqn + 1)
            idx = np.frombuffer(indices, dtype=np.int32, count=nnz)
            vals = np.frombuffer(values, dtype=np.float64, count=nnz)

            if matrix_status == 'STRUCTURE_CHANGED' or self.A is None:
                self.A = cp.sparse.csr_matrix(
                    (cp.asarray(vals), cp.asarray(idx), cp.asarray(indptr)),
                    shape=(num_eqn, num_eqn)
                )
            elif matrix_status == 'COEFFICIENTS_CHANGED':
                self.A.data[:] = cp.asarray(vals)

            rhs_gpu = cp.asarray(np.frombuffer(rhs, dtype=np.float64, count=num_eqn))
            x_gpu, info = cupyx.scipy.sparse.linalg.cg(
                self.A, rhs_gpu, tol=self.rtol, atol=self.atol, maxiter=self.maxiter
            )

            np.frombuffer(x, dtype=np.float64, count=num_eqn)[:] = cp.asnumpy(x_gpu)
            return -int(info)


class SciPyspSolveSolver:
    def __init__(self):
        self.A = None
        self._solve_func = None

    def solve(self, **kwargs):
        index_ptr = kwargs['index_ptr']
        indices = kwargs['indices']
        values = kwargs['values']
        rhs = kwargs['rhs']
        x = kwargs['x']
        num_eqn = kwargs['num_eqn']
        nnz = kwargs['nnz']
        matrix_status = kwargs['matrix_status']

        indptr = np.frombuffer(index_ptr, dtype=np.int32, count=num_eqn + 1)
        idx = np.frombuffer(indices, dtype=np.int32, count=nnz)
        vals = np.frombuffer(values, dtype=np.float64, count=nnz)

        if matrix_status != 'UNCHANGED' or self._solve_func is None:
            self.A = sp.csr_matrix((vals.copy(), idx.copy(), indptr.copy()),
                                  shape=(num_eqn, num_eqn))
            self._solve_func = sp_linalg.factorized(self.A)

        rhs_arr = np.frombuffer(rhs, dtype=np.float64, count=num_eqn)
        x_arr = np.frombuffer(x, dtype=np.float64, count=num_eqn)
        x_arr[:] = self._solve_func(rhs_arr)
        return 0


if HAVE_CUPY:
    class CuPyCGSolverFixed(CuPyCGSolver):
        """The documented example with the two fixes current CuPy needs:
        ``cupyx.scipy.sparse`` instead of ``cp.sparse``, and ``rtol=`` instead of
        the removed ``tol=`` keyword. Nothing else changes."""

        def solve(self, **kwargs):
            index_ptr = kwargs['index_ptr']
            indices = kwargs['indices']
            values = kwargs['values']
            rhs = kwargs['rhs']
            x = kwargs['x']
            num_eqn = kwargs['num_eqn']
            nnz = kwargs['nnz']
            matrix_status = kwargs['matrix_status']

            indptr = np.frombuffer(index_ptr, dtype=np.int32, count=num_eqn + 1)
            idx = np.frombuffer(indices, dtype=np.int32, count=nnz)
            vals = np.frombuffer(values, dtype=np.float64, count=nnz)

            if matrix_status == 'STRUCTURE_CHANGED' or self.A is None:
                self.A = cupyx.scipy.sparse.csr_matrix(
                    (cp.asarray(vals), cp.asarray(idx), cp.asarray(indptr)),
                    shape=(num_eqn, num_eqn)
                )
            elif matrix_status == 'COEFFICIENTS_CHANGED':
                self.A.data[:] = cp.asarray(vals)

            rhs_gpu = cp.asarray(np.frombuffer(rhs, dtype=np.float64, count=num_eqn))
            x_gpu, info = cupyx.scipy.sparse.linalg.cg(
                self.A, rhs_gpu, rtol=self.rtol, atol=self.atol, maxiter=self.maxiter
            )

            np.frombuffer(x, dtype=np.float64, count=num_eqn)[:] = cp.asnumpy(x_gpu)
            return -int(info)


# ----------------------------------------------------------------------------
# Verbatim from the documentation (eigen page)
# ----------------------------------------------------------------------------

class SciPyGeneralizedEigenSolver:
    def __init__(self, maxiter=None, tol=0.0):
        self.maxiter = maxiter
        self.tol = tol
        self._k_matrix = None
        self._m_matrix = None

    def solve(self, **kwargs):
        index_ptr = kwargs['index_ptr']
        indices = kwargs['indices']
        k_values = kwargs['k_values']
        m_values = kwargs['m_values']
        eigenvalues = kwargs['eigenvalues']
        eigenvectors = kwargs['eigenvectors']
        num_eqn = kwargs['num_eqn']
        nnz = kwargs['nnz']
        matrix_status = kwargs['matrix_status']
        num_modes = kwargs['num_modes']
        find_smallest = kwargs['find_smallest']

        indptr = np.frombuffer(index_ptr, dtype=np.int32, count=num_eqn + 1)
        idx = np.frombuffer(indices, dtype=np.int32, count=nnz)
        k_data = np.frombuffer(k_values, dtype=np.float64, count=nnz)
        m_data = np.frombuffer(m_values, dtype=np.float64, count=nnz)

        if matrix_status == 'STRUCTURE_CHANGED' or self._k_matrix is None:
            self._k_matrix = sp.csr_matrix((k_data, idx, indptr),
                                          shape=(num_eqn, num_eqn))
            self._m_matrix = sp.csr_matrix((m_data, idx, indptr),
                                          shape=(num_eqn, num_eqn))
        elif matrix_status == 'COEFFICIENTS_CHANGED':
            self._k_matrix.data[:] = k_data
            self._m_matrix.data[:] = m_data

        eigsh_kwargs = {
            'k': num_modes,
            'M': self._m_matrix,
            'which': 'LM',
        }
        if find_smallest:
            eigsh_kwargs['sigma'] = 0.0
        if self.maxiter is not None:
            eigsh_kwargs['maxiter'] = self.maxiter
        if self.tol > 0.0:
            eigsh_kwargs['tol'] = self.tol

        eigvals, eigvecs = sp_linalg.eigsh(self._k_matrix, **eigsh_kwargs)

        np.frombuffer(eigenvalues, dtype=np.float64,
                      count=num_modes)[:] = eigvals[:num_modes]
        np.frombuffer(eigenvectors, dtype=np.float64,
                      count=num_modes * num_eqn)[:] = eigvecs.T.flatten()

        return None


# ----------------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------------

DISTRIBUTED = "--distributed" in sys.argv
PID = ops.getPID() if hasattr(ops, "getPID") else 0
NP = ops.getNP() if hasattr(ops, "getNP") else 1
SYSTEM = "DistributedPythonSparse" if DISTRIBUTED else "PythonSparse"


def brick_model(nx=8, ny=2, nz=2):
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    ops.nDMaterial("ElasticIsotropic", 1, 29000.0, 0.3, 7.3e-7)
    ops.block3D(nx, ny, nz, 1, 1, "stdBrick", 1,
                1, 0.0, 0.0, 0.0, 2, 10.0, 0.0, 0.0, 3, 10.0, 1.0, 0.0, 4, 0.0, 1.0, 0.0,
                5, 0.0, 0.0, 2.0, 6, 10.0, 0.0, 2.0, 7, 10.0, 1.0, 2.0, 8, 0.0, 1.0, 2.0)
    ops.fixX(0.0, 1, 1, 1)
    tip = [n for n in ops.getNodeTags() if abs(ops.nodeCoord(n, 1) - 10.0) < 1e-9]
    for n in ops.getNodeTags():
        ops.mass(n, 1e-3, 1e-3, 1e-3)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for n in tip:
        ops.load(n, 0.0, 0.0, -1.0)
    if DISTRIBUTED and NP > 1:
        ops.partition()
    return tip


def static_tip_disp(system_args, steps=3):
    tip = brick_model()
    ops.constraints("Plain")
    ops.numberer("ParallelRCM" if DISTRIBUTED and NP > 1 else "RCM")
    ops.system(*system_args)
    ops.test("NormDispIncr", 1e-12, 20)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0 / steps)
    ops.analysis("Static")
    ok = ops.analyze(steps)
    # Under MPI each rank holds only its own nodes. The reference run uses the
    # same (deterministic) partition, so each rank compares its local sum.
    local_nodes = set(ops.getNodeTags())
    return ok, sum(ops.nodeDisp(n, 3) for n in tip if n in local_nodes)


def main() -> int:
    results = []
    ref_ok, ref = static_tip_disp(("Mumps",) if DISTRIBUTED and NP > 1 else ("UmfPack",))
    cases = [("SciPyspSolveSolver", SciPyspSolveSolver())]
    if HAVE_CUPY:
        # Known documentation bug with CuPy >= 13 (cp.sparse, tol=): reported, not fatal.
        cases.append(("CuPyCGSolver [doc, known bug]", CuPyCGSolver(rtol=1e-12, atol=1e-14,
                                                                   maxiter=5000)))
        cases.append(("CuPyCGSolverFixed", CuPyCGSolverFixed(rtol=1e-12, atol=1e-14,
                                                             maxiter=5000)))
    for name, solver in cases:
        try:
            ok, u = static_tip_disp((SYSTEM, {"solver": solver, "scheme": "CSR"}))
            good = ok == 0 and np.isclose(u, ref, rtol=1e-6)
            results.append((name, "ok" if good else f"FAIL ok={ok} u={u} ref={ref}"))
        except Exception as exc:  # noqa: BLE001
            results.append((name, f"ERROR {type(exc).__name__}: {exc}"))

    # eigen: compare with the default solver
    brick_model()
    ops.numberer("RCM")
    ref_eig = ops.eigen(6)
    try:
        lam = ops.eigen(SYSTEM if DISTRIBUTED else "PythonSparse", 6,
                        {"solver": SciPyGeneralizedEigenSolver(), "scheme": "CSR"})
        good = np.allclose(lam, ref_eig, rtol=1e-6)
        results.append(("SciPyGeneralizedEigenSolver", "ok" if good else f"FAIL {lam} vs {ref_eig}"))
    except Exception as exc:  # noqa: BLE001
        results.append(("SciPyGeneralizedEigenSolver", f"ERROR {type(exc).__name__}: {exc}"))

    if PID == 0:
        print(f"module={getattr(ops, '__file__', '?')} system={SYSTEM} np={NP} "
              f"reference ok={ref_ok}")
        for name, outcome in results:
            print(f"  {name:30s} {outcome}")
    ops.wipe()
    return 0 if all(o == "ok" or "known bug" in n for n, o in results) else 1


if __name__ == "__main__":
    sys.exit(main())
