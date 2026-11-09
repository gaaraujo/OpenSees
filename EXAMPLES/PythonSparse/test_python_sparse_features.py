"""Correctness checks for system/eigen PythonSparse (serial).

Run with a locally built module::

    OPENSEESPY_PATH=<dir with opensees.pyd|.so> python test_python_sparse_features.py

or with the installed ``openseespy`` (features missing from that build are
skipped and reported). Every check compares PythonSparse against a compiled
OpenSees system on the same model, so it needs only NumPy and SciPy.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

if os.environ.get("OPENSEESPY_PATH"):
    sys.path.insert(0, os.environ["OPENSEESPY_PATH"])
    import opensees as ops  # noqa: E402
else:
    import openseespy.opensees as ops  # noqa: E402

CHECKS: list[tuple[str, object]] = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


def run_checks() -> list[tuple[str, str]]:
    results = []
    for name, fn in CHECKS:
        try:
            outcome = fn()
            results.append((name, outcome or "ok"))
        except AssertionError as exc:
            results.append((name, f"FAIL: {exc}"))
        except Exception as exc:  # noqa: BLE001
            results.append((name, f"ERROR: {type(exc).__name__}: {exc}"))
    return results


# --------------------------------------------------------------------------
# Reference Python solver: builds the matrix from the buffers and solves it.
# --------------------------------------------------------------------------

class RecordingSolver:
    """Direct solver that records what OpenSees passed in."""

    def __init__(self, fail_first: int = 0):
        self.statuses: list[str] = []
        self.matrix_types: list[str | None] = []
        self.formats: list[str] = []
        self.asym: float = 0.0
        self.fail_first = fail_first
        self.last_A = None

    def _matrix(self, kw):
        n, nnz = kw["num_eqn"], kw["nnz"]
        vals = np.frombuffer(kw["values"], dtype=np.float64, count=nnz)
        scheme = kw["storage_scheme"]
        if scheme == "COO":
            row = np.frombuffer(kw["row"], dtype=np.int32, count=nnz)
            col = np.frombuffer(kw["col"], dtype=np.int32, count=nnz)
            A = sp.coo_matrix((vals, (row, col)), shape=(n, n)).tocsr()
        else:
            ptr = np.frombuffer(kw["index_ptr"], dtype=np.int32, count=n + 1)
            ind = np.frombuffer(kw["indices"], dtype=np.int32, count=nnz)
            cls = sp.csr_matrix if scheme == "CSR" else sp.csc_matrix
            A = cls((vals.copy(), ind.copy(), ptr.copy()), shape=(n, n)).tocsr()
        if kw.get("matrix_type") == "SYMMETRIC_LOWER":
            low = sp.tril(A)
            assert abs(sp.triu(A, 1)).sum() == 0, "upper entries present in lower storage"
            A = (low + sp.tril(A, -1).T).tocsr()
        return A

    def solve(self, **kw):
        self.statuses.append(kw["matrix_status"])
        self.matrix_types.append(kw.get("matrix_type"))
        self.formats.append(getattr(kw["values"], "format", "?"))
        if self.fail_first > 0:
            self.fail_first -= 1
            return -1
        A = self._matrix(kw)
        self.last_A = A
        if A.shape[0]:
            self.asym = max(self.asym, abs(A - A.T).max())
        b = np.frombuffer(kw["rhs"], dtype=np.float64, count=kw["num_eqn"])
        x = np.frombuffer(kw["x"], dtype=np.float64, count=kw["num_eqn"])
        x[:] = spla.spsolve(A.tocsc(), b)
        return 0


class PositionalSolver:
    """solve() written for OpenSees 3.8.0 with an explicit signature."""

    def __init__(self):
        self.inner = RecordingSolver()

    def solve(self, index_ptr, indices, values, rhs, x, matrix_status,
              storage_scheme, num_eqn, nnz):
        return self.inner.solve(index_ptr=index_ptr, indices=indices, values=values,
                                rhs=rhs, x=x, matrix_status=matrix_status,
                                storage_scheme=storage_scheme, num_eqn=num_eqn, nnz=nnz)


class ReturnSolver(RecordingSolver):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def solve(self, **kw):
        super().solve(**kw)
        return self.value


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def brick_model(nx=4, ny=2, nz=2):
    """Symmetric elastic brick cantilever under a tip load."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    ops.nDMaterial("ElasticIsotropic", 1, 29000.0, 0.3, 7.3e-7)
    ops.block3D(nx, ny, nz, 1, 1, "stdBrick", 1,
                1, 0.0, 0.0, 0.0, 2, 10.0, 0.0, 0.0, 3, 10.0, 1.0, 0.0, 4, 0.0, 1.0, 0.0,
                5, 0.0, 0.0, 2.0, 6, 10.0, 0.0, 2.0, 7, 10.0, 1.0, 2.0, 8, 0.0, 1.0, 2.0)
    ops.fixX(0.0, 1, 1, 1)
    tip = [n for n in ops.getNodeTags() if abs(ops.nodeCoord(n, 1) - 10.0) < 1e-9]
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for n in tip:
        ops.load(n, 0.0, 0.0, -1.0)
    return tip[-1]


def frame_model():
    """Nonlinear 2-D cantilever column (Steel01 fibers, P-Delta): Newton iterates."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, 120.0)
    ops.node(3, 0.0, 240.0)
    ops.fix(1, 1, 1, 1)
    ops.uniaxialMaterial("Steel01", 1, 50.0, 29000.0, 0.02)
    ops.section("Fiber", 1)
    ops.patch("rect", 1, 10, 1, -6.0, -6.0, 6.0, 6.0)
    ops.geomTransf("PDelta", 1)
    ops.beamIntegration("Lobatto", 1, 1, 5)
    ops.element("forceBeamColumn", 1, 1, 2, 1, 1)
    ops.element("forceBeamColumn", 2, 2, 3, 1, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(3, 60.0, -200.0, 0.0)
    return 3


def contact_model():
    """Frictional contact (zeroLengthContact2D) pushed past the slip limit; the
    sliding tangent couples shear to the normal force and is nonsymmetric."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, 0.0)
    ops.node(3, 0.0, 0.0)
    ops.fix(1, 1, 1)
    ops.fix(3, 1, 1)
    ops.element("zeroLengthContact2D", 1, 2, 1, 1.0e6, 1.0e6, 0.1, "-normal", 0.0, 1.0)
    ops.uniaxialMaterial("Elastic", 1, 5.0)
    ops.element("zeroLength", 2, 3, 2, "-mat", 1, "-dir", 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 40.0, -200.0)
    return 2


def run_static(build, system_args, algorithm="Newton", steps=5):
    node = build()
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system(*system_args)
    ops.test("NormDispIncr", 1e-10, 50)
    ops.algorithm(algorithm)
    ops.integrator("LoadControl", 1.0 / steps)
    ops.analysis("Static")
    ok = ops.analyze(steps)
    return ok, np.array(ops.nodeDisp(node))


def run_transient(build, system_args, algorithm="Newton", steps=5):
    node = build()
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system(*system_args)
    ops.test("NormDispIncr", 1e-10, 50)
    ops.algorithm(algorithm)
    ops.integrator("Newmark", 0.6, 0.3025)
    ops.analysis("Transient")
    ok = ops.analyze(steps, 0.01)
    return ok, np.array(ops.nodeDisp(node))


def ps(solver, **opts):
    cfg = {"solver": solver}
    cfg.update(opts)
    return ("PythonSparse", cfg)


def supports_new_options() -> bool:
    """True if this build passes matrix_type when asked (feature detection)."""
    s = RecordingSolver()
    run_static(lambda: brick_model(1, 1, 1), ps(s, scheme="CSR", matrix_type="SYMMETRIC_LOWER"),
               steps=1)
    return s.matrix_types[:1] == ["SYMMETRIC_LOWER"]


NEW = None


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

@check("frame: CSR/CSC/COO Newton match UmfPack (static, nonlinear)")
def _():
    ok, ref = run_static(frame_model, ("UmfPack",))
    assert ok == 0
    for scheme in ("CSR", "CSC", "COO"):
        ok, u = run_static(frame_model, ps(RecordingSolver(), scheme=scheme))
        assert ok == 0, scheme
        np.testing.assert_allclose(u, ref, rtol=1e-9, atol=1e-12, err_msg=scheme)


@check("frame: ModifiedNewton sees UNCHANGED between tangent updates")
def _():
    ok, ref = run_static(frame_model, ("UmfPack",), algorithm="ModifiedNewton")
    s = RecordingSolver()
    ok2, u = run_static(frame_model, ps(s, scheme="CSR"), algorithm="ModifiedNewton")
    assert ok == 0 and ok2 == 0
    np.testing.assert_allclose(u, ref, rtol=1e-8, atol=1e-12)
    assert "UNCHANGED" in s.statuses, s.statuses[:10]
    assert s.statuses[0] == "STRUCTURE_CHANGED"


@check("contact: nonsymmetric static Newton, CSR/CSC/COO match UmfPack")
def _():
    ok, ref = run_static(contact_model, ("UmfPack",), steps=10)
    assert ok == 0
    for scheme in ("CSR", "CSC", "COO"):
        s = RecordingSolver()
        ok, u = run_static(contact_model, ps(s, scheme=scheme), steps=10)
        assert ok == 0, scheme
        assert s.asym > 1e-8, f"tangent unexpectedly symmetric ({s.asym})"
        np.testing.assert_allclose(u, ref, rtol=1e-9, atol=1e-14, err_msg=scheme)
    return f"ok (max |A - A^T| = {s.asym:.3g})"


@check("brick: SYMMETRIC_LOWER matches GENERAL for CSR/CSC/COO")
def _():
    if not NEW:
        return "skipped (build has no matrix_type)"
    ok, ref = run_static(brick_model, ps(RecordingSolver(), scheme="CSR"), steps=1)
    assert ok == 0
    for scheme in ("CSR", "CSC", "COO"):
        s = RecordingSolver()
        ok, u = run_static(brick_model, ps(s, scheme=scheme, matrix_type="SYMMETRIC_LOWER"),
                           steps=1)
        assert ok == 0, scheme
        assert s.matrix_types[0] == "SYMMETRIC_LOWER"
        np.testing.assert_allclose(u, ref, rtol=1e-10, err_msg=scheme)


@check("matrix_type is not passed unless requested; positional solve() works")
def _():
    s = PositionalSolver()
    ok, _u = run_static(brick_model, ps(s, scheme="CSR"), steps=1)
    assert ok == 0
    assert s.inner.matrix_types == [None] * len(s.inner.matrix_types)


@check("typed buffers: format 'd' when typed=True, 'B' by default")
def _():
    s = RecordingSolver()
    run_static(lambda: brick_model(1, 1, 1), ps(s, scheme="CSR"), steps=1)
    assert s.formats[0] == "B", s.formats[0]
    if not NEW:
        return "skipped typed=True (build has no option)"
    s = RecordingSolver()
    ok, _u = run_static(lambda: brick_model(1, 1, 1), ps(s, scheme="CSR", typed=True), steps=1)
    assert ok == 0 and s.formats[0] == "d", s.formats[0]


@check("return values: None/0/np.int64(0)/True succeed; False/-3/np.int32(-2)/'x' fail")
def _():
    good = [None, 0, np.int64(0), True]
    bad = [False, -3, np.int32(-2), "x"] if NEW else [-3]
    for value in good:
        ok, _u = run_static(lambda: brick_model(1, 1, 1), ps(ReturnSolver(value)), steps=1)
        assert ok == 0, f"{value!r} should succeed"
    for value in bad:
        ok, _u = run_static(lambda: brick_model(1, 1, 1), ps(ReturnSolver(value)), steps=1)
        assert ok < 0, f"{value!r} should fail"


@check("failed solve keeps matrix_status (next call refactorizes) for CSR and COO")
def _():
    for scheme in ("CSR", "COO"):
        s = RecordingSolver(fail_first=1)
        brick_model(1, 1, 1)
        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system(*ps(s, scheme=scheme))
        ops.test("NormDispIncr", 1e-10, 10)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        assert ops.analyze(1) < 0
        ops.analyze(1)
        assert s.statuses[:2] == ["STRUCTURE_CHANGED", "STRUCTURE_CHANGED"], (scheme, s.statuses)


class EigenSolver:
    def __init__(self):
        self.statuses = []

    def solve(self, **kw):
        self.statuses.append(kw["matrix_status"])
        n, nnz, k = kw["num_eqn"], kw["nnz"], kw["num_modes"]
        ptr = np.frombuffer(kw["index_ptr"], dtype=np.int32, count=n + 1)
        ind = np.frombuffer(kw["indices"], dtype=np.int32, count=nnz)
        cls = sp.csr_matrix if kw["storage_scheme"] == "CSR" else sp.csc_matrix
        K = cls((np.frombuffer(kw["k_values"], dtype=np.float64, count=nnz), ind, ptr), shape=(n, n))
        M = cls((np.frombuffer(kw["m_values"], dtype=np.float64, count=nnz), ind, ptr), shape=(n, n))
        w, v = spla.eigsh(K.tocsc(), k=k, M=M.tocsc(), sigma=0.0, which="LM")
        order = np.argsort(w)
        np.frombuffer(kw["eigenvalues"], dtype=np.float64, count=k)[:] = w[order]
        np.frombuffer(kw["eigenvectors"], dtype=np.float64, count=k * n)[:] = v[:, order].T.ravel()
        return 0


@check("eigen: compressed PythonSparse matches default eigen; repeated calls")
def _():
    brick_model(3, 1, 1)
    ops.mass(ops.getNodeTags()[-1], 1.0, 1.0, 1.0)
    ref = ops.eigen(4)
    s = EigenSolver()
    lam = ops.eigen("PythonSparse", 4, {"solver": s, "scheme": "CSR"})
    lam2 = ops.eigen("PythonSparse", 4, {"solver": s, "scheme": "CSR"})
    np.testing.assert_allclose(lam, ref, rtol=1e-8)
    np.testing.assert_allclose(lam2, ref, rtol=1e-8)
    return f"ok (statuses {s.statuses})"


class COOEigenSolver:
    """COO eigen protocol: return (eigenvalues, eigenvectors) as lists."""

    def __init__(self):
        self.statuses = []

    def solve(self, **kw):
        self.statuses.append(kw["matrix_status"])
        n, nnz, k = kw["num_eqn"], kw["nnz"], kw["num_modes"]
        row = np.frombuffer(kw["row_indices"], dtype=np.int32, count=nnz)
        col = np.frombuffer(kw["col_indices"], dtype=np.int32, count=nnz)
        K = sp.coo_matrix((np.frombuffer(kw["k_values"], dtype=np.float64, count=nnz), (row, col)),
                          shape=(n, n)).tocsc()
        M = sp.coo_matrix((np.frombuffer(kw["m_values"], dtype=np.float64, count=nnz), (row, col)),
                          shape=(n, n)).tocsc()
        w, v = spla.eigsh(K, k=k, M=M, sigma=0.0, which="LM")
        order = np.argsort(w)
        return [float(x) for x in w[order]], [[float(y) for y in v[:, j]] for j in order]


@check("eigen COO: first call after setSize reports STRUCTURE_CHANGED")
def _():
    brick_model(3, 1, 1)
    ops.mass(ops.getNodeTags()[-1], 1.0, 1.0, 1.0)
    ref = ops.eigen(3)
    s = COOEigenSolver()
    lam = ops.eigen("PythonSparse", 3, {"solver": s, "scheme": "COO"})
    np.testing.assert_allclose(lam, ref, rtol=1e-8)
    assert s.statuses[0] == "STRUCTURE_CHANGED", s.statuses


def main() -> int:
    global NEW
    NEW = supports_new_options()
    print(f"module: {getattr(ops, '__file__', '?')}; new options: {NEW}")
    results = run_checks()
    width = max(len(n) for n, _ in results)
    failed = 0
    for name, outcome in results:
        print(f"{name:{width}s}  {outcome}")
        failed += outcome.startswith(("FAIL", "ERROR"))
    ops.wipe()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
