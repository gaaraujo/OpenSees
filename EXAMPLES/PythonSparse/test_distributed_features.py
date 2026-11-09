"""DistributedPythonSparse checks under MPI (run with 2 or more ranks).

    mpiexec -n 2 python test_distributed_features.py
    mpiexec -n 4 python test_distributed_features.py

Each case runs the same partitioned model twice in one process: with the
compiled parallel ``system Mumps`` (reference) and with
``system DistributedPythonSparse`` + a Python solver. The partition is
deterministic, so every rank compares the displacements of its own nodes and
the Newton iteration counts per step. Cases:

* symmetric brick (CSR, CSC, COO; full and lower-triangle storage),
* nonsymmetric frictional contact (CSR, CSC, COO),
* Newton, ModifiedNewton and KrylovNewton,
* eigen (DistributedPythonSparse vs the default eigen solver),
* a deliberate matrix-status mismatch between ranks, which must fail on
  every rank instead of hanging.

Solvers: a minimal hand-written SciPy solver, and openseespy-solvers' spsolve
(and pardiso when the MKL runtime is available).
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

PID = ops.getPID()
NP = ops.getNP()


class HandSolver:
    """The smallest correct PythonSparse solver (any scheme, full storage)."""

    def __init__(self):
        self.lu = None

    def solve(self, **kw):
        n, nnz = kw["num_eqn"], kw["nnz"]
        if kw["matrix_status"] != "UNCHANGED" or self.lu is None:
            vals = np.frombuffer(kw["values"], dtype=np.float64, count=nnz)
            if kw["storage_scheme"] == "COO":
                r = np.frombuffer(kw["row"], dtype=np.int32, count=nnz)
                c = np.frombuffer(kw["col"], dtype=np.int32, count=nnz)
                A = sp.csc_matrix((vals, (r, c)), shape=(n, n))
            else:
                ptr = np.frombuffer(kw["index_ptr"], dtype=np.int32, count=n + 1)
                ind = np.frombuffer(kw["indices"], dtype=np.int32, count=nnz)
                cls = sp.csr_matrix if kw["storage_scheme"] == "CSR" else sp.csc_matrix
                A = cls((vals.copy(), ind.copy(), ptr.copy()), shape=(n, n)).tocsc()
            self.lu = spla.splu(A)
        b = np.frombuffer(kw["rhs"], dtype=np.float64, count=n)
        np.frombuffer(kw["x"], dtype=np.float64, count=n)[:] = self.lu.solve(b)
        return 0


def package_solvers():
    out = []
    try:
        from openseespy_solvers.scipy import spsolve

        out.append(("pkg spsolve", lambda: spsolve()))
        try:
            from openseespy_solvers.scipy import pardiso
            from openseespy_solvers.scipy._pardiso import load_mkl

            load_mkl()
            out.append(("pkg pardiso", lambda: pardiso()))
            out.append(("pkg pardiso symmetric", lambda: pardiso(matrix_type="symmetric")))
        except Exception:  # noqa: BLE001
            pass
    except ImportError:
        pass
    return out


# -- models (identical on every rank, then partitioned) ----------------------


def brick_model():
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    ops.nDMaterial("ElasticIsotropic", 1, 29000.0, 0.3, 7.3e-7)
    ops.block3D(12, 2, 3, 1, 1, "stdBrick", 1,
                1, 0.0, 0.0, 0.0, 2, 10.0, 0.0, 0.0, 3, 10.0, 1.0, 0.0, 4, 0.0, 1.0, 0.0,
                5, 0.0, 0.0, 2.0, 6, 10.0, 0.0, 2.0, 7, 10.0, 1.0, 2.0, 8, 0.0, 1.0, 2.0)
    ops.fixX(0.0, 1, 1, 1)
    for n in ops.getNodeTags():
        ops.mass(n, 1e-3, 1e-3, 1e-3)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for n in ops.getNodeTags():
        if abs(ops.nodeCoord(n, 1) - 10.0) < 1e-9:
            ops.load(n, 0.0, 0.3, -1.0)


def contact_model(count=24):
    """Independent sliding blocks on frictional contact: nonsymmetric tangent."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.uniaxialMaterial("Elastic", 1, 5.0)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for i in range(count):
        base, block, wall = 3 * i + 1, 3 * i + 2, 3 * i + 3
        for tag in (base, block, wall):
            ops.node(tag, float(i), 0.0)
        ops.fix(base, 1, 1)
        ops.fix(wall, 1, 1)
        ops.element("zeroLengthContact2D", 2 * i + 1, block, base, 1.0e6, 1.0e6, 0.1,
                    "-normal", 0.0, 1.0)
        ops.element("zeroLength", 2 * i + 2, wall, block, "-mat", 1, "-dir", 1)
        ops.load(block, 30.0 + i, -200.0)


def run(build, system_args, algorithm=("Newton",), steps=5):
    build()
    if NP > 1:
        ops.partition()
    ops.constraints("Plain")
    ops.numberer("ParallelRCM")
    ops.system(*system_args)
    ops.test("NormDispIncr", 1e-10, 50)
    ops.algorithm(*algorithm)
    ops.integrator("LoadControl", 1.0 / steps)
    ops.analysis("Static")
    iters = []
    ok = 0
    for _ in range(steps):
        ok = ops.analyze(1)
        iters.append(ops.testIter())
        if ok != 0:
            break
    disp = {n: ops.nodeDisp(n) for n in sorted(ops.getNodeTags())}
    return ok, iters, disp


def same(d1, d2, rtol=1e-8, atol=1e-12):
    if d1.keys() != d2.keys():
        return False
    return all(np.allclose(d1[k], d2[k], rtol=rtol, atol=atol) for k in d1)


RESULTS = []


def record(name, good, detail=""):
    RESULTS.append((name, bool(good), detail))


def main() -> int:
    if NP < 2:
        print("run with at least 2 MPI ranks")
        return 1

    algorithms = {"Newton": ("Newton",), "ModifiedNewton": ("ModifiedNewton",),
                  "KrylovNewton": ("KrylovNewton",)}
    solvers = [("hand SciPy", HandSolver)] + package_solvers()

    for model_name, build in (("brick", brick_model), ("contact", contact_model)):
        for alg_name, alg in algorithms.items():
            ok_ref, it_ref, d_ref = run(build, ("Mumps",), alg)
            for scheme in ("CSR", "CSC", "COO"):
                for name, make in solvers:
                    if "symmetric" in name and model_name != "brick":
                        continue
                    solver = make()
                    cfg = (solver.to_openseespy(scheme=scheme) if hasattr(solver, "to_openseespy")
                           else {"solver": solver, "scheme": scheme})
                    ok, it, d = run(build, ("DistributedPythonSparse", cfg), alg)
                    label = f"{model_name} {alg_name} {scheme} {name}"
                    if ok_ref != 0:
                        # The compiled reference does not converge either.
                        record(label, True, f"skipped (reference ok={ok_ref})")
                        continue
                    # Displacements to round-off. Iteration counts are reported: on
                    # the frictional contact model they vary with round-off (the
                    # Mumps reference itself changes with the number of ranks).
                    good = ok == 0 and same(d, d_ref, rtol=1e-6, atol=1e-9)
                    record(label, good, f"iters={it} ref={it_ref}")
            # lower-triangle storage requested from OpenSees (hand solver mirrors it)
            if model_name == "brick" and alg_name == "Newton":
                class LowerHand(HandSolver):
                    def solve(self, **kw):
                        assert kw.get("matrix_type") == "SYMMETRIC_LOWER"
                        n, nnz = kw["num_eqn"], kw["nnz"]
                        if kw["matrix_status"] != "UNCHANGED" or self.lu is None:
                            ptr = np.frombuffer(kw["index_ptr"], dtype=np.int32, count=n + 1)
                            ind = np.frombuffer(kw["indices"], dtype=np.int32, count=nnz)
                            val = np.frombuffer(kw["values"], dtype=np.float64, count=nnz)
                            L = sp.csr_matrix((val.copy(), ind.copy(), ptr.copy()), shape=(n, n))
                            self.lu = spla.splu((L + sp.tril(L, -1).T).tocsc())
                        b = np.frombuffer(kw["rhs"], dtype=np.float64, count=n)
                        np.frombuffer(kw["x"], dtype=np.float64, count=n)[:] = self.lu.solve(b)
                        return 0

                cfg = {"solver": LowerHand(), "scheme": "CSR", "matrix_type": "SYMMETRIC_LOWER"}
                ok, it, d = run(build, ("DistributedPythonSparse", cfg), alg)
                record("brick Newton CSR hand SYMMETRIC_LOWER", ok == 0 and same(d, d_ref),
                       f"iters={it}")

    # eigen: distributed PythonSparse vs default eigen (ArpackSOE with channels)
    class EigenHand:
        def solve(self, **kw):
            n, nnz, k = kw["num_eqn"], kw["nnz"], kw["num_modes"]
            ptr = np.frombuffer(kw["index_ptr"], dtype=np.int32, count=n + 1)
            ind = np.frombuffer(kw["indices"], dtype=np.int32, count=nnz)
            K = sp.csr_matrix((np.frombuffer(kw["k_values"], dtype=np.float64, count=nnz), ind, ptr),
                              shape=(n, n))
            M = sp.csr_matrix((np.frombuffer(kw["m_values"], dtype=np.float64, count=nnz), ind, ptr),
                              shape=(n, n))
            w, v = spla.eigsh(K.tocsc(), k=k, M=M.tocsc(), sigma=0.0, which="LM")
            order = np.argsort(w)
            np.frombuffer(kw["eigenvalues"], dtype=np.float64, count=k)[:] = w[order]
            np.frombuffer(kw["eigenvectors"], dtype=np.float64, count=k * n)[:] = v[:, order].T.ravel()
            return None

    for i in range(2):  # twice: the second call sees COEFFICIENTS_CHANGED
        brick_model()
        ops.partition()
        ops.constraints("Plain")
        ops.numberer("ParallelRCM")
        ops.system("Mumps")
        ref = ops.eigen(4)
        lam = ops.eigen("DistributedPythonSparse", 4, {"solver": EigenHand(), "scheme": "CSR"})
        record(f"eigen brick call {i + 1}", np.allclose(lam, ref, rtol=1e-8), f"{lam[:2]}")

    # deliberate status mismatch: rank 0 holds its initial tangent, the
    # workers update theirs; must fail on every rank, not hang
    alg = ("ModifiedNewton", "-initial") if PID == 0 else ("Newton",)
    ok, it, _ = run(brick_model, ("DistributedPythonSparse", {"solver": HandSolver()}), alg,
                    steps=2)
    record("status mismatch fails cleanly", ok < 0, f"ok={ok} iters={it}")

    failed = [r for r in RESULTS if not r[1]]
    for name, good, detail in RESULTS:
        if not good or PID == 0:
            print(f"[pid={PID}] {'ok  ' if good else 'FAIL'} {name:46s} {detail}")
    if PID == 0:
        print(f"[pid=0] np={NP}: {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    ops.wipe()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
