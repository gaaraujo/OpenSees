"""Run the PR #1676 PythonSparse benchmarks at small sizes as a correctness check.

EXAMPLES/SolverBenchmark/benchmark_python_sparse.py and
benchmark_python_sparse_eigen.py (OpenSees PR #1676) define PythonSparse
solvers next to compiled ones. This driver imports both scripts unchanged with a
chosen OpenSees module and, on their own models and analysis settings:

* runs every solver they define (BandSPD, UmfPack, CuPyCG; genBandArpack,
  SciPyEigsh), and
* adds every solver of the ``openseespy-solvers`` package, when installed,

then checks each result against a compiled reference (UmfPack for the static
analysis; ``eigen`` on a freshly built model for the eigen analysis).

Usage::

    OPENSEESPY_PATH=<dir with opensees module> python run_pr1676_examples.py

Note: the eigen script's run_benchmark() performs a ``system Diagonal`` static
step before calling eigen. In that state genBandArpack returns wrong
eigenvalues (OpenSeesPy 3.8.0 as well), so the reference here is computed on a
freshly built model instead.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "SolverBenchmark"

if os.environ.get("OPENSEESPY_PATH"):
    sys.path.insert(0, os.environ["OPENSEESPY_PATH"])
    import opensees as _ops  # noqa: E402

    pkg = types.ModuleType("openseespy")
    pkg.opensees = _ops
    sys.modules["openseespy"] = pkg
    sys.modules["openseespy.opensees"] = _ops


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


def gpu_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


def package_linear_solvers() -> list[tuple[str, object]]:
    """(name, factory) for every linear solver the package offers."""
    try:
        import openseespy_solvers as oss  # noqa: F401
    except ImportError:
        return []
    from openseespy_solvers import hybrid
    from openseespy_solvers.scipy import cg, gmres, precond, spsolve

    out = [
        ("pkg scipy.spsolve", lambda: spsolve()),
        ("pkg scipy.cg+ilu", lambda: cg(rtol=1e-8, M=precond.ilu)),
        ("pkg scipy.gmres+ilu", lambda: gmres(rtol=1e-8, restart=50, M=precond.ilu)),
        ("pkg hybrid(scipy.spsolve)", lambda: hybrid(spsolve(), rtol=1e-8)),
    ]
    try:
        from openseespy_solvers.scipy import umfpack

        umfpack()
        out.append(("pkg scipy.umfpack", lambda: umfpack()))
    except ImportError:
        pass
    if gpu_available():
        from openseespy_solvers import cupy as ocp
        from openseespy_solvers.cupy import precond as cupy_precond

        out += [
            ("pkg cupy.cg+jacobi", lambda: ocp.cg(rtol=1e-8, M=cupy_precond.jacobi)),
            ("pkg cupy.gmres+jacobi", lambda: ocp.gmres(rtol=1e-8, restart=50,
                                                         M=cupy_precond.jacobi)),
            ("pkg cupy.spsolve", lambda: ocp.spsolve()),
        ]
        try:
            from openseespy_solvers.nvmath import direct_solver

            out += [
                ("pkg nvmath general", lambda: direct_solver()),
                ("pkg nvmath symmetric", lambda: direct_solver(matrix_type="symmetric")),
                ("pkg nvmath spd", lambda: direct_solver(matrix_type="spd")),
                ("pkg hybrid(nvmath)", lambda: hybrid(direct_solver(), rtol=1e-8)),
            ]
        except ImportError:
            pass
    return out


def package_eigen_solvers() -> list[tuple[str, object]]:
    try:
        from openseespy_solvers.scipy import eigsh
    except ImportError:
        return []
    out = [("pkg scipy.eigsh", lambda: eigsh())]
    if gpu_available():
        from openseespy_solvers import cupy as ocp
        from openseespy_solvers.cupy import precond as cupy_precond

        out.append(("pkg cupy.eigsh (general)", lambda: ocp.eigsh(mass_mode="general")))
    return out


def linear_checks(results: list) -> None:
    bench = load("benchmark_python_sparse")
    solvers = [("BandSPD", bench.setup_bandspd, "RCM"), ("UmfPack", bench.setup_umfpack, "Plain"),
               ("CuPyCG [PR, cp.sparse bug]", bench.setup_cupy_cg, "RCM")]
    ops = bench.ops
    for name, factory in package_linear_solvers():
        def setup(factory=factory):
            ops.system("PythonSparse", factory().to_openseespy())
        solvers.append((name, setup, "RCM"))

    for factor in (2.0, 4.0):
        mesh_size, counts = bench.counts_from_mesh_factor(factor)
        rows = {}
        for name, setup, numberer in solvers:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rows[name] = bench.run_benchmark(name, setup, numberer, factor, mesh_size,
                                                     counts)
            except Exception as exc:  # noqa: BLE001
                results.append((f"linear f={factor:g} {name}",
                                f"ERROR {type(exc).__name__}: {exc}"))
        ref = rows["UmfPack"]
        for name, row in rows.items():
            label = f"linear f={factor:g} {name}"
            if row.status != 0:
                results.append((label, f"FAIL status={row.status}"))
            elif not np.isclose(row.displacement_z, ref.displacement_z, rtol=1e-5):
                results.append((label, f"FAIL dz={row.displacement_z} vs {ref.displacement_z}"))
            else:
                results.append((label, f"ok neq={row.num_equations} t={row.time_seconds:.3f}s"))


def eigen_checks(results: list) -> None:
    bench = load("benchmark_python_sparse_eigen")
    ops = bench.ops
    solvers = [("genBandArpack [PR flow]", bench.run_gen_band_arpack),
               ("SciPyEigsh", bench.run_scipy_eigsh)]
    for name, factory in package_eigen_solvers():
        def run(num_modes, factory=factory):
            return ops.eigen("PythonSparse", num_modes, factory().to_openseespy())
        solvers.append((name, run))

    for factor in (1.0, 2.0):
        mesh_size, counts = bench.counts_from_mesh_factor(factor)
        bench.build_solid_bar_model(*counts)
        ref = ops.eigen(5)
        for name, run in solvers:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row = bench.run_benchmark(name, run, factor, mesh_size, counts)
            lam = row.eigenvalues_all if hasattr(row, "eigenvalues_all") else None
            lam1 = row.eigenvalue_mode1
            good = row.status == 0 and lam1 is not None and np.isclose(lam1, ref[0], rtol=1e-6)
            label = f"eigen f={factor:g} {name}"
            results.append((label, f"ok t={row.time_seconds:.3f}s" if good
                            else f"FAIL status={row.status} lam1={lam1} ref={ref[0]}"))
            del lam


def main() -> int:
    results: list = []
    linear_checks(results)
    eigen_checks(results)
    mod = sys.modules.get("openseespy.opensees")
    print(f"module={getattr(mod, '__file__', '?')}")
    for name, outcome in results:
        print(f"  {name:38s} {outcome}")
    expected = ("[PR, cp.sparse bug]", "[PR flow]")
    return 0 if all(o.startswith("ok") or n.endswith(expected) for n, o in results) else 1


if __name__ == "__main__":
    sys.exit(main())
