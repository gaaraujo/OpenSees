"""Per-call overhead of the OpenSees -> Python round trip in system PythonSparse.

Runs a transient analysis with ``algorithm Linear -factorOnce`` so that every
step assembles only the right-hand side and calls the solver once with
``matrix_status='UNCHANGED'``. The per-step time with a Python ``solve`` that
does nothing is compared with the per-step time of a compiled banded solver
(BandGeneral, whose back substitution on a few equations is negligible).
The difference is the cost of one C++ -> Python call.

Usage::

    python bench_call_overhead.py [--steps 20000] [--repeat 5] [--nx 1]

Set OPENSEESPY_PATH to a directory holding a locally built ``opensees``
module to test it instead of the installed ``openseespy``.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

if os.environ.get("OPENSEESPY_PATH"):
    sys.path.insert(0, os.environ["OPENSEESPY_PATH"])
    import opensees as ops  # noqa: E402
else:
    import openseespy.opensees as ops  # noqa: E402


class NoOpSolver:
    """A PythonSparse solver that returns immediately (x is left as is)."""

    def solve(self, **kwargs):
        return 0


class NoOpSolverPositional:
    """Same, but with an explicit signature and no ``**kwargs``."""

    def solve(self, index_ptr, indices, values, rhs, x, matrix_status,
              storage_scheme, num_eqn, nnz):
        return 0


def build_model(nx: int) -> int:
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    ops.nDMaterial("ElasticIsotropic", 1, 29000.0, 0.3, 7.3e-7)
    ny = nz = 1
    ops.block3D(
        nx, ny, nz, 1, 1, "stdBrick", 1,
        1, 0.0, 0.0, 0.0,
        2, 10.0, 0.0, 0.0,
        3, 10.0, 1.0, 0.0,
        4, 0.0, 1.0, 0.0,
        5, 0.0, 0.0, 1.0,
        6, 10.0, 0.0, 1.0,
        7, 10.0, 1.0, 1.0,
        8, 0.0, 1.0, 1.0,
    )
    ops.fixX(0.0, 1, 1, 1)
    tip = [n for n in ops.getNodeTags() if abs(ops.nodeCoord(n, 1) - 10.0) < 1e-9]
    ops.timeSeries("Trig", 1, 0.0, 1e9, 0.1)
    ops.pattern("Plain", 1, 1)
    for n in tip:
        ops.load(n, 0.0, 0.0, -1.0)
    return len(ops.getNodeTags()) * 3


def run(system_args, steps: int) -> float:
    ops.wipeAnalysis()
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system(*system_args)
    ops.algorithm("Linear", "-factorOnce")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")
    ops.analyze(10, 1e-3)  # warm-up (first call has STRUCTURE_CHANGED)
    t0 = time.perf_counter()
    ok = ops.analyze(steps, 1e-3)
    elapsed = time.perf_counter() - t0
    if ok != 0:
        raise RuntimeError(f"analyze failed for {system_args[0]}")
    return elapsed / steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--nx", type=int, default=1)
    args = parser.parse_args()

    ndof = build_model(args.nx)
    cases = {
        "BandGeneral": lambda: ("BandGeneral",),
        "PythonSparse no-op (**kwargs)": lambda: (
            "PythonSparse", {"solver": NoOpSolver(), "scheme": "CSR"}),
        "PythonSparse no-op (positional)": lambda: (
            "PythonSparse", {"solver": NoOpSolverPositional(), "scheme": "CSR"}),
    }
    print(f"model: {ndof} nodal DOFs (before constraints), steps={args.steps}, "
          f"repeat={args.repeat}, module={getattr(ops, '__file__', '?')}")
    results = {}
    for name, make in cases.items():
        times = [run(make(), args.steps) for _ in range(args.repeat)]
        results[name] = times
        print(f"{name:34s} median {1e6 * statistics.median(times):8.2f} us/step"
              f"  (min {1e6 * min(times):.2f}, max {1e6 * max(times):.2f})")
    base = statistics.median(results["BandGeneral"])
    for name in list(cases)[1:]:
        over = statistics.median(results[name]) - base
        print(f"overhead {name:34s} {1e6 * over:8.2f} us/call")
    ops.wipe()


if __name__ == "__main__":
    main()
