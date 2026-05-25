"""
Benchmark several linear equation solvers on a 3D solid bar model.

The base geometry and commentary originate from Michael H. Scott's blog post:
https://portwooddigital.com/2021/12/19/three-dimensional-meshing/

We reuse that model, run a short 5-step quasi-static load history, and record
the run times for a list of OpenSees linear system solvers, including:
- CuPy CG solvers with optional Jacobi preconditioning
- SciPy solvers (direct, UMFPACK, CG)
- NvMath direct solver (uses stateful API with planning for efficient factorization reuse)
- Native OpenSees solvers (BandSPD, BandGeneral, UmfPack, etc.)

CuPy solvers are defined in ``solvers/cupy_sparse_solver.py``.
NvMath solver is defined in ``solvers/nvmath_sparse_solver.py``.
"""

# -----------------------------------------------------------------------------
# General Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# OpenSeesPy Import
# -----------------------------------------------------------------------------
# Locate the locally built OpenSeesPy module.
SCRIPT_PATH = Path(__file__).resolve()
FILENAME = SCRIPT_PATH.name
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
OPENSEESPY_BUILD = REPO_ROOT / "build-cuda"
print(f"[{FILENAME}] Importing OpenSeesPy from: {OPENSEESPY_BUILD}")
sys.path.append(str(OPENSEESPY_BUILD))
import opensees as ops

# -----------------------------------------------------------------------------
# Numerical Model
# -----------------------------------------------------------------------------

# Geometric properties
BAR_LENGTH = 10.0 # inches
BAR_HEIGHT = 2.0 # inches
BAR_THICKNESS = 1.0 # inches

# Material properties
ELASTIC_MODULUS = 29_000.0 # kip / in^2
POISSON_RATIO = 0.3
STEEL_DENSITY = 0.284e-3 / 386.4 # kip s^2 / in^4

def build_solid_bar_model(nx: int, ny: int, nz: int) -> None:
    """Create a structured hexahedral mesh using ``block3D``."""

    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    ops.nDMaterial("ElasticIsotropic", 1, ELASTIC_MODULUS, POISSON_RATIO, STEEL_DENSITY)

    # block3D expects corner coordinates in local order.
    eleType = "stdBrick"
    eleArgs = 1
    ops.block3D(
        nx, ny, nz, 1, 1, eleType, eleArgs, 
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

def apply_load_pattern(load_per_node: float) -> None:
    """Apply a vertical load to all nodes on the free face."""

    tStart, tEnd, period = 0.0, 6.0, 4.0
    ops.timeSeries('Trig', 1, tStart, tEnd, period, '-factor', 1.0)
    ops.pattern("Plain", 1, 1)
    far_x_nodes = [
        node for node in ops.getNodeTags() if math.isclose(ops.nodeCoord(node, 1), BAR_LENGTH, abs_tol=1e-9)
    ]
    for node in far_x_nodes:
        ops.load(node, 0.0, 0.0, -load_per_node)

# -----------------------------------------------------------------------------
# Analysis configuration
# -----------------------------------------------------------------------------

def configure_static_analysis(solver: SolverSpec, num_steps: int, tol: float, max_iter: int) -> None:
    """Configure the analysis for a given solver."""
    ops.constraints("Plain")
    ops.numberer(solver.numberer)
    solver.setup()
    ops.integrator("LoadControl", 1.0 / num_steps)
    ops.test("NormUnbalance", tol, max_iter)
    ops.algorithm("ModifiedNewton", "-FactorOnce")
    # ops.algorithm("KrylovNewton", "-increment", "initial", "-iterate", "noTangent", "-maxDim", 6)
    ops.analysis("Static")

def configure_transient_analysis(solver: SolverSpec, num_steps: int, tol: float, max_iter: int) -> None:
    """Configure the analysis for a given solver."""
    zeta = 0.01
    T1 = estimate_fundamental_period(BAR_THICKNESS, BAR_HEIGHT, BAR_LENGTH, ELASTIC_MODULUS, STEEL_DENSITY)
    omega1 = 2 * math.pi / T1
    ops.rayleigh(2 * zeta * omega1, 0.0, 0.0, 0.0)
    ops.constraints("Plain")
    ops.numberer(solver.numberer)
    solver.setup()
    ops.integrator("Newmark", 0.5, 0.25)
    ops.test("NormUnbalance", tol, max_iter)
    ops.algorithm("ModifiedNewton", "-FactorOnce")
    # ops.algorithm("KrylovNewton", "-increment", "initial", "-iterate", "noTangent", "-maxDim", 6)
    ops.analysis("Transient")

def estimate_fundamental_period(b: float, h: float, L: float, E: float, rho: float) -> float:
    """Estimate the fundamental period of a cantilever beam."""
    A = b * h
    I = b * h**3 / 12.0
    m = rho * A
    beta1 = 1.87510406871196 # Exact cantilever constant
    C = 2 * math.pi / (beta1**2)
    return C * math.sqrt((m * L**4) / (E * I)) # Fundamental period

def analyze_case(
    solver_name: str,
    solver: SolverSpec,
    mesh_factor: float,
    mesh_size: float,
    counts: Tuple[int, int, int],
    args: argparse.Namespace,
) -> BenchmarkRow:
    nx, ny, nz = counts
    build_solid_bar_model(nx, ny, nz)
    apply_load_pattern(args.load)
    
    # Static analysis
    configure_static_analysis(solver, args.num_steps, args.tol, args.max_iter)
    start_static = time.perf_counter()
    status_static = ops.analyze(args.num_steps)
    runtime_static = time.perf_counter() - start_static

    # Transient analysis
    ops.wipeAnalysis()
    ops.remove("loadPattern", 1)
    configure_transient_analysis(solver, args.num_steps, args.tol, args.max_iter)
    T1 = estimate_fundamental_period(BAR_THICKNESS, BAR_HEIGHT, BAR_LENGTH, ELASTIC_MODULUS, STEEL_DENSITY)
    dt = T1 / args.num_steps
    start_transient = time.perf_counter()
    status_transient = ops.analyze(args.num_steps, dt)
    runtime_transient = time.perf_counter() - start_transient

    try:
        neq = ops.systemSize()
    except AttributeError:
        neq = -1

    # Use transient analysis status for displacement check (final state)
    displacement = None
    if status_transient == 0:
        for node in ops.getNodeTags():
            x = ops.nodeCoord(node, 1)
            y = ops.nodeCoord(node, 2)
            z = ops.nodeCoord(node, 3)
            if np.isclose(
                [x, y, z],
                [BAR_LENGTH, BAR_THICKNESS / 2.0, BAR_HEIGHT / 2.0],
                atol=1e-9,
            ).all():
                dx = ops.nodeDisp(node, 1)
                dy = ops.nodeDisp(node, 2)
                dz = ops.nodeDisp(node, 3)
                displacement = (dx, dy, dz)
                break

    if displacement is not None:
        dx, dy, dz = displacement
    else:
        dx = dy = dz = None

    return BenchmarkRow(
        mesh_factor,
        mesh_size,
        solver_name,
        neq,
        status_static,
        status_transient,
        runtime_static,
        runtime_transient,
        dx,
        dy,
        dz,
    )

# -----------------------------------------------------------------------------
# Benchmark parameters
# -----------------------------------------------------------------------------
# Mesh refinements requested by the user: c = t / factor
DEFAULT_MESH_FACTORS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)

# Default solver list (in the requested order).
DEFAULT_SOLVERS = (
    "BandSPD",
    "UmfPack",
    "cuDSS",
    "AmgX-PCG-JacobiL1",
    "CuPyCG",
    "CuPyCG-Jacobi",
    "NvMathDirect",
)

def build_solver_catalog() -> Dict[str, SolverSpec]:
    catalog: Dict[str, SolverSpec] = {
        "ProfileSPD": SolverSpec(lambda: ops.system("ProfileSPD"), numberer="RCM"),
        "BandSPD": SolverSpec(lambda: ops.system("BandSPD"), numberer="RCM"),
        "SparseSPD": SolverSpec(lambda: ops.system("SparseSPD"), numberer="Plain"),
        "BandGeneral": SolverSpec(lambda: ops.system("BandGeneral"), numberer="RCM"),
        "SparseGeneral": SolverSpec(lambda: ops.system("SparseGeneral"), numberer="Plain"),
        "UmfPack": SolverSpec(lambda: ops.system("UmfPack"), numberer="Plain"),
    }

    catalog["cuDSS"] = SolverSpec(
        lambda: ops.system("cuDSS"),
        numberer="RCM",
    )

    import json
    amgx_config = json.dumps({
        "config_version": 2,	
        "solver": "PCG",
        "max_iters": 10000,
        "convergence": "COMBINED_REL_INI_ABS",
        "tolerance": 1e-12,
        "alt_rel_tolerance": 1e-7,
        "use_scalar_norm": 1,
        "norm": "L2",
        "monitor_residual": 1,
        "preconditioner": {"solver": "JACOBI_L1", "max_iters": 1},
    })
    catalog["AmgX-PCG-JacobiL1"] = SolverSpec(
        lambda: ops.system(
            "AmgX",
            dict(
                configOptions=amgx_config,
                verbose=False,
                blockSize=1,
            ),
        ),
        numberer="RCM",
    )

    def _setup_cupy_cg_none():
        try:
            from pythonSolvers.cupy_sparse_solver import CuPySparseCGSolver  # type: ignore
        except ImportError as exc:
            raise RuntimeError("CuPy solver requested but solvers/cupy_sparse_solver.py is missing") from exc
        solver = CuPySparseCGSolver(rtol=1.0e-7, atol=1.0e-12, maxiter=None, preconditioner_type=None)
        ops.system("PythonSparse", {"solver": solver, "scheme": "CSR"})

    def _setup_cupy_cg_jacobi():
        try:
            from pythonSolvers.cupy_sparse_solver import CuPySparseCGSolver  # type: ignore
        except ImportError as exc:
            raise RuntimeError("CuPy solver requested but solvers/cupy_sparse_solver.py is missing") from exc
        solver = CuPySparseCGSolver(rtol=1.0e-7, atol=1.0e-12, maxiter=None, preconditioner_type="jacobi")
        ops.system("PythonSparse", {"solver": solver, "scheme": "CSR"})

    def _setup_cupy_direct():
        try:
            from pythonSolvers.cupy_sparse_solver import CuPySparseDirectSolver  # type: ignore
        except ImportError as exc:
            raise RuntimeError("CuPy solver requested but solvers/cupy_sparse_solver.py is missing") from exc

        solver = CuPySparseDirectSolver()
        ops.system("PythonSparse", {"solver": solver, "scheme": "CSC"})

    catalog["CuPyDirect"] = SolverSpec(_setup_cupy_direct, numberer="Plain")
    catalog["CuPyCG"] = SolverSpec(_setup_cupy_cg_none, numberer="RCM")  # Default: non-preconditioned CG
    catalog["CuPyCG-Jacobi"] = SolverSpec(_setup_cupy_cg_jacobi, numberer="RCM")

    def _setup_scipy_direct():
        try:
            from pythonSolvers.scipy_sparse_solver import SciPySparseDirectSolver  # type: ignore
        except ImportError as exc:
            raise RuntimeError("SciPy direct solver requested but solvers/scipy_sparse_solver.py is unavailable") from exc

        solver = SciPySparseDirectSolver()
        ops.system("PythonSparse", {"solver": solver, "scheme": "CSC"})

    def _setup_scipy_umfpack():
        try:
            from pythonSolvers.scipy_sparse_solver import SciPySparseDirectSolver  # type: ignore
        except ImportError as exc:
            raise RuntimeError("SciPy UMFPACK solver requested but solvers/scipy_sparse_solver.py is unavailable") from exc

        solver = SciPySparseDirectSolver(use_umfpack=True)
        ops.system("PythonSparse", {"solver": solver, "scheme": "CSC"})

    def _setup_scipy_cg():
        try:
            from pythonSolvers.scipy_sparse_solver import SciPySparseCGSolver  # type: ignore
        except ImportError as exc:
            raise RuntimeError("SciPy CG solver requested but solvers/scipy_sparse_solver.py is unavailable") from exc

        solver = SciPySparseCGSolver(rtol=1.0e-7, atol=1.0e-12, maxiter=10000, preconditioner_type="jacobi")
        ops.system("PythonSparse", {"solver": solver, "scheme": "CSR"})

    catalog["SciPyDirect"] = SolverSpec(_setup_scipy_direct, numberer="Plain")
    catalog["SciPyUmfpack"] = SolverSpec(_setup_scipy_umfpack, numberer="Plain")
    catalog["SciPyCG"] = SolverSpec(_setup_scipy_cg, numberer="RCM")

    def _setup_nvmath_direct():
        try:
            from pythonSolvers.nvmath_sparse_solver import NvMathSparseDirectSolver  # type: ignore
            import cupyx.scipy.sparse as cpsparse
        except ImportError as exc:
            raise RuntimeError("NvMath solver requested but solvers/nvmath_sparse_solver.py is unavailable") from exc

        solver = NvMathSparseDirectSolver(sp_module=cpsparse)
        ops.system("PythonSparse", {"solver": solver, "scheme": "CSR"})

    def _setup_nvmath_direct_hybrid():
        try:
            from pythonSolvers.nvmath_sparse_solver import NvMathSparseDirectSolver  # type: ignore
            import cupyx.scipy.sparse as cpsparse
            import nvmath
        except ImportError as exc:
            raise RuntimeError("NvMath hybrid solver requested but solvers/nvmath_sparse_solver.py is unavailable") from exc

        execution = nvmath.sparse.advanced.ExecutionHybrid(num_threads=8)
        solver = NvMathSparseDirectSolver(sp_module=cpsparse, execution=execution)
        ops.system("PythonSparse", {"solver": solver, "scheme": "CSR"})

    catalog["NvMathDirect"] = SolverSpec(_setup_nvmath_direct, numberer="Plain")
    catalog["NvMathDirectHybrid"] = SolverSpec(_setup_nvmath_direct_hybrid, numberer="Plain")
    return catalog

# -----------------------------------------------------------------------------
# Utility classes and functions
# -----------------------------------------------------------------------------
CSV_HEADER = (
    "solver",
    "mesh_factor",
    "mesh_c",
    "num_equations",
    "status_static",
    "status_transient",
    "time_static_seconds",
    "time_transient_seconds",
    "displacement_x",
    "displacement_y",
    "displacement_z",
)


@dataclass
class BenchmarkRow:
    mesh_factor: float
    mesh_size: float
    solver_label: str
    neq: int
    status_static: int
    status_transient: int
    runtime_static: float
    runtime_transient: float
    displacement_x: Optional[float] = None
    displacement_y: Optional[float] = None
    displacement_z: Optional[float] = None


@dataclass
class SolverSpec:
    """Encapsulates how to configure ``ops.system`` for a solver."""

    setup: Callable[[], None]
    numberer: str = "RCM"

def counts_from_mesh_factor(factor: float) -> Tuple[float, Tuple[int, int, int]]:
    """Return mesh size and brick counts for a given refinement factor."""

    mesh_size = BAR_THICKNESS / factor

    def count(dim: float) -> int:
        return max(1, int(math.ceil(dim / mesh_size)))

    return mesh_size, (count(BAR_LENGTH), count(BAR_THICKNESS), count(BAR_HEIGHT))


def _resolve_csv_path(raw: Optional[Path]) -> Optional[Path]:
    """Resolve user input into a concrete CSV destination."""

    if raw is None:
        return None
    if raw.is_absolute():
        return raw

    parts = raw.parts
    if len(parts) == 1 and parts[0] not in (".", ".."):
        return SCRIPT_DIR / raw.name
    return Path.cwd() / raw


class CSVLogger:
    """Thin wrapper that writes benchmark rows as they are produced."""

    def __init__(self, target: Optional[Path]):
        self.path = _resolve_csv_path(target)
        self._file = None
        self._writer = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(CSV_HEADER)
            self._file.flush()

    def write(self, row: BenchmarkRow) -> None:
        if self._writer is None:
            return
        if row.displacement_x is not None:
            dx = row.displacement_x
            dy = row.displacement_y
            dz = row.displacement_z
        else:
            dx = dy = dz = ""
        self._writer.writerow(
            (
                row.solver_label,
                row.mesh_factor,
                row.mesh_size,
                row.neq,
                row.status_static,
                row.status_transient,
                row.runtime_static,
                row.runtime_transient,
                dx,
                dy,
                dz,
            )
        )
        assert self._file is not None
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def target(self) -> Optional[Path]:
        return self.path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 3D solid bar tutorial model with multiple mesh refinements "
            "and compare linear equation solver runtime."
        )
    )
    parser.add_argument(
        "--mesh-factors",
        type=float,
        nargs="+",
        default=list(DEFAULT_MESH_FACTORS),
        help="Mesh spacing denominators so that c = thickness / factor. "
        "Defaults to %(default)s.",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=list(DEFAULT_SOLVERS),
        help=(
            "Subset/order of solvers to run. "
            "Choices: BandSPD, BandGeneral, UmfPack, cuDSS, AmgX, "
            "CuPyCG, CuPyCG-None, CuPyCG-Jacobi, CuPyDirect, "
            "SciPyDirect, SciPyUmfpack, SciPyCG, NvMathDirect, NvMathDirectHybrid."
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="Number of pseudo time steps for the static analysis (default: %(default)s).",
    )
    parser.add_argument(
        "--load",
        type=float,
        default=25.0,
        help="Magnitude of the vertical load applied to each node on the free face.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-7,
        help="Test tolerance for the NormUnbalance criterion.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="Maximum test iterations per step.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write a CSV summary of the runs.",
    )
    return parser.parse_args()


def select_solvers(requested: Sequence[str], catalog: Dict[str, SolverSpec]) -> List[Tuple[str, SolverSpec]]:
    chosen: List[Tuple[str, SolverSpec]] = []
    for name in requested:
        spec = catalog.get(name)
        if spec is None:
            print(f"[warning] Solver '{name}' is not recognized; skipping.")
            continue
        chosen.append((name, spec))

    if not chosen:
        raise SystemExit("No valid solvers requested.")
    return chosen


def print_header(args: argparse.Namespace, width: int) -> None:
    print("\n=== Solid bar solver benchmark ===")
    print(f"Mesh factors: {args.mesh_factors}")
    print(f"Number of steps: {args.num_steps}, load per node: {args.load}")
    print(
        f"{'Solver':<{width}}"
        f"{'mesh factor':>12}{'mesh c':>10}{'neq':>10}"
        f"{'stat':>6}{'tran':>6}"
        f"{'time_static':>12}{'time_transient':>15}"
        f"{'displ (x)':>15}{'displ (y)':>15}{'displ (z)':>15}"
    )


def report_row(row: BenchmarkRow, width: int, note: Optional[str] = None) -> None:
    display = row.solver_label
    if row.displacement_x is not None:
        dx = row.displacement_x
        dy = row.displacement_y
        dz = row.displacement_z
        line = (
            f"{display:<{width}}"
            f"{row.mesh_factor:>12.2f}{row.mesh_size:>10.4f}"
            f"{row.neq:>10}"
            f"{row.status_static:>6}{row.status_transient:>6}"
            f"{row.runtime_static:>12.4f}{row.runtime_transient:>15.4f}"
            f"{dx:>15.6e}{dy:>15.6e}{dz:>15.6e}"
        )
    else:
        line = (
            f"{display:<{width}}"
            f"{row.mesh_factor:>12.2f}{row.mesh_size:>10.4f}"
            f"{row.neq:>10}"
            f"{row.status_static:>6}{row.status_transient:>6}"
            f"{row.runtime_static:>12.4f}{row.runtime_transient:>15.4f}"
            f"{'N/A':>15}{'N/A':>15}{'N/A':>15}"
        )
    if note:
        line = f"{line}  <-- {note}"
    print(line)


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    catalog = build_solver_catalog()
    solvers = select_solvers(args.solvers, catalog)

    solver_col_width = 22
    print_header(args, solver_col_width)
    csv_logger = CSVLogger(args.csv)

    try:
        for factor in args.mesh_factors:
            mesh_size, counts = counts_from_mesh_factor(factor)
            for solver_name, solver in solvers:
                try:
                    row = analyze_case(solver_name, solver, factor, mesh_size, counts, args)
                    report_row(row, solver_col_width)
                except Exception as exc:  # pylint: disable=broad-except
                    row = BenchmarkRow(
                        factor, mesh_size, solver_name, -1, -999, -999,
                        float("nan"), float("nan")
                    )
                    report_row(row, solver_col_width, note=str(exc))
                finally:
                    csv_logger.write(row)
    finally:
        csv_logger.close()
        if csv_logger.target is not None:
            print(f"\nWrote CSV summary to {csv_logger.target}")


if __name__ == "__main__":
    main()
