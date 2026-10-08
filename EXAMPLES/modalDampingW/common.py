"""Shared helpers for modal damping example drivers (not model geometry)."""

import time
from pathlib import Path

import numpy as np

MODAL_CASES = (
    ("modalDampingQ", "BandGeneral"),
    ("modalDamping", "BandGeneral"),
    ("modalDampingW", "BandGeneral"),
    ("modalDamping", "FullGeneral"),
)


def apply_modal_damping(ops, case_tag: str, *zetas):
    if case_tag == "modalDampingQ":
        ops.modalDampingQ(*zetas)
    elif case_tag == "modalDampingW":
        ops.modalDamping("-woodbury", *zetas)
    elif case_tag == "modalDamping":
        ops.modalDamping("-legacy", *zetas)
    else:
        raise ValueError(f"unknown modal damping case: {case_tag}")


def load_opensees():
    repo = Path(__file__).resolve().parents[2]
    for sub in ("build/Release", "build"):
        folder = repo / sub
        if (folder / "opensees.so").is_file():
            import sys

            sys.path.insert(0, str(folder))
            import opensees as ops

            return ops
    import openseespy.opensees as ops

    return ops


def reset_output_dirs(example_dir: Path):
    for sub in ("results", "logs", "figures"):
        d = example_dir / sub
        if d.exists():
            import shutil

            shutil.rmtree(d)
        d.mkdir(parents=True)
    return example_dir / "results", example_dir / "logs"


def count_record_lines(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def excitation_steps(gm_path: Path, dt: float, t_exc: float | None = None, t_free_frac: float = 0.0):
    """Return (n_exc, n_free). Use t_exc seconds or full file length; optional free tail as fraction of excitation."""
    n_exc = count_record_lines(gm_path) if t_exc is None else int(t_exc / dt)
    n_free = int(t_free_frac * n_exc) if t_free_frac else 0
    return n_exc, n_free


def add_path_gm(ops, gm_path: Path, dt: float, factor: float, ts_tag: int = 1, pattern_tag: int = 1):
    ops.timeSeries("Path", ts_tag, "-filePath", str(gm_path), "-dt", dt, "-factor", factor)
    ops.pattern("UniformExcitation", pattern_tag, 1, "-accel", ts_tag)


def record_nodes(ops, tag: str, nodes, dof: int, results_dir: Path):
    for resp in ("disp", "vel", "accel"):
        ops.recorder("Node", "-file", str(results_dir / f"{tag}_{resp}.out"), "-time", "-node", *nodes, "-dof", dof, resp)


def _test_iter(ops):
    x = ops.testIter()
    return int(x[0] if isinstance(x, (list, tuple)) else x)


def _test_norms(ops):
    x = ops.testNorms()
    return list(x) if isinstance(x, (list, tuple)) else [float(x)]


def analyze_transient(
    ops,
    tag: str,
    dt: float,
    n_exc: int,
    n_free: int,
    results_dir: Path,
    *,
    pattern_tag: int = 1,
):
    """Step transient analysis; remove load pattern after excitation; write convergence file."""
    n_steps = n_exc + n_free
    t0 = time.perf_counter()
    iters_per_step = []
    tol_per_step = []
    for step in range(1, n_steps + 1):
        if step == n_exc + 1:
            ops.remove("loadPattern", pattern_tag)
        ok = ops.analyze(1, dt)
        if ok < 0:
            elapsed = time.perf_counter() - t0
            print(f"  {tag}: analyze failed at step {step} ({elapsed:.2f} s)", flush=True)
            break
        iters = _test_iter(ops)
        norms = _test_norms(ops)
        iters_per_step.append(iters)
        tol_per_step.append(
            norms[iters - 1] if iters and iters <= len(norms) else (norms[-1] if norms else float("nan"))
        )
    n = len(iters_per_step)
    t = (np.arange(1, n + 1) * dt).reshape(-1, 1)
    np.savetxt(
        results_dir / f"{tag}_convergence.dat",
        np.hstack([t, np.array(iters_per_step).reshape(-1, 1), np.array(tol_per_step).reshape(-1, 1)]),
        header="time iters final_norm",
        comments="# ",
    )
    if len(iters_per_step) == n_steps:
        elapsed = time.perf_counter() - t0
        print(f"  {tag}: successful ({elapsed:.2f} s)", flush=True)
