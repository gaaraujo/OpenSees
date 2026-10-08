#!/usr/bin/env python3
# Two-story steel MRF (Kolay & Ricles) — modalDampingW example
# Run: python3 main.py

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import model
from plotResults import run as plot_results
from common import (
    MODAL_CASES,
    add_path_gm,
    analyze_transient,
    apply_modal_damping,
    excitation_steps,
    load_opensees,
    record_nodes,
    reset_output_dirs,
)

ops = load_opensees()

# --- model ---
zeta = 0.02
nmodes = 2
record_nodes_list = [17, 16, 8]  # base, floor 1 diaphragm, roof

# --- ground motion ---
dt = 0.01
gm_scale = 3.0
gm = HERE / "RSN960_NORTHR_LOS270.txt"
if not gm.is_file():
    sys.exit(f"ERROR: missing {gm}")
n_exc, n_free = excitation_steps(gm, dt, t_free_frac=1.0)

results, logs = reset_output_dirs(HERE)

for modal_cmd, system in MODAL_CASES:
    tag = f"{modal_cmd}_{system}"
    ops.logFile(str(logs / f"opensees_{tag}.log"), "-noEcho")

    model.create_model(ops)
    model.apply_gravity(ops)
    model.eigen_after_gravity(ops, nmodes)
    apply_modal_damping(ops, modal_cmd, *[zeta] * nmodes)

    # tags 2: gravity static analysis uses timeSeries/pattern 1 in model.apply_gravity
    add_path_gm(ops, gm, dt, model.g * gm_scale, ts_tag=2, pattern_tag=2)

    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system(system)
    ops.test("NormUnbalance", 1e-8, 50, 2)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient", "-noWarnings")

    record_nodes(ops, tag, record_nodes_list, 1, results)
    analyze_transient(ops, tag, dt, n_exc, n_free, results, pattern_tag=2)

sys.exit(plot_results(HERE))
