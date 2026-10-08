#!/usr/bin/env python3
# Four-story shear frame (modalDampingW example) — https://openseesdigital.com/2019/09/12/be-careful-with-modal-damping/
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
k = 610.0
m = 1.0352
uy = 0.02
b = 0.01
zeta = 0.02
nmodes = 1

# --- ground motion ---
dt = 0.02
g = 9.81
tabas_scale = 1.0
gm = HERE / "tabasFN.txt"
if not gm.is_file():
    sys.exit(f"ERROR: missing {gm}")
n_exc, n_free = excitation_steps(gm, dt, t_exc=10.0, t_free_frac=1.0)

results, logs = reset_output_dirs(HERE)

for modal_cmd, system in MODAL_CASES:
    tag = f"{modal_cmd}_{system}"
    ops.logFile(str(logs / f"opensees_{tag}.log"), "-noEcho")

    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    nodes = model.build(ops, k, m, uy, b)

    ops.eigen(nmodes)
    apply_modal_damping(ops, modal_cmd, *[zeta] * nmodes)

    add_path_gm(ops, gm, dt, g * tabas_scale)
    ops.numberer("Plain")
    ops.system(system)
    ops.test("NormUnbalance", 1e-8, 10, 1)
    ops.analysis("Transient", "-noWarnings")

    record_nodes(ops, tag, nodes, 1, results)
    analyze_transient(ops, tag, dt, n_exc, n_free, results)

sys.exit(plot_results(HERE))
