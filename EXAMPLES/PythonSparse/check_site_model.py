"""Validate the bench_site.py soil column before using it in benchmarks.

1. Elastic gravity settlement at the top vs the 1D constrained-compression value
   (laminar sides => no lateral strain): u_top = sum over layers of
   integral(sigma_v / M) dz, with M = E (1 - nu) / ((1 + nu) (1 - 2 nu)).
2. Compiled UmfPack vs a hand-written PythonSparse solver: same displacements
   through gravity and the dynamic window.
3. The Tcl twin through OpenSees.exe (``--tcl-exe``) gives the same numbers.
4. The PIMY variant: runs, iterations per step, tangent asymmetry.

    python check_site_model.py [--tcl-exe path\\to\\OpenSees.exe]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

if os.environ.get("OPENSEESPY_PATH"):
    sys.path.insert(0, os.environ["OPENSEESPY_PATH"])
    import opensees as ops  # noqa: E402
else:
    import openseespy.opensees as ops  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_site as bs  # noqa: E402


class Probe:
    """Hand-written direct solver that also records tangent asymmetry."""

    def __init__(self):
        self.lu = None
        self.asym = 0.0
        self.factorizations = 0

    def solve(self, **kw):
        n, nnz = kw["num_eqn"], kw["nnz"]
        if kw["matrix_status"] != "UNCHANGED" or self.lu is None:
            A = sp.csr_matrix((np.frombuffer(kw["values"], np.float64, nnz).copy(),
                               np.frombuffer(kw["indices"], np.int32, nnz),
                               np.frombuffer(kw["index_ptr"], np.int32, n + 1)), shape=(n, n))
            self.asym = max(self.asym, abs(A - A.T).max() / abs(A).max())
            self.lu = spla.splu(A.tocsc())
            self.factorizations += 1
        np.frombuffer(kw["x"], np.float64, n)[:] = self.lu.solve(
            np.frombuffer(kw["rhs"], np.float64, n))
        return 0


def top_center_node(refine):
    xs, ys, zs, _ = bs.grid(refine)
    nx, ny = len(xs) - 1, len(ys) - 1
    i, j, k = nx // 2, ny // 2, len(zs) - 1
    return 1 + i + (nx + 1) * (j + (ny + 1) * k)


def analytic_settlement():
    props = bs.layer_props("elastic")
    z_top = bs.ZMIN + sum(L[4] for L in bs.LAYERS)
    u = 0.0
    z0 = bs.ZMIN
    gammas = [p[2] * 9.81 for p in props]  # kN/m^3 (rho in t/m^3)
    tops = []
    for L in bs.LAYERS:
        tops.append(z0 + L[4])
        z0 += L[4]
    for k, (E, nu, _rho, _G) in enumerate(props):
        M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        zb, zt = tops[k] - bs.LAYERS[k][4], tops[k]
        # overburden at the layer top from the layers above
        above = sum(gammas[m] * bs.LAYERS[m][4] for m in range(k + 1, len(props)))
        g = gammas[k]
        # sigma_v(z) = above + g (zt - z); integral over [zb, zt] / M
        u += (above * (zt - zb) + g * (zt - zb) ** 2 / 2.0) / M
    assert abs(tops[-1] - z_top) < 1e-9
    return -u


def run_py(system_args, material="elastic", refine=1, dyn_steps=50):
    bs.build(ops, refine, material)
    node = top_center_node(refine)
    out = bs.analysis(ops, system_args, ("Newton",), material, dyn_steps=dyn_steps)
    disp = ops.nodeDisp(node)
    neq = ops.systemSize()
    ops.wipe()
    return out, disp, neq


def run_tcl(exe, system_line, material="elastic", refine=1, dyn_steps=50):
    node = top_center_node(refine)
    script = (bs.tcl(refine, material)
              + bs.tcl_analysis(system_line, "Newton", material, dyn_steps)
              + f'puts "DISP [nodeDisp {node}]"\n')
    with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        res = subprocess.run([exe, path], capture_output=True, text=True, timeout=3600)
    finally:
        os.unlink(path)
    text = res.stdout + res.stderr
    disp = [float(v) for line in text.splitlines() if line.startswith("DISP")
            for v in line.split()[1:]]
    results = [line for line in text.splitlines() if line.startswith("RESULT")]
    return results, disp, text[-600:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcl-exe", default=None)
    args = ap.parse_args()
    ok_all = True

    # 1. gravity settlement (elastic), gravity phase only
    bs.build(ops, 1, "elastic")
    node = top_center_node(1)
    probe = Probe()
    out = bs.analysis(ops, ("PythonSparse", {"solver": probe, "scheme": "CSR"}), ("Newton",),
                      "elastic", dyn_steps=0)
    uz = ops.nodeDisp(node, 3)
    ua = analytic_settlement()
    neq = ops.systemSize()
    ops.wipe()
    good = abs(uz - ua) <= 0.02 * abs(ua)
    ok_all &= good
    print(f"[1] gravity settlement: FE {uz:.6e} m, 1D analytic {ua:.6e} m "
          f"({100 * (uz - ua) / ua:+.2f}%), neq={neq}, gravity iters={out['gravity'][1]} "
          f"-> {'ok' if good else 'FAIL'}")

    # 2. compiled vs PythonSparse (elastic, gravity + dynamic)
    o1, d1, neq = run_py(("UmfPack",))
    p2 = Probe()
    o2, d2, _ = run_py(("PythonSparse", {"solver": p2, "scheme": "CSR"}))
    good = all(v[0] == 0 for v in o1.values()) and np.allclose(d1, d2, rtol=1e-8, atol=1e-12)
    ok_all &= good
    print(f"[2] UmfPack vs PythonSparse: top disp {np.array(d1)} vs {np.array(d2)}; "
          f"dynamic iters {o1['dynamic'][1][:5]}...; asym {p2.asym:.1e} "
          f"-> {'ok' if good else 'FAIL'}")

    # 3. Tcl twin
    if args.tcl_exe:
        results, d3, tail = run_tcl(args.tcl_exe, "system UmfPack")
        good = len(d3) == 3 and np.allclose(d1, d3, rtol=1e-8, atol=1e-12)
        ok_all &= good
        print(f"[3] Tcl OpenSees.exe: top disp {np.array(d3)} ({len(results)} phases) "
              f"-> {'ok' if good else 'FAIL'}" + ("" if good else "\n" + tail))

    # 4. PIMY variant
    p4 = Probe()
    o4, d4, neq = run_py(("PythonSparse", {"solver": p4, "scheme": "CSR"}), material="pimy",
                         dyn_steps=20)
    phases = {k: (v[0], sum(v[1]), max(v[1]) if v[1] else 0) for k, v in o4.items()}
    good = all(v[0] == 0 for v in phases.values())
    ok_all &= good
    print(f"[4] PIMY: phases (ok, total iters, max iters/step) {phases}; "
          f"factorizations {p4.factorizations}; asym {p4.asym:.1e}; top disp {np.array(d4)} "
          f"-> {'ok' if good else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
