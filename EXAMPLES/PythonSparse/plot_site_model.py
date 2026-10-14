"""Confidence figures for the bench_site.py soil column (not part of the benchmark).

Runs the column under a full velocity record with the compiled UmfPack system
and plots:

1. schematic of the mesh, layers, laminar sides, base dashpot and input force;
2. absolute acceleration at the top and at the base (elastic and PIMY);
3. lateral displacement profile relative to the base at peak top response;
4. tau_zx - gamma_zx hysteresis in the soft and medium clay (PIMY);
5. check: the one-element-wide column (used for 1-4) matches the benchmark's
   10 x 10 plan mesh (laminar sides make the response independent of it).

    python plot_site_model.py --motion velocityHistory.out --out figures/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

if os.environ.get("OPENSEESPY_PATH"):
    sys.path.insert(0, os.environ["OPENSEESPY_PATH"])
    import opensees as ops  # noqa: E402
else:
    import openseespy.opensees as ops  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_site as bs  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"          # reference palette slots 1-2
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]  # light -> dark

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "legend.frameon": False, "figure.dpi": 150,
})


def node_id(nx, ny, i, j, k):
    return 1 + i + (nx + 1) * (j + (ny + 1) * k)


def run(material, motion, plan, duration, dt=0.001, hyst_cells=(), start_time=0.0):
    """Full-record run; returns time, top/base absolute accel, profile, hysteresis."""
    bs.build(ops, 1, material, plan=plan)
    xs, ys, zs, cells = bs.grid(1, plan)
    nx, ny = len(xs) - 1, len(ys) - 1
    top = node_id(nx, ny, nx // 2, ny // 2, len(zs) - 1)
    base = bs.base_master()
    column = [node_id(nx, ny, 0, 0, k) for k in range(len(zs))]
    eles = {name: 1 + (nx * ny) * k for name, k in hyst_cells}   # one element per level
    rec = {"t": [], "a_top": [], "a_base": [], "prof": [], "hyst": {n: [] for n in eles}}

    def on_step():
        rec["t"].append(ops.getTime())
        rec["a_top"].append(ops.nodeAccel(top, 1))
        rec["a_base"].append(ops.nodeAccel(base, 1))
        rec["prof"].append([ops.nodeDisp(n, 1) for n in column])
        for name, e in eles.items():
            s = ops.eleResponse(e, "material", 1, "stress")
            g = ops.eleResponse(e, "material", 1, "strain")
            rec["hyst"][name].append((g[5], s[5]))

    out = bs.analysis(ops, ("UmfPack",), ("Newton",), material, motion=motion,
                      dyn_steps=int(round(duration / dt)), start_time=start_time, dyn_dt=dt,
                      on_step=on_step)
    ops.wipe()
    ok = {k: v[0] for k, v in out.items()}
    rec = {k: (np.array(v) if k != "hyst" else {n: np.array(h) for n, h in v.items()})
           for k, v in rec.items()}
    rec["zs"] = np.array(zs)
    rec["ok"] = ok
    rec["iters"] = {k: v[1] for k, v in out.items()}
    return rec


def schematic(path):
    xs, ys, zs, cells = bs.grid(1)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    ax.grid(False)
    Gs = sorted({L[1] for L in bs.LAYERS})
    z0 = bs.ZMIN
    for k, L in enumerate(bs.LAYERS):
        color = RAMP[1 + 2 * Gs.index(L[1])]          # equal stiffness -> equal shade
        ax.add_patch(Rectangle((bs.XMIN, z0), bs.XMAX - bs.XMIN, L[4], color=color, lw=0))
        _rho, G_c, _B, c_u, _g = bs.PIMY_CLAYS[bs.PIMY_PROFILE[k]]
        ax.text(bs.XMAX + 0.5, z0 + L[4] / 2,
                f"{L[0]} (dz = {L[5]} m)\n"
                f"  elastic: G = {L[1] / 1e6:.0f} MPa\n"
                f"  PIMY: {bs.PIMY_PROFILE[k]} clay, G = {G_c / 1e3:.0f} MPa, c = {c_u:.0f} kPa",
                va="center", fontsize=7.5, color=INK2)
        z0 += L[4]
    for x in xs:
        ax.plot([x, x], [zs[0], zs[-1]], color="white", lw=0.4)
    for z in zs:
        ax.plot([xs[0], xs[-1]], [z, z], color="white", lw=0.4)
    for x in (xs[0], xs[-1]):  # laminar sides
        ax.plot([x, x], [zs[1], zs[-1]], color=INK, lw=1.6)
    ax.text(xs[0] - 0.3, -8, "laminar sides (equalDOF 1-3 per height)", rotation=90,
            ha="right", va="center", fontsize=7.5, color=INK)
    # base: vertical rollers, dashpot to a fixed node, input force
    ax.plot([xs[0], xs[-1]], [zs[0], zs[0]], color=INK, lw=2.0)
    for x in xs[::2]:
        ax.plot(x, zs[0] - 0.25, marker="o", ms=3, color=INK, mfc="white")
    ax.plot([xs[0] - 3.2, xs[0] - 0.4], [zs[0], zs[0]], color=INK, lw=1)
    ax.add_patch(Rectangle((xs[0] - 2.4, zs[0] - 0.35), 0.9, 0.7, fill=False, ec=INK, lw=1))
    ax.plot([xs[0] - 3.2] * 2, [zs[0] - 0.6, zs[0] + 0.6], color=INK, lw=2)
    ax.text(xs[0] - 0.5, zs[0] + 0.9,
            f"Lysmer dashpot (x, y)\nc = ρ·Vs·A = {bs.dashpot_coefficient():.3g} kN·s/m",
            ha="right", va="bottom", fontsize=7.5, color=INK)
    ax.text(0, zs[0] - 0.8, "base: fixed vertically, nodes tied horizontally", ha="center",
            va="top", fontsize=7.5, color=INK2)
    ax.add_patch(FancyArrowPatch((xs[0], zs[0] - 2.3), (xs[0] + 3.5, zs[0] - 2.3),
                                 arrowstyle="-|>", mutation_scale=12, color=ORANGE, lw=1.6))
    ax.text(xs[0] + 3.8, zs[0] - 2.3,
            f"F(t) = {bs.INPUT_FACTOR:g}·c·v(t) at the base "
            f"(rock Vs = {bs.ROCK_VS:.0f} m/s, ρ = {bs.ROCK_RHO} t/m³)",
            va="center", color=INK, fontsize=7.5)
    ax.set_xlim(bs.XMIN - 7.5, bs.XMAX + 15.5)
    ax.set_ylim(zs[0] - 3.2, zs[-1] + 0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    nx, ny = len(xs) - 1, len(ys) - 1
    ax.set_title(f"Soil column 10 m × 10 m × 18 m, {nx}×{ny}×{len(zs) - 1} stdBrick "
                 "(benchmark mesh), x–z section", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def histories(path, el, pimy):
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.0), sharex=True)
    for ax, rec, title in ((axes[0], el, "Elastic layers (Femora)"),
                           (axes[1], pimy, "PIMY clay (wiki values)")):
        g = 9.81
        ax.plot(rec["t"], rec["a_base"] / g, color=BLUE, label="base")
        ax.plot(rec["t"], rec["a_top"] / g, color=ORANGE, label="top", lw=1.0)
        pk_b, pk_t = np.max(np.abs(rec["a_base"])) / g, np.max(np.abs(rec["a_top"])) / g
        ax.set_title(f"{title}: peak |a| base {pk_b:.2f} g, top {pk_t:.2f} g "
                     f"(ratio {pk_t / pk_b:.2f})", loc="left", fontsize=9, color=INK)
        ax.set_ylabel("absolute accel. (g)")
        ax.legend(loc="upper right", ncols=2)
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def profiles(path, el, pimy):
    fig, ax = plt.subplots(figsize=(4.2, 5.2))
    for rec, color, label in ((el, BLUE, "elastic"), (pimy, ORANGE, "PIMY")):
        rel = rec["prof"] - rec["prof"][:, :1]
        k = int(np.argmax(np.abs(rel[:, -1])))
        ax.plot(rel[k] * 100, rec["zs"], color=color, marker="o", ms=3,
                label=f"{label} (t = {rec['t'][k]:.2f} s)")
    z0 = bs.ZMIN
    for L in bs.LAYERS:
        z0 += L[4]
        ax.axhline(z0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color="#c3c2b7", lw=0.8)
    ax.set_xlabel("lateral displacement relative to base (cm)")
    ax.set_ylabel("z (m)")
    ax.set_title("Deformed shape at peak top displacement", loc="left", fontsize=9, color=INK)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def hysteresis(path, pimy):
    names = list(pimy["hyst"])
    ncol = 2
    nrow = (len(names) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.4, 3.2 * nrow))
    for ax, name in zip(np.ravel(axes), names):
        h = pimy["hyst"][name]
        ax.plot(h[:, 0] * 100, h[:, 1], color=BLUE, lw=0.8)
        ax.set_xlabel("γ_zx (%)")
        ax.set_ylabel("τ_zx (kPa)")
        ax.set_title(name, loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plan_check(path, coarse, fine):
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    n = len(fine["t"])
    ax.plot(coarse["t"][:n], coarse["a_top"][:n] / 9.81, color=BLUE, label="1×1 plan")
    ax.plot(fine["t"], fine["a_top"] / 9.81, color=ORANGE, lw=1.0, ls="--",
            label="10×10 plan (benchmark mesh)")
    err = np.max(np.abs(coarse["a_top"][:n] - fine["a_top"])) / np.max(np.abs(fine["a_top"]))
    ax.set_title(f"Elastic column, top acceleration: 1×1 vs 10×10 plan "
                 f"(max difference {100 * err:.2g}% of peak)", loc="left", fontsize=9, color=INK)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("absolute accel. (g)")
    ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--motion-dt", type=float, default=0.005)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--duration", type=float, default=None, help="default: whole record")
    ap.add_argument("--check-duration", type=float, default=10.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    motion = bs.Motion.from_file(args.motion, args.motion_dt)
    duration = args.duration or (len(motion.values) - 1) * motion.dt

    schematic(out / "1_schematic.png")
    # top layer (soft clay) and middle of the loose layer (medium clay), 1x1 plan
    _, _, zs, cells = bs.grid(1, 1)
    top_cell = len(cells) - 1
    mid_cell = [k for k, c in enumerate(cells) if c == 3][len([c for c in cells if c == 3]) // 2]
    el = run("elastic", motion, 1, duration)
    print("elastic done", flush=True)
    pimy = run("pimy", motion, 1, duration,
               hyst_cells=(("layer 5, soft clay: top element", top_cell),
                           ("layer 4, medium clay: mid-depth", mid_cell),
                           ("layer 4, medium clay: bottom element", cells.index(3)),
                           ("layer 1, stiff clay: base element", 0)))
    histories(out / "2_top_vs_base_accel.png", el, pimy)
    profiles(out / "3_deformed_shape_at_peak.png", el, pimy)
    hysteresis(out / "4_pimy_hysteresis.png", pimy)
    fine = run("elastic", motion, None, args.check_duration)
    err = plan_check(out / "5_plan_mesh_check.png", el, fine)
    for name, rec in (("elastic", el), ("pimy", pimy), ("elastic 10x10", fine)):
        it = rec["iters"]
        print(f"{name}: ok {rec['ok']}; transient iters/step mean "
              f"{np.mean(it['transient']):.2f} max {max(it['transient'])}")
    print(f"plan-mesh check: max |a_top| difference {100 * err:.3g}% of peak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
