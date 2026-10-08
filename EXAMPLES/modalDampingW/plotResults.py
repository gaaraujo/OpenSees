#!/usr/bin/env python3
"""Shared plots for modalDampingW examples (four / forty / two story)."""

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# Publication style aligned with 11pt article (serif / Computer Modern mathtext).
FS = 10
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset": "cm",
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS - 1,
        "ytick.labelsize": FS - 1,
        "legend.fontsize": FS - 1,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)

CASES = [
    ("modalDampingQ", "BandGeneral"),
    ("modalDamping", "BandGeneral"),
    ("modalDampingW", "BandGeneral"),
    ("modalDamping", "FullGeneral"),
]
TAGS = [f"{cmd}_{sys}" for cmd, sys in CASES]
REF_TAG = "modalDamping_FullGeneral"
MDW_TAG = "modalDampingW_BandGeneral"
PLOT_TAGS = [REF_TAG] + [t for t in TAGS if t != REF_TAG]
CONV_TAGS = PLOT_TAGS
TEST_TOL = 1e-8
EPS = np.finfo(float).eps

RESP_SYMBOL = {
    "disp": r"u",
    "vel": r"\dot{u}",
    "accel": r"\ddot{u}",
}
RESP_UNITS = {"disp": "m", "vel": "m/s", "accel": "m/s²"}
DRIFT_PEAK_LABEL = r"peak $|\Delta u|$ (m)"


def resp_time_label(resp: str) -> str:
    return rf"${RESP_SYMBOL[resp]}$"


def resp_peak_label(resp: str) -> str:
    sym = RESP_SYMBOL[resp]
    return rf"peak $|{sym}|$ ({RESP_UNITS[resp]})"


def resp_error_label(resp: str) -> str:
    sym = RESP_SYMBOL[resp]
    return rf"$|{sym}|$ error vs. reference"


def resp_fft_error_label(resp: str) -> str:
    sym = RESP_SYMBOL[resp]
    return rf"$|\mathrm{{FFT}}({sym})|$ error vs. reference"


def legend_label(tag: str) -> str:
    """e.g. modalDampingW_BandGeneral -> modalDampingW + BandGeneral"""
    cmd, system = tag.rsplit("_", 1)
    if cmd == "modalDampingW":
        return f"modalDampingW + {system}"
    if cmd == "modalDamping":
        return f"modalDamping (legacy) + {system}"
    return f"{cmd} + {system}"


def style_axes(ax, *, logy: bool = False) -> None:
    ax.grid(True, which="both" if logy else "major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_legend(fig, handles, labels, ncol: int = 2) -> None:
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=ncol,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.2,
    )

# Paul Tol colorblind-friendly palette
COLORS = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

# (linestyle, color, linewidth) per case
STYLES = {
    "modalDamping_FullGeneral": ("-", "k", 1.7),
    "modalDampingW_BandGeneral": ("-.", COLORS["blue"], 1.1),
    "modalDamping_BandGeneral": (":", COLORS["red"], 1.2),
    "modalDampingQ_BandGeneral": ("--", COLORS["green"], 1.1),
}

MARKER_MS = 2.5


def plot_style(tag: str) -> tuple[str, str, float, int]:
    ls, col, lw = STYLES[tag]
    z = 1 if tag == REF_TAG else 2
    return ls, col, lw, z


def load_config(example_dir: Path):
    path = example_dir / "plot_config.py"
    spec = importlib.util.spec_from_file_location("plot_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(example_dir: Path) -> int:
    cfg = load_config(example_dir)
    results = example_dir / "results"
    figures = example_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def recorded_nodes(ncols):
        if ncols == len(cfg.RECORD_NODES):
            return list(cfg.RECORD_NODES[:ncols])
        if ncols == len(cfg.RECORD_NODES) - 1 and cfg.RECORD_NODES[0] == 0:
            return list(cfg.RECORD_NODES[1 : ncols + 1])
        return list(cfg.RECORD_NODES[:ncols])

    def col_index_for(ncols):
        return {node: j for j, node in enumerate(recorded_nodes(ncols))}

    def pick_floors(y):
        cix = col_index_for(y.shape[1])
        cols = [cix[n] for n in cfg.PLOT_FLOOR_NODES if n in cix]
        return y[:, cols]

    def floor_labels_for(ncols):
        # pick_floors() leaves ncols == len(PLOT_FLOOR_NODES); do not map via RECORD_NODES[:ncols]
        if ncols == len(cfg.PLOT_FLOOR_NODES):
            return list(cfg.FLOOR_LABELS)
        cix = col_index_for(ncols)
        return [lab for node, lab in zip(cfg.PLOT_FLOOR_NODES, cfg.FLOOR_LABELS) if node in cix]

    def profile_y_for_columns(ncols):
        """Y positions for peak profiles: use PROFILE_FLOORS, not node tags."""
        nodes = recorded_nodes(ncols)
        if len(cfg.PROFILE_FLOORS) == ncols:
            return np.array(cfg.PROFILE_FLOORS, dtype=float)
        return np.array(nodes, dtype=float)

    def profile_story_bounds(n_stories):
        """Lower and upper floor index for each story (drift is between them)."""
        if len(cfg.PROFILE_FLOORS) >= n_stories + 1:
            y_low = np.array(cfg.PROFILE_FLOORS[:n_stories], dtype=float)
            y_hi = np.array(cfg.PROFILE_FLOORS[1 : n_stories + 1], dtype=float)
            return y_low, y_hi
        rn = recorded_nodes(n_stories + 1)
        return np.array(rn[:-1], dtype=float), np.array(rn[1:], dtype=float)

    def plot_peak_drift_stairs(ax, peak_drift, y_low, y_hi, ls, col, lw, zorder, label=None):
        """Staircase: constant drift across each story, steps at floor levels."""
        xs, ys = [], []
        for k, d in enumerate(peak_drift):
            y0, y1 = float(y_low[k]), float(y_hi[k])
            if k == 0:
                xs.extend([0.0, d])
                ys.extend([y0, y0])
            else:
                xs.extend([peak_drift[k - 1], d])
                ys.extend([y0, y0])
            xs.extend([d, d])
            ys.extend([y0, y1])
        ax.plot(xs, ys, ls=ls, color=col, lw=lw, zorder=zorder, label=label)

    def save_fig(fig, path):
        fig.savefig(path)
        plt.close(fig)
        print(f"Wrote {path}", flush=True)

    def load(tag, resp):
        data = np.loadtxt(results / f"{tag}_{resp}.out")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data[:, 0], data[:, 1:]

    def align_with_ref(t_ref, y_ref, t, y, tag):
        n = min(len(t_ref), len(t), y_ref.shape[0], y.shape[0])
        if n < len(t_ref) or n < len(t):
            print(
                f"WARNING: {tag} vs {REF_TAG} — plotting errors over first {n} steps only",
                flush=True,
            )
        return t_ref[:n], y_ref[:n], t[:n], y[:n]

    def history_axes():
        n = len(cfg.PLOT_FLOOR_NODES)
        if n <= 2:
            return plt.subplots(1, n, figsize=(3.4 * n, 3.2), sharex=True, sharey=True, layout="constrained")
        return plt.subplots(2, 2, figsize=(6.8, 5.0), sharex=True, sharey=True, layout="constrained")

    # --- iteration check ---
    ref_path = results / f"{REF_TAG}_convergence.dat"
    mdw_path = results / f"{MDW_TAG}_convergence.dat"
    if ref_path.is_file() and mdw_path.is_file():
        ref_iters = np.loadtxt(ref_path, comments="#")[:, 1].astype(int)
        mdw_iters = np.loadtxt(mdw_path, comments="#")[:, 1].astype(int)
        n_cmp = min(len(ref_iters), len(mdw_iters))
        lines = [f"NR iterations per step: {MDW_TAG} vs {REF_TAG}\n"]
        mismatch = sum(ref_iters[i] != mdw_iters[i] for i in range(n_cmp))
        if mismatch == 0 and len(ref_iters) == len(mdw_iters):
            lines.append("OK: iteration count matches at every step.")
            print("OK: Woodbury modalDamping iterations match legacy FullGeneral at every step.", flush=True)
        else:
            lines.append(f"Reported {mismatch} step(s) with different iteration counts.")
        (results / "iter_comparison.txt").write_text("\n".join(lines) + "\n")

    # --- convergence ---
    fig, (ax_iter, ax_tol) = plt.subplots(2, 1, figsize=(6.2, 4.2), sharex=True, layout="constrained")
    for tag in CONV_TAGS:
        path = results / f"{tag}_convergence.dat"
        if not path.is_file():
            print(f"ERROR: missing {path}\nRun main.py first.", file=sys.stderr)
            return 1
        data = np.loadtxt(path, comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        t, iters, tol = data[:, 0], data[:, 1].astype(int), data[:, 2]
        ls, col, lw, z = plot_style(tag)
        ax_iter.step(t, iters, where="post", ls=ls, color=col, label=legend_label(tag), lw=lw, zorder=z)
        ax_tol.semilogy(
            t,
            np.maximum(tol, EPS),
            ls,
            color=col,
            label=legend_label(tag),
            lw=lw,
            marker="o",
            ms=MARKER_MS,
            zorder=z,
        )
    ax_tol.axhline(TEST_TOL, color=COLORS["grey"], ls="--", lw=0.9, zorder=0)
    ax_iter.set_ylim(bottom=0)
    ax_iter.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_iter.set_ylabel("NR iterations per step")
    ax_tol.set_ylabel(r"final $\|\mathbf{r}\|$")
    style_axes(ax_iter)
    style_axes(ax_tol, logy=True)
    handles, labels = ax_iter.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=COLORS["grey"], ls="--", lw=0.9))
    labels.append(rf"NormUnbalance tol ($10^{{-8}}$)")
    add_legend(fig, handles, labels)
    fig.supxlabel("time (s)")
    save_fig(fig, figures / "convergence_iters_tol.png")

    # --- peak profiles ---
    profile_panels = (
        ("disp", resp_peak_label("disp")),
        ("vel", resp_peak_label("vel")),
        ("accel", resp_peak_label("accel")),
    )
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.6), layout="constrained")
    ax_disp, ax_vel, ax_accel, ax_drift = axes
    for ax, (resp, xlab) in zip((ax_disp, ax_vel, ax_accel), profile_panels):
        ylim_lo, ylim_hi = None, None
        for tag in PLOT_TAGS:
            path = results / f"{tag}_{resp}.out"
            if not path.is_file():
                print(f"ERROR: missing {path}\nRun main.py first.", file=sys.stderr)
                return 1
            _, y = load(tag, resp)
            n = y.shape[1]
            floors = profile_y_for_columns(n)
            if ylim_lo is None:
                ylim_lo, ylim_hi = floors[0], floors[-1]
            ls, col, lw, z = plot_style(tag)
            peak = np.max(np.abs(y), axis=0)
            ax.plot(peak, floors, ls, color=col, label=legend_label(tag), lw=lw, zorder=z)
        ax.axvline(0.0, color=COLORS["grey"], lw=0.8, alpha=0.5, zorder=0)
        ax.set(xlabel=xlab, title=xlab)
        style_axes(ax)
        ax.set_ylim(ylim_lo, ylim_hi)

    drift_xlab = DRIFT_PEAK_LABEL
    for tag in PLOT_TAGS:
        _, disp = load(tag, "disp")
        drift = disp[:, 1:] - disp[:, :-1]
        n = drift.shape[1]
        y_low, y_hi = profile_story_bounds(n)
        peak_drift = np.max(np.abs(drift), axis=0)
        ls, col, lw, z = plot_style(tag)
        plot_peak_drift_stairs(ax_drift, peak_drift, y_low, y_hi, ls, col, lw, z, label=legend_label(tag))
    ax_drift.axvline(0.0, color=COLORS["grey"], lw=0.8, alpha=0.5, zorder=0)
    ax_drift.set(xlabel=drift_xlab, title=drift_xlab)
    if ylim_lo is not None:
        ax_drift.set_ylim(ylim_lo, ylim_hi)
    style_axes(ax_drift)
    ax_disp.set_ylabel(cfg.PROFILE_YLABEL)
    for ax in (ax_disp, ax_vel, ax_accel, ax_drift):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    add_legend(fig, *ax_disp.get_legend_handles_labels())
    save_fig(fig, figures / "profile_peak_resp.png")

    # --- histories, errors, FFT ---
    panels = (
        ("disp", "floor_disp.png", "floor_disp_error.png", "floor_disp_error_fft.png"),
        ("vel", "floor_vel.png", "floor_vel_error.png", "floor_vel_error_fft.png"),
        ("accel", "floor_accel.png", "floor_accel_error.png", "floor_accel_error_fft.png"),
    )

    def error_fft(t, err):
        n = len(err)
        dt = float(t[1] - t[0]) if n > 1 else 1.0
        spec = np.fft.rfft(err)
        freq = np.fft.rfftfreq(n, dt)
        mag = np.abs(spec) / n
        if n > 2:
            mag[1:-1] *= 2.0
        return freq, mag

    for resp, cmp_fname, err_fname, fft_fname in panels:
        ylab = resp_time_label(resp)
        fig, axes = history_axes()
        axes = np.atleast_1d(axes).ravel()
        for tag in PLOT_TAGS:
            path = results / f"{tag}_{resp}.out"
            if not path.is_file():
                print(f"ERROR: missing {path}\nRun main.py first.", file=sys.stderr)
                return 1
            t, y = load(tag, resp)
            y = pick_floors(y)
            ls, col, lw, z = plot_style(tag)
            for j, ax in enumerate(axes):
                ax.plot(t, y[:, j], ls, color=col, label=legend_label(tag) if j == 0 else None, lw=lw, zorder=z)
        _, y0 = load(REF_TAG, resp)
        flabs = floor_labels_for(y0.shape[1])
        for ax, flab in zip(axes, flabs):
            ax.axhline(0.0, color=COLORS["grey"], lw=0.8, alpha=0.5, zorder=0)
            ax.set_title(flab)
            style_axes(ax)
        fig.supylabel(ylab)
        add_legend(fig, *axes[0].get_legend_handles_labels())
        fig.supxlabel("time (s)")
        save_fig(fig, figures / cmp_fname)

        t_ref, y_ref = load(REF_TAG, resp)
        y_ref = pick_floors(y_ref)
        flabs = floor_labels_for(y_ref.shape[1])
        fig, axes = history_axes()
        axes = np.atleast_1d(axes).ravel()
        for tag in TAGS:
            if tag == REF_TAG:
                continue
            t, y = load(tag, resp)
            y = pick_floors(y)
            t_r, y_r, t_a, y_a = align_with_ref(t_ref, y_ref, t, y, tag)
            ls, col, lw, z = plot_style(tag)
            for j, ax in enumerate(axes):
                ax.semilogy(
                    t_a,
                    np.abs(y_a[:, j] - y_r[:, j]) + EPS,
                    ls,
                    color=col,
                    label=legend_label(tag) if j == 0 else None,
                    lw=lw,
                    zorder=z,
                )
        for ax, flab in zip(axes, flabs):
            ax.set_title(flab)
            style_axes(ax, logy=True)
        fig.supylabel(resp_error_label(resp))
        add_legend(fig, *axes[0].get_legend_handles_labels())
        fig.supxlabel("time (s)")
        save_fig(fig, figures / err_fname)

        _, y_ref = load(REF_TAG, resp)
        y_ref = pick_floors(y_ref)
        flabs = floor_labels_for(y_ref.shape[1])
        fig, axes = history_axes()
        axes = np.atleast_1d(axes).ravel()
        for tag in TAGS:
            if tag == REF_TAG:
                continue
            t, y = load(tag, resp)
            y = pick_floors(y)
            t_r, y_r, t_a, y_a = align_with_ref(t_ref, y_ref, t, y, tag)
            ls, col, lw, z = plot_style(tag)
            for j, ax in enumerate(axes):
                freq, mag = error_fft(t_a, y_a[:, j] - y_r[:, j])
                ax.semilogy(
                    freq[1:],
                    mag[1:] + EPS,
                    ls,
                    color=col,
                    label=legend_label(tag) if j == 0 else None,
                    lw=lw,
                    zorder=z,
                )
        for ax, flab in zip(axes, flabs):
            ax.set_title(flab)
            style_axes(ax, logy=True)
        fig.supylabel(resp_fft_error_label(resp))
        add_legend(fig, *axes[0].get_legend_handles_labels())
        fig.supxlabel("frequency (Hz)")
        save_fig(fig, figures / fft_fname)

    return 0


def main():
    return run(Path(__file__).resolve().parent)


if __name__ == "__main__":
    sys.exit(main())
