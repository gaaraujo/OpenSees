#!/usr/bin/env python3
"""PR #1676-style plot: number of equations vs wall time (from DPS sweep CSV/log).

Default: one figure per np (avoids legend clutter). Pass --combined for a single plot.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import EngFormatter

BACKEND_LABEL = {
    "cupy-cg": "CuPyCG",
    "umfpack": "scikit-UMFPACK",
    "nvmath": "nvmath/cuDSS",
    "umfpack-native": "UmfPack",
    "bandspd": "BandSPD",
    "mumps": "Mumps",
    "scipy-eigsh": "SciPyEigsh",
    "genBandArpack": "genBandArpack",
}

BACKEND_ORDER = [
    "CuPyCG",
    "scikit-UMFPACK",
    "nvmath/cuDSS",
    "UmfPack",
    "BandSPD",
    "Mumps",
    "SciPyEigsh",
    "genBandArpack",
]

BACKEND_COLORS = {
    "CuPyCG": "#1f77b4",
    "scikit-UMFPACK": "#d62728",
    "nvmath/cuDSS": "#2ca02c",
    "UmfPack": "#9467bd",
    "BandSPD": "#ff7f0e",
    "Mumps": "#8c564b",
    "SciPyEigsh": "#1f77b4",
    "genBandArpack": "#ff7f0e",
}


def load_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".log":
        return _from_log(path)
    df = pd.read_csv(path)
    if "num_equations" not in df.columns and "est_free_dofs" in df.columns:
        df["num_equations"] = df["est_free_dofs"]
    df = df[df["status"] != -999]
    df = df[(df["num_equations"] > 0) & (df["time_seconds"].notna())]
    if "backend" in df.columns:
        df["series"] = df["backend"].map(lambda b: BACKEND_LABEL.get(b, b))
    else:
        df["series"] = df["solver"].astype(str).str.replace(r"\s*\(np=\d+\)\s*$", "", regex=True)
    return df


def _from_log(path: Path) -> pd.DataFrame:
    rows = []
    np_cur = None
    factor = None
    for line in path.read_text().splitlines():
        m = re.search(r"mpirun -np (\d+)", line)
        if m:
            np_cur = int(m.group(1))
            continue
        m = re.match(r">>> np=(\d+) backend=(\S+) factor=([0-9.]+)", line)
        if m:
            np_cur = int(m.group(1))
            factor = float(m.group(3))
            continue
        m = re.match(r"<<< (\S+) status=0 neq=(\d+) ~dofs=(\d+) time=([0-9.]+)s", line)
        if m and np_cur is not None:
            be = m.group(1)
            rows.append(
                {
                    "backend": be,
                    "series": BACKEND_LABEL.get(be, be),
                    "np": np_cur,
                    "mesh_factor": factor,
                    "num_equations": int(m.group(2)),
                    "time_seconds": float(m.group(4)),
                    "status": 0,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(["backend", "np", "mesh_factor"], keep="last")
    return df


def _plot_one(df: pd.DataFrame, title: str, out: Path, logy: bool) -> None:
    plt.figure(figsize=(9, 5.5))
    names = [n for n in BACKEND_ORDER if (df["series"] == n).any()]
    for name in names:
        sub = df[df["series"] == name].sort_values("num_equations")
        plt.plot(
            sub["num_equations"],
            sub["time_seconds"],
            marker="o",
            label=name,
            color=BACKEND_COLORS.get(name),
            linewidth=2,
            markersize=6,
        )
    plt.xlabel("Number of Equations", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(EngFormatter(unit=""))
    if logy:
        plt.yscale("log")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out.resolve()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="CSV or dps_linear.log")
    ap.add_argument(
        "png",
        type=Path,
        help="output PNG, or prefix when using --per-np (writes <prefix>_npN.png)",
    )
    ap.add_argument("--title", default="DistributedPythonSparse Solver Performance")
    ap.add_argument("--logy", action="store_true")
    ap.add_argument(
        "--per-np",
        action="store_true",
        default=True,
        help="one figure per np (default)",
    )
    ap.add_argument(
        "--combined",
        action="store_true",
        help="single figure with all np series",
    )
    args = ap.parse_args()
    if args.combined:
        args.per_np = False

    df = load_df(args.csv)
    if df.empty:
        print("No plottable rows", file=sys.stderr)
        return 1
    if "np" not in df.columns:
        print("CSV/log needs an 'np' column for --per-np", file=sys.stderr)
        return 1

    if args.per_np:
        stem = args.png.with_suffix("")
        for np_ in sorted(df["np"].unique()):
            sub = df[df["np"] == np_]
            out = Path(f"{stem}_np{int(np_)}.png")
            _plot_one(sub, f"{args.title} (np={int(np_)})", out, args.logy)
    else:
        # combined: label by full solver name including np
        df = df.copy()
        df["series"] = df.apply(
            lambda r: f"{r['series']} (np={int(r['np'])})", axis=1
        )
        _plot_one(df, args.title, args.png, args.logy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
