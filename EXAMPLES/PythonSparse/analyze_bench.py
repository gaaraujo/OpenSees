"""Summarize bench_conventional / bench_mp_tcl / bench_distributed_site JSONL output.

    python analyze_bench.py results_dir [--figures]

Per run (material, size, np, algorithm, config) it reports:

* per-iteration time of the steady phases (gravity_plastic + transient; the
  first step of each phase, which carries symbolic analysis / GPU setup, is
  excluded), its median per step, and the first-step (setup) time;
* for PythonSparse runs, the solver's share of each step ("solver/it"); the
  remainder ("rest/it") is state determination, assembly and constraint
  handling, which is the same for every solver at a given size;
* Newton iterations per phase, linear-solver failures and the worst true
  relative residual for iterative / reduced-precision solvers.

For compiled runs without a solver split, "solver/it" is estimated as
step/it minus the median rest/it of the PythonSparse runs at the same size
(marked ~).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

STEADY = ("gravity_plastic", "transient")


def load(paths):
    recs = []
    for path in paths:
        meta = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "meta" in r:
                meta = r["meta"]
                continue
            r["_file"] = Path(path).name
            r["_meta"] = meta
            r.setdefault("np", 1)
            recs.append(r)
    return recs


def per_iteration(rec):
    """(step seconds per Newton iteration, solver seconds per iteration or None, setup s)."""
    phases = rec.get("phases", {})
    st = it = sv = 0.0
    have_solver = False
    for name in STEADY:
        p = phases.get(name)
        if not p or p["ok"] != 0 and name == "transient" and not p["iters"]:
            continue
        if not p:
            continue
        steps = p["step_seconds"][1:]
        iters = p["iters"][1:]
        st += sum(steps)
        it += sum(iters)
        if "solver" in p and "solver total" in p["solver"]:
            have_solver = True
            sv += sum(p["solver"]["solver total"][1:])
    first = [phases[n]["step_seconds"][0] for n in phases if phases[n]["step_seconds"]]
    if it == 0:
        return None, None, (first[0] if first else None)
    return st / it, (sv / it if have_solver else None), (first[0] if first else None)


def inexact_stats(rec):
    fails, relres, lin_it = 0, 0.0, 0
    for p in rec.get("phases", {}).values():
        s = p.get("solver", {})
        fails += int(sum(s.get("linear failures", [])))
        lin_it += int(sum(s.get("linear iterations", [])))
        if s.get("relres max"):
            relres = max(relres, max(s["relres max"]))
    return fails, relres, lin_it


def table(recs):
    rest = defaultdict(list)
    rows = []
    for r in recs:
        if "phases" not in r:
            rows.append((r, None, None, None))
            continue
        step_it, solver_it, setup = per_iteration(r)
        if step_it is not None and solver_it is not None and r.get("np", 1) == 1:
            rest[(r["material"], r["size"], r["_meta"].get("platform", "")[:7])].append(
                step_it - solver_it)
        rows.append((r, step_it, solver_it, setup))
    rest_med = {k: float(np.median(v)) for k, v in rest.items()}
    return rows, rest_med


def fmt(v, scale=1e3, w=8, prec=1):
    return " " * (w - 1) + "-" if v is None else f"{scale * v:{w}.{prec}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    files = sorted(Path(args.dir).glob("*.jsonl"))
    recs = load(files)
    rows, rest_med = table(recs)
    order = {s: i for i, s in enumerate(["6k", "27k", "54k", "126k", "189k", "457k"])}
    rows.sort(key=lambda x: (x[0]["_file"], x[0].get("material", ""), order.get(x[0]["size"], 9),
                             x[0].get("np", 1), x[0].get("algorithm", ""),
                             (x[1] if x[1] is not None else 1e9)))
    print(f"{'file':22s} {'size':5s} {'np':>3s} {'algorithm':22s} {'config':34s} {'ok':>4s} "
          f"{'step/it':>8s} {'solver/it':>9s} {'rest/it':>8s} {'setup':>8s} "
          f"{'iters g/p/t':>12s} {'fail':>4s} {'relres':>8s}  (ms)")
    for r, step_it, solver_it, setup in rows:
        if step_it is None and "phases" not in r:
            print(f"{r['_file'][:22]:22s} {r['size']:5s} {r.get('np', 1):3d} "
                  f"{r.get('algorithm', ''):22s} {r['config'][:34]:34s} {r['ok']:4d} "
                  f"{r.get('error', '')[:70]!r}")
            continue
        key = (r["material"], r["size"], r["_meta"].get("platform", "")[:7])
        est = ""
        if solver_it is None and step_it is not None and r.get("np", 1) == 1 and key in rest_med:
            solver_it, est = step_it - rest_med[key], "~"
        iters = "/".join(str(sum(p["iters"])) for p in r["phases"].values())
        fails, relres, _ = inexact_stats(r)
        rest = None if (step_it is None or solver_it is None) else step_it - solver_it
        print(f"{r['_file'][:22]:22s} {r['size']:5s} {r.get('np', 1):3d} "
              f"{r.get('algorithm', ''):22s} {r['config'][:34]:34s} {r['ok']:4d} "
              f"{fmt(step_it)} {est:>1s}{fmt(solver_it, w=8)} {fmt(rest)} {fmt(setup)} "
              f"{iters:>12s} {fails:4d} {relres:8.1e}")
    if args.figures:
        figures(rows, rest_med, Path(args.dir))
    return 0


def figures(rows, rest_med, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    neq_of = {"6k": 6152}
    for r, *_ in rows:
        if r.get("neq", -1) > 0:
            neq_of[r["size"]] = r["neq"]
    groups = {
        "CPU direct: PARDISO symmetric (pkg)": ("pkg pardiso symmetric", "#2a78d6"),
        "CPU direct: UMFPACK (compiled)": ("compiled UmfPack", "#7fb3ee"),
        "GPU direct: cuDSS spd (pkg)": ("pkg cuDSS spd", "#eb6834"),
        "GPU direct: cuDSS spd (compiled, Tcl)": ("compiled CuDSS spd (Tcl)", "#f2a07b"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, (config, color) in groups.items():
        pts = sorted((neq_of.get(r["size"], np.nan), s, sv) for r, s, sv, _ in rows
                     if r.get("config") == config and r.get("np", 1) == 1
                     and r.get("algorithm") == "Newton" and s is not None
                     and r["_file"].startswith("serial"))
        if not pts:
            continue
        n = [p[0] for p in pts]
        axes[0].plot(n, [1e3 * p[1] for p in pts], marker="o", color=color, label=label)
        sv = [p[2] if p[2] is not None else
              p[1] - rest_med.get(("pimy", r_size(neq_of, p[0]), "Windows"), np.nan) for p in pts]
        axes[1].plot(n, [1e3 * v for v in sv], marker="o", color=color, label=label)
    for ax, title in zip(axes, ("time per Newton iteration (whole step)",
                                "solver time per Newton iteration")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("equations")
        ax.set_ylabel("ms")
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(True, which="both", color="#e1e0d9", lw=0.6)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "crossover_serial.png", dpi=150)


def r_size(neq_of, n):
    for k, v in neq_of.items():
        if v == n:
            return k
    return "?"


if __name__ == "__main__":
    raise SystemExit(main())
