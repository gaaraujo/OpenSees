"""Partitioned soil-column benchmark: parallel Mumps vs DistributedPythonSparse.

Same column, phases, window, algorithms and per-step records as
bench_conventional.py, run on ``np`` MPI ranks: the model is partitioned
(METIS), every rank does state determination for its subdomain, and either
compiled parallel ``system Mumps`` solves across the ranks or
``system DistributedPythonSparse`` gathers the system on rank 0, where the
Python solver runs. Transformation constraints (the laminar and base ties
cross subdomains; check_site_mpi.py verifies this).

The driver runs each configuration as its own ``mpiexec`` job with a hard
timeout and appends one JSON line per run (rank 0's timings: ``analyze`` is
collective, so rank 0's step time covers every rank)::

    python bench_distributed_site.py --motion velocityHistory.out --np 4 8 \
        --out results.jsonl [--sizes 54k 126k] [--only Mumps cuDSS]

Needs the -DOPENSEESPY_MPI=ON module (OPENSEESPY_PATH) and mpiexec on PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_conventional as bc  # noqa: E402
import bench_site as bs  # noqa: E402

ops = bc.ops


def configurations():
    """Mumps plus every PythonSparse configuration of bench_conventional, distributed."""
    cfgs = {"compiled Mumps": lambda: ("Mumps",)}
    for name, make in bc.configurations().items():
        if not name.startswith("compiled"):
            cfgs[name] = (lambda make=make: ("DistributedPythonSparse", make()[1]))
    return cfgs


def run_single(material, size, alg_name, config, motion, steps):
    refine, plan = bc.SIZES[size]
    make = configurations()[config]
    t_build = time.perf_counter()
    bs.build(ops, refine, material, plan=plan)
    bs.add_input(ops, motion)
    ops.partition()
    t_build = time.perf_counter() - t_build
    args = make()
    timers = bc.StepTimers()
    if args[0] == "DistributedPythonSparse":
        cfg = dict(args[1])
        cfg["solver"] = bc.instrument(cfg["solver"], timers, stages=config.startswith("pkg"))
        args = (args[0], cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = bs.analysis(ops, args, bc.ALGORITHMS[alg_name], material, motion=motion,
                          dyn_steps=steps, step_probe=timers.probe,
                          constraints=("Transformation",), numberer=("ParallelRCM",),
                          input_defined=True)
    solver_steps = timers.per_phase()
    phases = {}
    for name, (ok, iters, seconds, step_seconds) in out.items():
        phases[name] = {"ok": ok, "iters": iters, "seconds": seconds,
                        "step_seconds": step_seconds}
        if name in solver_steps:
            phases[name]["solver"] = solver_steps[name]
    local_nodes = len(ops.getNodeTags())
    ops.wipe()
    ok = 0 if all(p["ok"] == 0 for p in phases.values()) and "transient" in phases else -1
    return {"ok": ok, "build_seconds": t_build, "phases": phases, "rank0_nodes": local_nodes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--motion-dt", type=float, default=0.005)
    ap.add_argument("--out", default=None)
    ap.add_argument("--np", nargs="+", type=int, default=[4])
    ap.add_argument("--sizes", nargs="+", default=["54k"], choices=list(bc.SIZES))
    ap.add_argument("--materials", nargs="+", default=["pimy"], choices=["elastic", "pimy"])
    ap.add_argument("--algorithms", nargs="+", default=list(bc.ALGORITHMS),
                    choices=list(bc.ALGORITHMS))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=None)
    ap.add_argument("--max-seconds", type=float, default=1800.0)
    ap.add_argument("--mpiexec", default="mpiexec")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--single", nargs=4, metavar=("MATERIAL", "SIZE", "ALGORITHM", "CONFIG"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--result-file", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--no-resume", action="store_true",
                    help="rerun configurations already recorded in --out (default: skip them)")
    args = ap.parse_args()
    motion = bs.Motion.from_file(args.motion, args.motion_dt)

    if args.single:
        rec = run_single(*args.single, motion, args.steps)
        if ops.getPID() == 0:
            # stdout of several ranks can interleave within a line; the file cannot
            if args.result_file:
                Path(args.result_file).write_text(json.dumps(rec) + "\n")
            print("JSON " + json.dumps(rec), flush=True)
        return 0

    names = [n for n in configurations()
             if (not args.only or any(s in n for s in args.only))
             and not (args.skip and any(s in n for s in args.skip))]
    if args.list:
        print("\n".join(names))
        return 0
    if not args.out:
        ap.error("--out is required")
    meta = {**bs.run_stamp(), "platform": platform.platform(), "python": platform.python_version(),
            "opensees": getattr(ops, "__file__", "?"), "motion": motion.name,
            "start_time": motion.strong_start(), "steps": args.steps,
            "dyn_test": bs.DYN_TEST, "gravity_test": bs.GRAVITY_TEST, "dyn_dt": bs.DYN_DT,
            "distributed": True}
    # resume after a requeue: runs recorded under an equivalent meta line are not repeated
    same = {k: meta[k] for k in ("opensees", "motion", "start_time", "steps", "dyn_test",
                                 "gravity_test", "dyn_dt")}
    done = {} if args.no_resume else bs.done_runs(
        args.out, same, ("material", "size", "np", "algorithm", "config"))
    out = open(args.out, "a", encoding="utf-8")
    out.write(json.dumps({"meta": meta}) + "\n")
    skipped = set()
    for material in args.materials:
        for size in args.sizes:
            for nproc in args.np:
                for alg in args.algorithms:
                    for name in names:
                        key = (material, nproc, alg, name)
                        if key in skipped:
                            continue
                        prev = done.get((material, size, nproc, alg, name))
                        if prev is not None:
                            print(f"{material:7s} {size:5s} np={nproc:<3d} {alg:22s} {name:26s} "
                                  f"already recorded (ok={prev['ok']}), skipped", flush=True)
                            if prev["ok"] == -997:
                                skipped.add(key)
                            continue
                        fd, result_file = tempfile.mkstemp(prefix="bench_dps_", suffix=".json")
                        os.close(fd)
                        cmd = [args.mpiexec, "-n", str(nproc), sys.executable,
                               str(Path(__file__).resolve()), "--motion", args.motion,
                               "--motion-dt", str(args.motion_dt), "--steps", str(args.steps),
                               "--result-file", result_file,
                               "--single", material, size, alg, name]
                        t0 = time.perf_counter()
                        try:
                            res = subprocess.run(cmd, capture_output=True, text=True,
                                                 timeout=args.max_seconds)
                            text = Path(result_file).read_text().strip()
                            rec = (json.loads(text) if text else
                                   {"ok": -999, "error": (res.stdout + res.stderr)[-600:]})
                        except subprocess.TimeoutExpired:
                            rec = {"ok": -997, "error": f"timeout after {args.max_seconds} s"}
                            skipped.add(key)
                        finally:
                            os.unlink(result_file)
                        rec.update({"material": material, "size": size, "np": nproc,
                                    "algorithm": alg, "config": name,
                                    "wall_seconds": time.perf_counter() - t0})
                        if "phases" in rec:
                            rec["neq"] = -1  # systemSize() is per-rank under MPI
                        out.write(json.dumps(rec) + "\n")
                        out.flush()
                        print(f"{material:7s} {size:5s} np={nproc:<3d} {alg:22s} {name:26s} "
                              f"ok={rec['ok']:4d} {bc.summary(rec)}", flush=True)
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
