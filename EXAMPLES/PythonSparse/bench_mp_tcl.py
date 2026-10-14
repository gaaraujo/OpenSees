"""Partitioned soil column through Tcl OpenSeesMP: DistributedCuDSS and Mumps.

The Tcl twin of bench_site.py, run with ``mpiexec -n N OpenSeesMP.exe``: the
column and the base input are built on every rank, ``partition`` (METIS)
splits it, then the same phases as bench_conventional.py run with a compiled
parallel system (``system DistributedCuDSS``: gather to rank 0, cuDSS on one
GPU; ``system Mumps``: distributed MUMPS). One JSON line per run with rank 0's
per-step times and Newton iterations.

    python bench_mp_tcl.py --exe OpenSeesMP.exe --motion velocityHistory.out \
        --np 4 8 --sizes 6k 54k --out results.jsonl
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_site as bs  # noqa: E402

# same tables as bench_conventional.py (kept here so this driver needs no opensees module)
ALGORITHMS = {
    "Newton": "Newton",
    "KrylovNewton": "KrylovNewton",
    "KrylovNewton-initial": "KrylovNewton -iterate initial -increment initial",
    "ModifiedNewton-initial": "ModifiedNewton -initial",
}
SIZES = {"6k": (1, 10), "27k": (1, 20), "54k": (2, 20), "126k": (2, 30), "189k": (3, 30),
         "457k": (4, 40)}
SYSTEMS = {
    "Tcl MP DistributedCuDSS": "system DistributedCuDSS",
    "Tcl MP DistributedCuDSS spd": "system DistributedCuDSS -matrixType spd",
    "Tcl MP DistributedCuDSS dFFI": "system DistributedCuDSS -precision dFFI",
    "Tcl MP Mumps": "system Mumps",
    "Tcl MP DistributedAmgX AMG":
        "system DistributedAmgX -solver PCG -preconditioner AMG -maxIters 2000 -relTolerance 1e-8",
}


RESULT_FILE = "result.txt"   # rank 0 writes its RESULT lines here (in the run directory)


def script(material, size, alg, system_line, motion, steps):
    refine, plan = SIZES[size]
    return (bs.tcl(refine, material, plan) + bs.tcl_input(motion) + "partition\n"
            + bs.tcl_analysis(system_line, ALGORITHMS[alg], material, motion=motion,
                              dyn_steps=steps, numberer="ParallelRCM", input_defined=True,
                              rank0_only=True, result_file=RESULT_FILE))


def parse(text):
    phases, neq = {}, -1
    for line in text.splitlines():
        if not line.startswith("RESULT"):
            continue
        parts = line.split()
        fields = dict(item.split("=", 1) for item in parts[2:] if "=" in item)
        neq = int(fields["neq"])
        phases[parts[1]] = {"ok": int(fields["ok"]),
                            "iters": [int(v) for v in fields.get("iters", "").split(",") if v],
                            "seconds": float(fields["seconds"]),
                            "step_seconds": [float(v) for v in
                                             fields.get("times", "").split(",") if v]}
    return phases, neq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--mpiexec", default="mpiexec")
    ap.add_argument("--motion", required=True)
    ap.add_argument("--motion-dt", type=float, default=0.005)
    ap.add_argument("--out", required=True)
    ap.add_argument("--np", nargs="+", type=int, default=[4])
    ap.add_argument("--sizes", nargs="+", default=["6k"], choices=list(SIZES))
    ap.add_argument("--materials", nargs="+", default=["pimy"])
    ap.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=list(ALGORITHMS))
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--max-seconds", type=float, default=1800.0)
    args = ap.parse_args()
    motion = bs.Motion.from_file(args.motion, args.motion_dt)
    env = dict(os.environ, I_MPI_PIN="on", I_MPI_PIN_CELL="core", I_MPI_FABRICS="shm",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if os.name != "nt":
        # Inside a Slurm job (and a container) Hydra must fork the ranks locally
        # instead of launching them through Slurm.
        env.setdefault("I_MPI_HYDRA_BOOTSTRAP", "fork")
    out = open(args.out, "a", encoding="utf-8")
    out.write(json.dumps({"meta": {**bs.run_stamp(), "platform": platform.platform(), "exe": args.exe,
                                   "motion": motion.name, "start_time": motion.strong_start(),
                                   "steps": args.steps, "dyn_test": bs.DYN_TEST,
                                   "distributed": True, "tcl": True}}) + "\n")
    skipped = set()
    for material in args.materials:
        for size in args.sizes:
            for nproc in args.np:
                for alg in args.algorithms:
                    for name in args.systems:
                        key = (material, nproc, alg, name)
                        if key in skipped:
                            continue
                        workdir = tempfile.mkdtemp(prefix="bench_mp_")
                        path = os.path.join(workdir, "run.tcl")
                        with open(path, "w") as f:
                            f.write(script(material, size, alg, SYSTEMS[name], motion,
                                           args.steps))
                        t0 = time.perf_counter()
                        try:
                            res = subprocess.run([args.mpiexec, "-n", str(nproc), args.exe,
                                                  path], capture_output=True, text=True,
                                                 timeout=args.max_seconds, env=env,
                                                 cwd=workdir)
                            rf = os.path.join(workdir, RESULT_FILE)
                            try:
                                phases, neq = parse(open(rf).read() if os.path.exists(rf) else "")
                            except (KeyError, ValueError) as exc:
                                phases, neq = {}, -1
                                res.stderr += f"\nRESULT parse failed: {exc}"
                            ok = (0 if phases and all(p["ok"] == 0 for p in phases.values())
                                  and "transient" in phases else -1)
                            rec = {"ok": ok, "neq": neq, "phases": phases}
                            if ok != 0:
                                rec["error"] = (res.stdout + res.stderr)[-1500:]
                            if not phases:
                                rec = {"ok": -999, "error": (res.stdout + res.stderr)[-600:]}
                        except subprocess.TimeoutExpired:
                            rec = {"ok": -997, "error": f"timeout after {args.max_seconds} s"}
                            skipped.add(key)
                            kill = (["taskkill", "/F", "/IM", Path(args.exe).name] if os.name == "nt"
                                    else ["pkill", "-f", args.exe])
                            subprocess.run(kill, capture_output=True)
                        rec.update({"material": material, "size": size, "np": nproc,
                                    "algorithm": alg, "config": name,
                                    "wall_seconds": time.perf_counter() - t0})
                        out.write(json.dumps(rec) + "\n")
                        out.flush()
                        msg = rec.get("error", "")[:200] if rec["ok"] < -1 else " ".join(
                            f"{k}: ok={p['ok']} it={sum(p['iters'])} t={p['seconds']:.2f}s"
                            for k, p in rec["phases"].items())
                        print(f"{material:5s} {size:5s} np={nproc:<3d} {alg:22s} {name:30s} "
                              f"ok={rec['ok']:4d} {msg}", flush=True)
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
