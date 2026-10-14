"""Does the bench_site.py soil column run correctly when partitioned?

The column ties every perimeter node at a given height to one retained node
(laminar sides), ties the base nodes horizontally to node 1, and attaches the
dashpots to a ghost node; after METIS partitioning those constraints cross
subdomains. Each rank first runs the unpartitioned column on its own (serial
UmfPack reference, OpenSeesMP independent-model mode), then the partitioned
column with compiled parallel Mumps and with DistributedPythonSparse, and
compares the displacements of its local nodes and the Newton iteration counts.

    mpiexec -n 2 python check_site_mpi.py [--plan 4] [--steps 20]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

if os.environ.get("OPENSEESPY_PATH"):
    sys.path.insert(0, os.environ["OPENSEESPY_PATH"])
    import opensees as ops  # noqa: E402
else:
    import openseespy.opensees as ops  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_site as bs  # noqa: E402
from test_distributed_features import HandSolver  # noqa: E402

PID, NP = ops.getPID(), ops.getNP()


def run(material, plan, steps, system_args, constraints, partitioned, motion):
    bs.build(ops, 1, material, plan=plan)
    bs.add_input(ops, motion)
    if partitioned:
        ops.partition()
    numberer = ("ParallelRCM",) if partitioned else ("RCM",)
    try:
        out = bs.analysis(ops, system_args, ("Newton",), material, motion=motion,
                          dyn_steps=steps, constraints=constraints, numberer=numberer,
                          input_defined=True)
    except Exception as exc:  # noqa: BLE001
        ops.wipe()
        return None, f"{type(exc).__name__}: {exc}", {}
    disp = {n: ops.nodeDisp(n) for n in ops.getNodeTags()}
    ops.wipe()
    return out, None, disp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--motion", default=None)
    ap.add_argument("--motion-dt", type=float, default=0.005)
    ap.add_argument("--tol", type=float, default=None,
                    help="override the gravity and transient NormDispIncr tolerances")
    args = ap.parse_args()
    if args.tol:
        bs.GRAVITY_TEST = ("NormDispIncr", args.tol, 50)
        bs.DYN_TEST = ("NormDispIncr", args.tol, 50)
    motion = bs.Motion.from_file(args.motion, args.motion_dt) if args.motion else None
    ok_all = True
    for material in ("elastic", "pimy"):
        ref, err, d_ref = run(material, args.plan, args.steps, ("UmfPack",),
                              ("Transformation",), False, motion)
        assert err is None and all(v[0] == 0 for v in ref.values()), err
        it_ref = {k: v[1] for k, v in ref.items()}
        cases = [
            ("Mumps, Transformation", ("Mumps",), ("Transformation",)),
            ("DistributedPythonSparse, Transformation",
             ("DistributedPythonSparse", {"solver": HandSolver(), "scheme": "CSR"}),
             ("Transformation",)),
            ("Mumps, Penalty 1e12", ("Mumps",), ("Penalty", 1e12, 1e12)),
            ("DistributedPythonSparse, Penalty 1e12",
             ("DistributedPythonSparse", {"solver": HandSolver(), "scheme": "CSR"}),
             ("Penalty", 1e12, 1e12)),
        ]
        for name, system_args, constraints in cases:
            out, err, d = run(material, args.plan, args.steps, system_args, constraints,
                              NP > 1, motion)
            if err is not None:
                ok_all = False
                print(f"[rank {PID}/{NP}] {material:7s} {name:42s} ERROR {err}", flush=True)
                continue
            phases_ok = all(v[0] == 0 for v in out.values())
            its = {k: v[1] for k, v in out.items()}
            missing = [n for n in d if n not in d_ref]
            diff = max((np.max(np.abs(np.subtract(d[n], d_ref[n]))) for n in d if n in d_ref),
                       default=np.nan)
            scale = max(np.max(np.abs(v)) for v in d_ref.values())
            rel = diff / scale
            tol = 1e-8 if constraints[0] == "Transformation" else 1e-5
            good = phases_ok and not missing and rel <= tol
            same_iters = its == it_ref
            ok_all &= good
            print(f"[rank {PID}/{NP}] {material:7s} {name:42s} local nodes {len(d):5d} "
                  f"max|du|/max|u| {rel:.1e} iters {'same' if same_iters else 'DIFFER'} "
                  f"(total per phase {[sum(v) for v in its.values()]} vs "
                  f"{[sum(v) for v in it_ref.values()]}) -> {'ok' if good else 'FAIL'}",
                  flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
