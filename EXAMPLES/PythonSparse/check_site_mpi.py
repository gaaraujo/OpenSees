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
    ap.add_argument("--dyn-tol", type=float, default=None,
                    help="override the transient NormDispIncr tolerance (PIMY gravity "
                         "does not converge much below its default)")
    args = ap.parse_args()
    if args.dyn_tol:
        bs.DYN_TEST = ("NormDispIncr", args.dyn_tol, 50)
    motion = bs.Motion.from_file(args.motion, args.motion_dt) if args.motion else None
    ok_all = True

    def rel_diff(d, d_ref):
        diff = max(np.max(np.abs(np.subtract(d[n], d_ref[n]))) for n in d)
        return diff / max(np.max(np.abs(v)) for v in d_ref.values())

    for material in ("elastic", "pimy"):
        ref, err, d_ref = run(material, args.plan, args.steps, ("UmfPack",),
                              ("Transformation",), False, motion)
        assert err is None and all(v[0] == 0 for v in ref.values()), err
        it_ref = {k: v[1] for k, v in ref.items()}
        for constraints in (("Transformation",), ("Penalty", 1e12, 1e12)):
            mumps = None
            for solver in ("Mumps", "DistributedPythonSparse"):
                system_args = ((solver,) if solver == "Mumps" else
                               (solver, {"solver": HandSolver(), "scheme": "CSR"}))
                name = f"{solver}, {constraints[0]}"
                out, err, d = run(material, args.plan, args.steps, system_args, constraints,
                                  NP > 1, motion)
                if err is not None:
                    ok_all = False
                    print(f"[rank {PID}/{NP}] {material:7s} {name:38s} ERROR {err}", flush=True)
                    continue
                its = {k: v[1] for k, v in out.items()}
                good = all(v[0] == 0 for v in out.values()) and all(n in d_ref for n in d)
                rel = rel_diff(d, d_ref)
                # Serial vs partitioned: exact for the elastic column; for PIMY the
                # path-dependent response differs at the level of the convergence
                # tolerance. Penalty adds its own constraint error.
                tol_ref = (1e-8 if material == "elastic" else 1e-4) \
                    if constraints[0] == "Transformation" else 1e-4
                good &= rel <= tol_ref
                line = (f"[rank {PID}/{NP}] {material:7s} {name:38s} nodes {len(d):4d} "
                        f"vs serial {rel:.1e} iters/phase {[sum(v) for v in its.values()]} "
                        f"(serial {[sum(v) for v in it_ref.values()]})")
                if solver == "Mumps":
                    mumps = (d, its)
                elif mumps is not None:
                    # DistributedPythonSparse vs Mumps on the same partition: same
                    # equations, both direct; must agree to round-off.
                    # PIMY: different round-off moves the converged state by about
                    # the test tolerance and can add or save one Newton iteration
                    # near the threshold; with --dyn-tol 1e-10 the states agree to
                    # ~1e-10.
                    rel_m = rel_diff(d, mumps[0])
                    if material == "elastic":
                        good &= rel_m <= 1e-8 and its == mumps[1]
                    else:
                        tight = bs.DYN_TEST[1] <= 1e-9
                        close = all(abs(sum(its[k]) - sum(mumps[1][k]))
                                    <= max(1, 0.03 * sum(mumps[1][k])) for k in its)
                        good &= rel_m <= (1e-8 if tight else 1e-4) and close
                    line += (f" vs Mumps {rel_m:.1e}, iters "
                             f"{'same' if its == mumps[1] else 'differ'}")
                ok_all &= good
                print(line + f" -> {'ok' if good else 'FAIL'}", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
