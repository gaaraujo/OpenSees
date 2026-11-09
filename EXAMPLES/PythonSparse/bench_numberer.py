"""Cost of the OpenSees numberer: first elastic-gravity step with RCM vs Plain.

The numberer runs at analysis setup (and after wipeAnalysis / structure
changes), not per iteration. This times, per size, the first analyze() call
(numbering + graph + symbolic analysis + first factorization) and the second
(steady) one, for ``numberer RCM`` and ``numberer Plain``, with PARDISO
symmetric (which reorders internally either way).

    python bench_numberer.py --motion velocityHistory.out --sizes 6k 54k 126k
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_conventional as bc  # noqa: E402
import bench_site as bs  # noqa: E402

ops = bc.ops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--sizes", nargs="+", default=["6k", "54k"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from openseespy_solvers.scipy import pardiso

    rows = []
    for size in args.sizes:
        refine, plan = bc.SIZES[size]
        for numberer in ("RCM", "Plain"):
            bs.build(ops, refine, "pimy", plan=plan)
            ops.constraints("Transformation")
            t = time.perf_counter()
            ops.numberer(numberer)
            ops.system("PythonSparse", pardiso(matrix_type="symmetric").to_openseespy())
            ops.test(*bs.GRAVITY_TEST)
            ops.algorithm("Newton")
            ops.integrator("Newmark", bs.NEWMARK_GAMMA, bs.NEWMARK_BETA)
            ops.analysis("Transient")
            for tag in range(1, len(bs.LAYERS) + 1):
                ops.updateMaterialStage("-material", tag, "-stage", 0)
            t_def = time.perf_counter() - t
            times = []
            for _ in range(2):
                t = time.perf_counter()
                ops.analyze(1, bs.GRAVITY_DT)
                times.append(time.perf_counter() - t)
            iters = ops.testIter()
            ops.wipe()
            row = {"size": size, "numberer": numberer, "define_s": t_def,
                   "first_step_s": times[0], "second_step_s": times[1], "iters_step2": iters}
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
