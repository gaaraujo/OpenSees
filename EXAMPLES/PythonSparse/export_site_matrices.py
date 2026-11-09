"""Export bench_site.py tangents (A.npz, b.npy) for bench_matrices.py.

Saves, for each size of the PIMY column under Newton: the last plastic-gravity
tangent ("gravity") and the first-iteration tangent of transient step
``--step + 1`` ("eq").

    python export_site_matrices.py --motion velocityHistory.out --out dir --sizes 6k 27k
"""

from __future__ import annotations

import argparse
import os
import sys
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
import bench_conventional as bc  # noqa: E402
import bench_site as bs  # noqa: E402


class Dump:
    """SuperLU solve; keeps the latest system and saves the requested ones."""

    def __init__(self, out, size, step):
        self.out, self.size, self.step = out, size, step
        self.transient_steps, self.last, self.saved_eq = 0, None, False
        self.gravity = None

    def probe(self, phase):
        if phase == "gravity_plastic":
            self.gravity = self.last          # last solve of the latest plastic step
        elif phase == "transient":
            self.transient_steps += 1

    def solve(self, **kw):
        ptr, ind, val, n = bc._csr(kw)
        A = sp.csr_matrix((val.copy(), ind.copy(), ptr.copy()), shape=(n, n))
        b = np.frombuffer(kw["rhs"], np.float64, n).copy()
        np.frombuffer(kw["x"], np.float64, n)[:] = spla.splu(A.tocsc()).solve(b)
        self.last = (A, b)
        if self.transient_steps == self.step and not self.saved_eq:
            self.save("eq", A, b)
            self.saved_eq = True
        return 0

    def save(self, state, A, b):
        stem = Path(self.out) / f"site_{self.size}_{state}"
        sp.save_npz(f"{stem}_A.npz", A)
        np.save(f"{stem}_b.npy", b)
        print(f"saved {stem} n={A.shape[0]} nnz={A.nnz}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", nargs="+", default=["6k"])
    ap.add_argument("--step", type=int, default=20)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    motion = bs.Motion.from_file(args.motion, 0.005)
    for size in args.sizes:
        refine, plan = bc.SIZES[size]
        bs.build(ops, refine, "pimy", plan=plan)
        d = Dump(args.out, size, args.step)
        bs.analysis(ops, ("PythonSparse", {"solver": d, "scheme": "CSR"}), ("Newton",), "pimy",
                    motion=motion, dyn_steps=args.step + 1, step_probe=d.probe)
        if d.gravity is not None:
            d.save("gravity", *d.gravity)
        ops.wipe()
    return 0


if __name__ == "__main__":
    sys.exit(main())
