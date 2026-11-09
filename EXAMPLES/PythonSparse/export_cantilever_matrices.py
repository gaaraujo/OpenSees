"""Export stiffness matrices of a scalable elastic cantilever (A.npz, b.npy).

The plane cantilever of the STAIRlab gallery example plane-0002 (L = 50,
d = 10, t = 1, E = 1e4, nu = 0.25, plane stress, 4-node quads, fixed root,
transverse tip load), meshed with ``k * (5, 1) * nbase`` elements, and a 3D
version (L = 50, 10 x 10 section, stdBrick, clamped root, tip load). A static,
linear, bending-dominated operator with no mass term: the hard case for
diagonal preconditioning and the textbook case for AMG.

    python export_cantilever_matrices.py --out dir --k2d 1 4 16 --k3d 1 2 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

if os.environ.get("OPENSEESPY_PATH"):
    sys.path.insert(0, os.environ["OPENSEESPY_PATH"])
    import opensees as ops  # noqa: E402
else:
    import openseespy.opensees as ops  # noqa: E402

L, D, E, NU = 50.0, 10.0, 1.0e4, 0.25


class Dump:
    def __init__(self, stem):
        self.stem = stem

    def solve(self, **kw):
        n, nnz = kw["num_eqn"], kw["nnz"]
        A = sp.csr_matrix((np.frombuffer(kw["values"], np.float64, nnz).copy(),
                           np.frombuffer(kw["indices"], np.int32, nnz).copy(),
                           np.frombuffer(kw["index_ptr"], np.int32, n + 1).copy()), shape=(n, n))
        b = np.frombuffer(kw["rhs"], np.float64, n).copy()
        sp.save_npz(f"{self.stem}_A.npz", A)
        np.save(f"{self.stem}_b.npy", b)
        print(f"saved {self.stem} n={n} nnz={A.nnz}", flush=True)
        np.frombuffer(kw["x"], np.float64, n)[:] = 0.0   # no solve needed
        return 0


def static_export(stem):
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("PythonSparse", {"solver": Dump(stem), "scheme": "CSR"})
    ops.test("NormUnbalance", 1e300, 1)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    ops.analyze(1)
    ops.wipe()


def cantilever_2d(k, out):
    nx, ny = 12 * k, 4 * k            # the gallery mesh (12 x 4) refined k times
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.nDMaterial("ElasticIsotropic", 1, E, NU)
    ops.block2D(nx, ny, 1, 1, "quad", 1.0, "PlaneStress", 1,
                1, 0.0, -D / 2, 2, L, -D / 2, 3, L, D / 2, 4, 0.0, D / 2)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    tip = []
    for n in ops.getNodeTags():
        x, y = ops.nodeCoord(n)
        if abs(x) < 1e-9:
            ops.fix(n, 1, 1)
        elif abs(x - L) < 1e-9:
            tip.append(n)
    for n in tip:
        ops.load(n, 0.0, -1.0 / len(tip))
    static_export(Path(out) / f"cant2d_k{k}")


def cantilever_3d(k, out):
    nx, ny, nz = 20 * k, 4 * k, 4 * k
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    ops.nDMaterial("ElasticIsotropic", 1, E, NU)
    ops.block3D(nx, ny, nz, 1, 1, "stdBrick", 1,
                1, 0.0, -D / 2, -D / 2, 2, L, -D / 2, -D / 2, 3, L, D / 2, -D / 2,
                4, 0.0, D / 2, -D / 2, 5, 0.0, -D / 2, D / 2, 6, L, -D / 2, D / 2,
                7, L, D / 2, D / 2, 8, 0.0, D / 2, D / 2)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    tip = []
    for n in ops.getNodeTags():
        x = ops.nodeCoord(n, 1)
        if abs(x) < 1e-9:
            ops.fix(n, 1, 1, 1)
        elif abs(x - L) < 1e-9:
            tip.append(n)
    for n in tip:
        ops.load(n, 0.0, -1.0 / len(tip), 0.0)
    static_export(Path(out) / f"cant3d_k{k}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k2d", nargs="*", type=int, default=[1, 4, 16, 32])
    ap.add_argument("--k3d", nargs="*", type=int, default=[1, 2, 4, 6])
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for k in args.k2d:
        cantilever_2d(k, args.out)
    for k in args.k3d:
        cantilever_3d(k, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
