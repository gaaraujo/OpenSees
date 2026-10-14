"""Worker ranks must not touch the GPU under DistributedPythonSparse.

    mpiexec -n 2 python check_worker_gpu.py

Every rank builds an nvmath (cuDSS) solver, as a user script would, but only
rank 0 calls it. Afterwards each rank reports how much device memory CuPy's
pool holds: rank 0 > 0, workers 0.
"""

import os
import sys

if os.environ.get("OPENSEESPY_PATH"):
    sys.path.insert(0, os.environ["OPENSEESPY_PATH"])
    import opensees as ops
else:
    import openseespy.opensees as ops

import cupy as cp

from openseespy_solvers.nvmath import direct_solver

pid = ops.getPID()
solver = direct_solver()  # every rank constructs it

ops.model("basic", "-ndm", 3, "-ndf", 3)
ops.nDMaterial("ElasticIsotropic", 1, 29000.0, 0.3)
ops.block3D(12, 2, 3, 1, 1, "stdBrick", 1,
            1, 0.0, 0.0, 0.0, 2, 10.0, 0.0, 0.0, 3, 10.0, 1.0, 0.0, 4, 0.0, 1.0, 0.0,
            5, 0.0, 0.0, 2.0, 6, 10.0, 0.0, 2.0, 7, 10.0, 1.0, 2.0, 8, 0.0, 1.0, 2.0)
ops.fixX(0.0, 1, 1, 1)
ops.timeSeries("Linear", 1)
ops.pattern("Plain", 1, 1)
for n in ops.getNodeTags():
    if abs(ops.nodeCoord(n, 1) - 10.0) < 1e-9:
        ops.load(n, 0.0, 0.0, -1.0)
ops.partition()
ops.constraints("Plain")
ops.numberer("ParallelRCM")
ops.system("DistributedPythonSparse", solver.to_openseespy())
ops.test("NormDispIncr", 1e-10, 10)
ops.algorithm("Newton")
ops.integrator("LoadControl", 0.5)
ops.analysis("Static")
ok = ops.analyze(2)

used = cp.get_default_memory_pool().total_bytes()
expected = used > 0 if pid == 0 else used == 0
print(f"[pid={pid}] analyze={ok} cupy pool bytes={used} -> {'ok' if expected and ok == 0 else 'FAIL'}")
ops.wipe()
