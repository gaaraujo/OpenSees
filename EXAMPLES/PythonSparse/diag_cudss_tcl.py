"""Diagnose the compiled ``system CuDSS`` cost: verbose run on the soil column.

Writes the Tcl twin of bench_site.py at one size with ``system CuDSS
-verbose 1`` (elastic gravity only, 10 steps), runs it with the given
OpenSees.exe, and saves the full output (cuDSS phase timings, memory) next to
the per-step times.

    python diag_cudss_tcl.py --exe OpenSees.exe --size 54k --out diag_54k.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_site as bs  # noqa: E402

SIZES = {"6k": (1, 10), "27k": (1, 20), "54k": (2, 20), "126k": (2, 30)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--size", default="54k")
    ap.add_argument("--system", default="system CuDSS -matrixType spd -verbose 1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    refine, plan = SIZES[args.size]
    script = (bs.tcl(refine, "pimy", plan)
              + bs.tcl_analysis(args.system, "Newton", "pimy", dyn_steps=0))
    with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as f:
        f.write(script)
    res = subprocess.run([args.exe, f.name], capture_output=True, text=True, timeout=3600)
    Path(args.out).write_text(f"# {args.system} size {args.size}\n" + res.stdout + res.stderr,
                              encoding="utf-8")
    print((res.stdout + res.stderr)[-3000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
