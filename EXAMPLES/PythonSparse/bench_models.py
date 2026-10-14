"""Benchmark model for PythonSparse (OpenSeesPy) and its Tcl twin.

Both versions build the same mesh in the same order, so equation numbering
(RCM) and the assembled matrices are identical between OpenSeesPy and the
Tcl interpreter (used for compiled systems such as ``system CuDSS``).

Model
-----
``plastic``  J2-plastic brick cantilever (symmetric tangent) under a tip load
             ramped linearly to ``TIP_LOAD`` over the analysis (``STEPS``
             steps of ``DT``), transient Newmark.
"""

from __future__ import annotations

E, NU = 29000.0, 0.3
K_BULK = E / (3.0 * (1.0 - 2.0 * NU))
G_SHEAR = E / (2.0 * (1.0 + NU))
LENGTH, WIDTH, HEIGHT = 40.0, 4.0, 6.0

STEPS, DT = 10, 0.01
TIP_LOAD = 30.0  # total downward tip load reached at t = STEPS * DT
# Linear time series scaled so the load factor is 1.0 at the end of the analysis.
SERIES_FACTOR = 1.0 / (STEPS * DT)

# (nx, ny, nz) -> about 2k, 12k and 54k equations
SIZES = {"2k": (20, 4, 6), "12k": (40, 8, 10), "54k": (80, 12, 16)}


def _corners():
    L, W, H = LENGTH, WIDTH, HEIGHT
    return [(0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0),
            (0, 0, H), (L, 0, H), (L, W, H), (0, W, H)]


def build(ops, model: str, size: str) -> None:
    """Build ``model`` of ``size`` in OpenSeesPy (``ops`` module)."""
    if model != "plastic":
        raise ValueError(f"unknown model {model!r}")
    nx, ny, nz = SIZES[size]
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    ops.nDMaterial("J2Plasticity", 1, K_BULK, G_SHEAR, 36.0, 50.0, 10.0, 200.0)
    args = []
    for i, c in enumerate(_corners(), start=1):
        args += [i, *map(float, c)]
    ops.block3D(nx, ny, nz, 1, 1, "stdBrick", 1, *args)
    nodes = ops.getNodeTags()
    for n in nodes:
        ops.mass(n, 1e-4, 1e-4, 1e-4)
    ops.fixX(0.0, 1, 1, 1)
    ops.timeSeries("Linear", 1, "-factor", SERIES_FACTOR)
    ops.pattern("Plain", 1, 1)
    tip = [n for n in nodes if abs(ops.nodeCoord(n, 1) - LENGTH) < 1e-9]
    for n in tip:
        ops.load(n, 0.0, 0.0, -TIP_LOAD / len(tip))


def tcl(model: str, size: str) -> str:
    """Tcl script building the same model (no analysis)."""
    if model != "plastic":
        raise ValueError(f"unknown model {model!r}")
    nx, ny, nz = SIZES[size]
    corners = " ".join(f"{i} {c[0]} {c[1]} {c[2]}" for i, c in enumerate(_corners(), 1))
    lines = [
        "wipe",
        "model basic -ndm 3 -ndf 3",
        f"nDMaterial J2Plasticity 1 {K_BULK!r} {G_SHEAR!r} 36.0 50.0 10.0 200.0",
        f"block3D {nx} {ny} {nz} 1 1 stdBrick 1 {{ {corners} }}",
        "set nodes [getNodeTags]",
        "foreach n $nodes { mass $n 1e-4 1e-4 1e-4 }",
        "fixX 0.0 1 1 1",
        f"timeSeries Linear 1 -factor {SERIES_FACTOR!r}",
        "pattern Plain 1 1 {",
        "  set tip {}",
        f"  foreach n $nodes {{ if {{abs([nodeCoord $n 1] - {LENGTH})< 1e-9}} {{ lappend tip $n }} }}",
        f"  foreach n $tip {{ load $n 0.0 0.0 [expr -{TIP_LOAD} / [llength $tip]] }}",
        "}",
    ]
    return "\n".join(lines) + "\n"


def analysis(ops, system_args, algorithm, steps=STEPS, dt=DT):
    """Transient Newmark analysis; returns (ok, iterations per step)."""
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system(*system_args)
    ops.test("NormDispIncr", 1e-8, 50)
    ops.algorithm(*algorithm)
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")
    iters = []
    ok = 0
    for _ in range(steps):
        ok = ops.analyze(1, dt)
        iters.append(ops.testIter())
        if ok != 0:
            break
    return ok, iters


def tcl_analysis(system_line: str, algorithm: str, steps=STEPS, dt=DT) -> str:
    return "\n".join([
        "constraints Plain", "numberer RCM", system_line,
        "test NormDispIncr 1e-8 50", f"algorithm {algorithm}",
        "integrator Newmark 0.5 0.25", "analysis Transient",
        "set iters {}", "set t0 [clock microseconds]", "set ok 0",
        f"for {{set i 0}} {{$i < {steps}}} {{incr i}} {{",
        f"  set ok [analyze 1 {dt}]", "  lappend iters [testIter]",
        "  if {$ok != 0} break", "}",
        "set t1 [clock microseconds]",
        'puts "RESULT ok=$ok seconds=[expr ($t1 - $t0) * 1e-6] iters=$iters neq=[systemSize]"',
    ]) + "\n"
