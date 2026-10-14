"""Benchmark models for PythonSparse (OpenSeesPy) and their Tcl twins.

Both versions build the same mesh in the same order, so equation numbering
(RCM) and the assembled matrices are identical between OpenSeesPy and the
Tcl interpreter (used for compiled systems such as ``system CuDSS``).

Models
------
``plastic``  J2-plastic brick cantilever (symmetric tangent), transient.
``sliding``  the same brick resting on frictional contact elements at its base,
             with soft lateral springs, pushed past the slip limit (nonsymmetric
             tangent once the base slides).
"""

from __future__ import annotations

E, NU = 29000.0, 0.3
K_BULK = E / (3.0 * (1.0 - 2.0 * NU))
G_SHEAR = E / (2.0 * (1.0 + NU))
LENGTH, WIDTH, HEIGHT = 40.0, 4.0, 6.0

# (nx, ny, nz) -> about 2k, 12k and 54k equations
SIZES = {"2k": (20, 4, 6), "12k": (40, 8, 10), "54k": (80, 12, 16)}


def _corners():
    L, W, H = LENGTH, WIDTH, HEIGHT
    return [(0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0),
            (0, 0, H), (L, 0, H), (L, W, H), (0, W, H)]


def build(ops, model: str, size: str) -> None:
    """Build ``model`` of ``size`` in OpenSeesPy (``ops`` module)."""
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
    if model == "plastic":
        ops.fixX(0.0, 1, 1, 1)
    elif model == "sliding":
        _sliding_base(ops, nodes)
    else:
        raise ValueError(model)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    top = [n for n in nodes if abs(ops.nodeCoord(n, 3) - HEIGHT) < 1e-9]
    tip = [n for n in nodes if abs(ops.nodeCoord(n, 1) - LENGTH) < 1e-9]
    if model == "plastic":
        for n in tip:
            ops.load(n, 0.0, 0.0, -300.0 / len(tip))
    else:
        for n in top:
            ops.load(n, 60.0 / len(top), 0.0, -400.0 / len(top))


def _sliding_base(ops, nodes):
    base = [n for n in nodes if abs(ops.nodeCoord(n, 3)) < 1e-9]
    ops.uniaxialMaterial("Elastic", 10, 20.0)
    tag = max(nodes) + 1
    ele = 10_000_000
    for n in base:
        x, y = ops.nodeCoord(n, 1), ops.nodeCoord(n, 2)
        ops.node(tag, x, y, 0.0)
        ops.fix(tag, 1, 1, 1)
        ops.element("zeroLengthContact3D", ele, n, tag, 1.0e5, 1.0e5, 0.3, 0.0, 3)
        ops.element("zeroLength", ele + 1, tag, n, "-mat", 10, 10, "-dir", 1, 2)
        tag += 1
        ele += 2


def tcl(model: str, size: str) -> str:
    """Tcl script building the same model (no analysis)."""
    nx, ny, nz = SIZES[size]
    corners = " ".join(f"{i} {c[0]} {c[1]} {c[2]}" for i, c in enumerate(_corners(), 1))
    lines = [
        "wipe",
        "model basic -ndm 3 -ndf 3",
        f"nDMaterial J2Plasticity 1 {K_BULK!r} {G_SHEAR!r} 36.0 50.0 10.0 200.0",
        f"block3D {nx} {ny} {nz} 1 1 stdBrick 1 {{ {corners} }}",
        "set nodes [getNodeTags]",
        "foreach n $nodes { mass $n 1e-4 1e-4 1e-4 }",
    ]
    if model == "plastic":
        lines.append("fixX 0.0 1 1 1")
    else:
        lines += [
            "uniaxialMaterial Elastic 10 20.0",
            "set tag [expr [tcl::mathfunc::max {*}$nodes] + 1]",
            "set ele 10000000",
            "foreach n $nodes {",
            "  if {abs([nodeCoord $n 3]) < 1e-9} {",
            "    node $tag [nodeCoord $n 1] [nodeCoord $n 2] 0.0",
            "    fix $tag 1 1 1",
            "    element zeroLengthContact3D $ele $n $tag 1.0e5 1.0e5 0.3 0.0 3",
            "    element zeroLength [expr $ele + 1] $tag $n -mat 10 10 -dir 1 2",
            "    incr tag; incr ele 2",
            "  }",
            "}",
        ]
    lines += ["timeSeries Linear 1", "pattern Plain 1 1 {"]
    if model == "plastic":
        lines += [
            "  set tip {}",
            f"  foreach n $nodes {{ if {{abs([nodeCoord $n 1] - {LENGTH})< 1e-9}} {{ lappend tip $n }} }}",
            "  foreach n $tip { load $n 0.0 0.0 [expr -300.0 / [llength $tip]] }",
        ]
    else:
        lines += [
            "  set top {}",
            f"  foreach n $nodes {{ if {{abs([nodeCoord $n 3] - {HEIGHT})< 1e-9}} {{ lappend top $n }} }}",
            "  foreach n $top { load $n [expr 60.0 / [llength $top]] 0.0 [expr -400.0 / [llength $top]] }",
        ]
    lines.append("}")
    return "\n".join(lines) + "\n"


def analysis(ops, system_args, algorithm, steps=10, dt=0.01):
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


def tcl_analysis(system_line: str, algorithm: str, steps=10, dt=0.01) -> str:
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
