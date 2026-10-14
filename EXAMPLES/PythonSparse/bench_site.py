"""Soil-column site-response model for PythonSparse benchmarks (OpenSeesPy + Tcl).

Port of Femora's SiteResponse Example 3
(https://github.com/GeotechUW/Femora/tree/main/examples/SiteResponse/Example3):
a 10 m x 10 m x 18 m column of five layers on stdBrick elements with body
forces, frequency Rayleigh damping (3% at 2.76 and 13.84 Hz), laminar side
boundaries (equalDOF of the perimeter nodes at each height), fixed base, a
transient gravity stage (30 steps of dt = 1, Newmark gamma = 0.6) and uniform
base excitation in x at dt = 0.001.

Differences from the Femora script, all deliberate:

* the input motion is a short synthetic sine sweep (Femora ships its
  FrequencySweep record inside the package); benchmarks run a short window, so
  the motion only has to excite the column;
* ``refine`` divides dx, dy and every layer's dz to scale the model;
* ``material='pimy'`` turns the profile into clay with
  PressureIndependMultiYield (total stress, undrained), using the suggested
  values on the OpenSees wiki
  (https://opensees.berkeley.edu/wiki/index.php/PressureIndependMultiYield_Material):
  stiff clay for the three dense layers, medium clay for the loose layer, soft
  clay at the top. Gravity is elastic (material stage 0), then the plastic
  stage (stage 1) is switched on, as the wiki prescribes.

Both builders create nodes and elements in the same order, so equation
numbering (RCM) and the assembled matrices match between OpenSeesPy and Tcl.
"""

from __future__ import annotations

import math

XMIN, XMAX, YMIN, YMAX, ZMIN = -5.0, 5.0, -5.0, 5.0, -18.0
DX = DY = 1.0
# name, G (Pa), unit weight (kN/m^3), nu, thickness (m), dz (m) -- bottom to top
LAYERS = [
    ("Dense Ottawa1", 145.0e6, 19.9, 0.3, 2.6, 1.3),
    ("Dense Ottawa2", 145.0e6, 19.9, 0.3, 2.4, 1.2),
    ("Dense Ottawa3", 145.0e6, 19.9, 0.3, 5.0, 1.0),
    ("Loose Ottawa", 75.0e6, 19.1, 0.3, 6.0, 0.5),
    ("Dense Montrey", 42.0e6, 19.8, 0.3, 2.0, 0.5),
]
# PressureIndependMultiYield suggested values (OpenSees wiki):
# rho (t/m^3), refShearModul (kPa), refBulkModul (kPa), cohesion (kPa), peakShearStra
PIMY_CLAYS = {
    "soft": (1.3, 1.3e4, 6.5e4, 18.0, 0.1),
    "medium": (1.5, 6.0e4, 3.0e5, 37.0, 0.1),
    "stiff": (1.8, 1.5e5, 7.5e5, 75.0, 0.1),
}
PIMY_PROFILE = ["stiff", "stiff", "stiff", "medium", "soft"]  # bottom to top
# frictionAng, refPress, pressDependCoe: now required by OpenSees. phi = d = 0 as in
# the wiki table; p'r = 80 kPa is the pressure at which the table gives gamma_max.
PIMY_PHI, PIMY_REF_PRESS, PIMY_D = 0.0, 80.0, 0.0
PIMY_YIELD_SURFACES = 20  # wiki default; also required now

DAMP_XI, DAMP_F1, DAMP_F2 = 0.03, 2.76, 13.84
NEWMARK_GAMMA = 0.6
NEWMARK_BETA = (NEWMARK_GAMMA + 0.5) ** 2 / 4.0
GRAVITY_STEPS, GRAVITY_DT = 30, 1.0
DYN_DT = 0.001


def layer_props(material: str = "elastic"):
    """(E kPa, nu, rho t/m^3, G kPa) per layer.

    ``elastic``: as in the Femora script. ``pimy``: wiki clay values (E and nu
    from G and B, for reference only; PIMY takes G and B directly).
    """
    out = []
    if material == "pimy":
        for kind in PIMY_PROFILE:
            rho, G, B, _c, _g = PIMY_CLAYS[kind]
            nu = (3 * B - 2 * G) / (2 * (3 * B + G))
            out.append((9 * B * G / (3 * B + G), nu, rho, G))
        return out
    for _name, G, gamma, nu, _t, _dz in LAYERS:
        rho = gamma * 1000.0 / 9.81 / 1000.0
        E = 2.0 * G * (1.0 + nu) / 1000.0
        out.append((E, nu, rho, G / 1000.0))
    return out


def grid(refine: int = 1):
    nx = int(round((XMAX - XMIN) / DX)) * refine
    ny = int(round((YMAX - YMIN) / DY)) * refine
    xs = [XMIN + (XMAX - XMIN) * i / nx for i in range(nx + 1)]
    ys = [YMIN + (YMAX - YMIN) * j / ny for j in range(ny + 1)]
    zs = [ZMIN]
    layer_of_cell = []
    z = ZMIN
    for k, (_n, _G, _g, _nu, t, dz) in enumerate(LAYERS):
        ncell = int(round(t / dz)) * refine
        for c in range(ncell):
            zs.append(z + t * (c + 1) / ncell)
            layer_of_cell.append(k)
        z += t
    return xs, ys, zs, layer_of_cell


def _commands(refine: int, material: str):
    """Yield (command, args) in build order; shared by both builders."""
    xs, ys, zs, layer_of_cell = grid(refine)
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    props = layer_props(material)

    def nid(i, j, k):
        return 1 + i + (nx + 1) * (j + (ny + 1) * k)

    yield "model", ("basic", "-ndm", 3, "-ndf", 3)
    for k, (E, nu, rho, G) in enumerate(props):
        tag = k + 1
        if material == "elastic":
            yield "nDMaterial", ("ElasticIsotropic", tag, E, nu, rho)
        elif material == "pimy":
            rho_c, G_c, B_c, c, gmax = PIMY_CLAYS[PIMY_PROFILE[k]]
            yield "nDMaterial", ("PressureIndependMultiYield", tag, 3, rho_c, G_c, B_c, c, gmax,
                                 PIMY_PHI, PIMY_REF_PRESS, PIMY_D, PIMY_YIELD_SURFACES)
        else:
            raise ValueError(material)
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                yield "node", (nid(i, j, k), x, y, z)
    ele = 1
    for k in range(nz):
        mat = layer_of_cell[k] + 1
        rho = props[layer_of_cell[k]][2]
        for j in range(ny):
            for i in range(nx):
                n = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1),
                     nid(i, j + 1, k + 1)]
                yield "element", ("stdBrick", ele, *n, mat, 0.0, 0.0, -9.81 * rho)
                ele += 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            yield "fix", (nid(i, j, 0), 1, 1, 1)
    # laminar boundary: perimeter nodes at each height share dofs 1-3
    for k in range(1, nz + 1):
        perim = [nid(i, j, k) for j in range(ny + 1) for i in range(nx + 1)
                 if i in (0, nx) or j in (0, ny)]
        for slave in perim[1:]:
            yield "equalDOF", (perim[0], slave, 1, 2, 3)
    w1, w2 = 2 * math.pi * DAMP_F1, 2 * math.pi * DAMP_F2
    alpha_m = 2 * DAMP_XI * w1 * w2 / (w1 + w2)
    beta_k = 2 * DAMP_XI / (w1 + w2)
    yield "rayleigh", (alpha_m, beta_k, 0.0, 0.0)


def build(ops, refine: int = 1, material: str = "elastic") -> None:
    """Build the column in OpenSeesPy."""
    ops.wipe()
    for cmd, args in _commands(refine, material):
        getattr(ops, cmd)(*args)


def tcl(refine: int = 1, material: str = "elastic") -> str:
    """Tcl script building the same column."""
    lines = ["wipe"]
    for cmd, args in _commands(refine, material):
        lines.append(cmd + " " + " ".join(repr(a) if isinstance(a, float) else str(a)
                                          for a in args))
    return "\n".join(lines) + "\n"


def sweep_values(duration: float, dt: float = DYN_DT, f0=3.0, f1=10.0, amp=0.3):
    """Linear sine sweep (in g) from f0 to f1 Hz over ``duration``.

    No ramp: the record starts at zero acceleration (sin 0), so there is no jump,
    and short benchmark windows see full-amplitude shaking. f0 is near the
    column's fundamental frequency (about 3.7 Hz for the elastic profile).
    """
    n = int(round(duration / dt)) + 1
    vals = []
    for i in range(n):
        t = i * dt
        f = f0 + (f1 - f0) * t / max(duration, dt)
        vals.append(amp * math.sin(2 * math.pi * f * t))
    return vals


def _material_tags():
    return list(range(1, len(LAYERS) + 1))


def analysis(ops, system_args, algorithm, material="elastic", dyn_steps=50,
             plastic_gravity_steps=10):
    """Gravity stage then ``dyn_steps`` of base excitation.

    Returns a dict with ok, iterations per step and seconds for each phase.
    """
    import time

    def setup():
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.system(*system_args)
        ops.test("NormDispIncr", 1e-6, 50)
        ops.algorithm(*algorithm)
        ops.integrator("Newmark", NEWMARK_GAMMA, NEWMARK_BETA)
        ops.analysis("Transient")

    def run(steps, dt):
        iters, ok = [], 0
        t0 = time.perf_counter()
        for _ in range(steps):
            ok = ops.analyze(1, dt)
            iters.append(ops.testIter())
            if ok != 0:
                break
        return ok, iters, time.perf_counter() - t0

    out = {}
    setup()
    if material == "pimy":
        for tag in _material_tags():
            ops.updateMaterialStage("-material", tag, "-stage", 0)
    out["gravity"] = run(GRAVITY_STEPS, GRAVITY_DT)
    if material == "pimy" and out["gravity"][0] == 0:
        for tag in _material_tags():
            ops.updateMaterialStage("-material", tag, "-stage", 1)
        out["gravity_plastic"] = run(plastic_gravity_steps, GRAVITY_DT)
    if all(v[0] == 0 for v in out.values()):
        ops.setTime(0.0)
        vals = sweep_values(dyn_steps * DYN_DT + DYN_DT)
        ops.timeSeries("Path", 2, "-dt", DYN_DT, "-values", *vals, "-factor", 9.81)
        ops.pattern("UniformExcitation", 2, 1, "-accel", 2)
        out["dynamic"] = run(dyn_steps, DYN_DT)
    return out


def tcl_analysis(system_line, algorithm, material="elastic", dyn_steps=50,
                 plastic_gravity_steps=10):
    """Tcl twin of :func:`analysis`; prints one RESULT line per phase."""
    vals = " ".join(repr(v) for v in sweep_values(dyn_steps * DYN_DT + DYN_DT))
    stage = lambda s: "\n".join(  # noqa: E731
        f"updateMaterialStage -material {t} -stage {s}" for t in _material_tags())
    run = lambda name, steps, dt: "\n".join([  # noqa: E731
        "set iters {}", "set ok 0", "set t0 [clock microseconds]",
        f"for {{set i 0}} {{$i < {steps}}} {{incr i}} {{",
        f"  set ok [analyze 1 {dt}]", "  lappend iters [testIter]",
        "  if {$ok != 0} break", "}",
        f'puts "RESULT {name} ok=$ok seconds=[expr ([clock microseconds] - $t0) * 1e-6] '
        'neq=[systemSize] iters=$iters"',
        "if {$ok != 0} {exit}",
    ])
    lines = ["constraints Transformation", "numberer RCM", system_line,
             "test NormDispIncr 1e-6 50", f"algorithm {algorithm}",
             f"integrator Newmark {NEWMARK_GAMMA} {NEWMARK_BETA!r}", "analysis Transient"]
    if material == "pimy":
        lines.append(stage(0))
    lines.append(run("gravity", GRAVITY_STEPS, GRAVITY_DT))
    if material == "pimy":
        lines += [stage(1), run("gravity_plastic", plastic_gravity_steps, GRAVITY_DT)]
    lines += ["setTime 0.0",
              f"timeSeries Path 2 -dt {DYN_DT} -values {{{vals}}} -factor 9.81",
              "pattern UniformExcitation 2 1 -accel 2",
              run("dynamic", dyn_steps, DYN_DT)]
    return "\n".join(lines) + "\n"
