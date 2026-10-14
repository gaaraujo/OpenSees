"""Soil-column site-response model for PythonSparse benchmarks (OpenSeesPy + Tcl).

Geometry and elastic layers: Femora's SiteResponse Example 3
(https://github.com/GeotechUW/Femora/tree/main/examples/SiteResponse/Example3),
a 10 m x 10 m x 18 m column of five layers on stdBrick elements with body
forces and laminar side boundaries (perimeter nodes at each height share
dofs 1-3).

Base and input: the compliant-base method of McGann and Arduino's total-stress
site response example
(https://opensees.berkeley.edu/wiki/index.php/Site_Response_Analysis_of_a_Layered_Soil_Column_(Total_Stress_Analysis)),
the same as in OSU_SSI_Bridge (soil/BuildShinLysmer.tcl):

* base nodes fixed vertically only and tied horizontally (equalDOF 1 2);
* Lysmer-Kuhlemeyer dashpots in x and y between the base and a fixed node,
  c = rho_rock * Vs_rock * A_base (rock: Vs = 760 m/s, rho = 2.4 t/m^3, as in
  both sources);
* the motion enters as a horizontal force F(t) = factor * c * v(t) at the base
  (Joyner and Chen 1975) from a velocity record. ``factor`` = 2 follows
  OSU_SSI_Bridge (outcrop motion = twice the incident wave); the wiki example
  uses 1.

Analysis (as in the wiki example): gravity as a transient analysis with very
large steps (dt = 500 s), first elastic (10 steps), then, for ``pimy``, the
plastic stage (40 steps); then the transient with Newmark gamma = 0.5,
beta = 0.25 and Rayleigh damping (Femora's 3% at 2.76 and 13.84 Hz).

``material='pimy'`` uses PressureIndependMultiYield with the clay values
suggested on the OpenSees wiki (stiff / medium / soft clay, bottom to top).
``refine`` divides dx, dy and every layer's dz to scale the model.

Both builders create nodes and elements in the same order, so equation
numbering (RCM) and the assembled matrices match between OpenSeesPy and Tcl.
"""

from __future__ import annotations

import math
from pathlib import Path

XMIN, XMAX, YMIN, YMAX, ZMIN = -5.0, 5.0, -5.0, 5.0, -18.0
DX = DY = 1.0
# name, G (Pa), unit weight (kN/m^3), nu, thickness (m), dz (m) -- bottom to top.
# Geometry and elastic properties from the Femora site-response example 3
# (dense Ottawa x3, loose Ottawa, dense Monterey sand); the "pimy" variant keeps
# the geometry and replaces each layer with a wiki clay (PIMY_PROFILE).
LAYERS = [
    ("Layer 1", 145.0e6, 19.9, 0.3, 2.6, 1.3),
    ("Layer 2", 145.0e6, 19.9, 0.3, 2.4, 1.2),
    ("Layer 3", 145.0e6, 19.9, 0.3, 5.0, 1.0),
    ("Layer 4", 75.0e6, 19.1, 0.3, 6.0, 0.5),
    ("Layer 5", 42.0e6, 19.8, 0.3, 2.0, 0.5),
]
# PressureIndependMultiYield suggested values (OpenSees wiki):
# rho (t/m^3), refShearModul (kPa), refBulkModul (kPa), cohesion (kPa), peakShearStra
PIMY_CLAYS = {
    "soft": (1.3, 1.3e4, 6.5e4, 18.0, 0.1),
    "medium": (1.5, 6.0e4, 3.0e5, 37.0, 0.1),
    "stiff": (1.8, 1.5e5, 7.5e5, 75.0, 0.1),
}
PIMY_PROFILE = ["stiff", "stiff", "stiff", "medium", "soft"]  # bottom to top
# frictionAng, refPress, pressDependCoe, noYieldSurf: now required by OpenSees.
# phi = d = 0 as in the wiki table; p'r = 80 kPa is where the table gives gamma_max.
PIMY_PHI, PIMY_REF_PRESS, PIMY_D, PIMY_YIELD_SURFACES = 0.0, 80.0, 0.0, 20

ROCK_VS, ROCK_RHO = 760.0, 2.4          # m/s, t/m^3
INPUT_FACTOR = 2.0                      # F = INPUT_FACTOR * c * v (OSU convention)

DAMP_XI, DAMP_F1, DAMP_F2 = 0.03, 2.76, 13.84
NEWMARK_GAMMA, NEWMARK_BETA = 0.5, 0.25
GRAVITY_DT = 500.0
GRAVITY_ELASTIC_STEPS, GRAVITY_PLASTIC_STEPS = 10, 10  # plastic stage converges in 1 iter/step
GRAVITY_TEST = ("NormDispIncr", 1e-5, 30)
DYN_DT = 0.001
DYN_TEST = ("NormDispIncr", 1e-6, 30)

GHOST_NODE = 9_000_001
DASHPOT_MAT, DASHPOT_ELE_X, DASHPOT_ELE_Y = 900, 9_000_001, 9_000_002


def layer_props(material: str = "elastic"):
    """(E kPa, nu, rho t/m^3, G kPa) per layer."""
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


def grid(refine: int = 1, plan: int | None = None):
    """Node coordinates. ``plan`` overrides the number of elements per side in
    plan (laminar sides make the response independent of it)."""
    nx = plan if plan else int(round((XMAX - XMIN) / DX)) * refine
    ny = plan if plan else int(round((YMAX - YMIN) / DY)) * refine
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


def top_node(refine: int = 1, plan: int | None = None) -> int:
    """Surface corner node (0, 0, top); with laminar sides it carries the column's motion."""
    xs, ys, zs, _ = grid(refine, plan)
    return 1 + len(xs) * len(ys) * (len(zs) - 1)


def base_master(refine: int = 1) -> int:
    """Base node carrying the dashpots and the input force (corner node 1)."""
    return 1


def dashpot_coefficient() -> float:
    """c = rho_rock * Vs_rock * A_base (kN s/m)."""
    return ROCK_RHO * ROCK_VS * (XMAX - XMIN) * (YMAX - YMIN)


def _commands(refine: int, material: str, plan: int | None = None):
    """Yield (command, args) in build order; shared by both builders."""
    xs, ys, zs, layer_of_cell = grid(refine, plan)
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
    # Base: fixed vertically, tied horizontally to the master base node.
    master = base_master(refine)
    for j in range(ny + 1):
        for i in range(nx + 1):
            n = nid(i, j, 0)
            yield "fix", (n, 0, 0, 1)
            if n != master:
                yield "equalDOF", (master, n, 1, 2)
    # Laminar sides: perimeter nodes at each height share dofs 1-3.
    for k in range(1, nz + 1):
        perim = [nid(i, j, k) for j in range(ny + 1) for i in range(nx + 1)
                 if i in (0, nx) or j in (0, ny)]
        for slave in perim[1:]:
            yield "equalDOF", (perim[0], slave, 1, 2, 3)
    # Lysmer-Kuhlemeyer dashpots (x and y) from a fixed node to the base.
    yield "node", (GHOST_NODE, xs[0], ys[0], zs[0])
    yield "fix", (GHOST_NODE, 1, 1, 1)
    yield "uniaxialMaterial", ("Viscous", DASHPOT_MAT, dashpot_coefficient(), 1.0)
    yield "element", ("zeroLength", DASHPOT_ELE_X, GHOST_NODE, master,
                      "-mat", DASHPOT_MAT, "-dir", 1)
    yield "element", ("zeroLength", DASHPOT_ELE_Y, GHOST_NODE, master,
                      "-mat", DASHPOT_MAT, "-dir", 2)
    w1, w2 = 2 * math.pi * DAMP_F1, 2 * math.pi * DAMP_F2
    alpha_m = 2 * DAMP_XI * w1 * w2 / (w1 + w2)
    beta_k = 2 * DAMP_XI / (w1 + w2)
    yield "rayleigh", (alpha_m, beta_k, 0.0, 0.0)


def tcl_slabs(refine: int = 1, material: str = "elastic", plan: int | None = None,
              nparts: int = 2) -> list[str]:
    """Per-rank Tcl for OpenSeesMP without ``partition``: rank r builds only a horizontal slab
    of element layers, so memory scales with the model, not with ranks x model.

    Every rank defines the materials and Rayleigh damping; rank r the nodes of z-levels
    k0..k1 and the elements of layers k0..k1-1, so the boundary planes are shared with the
    neighbouring slabs. The laminar ties link nodes at one height, so a slab holds every tie
    of its levels; ties on a shared plane are defined on both ranks, so both subdomains
    eliminate the same constrained dofs. Rank 0 also holds the base (fixities, base ties,
    dashpots, ghost node); the input pattern (:func:`tcl_input`) goes on rank 0 as well.
    """
    xs, ys, zs, _ = grid(refine, plan)
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    if nparts > nz:
        raise ValueError(f"{nparts} slabs for {nz} element layers")
    bounds = [round(r * nz / nparts) for r in range(nparts + 1)]
    per_plane = (nx + 1) * (ny + 1)
    node_k = lambda tag: (tag - 1) // per_plane  # noqa: E731
    ele_k = lambda tag: (tag - 1) // (nx * ny)   # noqa: E731
    base = {GHOST_NODE, DASHPOT_ELE_X, DASHPOT_ELE_Y}

    def fmt(cmd, args):
        return cmd + " " + " ".join(repr(a) if isinstance(a, float) else str(a) for a in args)

    out = [["wipe"] for _ in range(nparts)]
    for cmd, args in _commands(refine, material, plan):
        if cmd in ("model", "nDMaterial", "uniaxialMaterial", "rayleigh"):
            ranks = range(nparts)
        elif cmd in ("node", "fix") and args[0] in base or cmd == "element" and args[1] in base:
            ranks = [0]
        elif cmd == "node" or cmd == "fix":
            k = node_k(args[0])
            ranks = [r for r in range(nparts) if bounds[r] <= k <= bounds[r + 1]]
        elif cmd == "equalDOF":
            k = node_k(args[1])
            ranks = [r for r in range(nparts) if bounds[r] <= k <= bounds[r + 1]]
        elif cmd == "element":
            k = ele_k(args[1])
            ranks = [r for r in range(nparts) if bounds[r] <= k < bounds[r + 1]]
        else:
            raise ValueError(f"tcl_slabs: unhandled command {cmd}")
        line = fmt(cmd, args)
        for r in ranks:
            out[r].append(line)
    return ["\n".join(lines) + "\n" for lines in out]


def tie_violation(ops, refine: int = 1, material: str = "elastic", plan: int | None = None) -> dict:
    """Largest |u_retained - u_constrained| over the equalDOF ties, absolute and relative to
    the largest tied displacement. Zero (to round-off) under Transformation; under Penalty
    it measures how well the penalty factor enforces the ties."""
    worst, umax = 0.0, 0.0
    for cmd, args in _commands(refine, material, plan):
        if cmd != "equalDOF":
            continue
        master, slave, *dofs = args
        for d in dofs:
            um, us = ops.nodeDisp(master, d), ops.nodeDisp(slave, d)
            worst = max(worst, abs(us - um))
            umax = max(umax, abs(um), abs(us))
    return {"max_abs": worst, "max_rel": worst / umax if umax > 0 else 0.0, "max_disp": umax}


def build(ops, refine: int = 1, material: str = "elastic", plan: int | None = None) -> None:
    """Build the column in OpenSeesPy."""
    ops.wipe()
    for cmd, args in _commands(refine, material, plan):
        getattr(ops, cmd)(*args)


def tcl(refine: int = 1, material: str = "elastic", plan: int | None = None) -> str:
    """Tcl script building the same column."""
    lines = ["wipe"]
    for cmd, args in _commands(refine, material, plan):
        lines.append(cmd + " " + " ".join(repr(a) if isinstance(a, float) else str(a)
                                          for a in args))
    return "\n".join(lines) + "\n"


def run_stamp() -> dict:
    """Time, host and code version for result headers (runs append, never overwrite)."""
    import datetime
    import socket
    import subprocess

    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse",
                                 "--short", "HEAD"], capture_output=True, text=True,
                                timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "?"
    return {"time": datetime.datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(), "commit": commit or "?"}


# ---------------------------------------------------------------------------
# Input motion
# ---------------------------------------------------------------------------


class Motion:
    """Velocity record (m/s) sampled at ``dt``."""

    def __init__(self, values, dt, name):
        self.values = list(values)
        self.dt = float(dt)
        self.name = name

    @classmethod
    def from_file(cls, path, dt):
        """One value per line or whitespace-separated (wiki velocityHistory.out: dt 0.005)."""
        vals = [float(v) for v in Path(path).read_text().split()]
        return cls(vals, dt, Path(path).name)

    @classmethod
    def sweep(cls, duration=10.0, dt=0.005, f0=1.0, f1=10.0, peak_velocity=0.3):
        """Synthetic sine-sweep velocity (fallback when no record is given)."""
        n = int(round(duration / dt)) + 1
        vals = [peak_velocity * math.sin(2 * math.pi * (f0 + (f1 - f0) * (i * dt) / duration / 2)
                                         * (i * dt)) for i in range(n)]
        return cls(vals, dt, f"sweep {f0}-{f1} Hz")

    def strong_start(self) -> float:
        """Record time of the velocity zero crossing just before the peak |v|."""
        k = max(range(len(self.values)), key=lambda i: abs(self.values[i]))
        while k > 0 and self.values[k - 1] * self.values[k] > 0:
            k -= 1
        return k * self.dt


def add_input(ops, motion=None):
    """Base force pattern F = INPUT_FACTOR * c * v(t) on the base master node.

    Defining it before gravity is harmless: gravity steps are at t >= GRAVITY_DT,
    past the end of the record, where the Path series is zero. Call it before
    ``ops.partition()`` under MPI, then pass ``input_defined=True`` to analysis.
    """
    motion = motion or Motion.sweep()
    assert GRAVITY_DT > (len(motion.values) - 1) * motion.dt
    ops.timeSeries("Path", 2, "-dt", motion.dt, "-values", *motion.values,
                   "-factor", INPUT_FACTOR * dashpot_coefficient())
    ops.pattern("Plain", 2, 2)
    ops.load(base_master(), 1.0, 0.0, 0.0)


def analysis(ops, system_args, algorithm, material="elastic", motion=None, dyn_steps=50,
             start_time=None, gravity_algorithm=None, dyn_dt=DYN_DT, on_step=None,
             constraints=("Transformation",), numberer=("RCM",), input_defined=False,
             step_probe=None):
    """Elastic gravity, plastic gravity (pimy), then ``dyn_steps`` of shaking.

    The transient window starts at record time ``start_time`` (default: the
    zero crossing of velocity before the peak, so the base force does not
    jump). Returns {phase: (ok, iterations per step, seconds, seconds per step)}.
    ``step_probe(phase)`` is called after every step of every phase (outside
    the timed region); ``on_step()`` after every transient step.
    """
    import time

    motion = motion or Motion.sweep()
    start_time = motion.strong_start() if start_time is None else start_time

    def setup(test, alg):
        ops.wipeAnalysis()
        ops.constraints(*constraints)
        ops.numberer(*numberer)
        ops.system(*system_args)
        ops.test(*test)
        ops.algorithm(*alg)
        ops.integrator("Newmark", NEWMARK_GAMMA, NEWMARK_BETA)
        ops.analysis("Transient")

    def run(phase, steps, dt, callback=None):
        iters, step_seconds, ok = [], [], 0
        for _ in range(steps):
            t = time.perf_counter()
            ok = ops.analyze(1, dt)
            step_seconds.append(time.perf_counter() - t)
            iters.append(ops.testIter())
            if step_probe is not None:
                step_probe(phase)
            if callback is not None:
                callback()
            if ok != 0:
                break
        return ok, iters, sum(step_seconds), step_seconds

    out = {}
    setup(GRAVITY_TEST, gravity_algorithm or algorithm)
    if material == "pimy":
        for tag in range(1, len(LAYERS) + 1):
            ops.updateMaterialStage("-material", tag, "-stage", 0)
    out["gravity_elastic"] = run("gravity_elastic", GRAVITY_ELASTIC_STEPS, GRAVITY_DT)
    if material == "pimy" and out["gravity_elastic"][0] == 0:
        for tag in range(1, len(LAYERS) + 1):
            ops.updateMaterialStage("-material", tag, "-stage", 1)
        out["gravity_plastic"] = run("gravity_plastic", GRAVITY_PLASTIC_STEPS, GRAVITY_DT)
    if any(v[0] != 0 for v in out.values()) or dyn_steps <= 0:
        return out
    ops.setTime(start_time)
    if not input_defined:
        add_input(ops, motion)
    setup(DYN_TEST, algorithm)
    out["transient"] = run("transient", dyn_steps, dyn_dt, on_step)
    return out


def tcl_input(motion=None) -> str:
    """Tcl twin of :func:`add_input` (define before ``partition`` under OpenSeesMP)."""
    motion = motion or Motion.sweep()
    vals = " ".join(repr(v) for v in motion.values)
    return (f"timeSeries Path 2 -dt {motion.dt!r} -values {{{vals}}} "
            f"-factor {INPUT_FACTOR * dashpot_coefficient()!r}\n"
            f"pattern Plain 2 2 {{ load {base_master()} 1.0 0.0 0.0 }}\n")


def tcl_analysis(system_line, algorithm, material="elastic", motion=None, dyn_steps=50,
                 start_time=None, dyn_dt=DYN_DT, numberer="RCM", input_defined=False,
                 rank0_only=False, constraints="Transformation", result_file=None,
                 dyn_integrator=None, dyn_algorithm=None, trace_node=None):
    """Tcl twin of :func:`analysis`; prints one RESULT line per phase:
    ``RESULT <phase> ok=.. seconds=.. neq=.. iters=i,i,.. times=s,s,..``
    (only from rank 0 with ``rank0_only``, for OpenSeesMP). With ``result_file`` the
    lines are also appended to that file, which other ranks' output cannot interleave.
    ``dyn_integrator`` / ``dyn_algorithm`` replace Newmark / ``algorithm`` in the transient
    phase only (gravity keeps them). ``trace_node`` (a node tag or a list of tags): the rank
    holding each node writes its x displacement after every transient step to
    trace_<pid>_<node>.txt (outside the timing)."""
    motion = motion or Motion.sweep()
    start_time = motion.strong_start() if start_time is None else start_time
    guard = "if {[getPID] == 0} " if rank0_only else "if {1} "

    def setup(test, alg=algorithm, integ=f"Newmark {NEWMARK_GAMMA} {NEWMARK_BETA}"):
        return "\n".join(["wipeAnalysis", f"constraints {constraints}", f"numberer {numberer}",
                          system_line, "test " + " ".join(map(str, test)),
                          f"algorithm {alg}", f"integrator {integ}", "analysis Transient"])

    def run(name, steps, dt):
        line = (f'"RESULT {name} ok=$ok seconds=[expr $total * 1e-6] '
                'neq=[systemSize] iters=[join $iters ,] times=[join $times ,]"')
        to_file = (f"; set fh [open {{{result_file}}} a]; puts $fh {line}; close $fh"
                   if result_file else "")
        nodes = ([trace_node] if isinstance(trace_node, int) else list(trace_node or [])) \
            if name == "transient" else []
        pre = [f"set has{n} [expr {{[lsearch -exact [getNodeTags] {n}] >= 0}}]; set tr{n} {{}}"
               for n in nodes]
        rec = [f"  if {{$has{n}}} {{lappend tr{n} [format %.10e [nodeDisp {n} 1]]}}" for n in nodes]
        post = [f"if {{$has{n}}} {{set fh [open trace_[getPID]_{n}.txt w]; puts $fh [join $tr{n} ,]; "
                "close $fh}" for n in nodes]
        return "\n".join([
            "set iters {}", "set times {}", "set ok 0", "set total 0", *pre,
            f"for {{set i 0}} {{$i < {steps}}} {{incr i}} {{",
            "  set t [clock microseconds]",
            f"  set ok [analyze 1 {dt}]",
            "  set dt_us [expr [clock microseconds] - $t]",
            "  set total [expr $total + $dt_us]",
            "  lappend times [expr $dt_us * 1e-6]", "  lappend iters [testIter]",
            *rec,
            "  if {$ok != 0} break", "}",
            guard + f"{{puts {line}{to_file}}}",
            *post,
            "if {$ok != 0} {exit}",
        ])

    stage = lambda s: "\n".join(  # noqa: E731
        f"updateMaterialStage -material {t} -stage {s}" for t in range(1, len(LAYERS) + 1))
    lines = [setup(GRAVITY_TEST)]
    if material == "pimy":
        lines.append(stage(0))
    lines.append(run("gravity_elastic", GRAVITY_ELASTIC_STEPS, GRAVITY_DT))
    if material == "pimy":
        lines += [stage(1), run("gravity_plastic", GRAVITY_PLASTIC_STEPS, GRAVITY_DT)]
    if dyn_steps > 0:
        lines.append(f"setTime {start_time!r}")
        if not input_defined:
            lines.append(tcl_input(motion))
        lines += [setup(DYN_TEST, dyn_algorithm or algorithm,
                        dyn_integrator or f"Newmark {NEWMARK_GAMMA} {NEWMARK_BETA}"),
                  run("transient", dyn_steps, dyn_dt)]
    return "\n".join(lines) + "\n"
