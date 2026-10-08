"""
Two-story steel MRF (Kolay & Ricles). Units: kN, m, s.
"""

import math

ROOF_NODE = 8

kN = m = s = 1.0
kip = 4.44822 * kN
inch = 0.0254 * m
foot = 12 * inch
lbf = kip / 1000
g = 9.80665 * m / s**2
kg = (kN / 1000.0) * s**2 / m  # physical kg → consistent mass (tonne) in kN–m–s
MPa = 1000.0 * kN / m**2
GPa = 1000 * MPa


def create_model(ops):
    """Geometry, materials, elements (no loads or analysis)."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    L, H = 6.0 * m, 3.0 * m
    Lb, Hc = L / 12, H / 6

    # floor 2
    for i, x in enumerate(
        (0, Lb, 2 * Lb, L / 2, L - 2 * Lb, L - Lb, L, L + 0.2 * L), start=1
    ):
        ops.node(i, x, 2 * H)
    # floor 1
    for i, x in enumerate(
        (0, Lb, 2 * Lb, L / 2, L - 2 * Lb, L - Lb, L, L + 0.2 * L), start=9
    ):
        ops.node(i, x, H)
    # bases
    ops.node(17, 0, 0)
    ops.node(18, L, 0)
    ops.node(19, L + 0.2 * L, 0)

    # column interior nodes (left 20-26, right 27-33)
    zcols = (
        (H - 2 * Hc) / 2,
        H - 2 * Hc,
        H - Hc,
        H + Hc,
        H + 2 * Hc,
        2 * H - 2 * Hc,
        2 * H - Hc,
    )
    for j, z in enumerate(zcols, start=20):
        ops.node(j, 0, z)
    for j, z in enumerate(zcols, start=27):
        ops.node(j, L, z)

    for n in (17, 18, 19):
        ops.fix(n, 1, 1, 0)
    ops.rigidDiaphragm(1, 8, 4)
    ops.rigidDiaphragm(1, 16, 12)
    ops.mass(8, 50.97e3 * kg, kg, 1 * kg * m)
    ops.mass(16, 50.97e3 * kg, kg, 1 * kg * m)

    ops.uniaxialMaterial("Steel01", 1, 345 * MPa, 200 * GPa, 0.01)
    ops.section("WFSection2d", 1, 1, 23.6 * inch, 0.395 * inch, 7.01 * inch, 0.505 * inch, 10, 3)
    ops.section("WFSection2d", 2, 1, 14.5 * inch, 0.59 * inch, 14.7 * inch, 0.94 * inch, 10, 3)

    ops.geomTransf("Linear", 1)
    ops.beamIntegration("Legendre", 1, 1, 2)
    rho_b = (55 * lbf / foot) / g
    beams = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15)]
    for e, (a, b) in enumerate(beams, start=1):
        ops.element("dispBeamColumn", e, a, b, 1, 1, "-cMass", "-mass", rho_b)

    ops.geomTransf("Linear", 2)
    ops.beamIntegration("Legendre", 2, 2, 2)
    rho_c = (120 * lbf / foot) / g
    cols = [
        (15, 17, 20), (16, 20, 21), (17, 21, 22), (18, 22, 9), (19, 9, 23),
        (20, 23, 24), (21, 24, 25), (22, 25, 26), (23, 26, 1),
        (24, 18, 27), (25, 27, 28), (26, 28, 29), (27, 29, 15), (28, 15, 30),
        (29, 30, 31), (30, 31, 32), (31, 32, 33), (32, 33, 7),
    ]
    for e, a, b in cols:
        ops.element("dispBeamColumn", e, a, b, 2, 2, "-cMass", "-mass", rho_c)

    ops.geomTransf("PDelta", 3)
    A, I = 9.76e-2 * m**2, 7.125e-4 * m**4
    for e, a, b in ((33, 19, 16), (34, 16, 8)):
        ops.element("elasticBeamColumn", e, a, b, A, 200 * GPa, I, 3, "-mass", 1e-3 * kg / m, "-cMass")


def apply_gravity(ops):
    """Static gravity load to constant (committed) state; leaves Static analysis active."""
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(8, 0, -500 * kN, 0)
    ops.load(16, 0, -500 * kN, 0)
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-6, 10)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    if ops.analyze(10) != 0:
        raise RuntimeError("gravity static analysis failed to converge")
    ops.loadConst("-time", 0.0)


def _eigenvalues(ops, n_modes):
    raw = ops.eigen(n_modes)
    if raw is None:
        raise RuntimeError("eigen returned no values")
    if isinstance(raw, (int, float)):
        return [float(raw)]
    return [float(lam) for lam in raw]


def eigen_after_gravity(ops, n_modes):
    """Generalized eigenvalues on post-gravity tangent (call while Static analysis is active)."""
    lambdas = _eigenvalues(ops, n_modes)
    if any(lam <= 0.0 for lam in lambdas):
        raise ValueError(f"eigen({n_modes}) returned non-positive λ: {lambdas}")
    omegas = [math.sqrt(lam) for lam in lambdas]
    periods = [2.0 * math.pi / w for w in omegas]
    return lambdas, omegas, periods


