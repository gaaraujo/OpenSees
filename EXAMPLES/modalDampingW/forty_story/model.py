"""40-story shear frame with linearly varying stiffness. See main.py for units and analysis."""


def build(ops, nstories, mass, k_bottom, k_top, uy, b):
    """Build nodes 0–N, story Steel01 springs with k linear from base to roof."""
    ops.node(0, 0)
    ops.fix(0, 1)
    for i in range(1, nstories + 1):
        ops.node(i, 0)
        ops.mass(i, mass)
        k = k_bottom + (k_top - k_bottom) * (i - 1) / (nstories - 1)
        fy = k * uy
        ops.uniaxialMaterial("Steel01", i, fy, k, b)
        ops.element("zeroLength", i, i - 1, i, "-mat", i, "-dir", 1)
    return list(range(0, nstories + 1))
