"""Four-story shear frame (Scott 2019). See main.py for units and analysis."""


def build(ops, k, m, uy, b):
    """Build nodes 0–4, Steel01 zeroLength springs, Scott story connectivity."""
    fy = k * uy
    ops.node(0, 0)
    ops.fix(0, 1)
    ops.node(1, 0)
    ops.mass(1, m)
    ops.node(2, 0)
    ops.mass(2, m)
    ops.node(3, 0)
    ops.mass(3, m)
    ops.node(4, 0)
    ops.mass(4, 0.5 * m)

    ops.uniaxialMaterial("Steel01", 1, fy, k, b)
    ops.element("zeroLength", 1, 0, 1, "-mat", 1, "-dir", 1)
    ops.element("zeroLength", 2, 1, 3, "-mat", 1, "-dir", 1)
    ops.element("zeroLength", 3, 3, 2, "-mat", 1, "-dir", 1)
    ops.element("zeroLength", 4, 3, 4, "-mat", 1, "-dir", 1)
    return [0, 1, 2, 3, 4]
