"""Equation 9 from Kampik et al., Energies 2024, 17, 2967.

The paper writes the conductor stress term as sigma_2 * A_c.  This module
uses ``tension`` for that product, as requested.  Use one consistent unit
system: for example, metres, kgf/m, and kgf, or metres, N/m, and N.
"""

from math import isfinite


__all__ = ["equation_9_sag", "equation_9_tension"]


def equation_9_sag(a0, w, L_ins, G_ins, tension):
    """Calculate sag directly from Equation 9.

    Parameters
    ----------
    a0 : float
        Actual horizontal conductor span between its attachment points.
    w : float
        Conductor weight per unit length.
    L_ins : float
        Length of one insulator set.  Equation 9 assumes identical sets at
        both ends of the span.
    G_ins : float
        Weight of one insulator set in the state being calculated.
    tension : float
        The product sigma * A from Equation 9.  This corresponds to the
        horizontal tension quantity used by the parabolic sag formula.

    Returns
    -------
    float
        Sag f from Equation 9.
    """

    values = (a0, w, L_ins, G_ins, tension)
    if not all(isfinite(value) for value in values):
        raise ValueError("All inputs must be finite numbers.")
    if a0 <= 0:
        raise ValueError("a0 must be positive.")
    if w <= 0:
        raise ValueError("w must be positive.")
    if L_ins < 0:
        raise ValueError("L_ins must be non-negative.")
    if G_ins < 0:
        raise ValueError("G_ins must be non-negative.")
    if tension <= 0:
        raise ValueError("tension must be positive.")

    insulator_term = L_ins * (G_ins + w * a0) / (2.0 * tension)
    conductor_term = a0**2 * w / (8.0 * tension)
    return float(insulator_term + conductor_term)


def equation_9_tension(a0, w, L_ins, G_ins, target_sag):
    """Solve Equation 9 algebraically for tension at a target sag.

    The parameters and assumptions are the same as for
    :func:`equation_9_sag`.  The returned value is sigma * A in the paper's
    notation and can be compared with ``result["Th"]`` in this project.

    Parameters
    ----------
    a0 : float
        Actual horizontal conductor span between its attachment points.
    w : float
        Conductor weight per unit length.
    L_ins : float
        Length of one insulator set.
    G_ins : float
        Weight of one insulator set in the state being calculated.
    target_sag : float
        Desired sag f in Equation 9.

    Returns
    -------
    float
        Tension sigma * A required by Equation 9.
    """

    values = (a0, w, L_ins, G_ins, target_sag)
    if not all(isfinite(value) for value in values):
        raise ValueError("All inputs must be finite numbers.")
    if a0 <= 0:
        raise ValueError("a0 must be positive.")
    if w <= 0:
        raise ValueError("w must be positive.")
    if L_ins < 0:
        raise ValueError("L_ins must be non-negative.")
    if G_ins < 0:
        raise ValueError("G_ins must be non-negative.")
    if target_sag <= 0:
        raise ValueError("target_sag must be positive.")

    numerator = 0.5 * L_ins * (G_ins + w * a0) + a0**2 * w / 8.0
    return float(numerator / target_sag)
