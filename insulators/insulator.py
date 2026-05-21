import numpy as np


def insulator_angle(Th, Tv, G):
    """
    Return the signed insulator angle theta in radians.

    Convention
    ----------
    theta > 0  -> above horizontal
    theta < 0  -> below horizontal

    Force convention
    ----------------
    positive vertical force = upward
    negative vertical force = downward

    Inputs
    ------
    Th : horizontal force magnitude
    Tv : signed vertical force of the conductor on the support / insulator
    G  : insulator weight entered by the user

    Notes
    -----
    The insulator weight is always treated internally as downward:
        G_signed = -abs(G)

    The angle is measured from the local horizontal
    pointing toward the span interior.
    """

    G_signed = -abs(G)

    theta = np.arctan2(Tv + G_signed, Th)

    return theta

def insulator_offset(L, theta):
    """
    Return the local offset of an insulator of length L.

    Convention
    ----------
    dx > 0 : toward the span interior
    dy > 0 : upward
    dy < 0 : downward

    Inputs
    ------
    L     : insulator length
    theta : signed insulator angle in radians

    Returns
    -------
    dx, dy
    """

    dx = L * np.cos(theta)
    dy = L * np.sin(theta)

    return dx, dy

def insulator_attachment_point(P, L, theta, side):
    """
    Return the global conductor attachment point of an insulator.

    Parameters
    ----------
    P : tuple
        Main support point, e.g. (x, y)
    L : float
        Insulator length
    theta : float
        Signed insulator angle in radians

        Convention:
        - theta > 0 -> above horizontal
        - theta < 0 -> below horizontal

    side : str
        "left"  -> support is on the left side of the span
        "right" -> support is on the right side of the span

    Returns
    -------
    tuple
        The conductor attachment point (x_new, y_new)
    """

    x, y = P

    dx, dy = insulator_offset(L, theta)

    if side == "left":
        return (x + dx, y + dy)

    if side == "right":
        return (x - dx, y + dy)

    raise ValueError("side must be 'left' or 'right'.")


if __name__ == "__main__":
    Tv = 100
    Th = 300
    G = 100
    L = 5
    P = (0,0)
    theta = insulator_angle(Th, Tv, G)
    
    print(theta)
    print(insulator_attachment_point(P, L, theta, "left"))
    print(insulator_attachment_point(P, L, theta, "right"))