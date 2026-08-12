from .line import line_between_points
from .catenary import catenary
from .sag import catenary_sag_vertical


def Th_for_target_sag(A, B, w, target_sag, tol=1e-6, max_iter=100):
    """
    Find the horizontal tension Th that gives the target vertical sag.

    Uses bisection

    Parameters
    ----------
    A, B : tuple
        Suspension points, e.g. A = (x1, y1), B = (x2, y2)
    w : float
        Conductor weight per unit length
    target_sag : float
        Desired maximum vertical sag relative to the chord AB
    tol : float
        Allowed difference between computed sag and target sag
    max_iter : int
        Maximum number of bisection iterations

    Returns
    -------
    float
        The horizontal tension Th
    """

    if w <= 0:
        raise ValueError("w must be positive.")

    if target_sag <= 0:
        raise ValueError("target_sag must be positive.")

    x1, y1 = A
    x2, y2 = B


    # Rough first guess from the parabolic formula
    span = abs(x2 - x1)
    Th_guess = w * span**2 / (8.0 * target_sag)

    # Start with a bracket around that guess
    Th_low = max(1e-6, Th_guess * 0.25)
    Th_high = Th_guess * 4.0

    line = line_between_points(A, B)

    def sag_for_Th(Th):
        cat = catenary(A, B, w, Th)
        sag_info = catenary_sag_vertical(cat, line)
        return sag_info["sag_max"]

    sag_low = sag_for_Th(Th_low)
    sag_high = sag_for_Th(Th_high)

    # low tension  -> larger sag
    # high tension -> smaller sag
    expand_count = 0
    while not (sag_low >= target_sag and sag_high <= target_sag):
        if expand_count > 50:
            raise RuntimeError("Could not bracket the solution for Th.")

        if sag_low < target_sag:
            Th_low *= 0.5
            sag_low = sag_for_Th(Th_low)

        if sag_high > target_sag:
            Th_high *= 2.0
            sag_high = sag_for_Th(Th_high)

        expand_count += 1

    for i in range(max_iter):
        Th_mid = 0.5 * (Th_low + Th_high)

        sag_mid = sag_for_Th(Th_mid)
        error = sag_mid - target_sag

        if abs(error) < tol:
            return Th_mid

        if sag_mid > target_sag:
            Th_low = Th_mid
        else:
            Th_high = Th_mid

    return 0.5 * (Th_low + Th_high)