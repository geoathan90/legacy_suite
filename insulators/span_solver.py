from .line import line_between_points
from .sag import catenary_sag_vertical
from .attachment_solver import get_attachment_points_for_Th


def solve_span_for_target_sag(
    A,
    B,
    w,
    target_sag,
    L_left,
    L_right,
    G_left,
    G_right,
    tol=1e-6,
    max_iter=100,
    attachment_tol=1e-9,
    attachment_max_iter=100,
):
    """
    Solve for the horizontal tension Th that gives the target vertical sag
    when rigid insulators are included.

    The target sag is measured as:
        maximum vertical distance between the actual conductor catenary
        and the straight line through the main support points A and B.

    Returns a dictionary with the final solved span state.
    """

    if w <= 0:
        raise ValueError("w must be positive.")

    if target_sag <= 0:
        raise ValueError("target_sag must be positive.")

    xA, yA = A
    xB, yB = B

    if xA == xB:
        raise ValueError("This version needs A and B to have different x-values.")

    if xA > xB:
        raise ValueError("A must be the left support and B the right support.")

    line_AB = line_between_points(A, B)

    # Rough first guess from the simple no-insulator parabolic formula
    span = xB - xA
    Th_guess = w * span**2 / (8.0 * target_sag)

    # Initial bracket
    Th_low = max(1e-6, Th_guess * 0.25)
    Th_high = Th_guess * 4.0

    def span_state_for_Th(Th):
        attachment_result = get_attachment_points_for_Th(
            A=A,
            B=B,
            w=w,
            Th=Th,
            L_left=L_left,
            L_right=L_right,
            G_left=G_left,
            G_right=G_right,
            tol=attachment_tol,
            max_iter=attachment_max_iter,
        )

        cat = attachment_result["catenary"]
        sag_info = catenary_sag_vertical(cat, line_AB)

        return sag_info["sag_max"], attachment_result, sag_info

    sag_low, attach_low, sag_info_low = span_state_for_Th(Th_low)
    sag_high, attach_high, sag_info_high = span_state_for_Th(Th_high)

    # low tension  -> larger sag
    # high tension -> smaller sag
    expand_count = 0
    while not (sag_low >= target_sag and sag_high <= target_sag):
        if expand_count > 50:
            raise RuntimeError("Could not bracket the solution for Th.")

        if sag_low < target_sag:
            Th_low *= 0.5
            sag_low, attach_low, sag_info_low = span_state_for_Th(Th_low)

        if sag_high > target_sag:
            Th_high *= 2.0
            sag_high, attach_high, sag_info_high = span_state_for_Th(Th_high)

        expand_count += 1

    for i in range(max_iter):
        Th_mid = 0.5 * (Th_low + Th_high)

        sag_mid, attach_mid, sag_info_mid = span_state_for_Th(Th_mid)
        error = sag_mid - target_sag

        if abs(error) < tol:
            return {
                "Th": Th_mid,
                "sag": sag_mid,
                "error": error,
                "iterations": i + 1,
                "converged": True,
                "C": attach_mid["C"],
                "D": attach_mid["D"],
                "theta_left": attach_mid["theta_left"],
                "theta_right": attach_mid["theta_right"],
                "catenary": attach_mid["catenary"],
                "Tv_left": attach_mid["Tv_left"],
                "Tv_right": attach_mid["Tv_right"],
                "attachment_iterations": attach_mid["iterations"],
                "attachment_converged": attach_mid["converged"],
                "attachment_max_change": attach_mid["max_change"],
                "x_sag": sag_info_mid["x_sag"],
                "y_curve": sag_info_mid["y_curve"],
                "y_line": sag_info_mid["y_line"],
                "line": line_AB,
            }

        if sag_mid > target_sag:
            Th_low = Th_mid
        else:
            Th_high = Th_mid

    # Best available result if outer tolerance was not reached
    Th_mid = 0.5 * (Th_low + Th_high)
    sag_mid, attach_mid, sag_info_mid = span_state_for_Th(Th_mid)
    error = sag_mid - target_sag

    return {
        "Th": Th_mid,
        "sag": sag_mid,
        "error": error,
        "iterations": max_iter,
        "converged": False,
        "C": attach_mid["C"],
        "D": attach_mid["D"],
        "theta_left": attach_mid["theta_left"],
        "theta_right": attach_mid["theta_right"],
        "catenary": attach_mid["catenary"],
        "Tv_left": attach_mid["Tv_left"],
        "Tv_right": attach_mid["Tv_right"],
        "attachment_iterations": attach_mid["iterations"],
        "attachment_converged": attach_mid["converged"],
        "attachment_max_change": attach_mid["max_change"],
        "x_sag": sag_info_mid["x_sag"],
        "y_curve": sag_info_mid["y_curve"],
        "y_line": sag_info_mid["y_line"],
        "line": line_AB,
    }
