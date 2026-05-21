from catenary import catenary, catenary_Tv_left, catenary_Tv_right
from insulator import insulator_angle, insulator_attachment_point


def get_attachment_points_for_Th(
    A,
    B,
    w,
    Th,
    L_left,
    L_right,
    G_left,
    G_right,
    tol=1e-9,
    max_iter=100,
):
    """
    For a given trial horizontal tension Th, iterate to find the
    actual conductor attachment points C and D.

    Assumptions
    -----------
    - A is the left main support point
    - B is the right main support point
    - x_A < x_B
    - the conductor is suspended between C and D
    - the left insulator is attached at A
    - the right insulator is attached at B

    Returns
    -------
    dict with:
        C
        D
        theta_left
        theta_right
        catenary

    Plus a few extra logs:
        Tv_left
        Tv_right
        iterations
        converged
        max_change
    """

    xA, yA = A
    xB, yB = B

    if xA >= xB:
        raise ValueError("A must be the left support and B the right support (x_A < x_B).")

    if w <= 0:
        raise ValueError("w must be positive.")

    if Th <= 0:
        raise ValueError("Th must be positive.")

    if L_left < 0 or L_right < 0:
        raise ValueError("Insulator lengths must be non-negative.")

    # First guess:
    # both insulators horizontal, pointing inward
    theta_left = 0.0
    theta_right = 0.0

    C = insulator_attachment_point(A, L_left, theta_left, "left")
    D = insulator_attachment_point(B, L_right, theta_right, "right")

    for i in range(max_iter):
        if C[0] >= D[0]:
            raise RuntimeError("Actual conductor span became zero or negative during iteration.")

        cat = catenary(C, D, w, Th)

        Tv_left = catenary_Tv_left(cat)
        Tv_right = catenary_Tv_right(cat)

        theta_left = insulator_angle(Th, Tv_left, G_left)
        theta_right = insulator_angle(Th, Tv_right, G_right)

        C_new = insulator_attachment_point(A, L_left, theta_left, "left")
        D_new = insulator_attachment_point(B, L_right, theta_right, "right")

        max_change = max(
            abs(C_new[0] - C[0]),
            abs(C_new[1] - C[1]),
            abs(D_new[0] - D[0]),
            abs(D_new[1] - D[1]),
        )

        C = C_new
        D = D_new

        if max_change < tol:
            cat = catenary(C, D, w, Th)

            return {
                "C": C,
                "D": D,
                "theta_left": theta_left,
                "theta_right": theta_right,
                "catenary": cat,
                "Tv_left": catenary_Tv_left(cat),
                "Tv_right": catenary_Tv_right(cat),
                "iterations": i + 1,
                "converged": True,
                "max_change": max_change,
            }

    # If tolerance was not reached, return the best available result anyway
    cat = catenary(C, D, w, Th)

    return {
        "C": C,
        "D": D,
        "theta_left": theta_left,
        "theta_right": theta_right,
        "catenary": cat,
        "Tv_left": catenary_Tv_left(cat),
        "Tv_right": catenary_Tv_right(cat),
        "iterations": max_iter,
        "converged": False,
        "max_change": max_change,
    }
