import numpy as np
from catenary import catenary_y
from line import line_y


def catenary_sag_vertical(cat, line):
    """
    Return the maximum vertical sag of the catenary
    relative to the given straight line.

    This version uses the analytic critical point of

        sag(x) = y_line(x) - y_catenary(x)

    instead of sampling.

    Returns a dictionary with:
        sag_max  : maximum vertical sag
        x_sag    : x-value where that maximum occurs
        y_curve  : catenary y-value at that x
        y_line   : line y-value at that x
    """

    alpha = cat["alpha"]
    x0 = cat["x0"]

    x1, y1 = cat["A"]
    x2, y2 = cat["B"]

    m = line["m"]

    def sag_at_x(x):
        y_curve = catenary_y(cat, x)
        y_line_value = line_y(line, x)
        sag_value = y_line_value - y_curve
        return sag_value, y_curve, y_line_value

    # Candidate interior point from:
    # d/dx [ line(x) - catenary(x) ] = 0
    # m - sinh((x-c)/a) = 0
    # x = x0 + alpha * asinh(m)
    x_crit = x0 + alpha * np.arcsinh(m)

    candidates = [x1, x2]

    if x1 <= x_crit <= x2:
        candidates.append(x_crit)

    best_x = None
    best_sag = None
    best_y_curve = None
    best_y_line = None

    for x in candidates:
        sag_value, y_curve, y_line_value = sag_at_x(x)

        if (best_sag is None) or (sag_value > best_sag):
            best_sag = sag_value
            best_x = x
            best_y_curve = y_curve
            best_y_line = y_line_value

    return {
        "sag_max": best_sag,
        "x_sag": best_x,
        "y_curve": best_y_curve,
        "y_line": best_y_line,
    }
