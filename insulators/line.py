def line_between_points(A, B):
    """
    Return the straight line passing through points A and B.

    The line is stored in the form:
        y = m*x + b
    """
    x1, y1 = A
    x2, y2 = B

    if x1 == x2:
        raise ValueError("This version does not handle vertical lines.")

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    return {
        "m": m,
        "b": b,
        "A": A,
        "B": B,
    }

def line_y(line, x):
    """
    Return the y-value of the line at x.
    """
    m = line["m"]
    b = line["b"]

    return m * x + b
