import numpy as np

################### essential functions  ########################

def catenary(A, B, w, Th):
    """
    Build the unique 2D catenary passing through A and B
    for a given unit weight w and horizontal tension Th.

    Returns a dictionary with the catenary parameters.
    """

    x1, y1 = A
    x2, y2 = B

    if x1 == x2:
        raise ValueError("This version needs A and B to have different x-values.")

    # Keep things left-to-right
    if x2 < x1:
        x1, y1, x2, y2 = x2, y2, x1, y1

    if w <= 0:
        raise ValueError("w must be positive.")

    if Th <= 0:
        raise ValueError("Th must be positive.")

    # Main catenary parameter
    alpha = Th / w

    # Helpful intermediate values
    xm = 0.5 * (x1 + x2)
    half_span = 0.5 * (x2 - x1)
    dy = y2 - y1

    # Solve for c using the two endpoint conditions
    temp = dy / (2.0 * alpha * np.sinh(half_span / alpha))
    u = np.arcsinh(temp)
    x0 = xm - alpha * u

    # Then solve for d
    y0 = y1 - alpha * np.cosh((x1 - x0) / alpha)

    return {
        "alpha": alpha,
        "x0": x0,
        "y0": y0,
        "A": (x1, y1),
        "B": (x2, y2),
        "Th": Th,
    }

def catenary_y(cat, x):
    """
    Return y-values of the catenary at x.
    """
    alpha = cat["alpha"]
    x0 = cat["x0"]
    y0 = cat["y0"]

    return alpha * np.cosh((x - x0) / alpha) + y0

def catenary_points(cat, n=100):
    """
    Sample n points along the catenary between A and B.
    """
    x1, y1 = cat["A"]
    x2, y2 = cat["B"]

    x = np.linspace(x1, x2, n)
    y = catenary_y(cat, x)

    return x, y

def catenary_low_point(cat): 
    """ Return the x-value lowest point (vertex) of the catenary. """ 
    xv = cat["x0"] 
    return xv

def catenary_length(cat):
    """
    Return the conductor length along the catenary
    between the two suspension points A and B.
    """
    alpha = cat["alpha"]
    x0 = cat["x0"]

    x1, y1 = cat["A"]
    x2, y2 = cat["B"]

    s = alpha * (
        np.sinh((x2 - x0) / alpha) -
        np.sinh((x1 - x0) / alpha)
    )

    return s

#############   helpers for insulator equilibrium  #########################

def catenary_slope(cat, x):
    """
    Return the slope dy/dx of the catenary at x.
    """
    alpha = cat["alpha"]
    x0 = cat["x0"]

    return np.sinh((x - x0) / alpha)

def catenary_Tv(cat, x):
    """
    Return the vertical component of the conductor tension at x.

    Positive  -> upward component
    Negative  -> downward component
    """
    
    Th = cat["Th"]
    
    return Th * catenary_slope(cat, x)

def catenary_Tv_left(cat):
    """
    Return the vertical force of the conductor on the left
    support / insulator.

    Convention:
    - positive = upward
    - negative = downward
    """
    x_left, y_left = cat["A"]
    Th = cat["Th"]
    
    return Th * catenary_slope(cat, x_left)

def catenary_Tv_right(cat):
    """
    Return the vertical force of the conductor on the right
    support / insulator.

    Convention:
    - positive = upward
    - negative = downward
    """
    x_right, y_right = cat["B"]
    Th = cat["Th"]
    
    return -Th * catenary_slope(cat, x_right)

def catenary_Ta(cat, x):
    """
    Return the total tension magnitude in the conductor at x.
    """
    Th = cat["Th"]
    Tv = catenary_Tv(cat, x)
    
    return np.sqrt(Th**2 + Tv**2)