
from sympy import symbols, Eq, solve, re, N
from math import isfinite, sinh, asinh, acosh
import numpy as np
import sys

# ---- Input ----
S =  50           # span (m)      290 316.72
dh =  0             # elevation difference h_R - h_L (m)     -95.12  -37.54
w = 1.303           #
w1 = w              # kg/m
w2 = w              #    
A = 5.47e-4         # area (m^2)
E = 5.132e9          # Young's modulus (Pa) |5.132e9 για Cardinal, 6.184e9 για τους άλλου αγωγούς
alpha = 1.935e-5    # thermal expansion (1m/°C)
T1 = 0          # initial temp (°C)
T2 = 40          # new temp (°C)
H1 = 2585           # initial horizontal tension (kg)   ex:2585 για τις μαλακίες τους | 1720 | 1040
H1_old = H1         #  archive
#H1 = float(sys.argv[1])

conductors = {"Linnet": [0.702,18.31e-3,1.98e-4,6.184e9,1.899e-5],
              "Grosbeak": [1.303,25.15e-3,3.74e-4,6.184e9,1.899e-5],
              "Cardinal": [1.823,30.42e-3,5.47e-4,5.132e9,1.935e-5],
              "SW_150": [0.44,9.5e-3,0.55e-4,19.33e9,1.152e-5],
              "SW_400": [0.77,12.6e-3,0.97e-4,19.33e9,1.152e-5]}

S1 = 377.52 #pyrgos 22    sags per temp 0 10 20 30 40 50 ICE, tensions per temp
dh1 = -85.42             # kk = [9.614, 10.0, 10.386, 10.767, 11.148, 11.52, 11.622], tt = [3378.0999295194506, 3247.70527224, 3127.0029580589257, 3016.351139816105, 2913.2627128094728, 2819.188604375, 2794.446112751678]   

S2 = 369.70 #pyrgos 25   # εδώ υπάρχει βασικό άνοιγμα, όχι τερματικό
dh2 = -70.70

S3 = 290.00 # pyrgos 3   # kk = [5.093, 5.392, 5.69, 6.002, 6.314, 6.626, 6.779], tt = [3762.8681523659925, 3554.2076224035604, 3368.064586994727, 3192.983588803732, 3035.2054957237883, 2892.2860700271654, 2827.0080395338546]
dh3 = -95.12

### note για Λάρισα ΙΙ: νομίζω έχουν υπολογίσει τα βάρη στους -10 με γυμνό αγωγό 

def solve_for_H2(S,H1,T1,T2,w1=w,w2=w):

    # Coefficients
    c1 = alpha * A * E * (T2 - T1) - H1 + (w1**2 * A * E * S**2) / (24.0 * H1**2)
    c2 = S**2 * w2**2 / 24.0
    c3 = alpha * A * E * (T2 - T1) * (S**2 * w2**2) / 24.0 - H1 * (S**2 * w2**2) / 24.0 - (w2**2 * A * E * S**2) / 24.0

    # Cubic: x^3 + c1*x^2 + c2*x + c3 = 0
    H2 = symbols('H2')  
    eq = Eq(H2**3 + c1*H2**2 + c2*H2 + c3, 0)   

    roots = solve(eq, H2)  
    #print(f"Found roots: {roots}")
    
    # pick valid root: real, positive, finite
    candidates = []
    for r in roots:
        rv = complex(N(r))
        if abs(rv.imag) < 1e-9 and rv.real > 0 and isfinite(rv.real):
            candidates.append(rv.real)

    if not candidates:
        raise RuntimeError("Δεν βρέθηκε θετική ρίζα, τσέκαρε τα input.")

    H2_sol = max(candidates)  ############### check ############ tensions are typically the largest positive real root

    return H2_sol

def sag_old(S, H, w):
    return w * S**2 / (8.0 * H)

def Th_from_sag_old(sag, S, w=w):
    return w * S**2 / (8.0 * sag)    

def sag(S, Th, w):
    S = np.asarray(S, dtype=float)
    Th = np.asarray(Th, dtype=float)
    w = np.asarray(w, dtype=float)

    a = Th / w
    return 2.0 * a * np.sinh(S / (4.0 * a))**2

def Th_from_sag(target_sag, S, w, tol=1e-10, max_iter=100):

    def f(Th):
        return sag(S, Th, w) - target_sag
    
    Th_low = 1e-12
    Th_high = w * S**2 / (8.0 * target_sag)

    # Make sure upper bound is actually on the low-sag side
    while f(Th_high) > 0:
        Th_high *= 2.0

    for _ in range(max_iter):
        Th_mid = 0.5 * (Th_low + Th_high)
        err = f(Th_mid)

        if abs(err) < tol:
            return Th_mid

        if err > 0:
            # sag too large -> need more tension
            Th_low = Th_mid
        else:
            # sag too small -> need less tension
            Th_high = Th_mid

    return Th_mid

def distance_lowest_point_r(S, dh, H=H1, w=w):
    S = np.asarray(S, dtype=float)
    dh = np.asarray(dh, dtype=float)
    H = np.asarray(H, dtype=float)
    w = np.asarray(w, dtype=float)

    a = H / w
    return -a * np.arcsinh(dh / (2.0 * a * np.sinh(S / (2.0 * a)))) + S / 2.0

def distance_lowest_point_l(S, dh, H=H1, w=w):
    S = np.asarray(S, dtype=float)
    dh = -np.asarray(dh, dtype=float)
    H = np.asarray(H, dtype=float)
    w = np.asarray(w, dtype=float)

    a = H / w
    return -a * np.arcsinh(dh / (2.0 * a * np.sinh(S / (2.0 * a)))) + S / 2.0

def height_at_x(x, S, H, w, dh, y0):
    x = np.asarray(x, dtype=float)
    S = np.asarray(S, dtype=float)
    H = np.asarray(H, dtype=float)
    w = np.asarray(w, dtype=float)
    dh = np.asarray(dh, dtype=float)
    y0 = np.asarray(y0, dtype=float)

    a = H / w
    xr = -a * np.arcsinh(dh / (2.0 * a * np.sinh(S / (2.0 * a)))) + S / 2.0
    return y0 + a * (np.cosh((x - xr) / a) - np.cosh(xr / a))

# def monopleyro_right(S,dh,H=H1,w=w):
#     monopleyro = -H/w*asinh(dh/2/(H/w)/sinh(S/2/(H/w)))-S/2
#     if monopleyro <= 0: print("δεν υπάρχει δεξιό μονόπλευρο φορτίο") 
#     else: return monopleyro

# def monopleyro_left(S,dh,H=H1,w=w):
#     dh = -dh
#     monopleyro = -H/w*asinh(dh/2/(H/w)/sinh(S/2/(H/w)))-S/2
#     if monopleyro <= 0: print("δεν υπάρχει αριστερό μονόπλευρο φορτίο") 
#     else: return monopleyro

def monopleyro_right(S, dh, H=H1, w=w, *, invalid="nan"):
    S = np.asarray(S, dtype=float)
    dh = np.asarray(dh, dtype=float)
    H = np.asarray(H, dtype=float)
    w = np.asarray(w, dtype=float)

    a = H / w
    val = -a * np.arcsinh(dh / (2.0 * a * np.sinh(S / (2.0 * a)))) - S / 2.0

    if invalid == "zero":
        return np.where(val > 0.0, val, 0.0)
    return np.where(val > 0.0, val, np.nan)

def monopleyro_left(S, dh, H=H1, w=w, *, invalid="nan"):
    S = np.asarray(S, dtype=float)
    dh = -np.asarray(dh, dtype=float)
    H = np.asarray(H, dtype=float)
    w = np.asarray(w, dtype=float)

    a = H / w
    val = -a * np.arcsinh(dh / (2.0 * a * np.sinh(S / (2.0 * a)))) - S / 2.0

    if invalid == "zero":
        return np.where(val > 0.0, val, 0.0)
    return np.where(val > 0.0, val, np.nan)


def synoliko_katakoryfo(S_l,dh_l,H_l,w_l,S_r,dh_r,H_r,w_r):
    return distance_lowest_point_r(S_r,dh_r,H_r,w_r) + distance_lowest_point_l(S_l,dh_l,H_l,w_l)

def conductor_length(S, Th, w, dh):
    S = np.asarray(S)
    H = np.asarray(Th)
    w = np.asarray(w)
    dh = np.asarray(dh)

    a = H / w

    #return 2.0 * a * np.sinh(S / (2.0 * a)) * np.cosh(np.arcsinh(dh / (2.0 * a * np.sinh(S / (2.0 * a)))))
    # deleted because it was numerically unstable. Kept for posterity.

    level_length = 2.0 * a * np.sinh(S / (2.0 * a))

    return np.hypot(level_length, dh) # essentially the same as sqrt(level_length**2 + dh**2), but better numerically
    
        
def Th_from_length(target_length, S, w, dh, tol=1e-10, max_iter=10000):

    def f(Th):
        return conductor_length(S, Th, w, dh) - target_length
    
    Th_low = 50
    Th_high = 5000 #w * S**2 / (8.0 * sag(S, Th_low, w))

    # Make sure lower bound is actually on the short-length side
    while f(Th_low) < 0:
        Th_low *= 0.5

    # Make sure upper bound is actually on the long-length side
    while f(Th_high) > 0:
        Th_high *= 2.0

    for _ in range(max_iter):
        Th_mid = 0.5 * (Th_low + Th_high)
        err = f(Th_mid)

        if abs(err) < tol:
            return Th_mid

        if err > 0:
            # length too long -> need more tension
            Th_low = Th_mid
        else:
            # length too short -> need less tension
            Th_high = Th_mid

    return Th_mid


if __name__ == "__main__":
    l = conductor_length(283.64,1550,0.769,0)

    extra_length = 0.50

    Th_elongated = Th_from_length(l + extra_length, 283.64, 0.769, 0)

    sag_elongated = sag(283.64, Th_elongated, 0.769)

    print(f"Length: {l:.6f} m")
    print(f"Tension for +{extra_length:.2f}m length: {Th_elongated:.2f} kg")
    print(f"Sag for original length: {sag(283.64, 1550, 0.769):.6f} m") 
    print(f"Sag for +{extra_length:.2f}m length: {sag_elongated:.6f} m")    


