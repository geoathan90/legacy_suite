
from sympy import symbols, Eq, solve, re, N
from math import isfinite, sinh, asinh, acosh
import numpy as np
import sys

# ---- Input ----
#S =  50           # span (m)      290 316.72
#dh =  0             # elevation difference h_R - h_L (m)     -95.12  -37.54
w = 1.823          #
w1 = w              # kg/m
w2 = w              #    
A = 5.47e-4         # area (m^2) 5.47e-4 for Cardinal
E_initial = 5.132e9  
E_final = 6.8529e9  # Young's modulus (kg/m2) |5.132e9 για Cardinal, 6.184e9 για τους άλλου αγωγούς
alpha = 1.935e-5    # thermal expansion (1m/°C)
T1 = 0              # initial temp (°C)
T2 = 40             # new temp (°C)
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

def solve_for_H2(S, H1, E, A, alpha, T1, T2, w1, w2):

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

def solve_for_H2_numeric(S, H1, E, A, alpha, T1, T2, w1, w2):
    c1 = alpha * A * E * (T2 - T1) - H1 + (w1**2 * A * E * S**2) / (24.0 * H1**2)
    c2 = S**2 * w2**2 / 24.0
    c3 = (
        alpha * A * E * (T2 - T1) * (S**2 * w2**2) / 24.0
        - H1 * (S**2 * w2**2) / 24.0
        - (w2**2 * A * E * S**2) / 24.0
    )

    roots = np.roots([1.0, c1, c2, c3])

    real_roots = roots[np.isclose(roots.imag, 0.0, atol=1e-7)].real
    positive_roots = real_roots[real_roots > 0.0]

    if len(positive_roots) == 0:
        raise RuntimeError("No positive real root found.")

    return positive_roots.max()

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

def Tv_A (S, H, w, dh):
    """
    vertical force at left support (point A)
    """
    S = np.asarray(S, dtype=float)
    H = np.asarray(H, dtype=float)
    w = np.asarray(w, dtype=float)
    dh = np.asarray(dh, dtype=float)

    a = H / w
    xv = -a * np.arcsinh(dh / (2.0 * a * np.sinh(S / (2.0 * a)))) + S / 2.0
    return H*np.sinh((0-xv)/a)

def Tv_B (S, H, w, dh):
    """
    vertical force at right support (point B)
    """
    S = np.asarray(S, dtype=float)
    H = np.asarray(H, dtype=float)
    w = np.asarray(w, dtype=float)
    dh = np.asarray(dh, dtype=float)

    a = H / w
    xv = -a * np.arcsinh(dh / (2.0 * a * np.sinh(S / (2.0 * a)))) + S / 2.0
    return -H*np.sinh((S-xv)/a)

def Taxial_A (S, H, w, dh):
    """
    axial force at left support (point A)
    """
    S = np.asarray(S, dtype=float)
    H = np.asarray(H, dtype=float)
    w = np.asarray(w, dtype=float)
    dh = np.asarray(dh, dtype=float)

    a = H / w
    return np.hypot(H, Tv_A(S, H, w, dh))

def Taxial_B (S, H, w, dh):
    """
    axial force at right support (point B)
    """
    S = np.asarray(S, dtype=float)
    H = np.asarray(H, dtype=float)
    w = np.asarray(w, dtype=float)
    dh = np.asarray(dh, dtype=float)

    a = H / w
    return np.hypot(H, Tv_B(S, H, w, dh))   

if __name__ == "__main__":

    ### TEST RAKITA
    # l = conductor_length(283.64,1550,0.769,0)

    # extra_length = 0.50

    # Th_elongated = Th_from_length(l + extra_length, 283.64, 0.769, 0)

    # sag_elongated = sag(283.64, Th_elongated, 0.769)

    # print(f"Length: {l:.6f} m")
    # print(f"Tension for +{extra_length:.2f}m length: {Th_elongated:.2f} kg")
    # print(f"Sag for original length: {sag(283.64, 1550, 0.769):.6f} m") 
    # print(f"Sag for +{extra_length:.2f}m length: {sag_elongated:.6f} m")    


    # ###############  TEST RAKITA 2
    # S = 593.70
    # dh = 131.65
    # sag_f = 27.86
    # w = 1.823
    # H = Th_from_sag_old(sag_f, S, w)
    

    # print(f"Για S={S} m, dh={dh} m, sag={sag} m, η οριζόντια τάση είναι H={H:.2f} kg")
    # print(f"Η κατακόρυφη δύναμη στη στήριξη Α είναι Tv_A={Tv_A(S, H, w, dh):.2f} kg")
    # print(f"Η κατακόρυφη δύναμη στη στήριξη Β είναι Tv_B={Tv_B(S, H, w, dh):.2f} kg")
    # print(f"Η αξονική δύναμη στη στήριξη Α είναι Taxial_A={Taxial_A(S, H, w, dh):.2f} kg")
    # print(f"Η αξονική δύναμη στη στήριξη Β είναι Taxial_B={Taxial_B(S, H, w, dh):.2f} kg")  

    #################

    # test for span 8/9 - support reactions

    # S = 345
    # dh = -103.34
    # H = 4515
    # w = 2.5

    # print(f"Η κατακόρυφη δύναμη του ΕΝΟΣ ΥΠΟΑΓΩΓΟΥ στη στήριξη Α είναι Tv_A={Tv_A(S, H, w, dh):.2f} kg")
    # print(f"Η κατακόρυφη δύναμη του ΕΝΟΣ ΥΠΟΑΓΩΓΟΥ στη στήριξη Β είναι Tv_B={Tv_B(S, H, w, dh):.2f} kg") 

    #################

    # test for tower 9, left side

    # w_bare = 1.823
    # w_ice = 2.623

    # S = 345
    # dh =103.34
    # H0 = 3494.416
    # H = solve_for_H2(S, H0, 0, -10, w_bare, w_bare)

    # print(f"H: {H:.2f} kg")
    # print(f"αριστερή πλευρά: {distance_lowest_point_l(S, dh, H, w_bare):.2f} m")

    # print(1.303*1500*1500/8/204.574)
    # print(1.303*1500*1500/8/212.9)
    

    #################

    # test for towers 22, 57, total vert

    temperature = -19

    w_bare = 1.823
    w_ice = 3.6

    # S1 = 215              ## tower 57
    # S2 = 190
    # dh1 = 62.45
    # dh2 = -35.31

    S1 = 527.48
    S2 = 220
    dh1 = 10.9
    dh2 = -73.03

    Th_initial = 3480.0
    Th_final = 3151.2 +100

    E = E_final
    Th = Th_initial
    #w_ice = w_bare #### flag for ice or not 

    H1 = solve_for_H2(422.29, Th, E, A, alpha, 0, temperature, w_bare, w_ice) # 390.47/422.29 for 57/22
    H2 = solve_for_H2(322.09, Th, E, A, alpha, 0, temperature, w_bare, w_ice) # 343.95/322.09 for 57/22

    vert = synoliko_katakoryfo(S1, dh1, H1, w_ice, S2, dh2, H2, w_ice)

    axial_left = Taxial_B(S1, H1, w_ice, dh1)
    axial_right = Taxial_B(S2, H2, w_ice, dh2)

    print(f"\n              ΑΓΩΓΟΣ ΦΑΣΗΣ")
    print(f"Θερμοκρασία: {temperature}°C, βάρος αγωγού: {w_ice:.3f} kg/m, ") 
    print(f"Μέτρο Ελαστικότητας: {E:.2e} kg/m2, Τάνυση ΒΑ στους 0°C: {Th:.2f} kg")
    print(f"H1: {H1:.2f} kg")
    print(f"H2: {H2:.2f} kg")
    print(f"συνολικό κατακόρυφο: {vert.round(2)} m")
    print(f"συνολική κατακόρυφη φόρτιση ημιγεφυρίου: {vert*w_ice*2:.2f} kg")
    print(f"αξονική πίσω: {axial_left:.2f} kg")
    print(f"αξονική μπροστά: {axial_right:.2f} kg")

    ###############

    w_bare = 0.769
    w_ice = 1.68 # 1.11 για 1/4" πάγο, 1.68 για 1/2" πάγο

    A = 9.6454e-5       # διατομή
    alpha = 1.152e-5    # θερμική διαστολή
    E = 19.33e9
    Th = 1810
    #w_ice = w_bare #### flag for ice or not 

    H1 = solve_for_H2(390.47, Th, E, A, alpha, 0, temperature, w_bare, w_ice) # 390.47
    H2 = solve_for_H2(343.95, Th, E, A, alpha, 0, temperature, w_bare, w_ice) # 343.95

    vert = synoliko_katakoryfo(S1, dh1, H1, w_ice, S2, dh2, H2, w_ice)

    axial_left = Taxial_B(S1, H1, w_ice, dh1)
    axial_right = Taxial_B(S2, H2, w_ice, dh2)


    print(f"\n              ΑΓΩΓΟΣ ΠΡΟΣΤΑΣΙΑΣ - OXI OPGW")
    print(f"Θερμοκρασία: {temperature}°C, βάρος αγωγού: {w_ice:.3f} kg/m, ") 
    print(f"Μέτρο Ελαστικότητας: {E:.2e} kg/m2, Τάνυση ΒΑ στους 0°C: {Th:.2f} kg")
    print(f"H1: {H1:.2f} kg")
    print(f"H2: {H2:.2f} kg")
    print(f"συνολικό κατακόρυφο: {vert.round(2)} m")
    print(f"συνολική κατακόρυφη φόρτιση ενός κερατίου: {vert*w_ice:.2f} kg")
    print(f"αξονική πίσω: {axial_left:.2f} kg")
    print(f"αξονική μπροστά: {axial_right:.2f} kg")

