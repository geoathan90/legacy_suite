import numpy as np
import pandas as pd

# ---- Inputs (same as yours) ----

G = 9.80665         # gravity

a = 290.00          # m (span)
dh = -95.12         # height difference
w = 1.3             # kg/m (per horizontal length)
S = 5.47e-4         # m^2 (surface area of cut conductor)
E = 6.18e9          # Pa  (Young's Modulus)
e = 1.935e-5        # 1/°C (thermal expansion coeff)
T1 = 0.0            # °C
T2 = 0.0            # °C
H1 = G * 2183.0     # kg (tension at T1)

T350 = 2183.0      # kg (tension at T2 for 350 ruling span)

df = pd.read_csv("output.csv")
#spans=list(df.columns.values)[1:]
spans=np.array([333, 523, 350, 410])
sags=df.iloc[0].to_list()[1:]
sags=np.array(sags)
heights = np.array([75, 60, -85, -70])
starting_tension = [T350 for _ in range(len(spans))]
starting_tension = np.array(starting_tension)   


############### Tabu Calculations ####################  

import numpy as np

def lengths(tension_list: np.ndarray) -> np.ndarray:
    
    s = np.asarray(spans, dtype=float)
    h = np.asarray(heights, dtype=float)
    t = np.asarray(tension_list, dtype=float)
    
    return np.sqrt(s*s + h*h + (s**4) * (w*w) / (12.0 * t*t))

def equivalent_spans(tension_list: np.ndarray) -> np.ndarray:
    """
    Returns a list of equivalent spans for each span in the dataframe
    """
    s = np.asarray(spans, dtype=float)
    h = np.asarray(heights, dtype=float)
    t = np.asarray(tension_list, dtype=float)

    return s + (2.0 * t * np.abs(h)) / (s * w)

def equivalent_sags(tension_list: np.ndarray) -> np.ndarray:
    """
    Returns a list of equivalent sags for each span in the dataframe
    """
    t = np.asarray(tension_list, dtype=float)
    eq_s = equivalent_spans(t)  # vectorized above
    
    return (eq_s**2) * w / (8.0 * t)

def axial_tension(tension_list:np.ndarray) -> np.ndarray:
    t = np.asarray(tension_list, dtype=float)
    eq_sag = equivalent_sags(t)          
    return t + w * eq_sag

def horizontal_tension(h0: float, axial: np.ndarray) -> np.ndarray:
    """
    """
    s = np.asarray(spans, dtype=float)
    h = np.asarray(heights, dtype=float)
    a = np.asarray(axial, dtype=float)

    n = s.size
    H = np.empty(n, dtype=float)
    H[0] = float(h0)

    # Precompute pieces that do NOT depend on axial[i-1]
    denom = 2.0 + (h*h) / (s*s)         # shape (n,)
    bh = w * h                          # w*height[i]
    c = -0.5 * (s*s) * (w*w)            # -(span[i]^2 * w^2)/2

    for i in range(1, n):
        a_prev = a[i-1]
        disc = a_prev*a_prev + a_prev*bh[i] + c[i]   # discriminant inside sqrt
        if disc < 0.0:  # numerical guard
            disc = 0.0
        H[i] = (a_prev + 0.5*bh[i] + np.sqrt(disc)) / denom[i]

    return H

def placeholder() -> None:
    """
    Placeholder function to be replaced by your own code
    """
    print("This is a placeholder function. Replace it with your own code.")

target_lengths = lengths(starting_tension)


###########  temperature CALCULATIONS  ####################

def Th2_for_span_tension(span: float, tension: float) -> float:
    """
    Solves H2^3 + ac*H2^2 + bc*H2 + cc = 0
    gets new H2 for T2 
    """
    tension = G * tension
    
    ac = e*S*E*(T2-T1)-tension+span**2*S*E*w**2/24/tension**2 
    bc = span**2*w**2/24 
    cc = e*(T2-T1)*S*E*w**2*span**2/24 - tension*w**2*span**2/24 - S*E*w**2*span**2/24

    coeffs = [1.0, ac, bc, cc]
    roots = np.roots(coeffs)

    # Filter real roots, pick the largest positive
    real_pos = roots[(np.isreal(roots)) & (roots.real > 0)].real
    return np.max(real_pos/G) if real_pos.size else np.nan

def Th_from_sag_span (sag: float, span: float) -> float:
    """
    Given sag, returns horizontal tension Th
    """
    return (w*span**2)/(8*sag)*(1+4/3*(sag/span)**2)

def tensions_for_temp() -> list:
    """
    Returns a list of tensions for each span in the dataframe
    """
    Tensions = []
    for i in range(len(spans)):
        span = float(spans[i])
        sag = float(sags[i])
        Tensions.append(Th_from_sag_span(sag, span))

    New_Tensions= []
    for i in range(len(spans)):
        span = float(spans[i])
        tension = Tensions[i]
        New_Tensions.append(Th2_for_span_tension(span, tension))
 
    return [round(x, 3) for x in New_Tensions]


########### HELPERS ####################

def data() -> pd.DataFrame:
    """
    Reads the output csv file and returns a dataframe
    """
    df = pd.read_csv("output.csv")
    spans=list(df.columns.values)[1:]
    sags=df.iloc[0].to_list()[1:]
    
    return df, spans, sags

# ---- iterate over many spans ----  THIS ENDED UP MAKING NO SENSE from a physics standpoint
#Spans = np.linspace(100.0, 800.0, 100)   
#Tensions = np.array([h2_for_span(a) for a in Spans])  
# #OR Tensions = np.vectorize(h2_for_span, otypes=[float])(Spans)