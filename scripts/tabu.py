import numpy as np
import pandas as pd 
import os

spans=np.array([333, 523, 350, 410])
heights = np.array([75, 60, -85, -70])

# Temperatures (°C) — same index positions used in all T-arrays below
TEMPS = np.array([0, 10, 20, 30, 40], dtype=float)

# All conductors in one dict
CONDUCTORS = {
    "Linnet": {
        "w": 0.7024,  # kg/m
        "T350": np.array([1527, 1457, 1386, 1327, 1269], dtype=float),
        "T500": np.array([1244, 1219, 1185, 1172, 1149], dtype=float),
    },
    "Grosbeak": {
        "w": 1.3,
        "T350": np.array([2183, 2102, 2028, 1960, 1893], dtype=float),
        "T500": np.array([2026, 1989, 1953, 1919, 1885], dtype=float),
    },
    "Cardinal": {
        "w": 1.823,
        "T350": np.array([3480, 3332, 3185, 3065, 2945], dtype=float),
        "T500": np.array([3105, 3045, 2980, 2925, 2870], dtype=float),
    },
    "SW150": {
        "w": 0.44,
        "T350": np.array([1118, 1074, 1031, 994, 957], dtype=float),
        "T500": np.array([892, 876, 860, 846, 832], dtype=float),
    },
    "SW400": {
        "w": 0.769,
        "T350": np.array([1810, 1740, 1670, 1610, 1550], dtype=float),
        "T500": np.array([1520, 1495, 1470, 1445, 1420], dtype=float),
    },
}

def select_conductor(name: str, ruling: int = 350):
    """
    Returns a dict with:
      - w (kg/m)
      - Tvec: the tensions vs temperature for the chosen ruling span (length = len(TEMPS))
    """
    data = CONDUCTORS[name]
    if ruling == 350:
        return {"w": data["w"], "Tvec": data["T350"]}
    elif ruling == 500:
        return {"w": data["w"], "Tvec": data["T500"]}
    else:
        raise ValueError("ruling must be 350 or 500")

def lengths(tension_list: np.ndarray) -> np.ndarray:
    
    s = np.asarray(spans, dtype=float)
    h = np.asarray(heights, dtype=float)
    t = np.asarray(tension_list, dtype=float)
    
    return np.sqrt(s*s + h*h + (s**4) * (w*w) / (12.0 * t*t))

def sags(tension_list: np.ndarray) -> np.ndarray:
    """
    Returns a list of sags for each span in the dataframe
    """
    t = np.asarray(tension_list, dtype=float)
    s = np.asarray(spans, dtype=float)
    
    return (s**2) * w / (8.0 * t) + (s**4) * (w**3) / (384 * t**3)

def increase(tension_list: np.ndarray) -> np.ndarray:

    t = np.asarray(tension_list, dtype=float)
    l = lengths(t)
    s = np.asarray(spans, dtype=float)

    return l/s

    # return l/ (s * (1 + s * w / t )**2 / 24)


def _forward_from_H0(H0: float):
    """
    One forward sweep given the initial horizontal tension H0.
    Uses:
      if height[i] > 0 : axial[i+1] = H[i] + w*eq_sag[i]
      else (<= 0)      : axial[i+1] = H[i] + w*(eq_sag[i] + height[i])
    """
    s = np.asarray(spans, dtype=float)
    h = np.asarray(heights, dtype=float)
    N = s.size

    H = np.empty(N, dtype=float)
    H[0] = float(H0)

    for i in range(0, N - 1):
        # Equivalent span & sag for current span i (using H[i])
        eq_span = s[i] + (2.0 * H[i] * abs(h[i])) / (s[i] * w)
        eq_sag  = (eq_span * eq_span) * w / (8.0 * H[i])

        # Axial tension at the NEXT tower (i+1), using sign-aware rule
        if h[i] > 0.0:
            axial_next = H[i] + w * eq_sag
        else:
            axial_next = H[i] + w * (eq_sag + h[i])  # h[i] <= 0

        # Horizontal tension for NEXT span (i+1)
        denom = 2.0 + (h[i + 1] * h[i + 1]) / (s[i + 1] * s[i + 1])
        disc  = (axial_next * axial_next
                 + axial_next * w * h[i + 1]
                 - 0.5 * (s[i + 1] * s[i + 1]) * (w * w))

        if disc < 0.0:
            print(f"Warning: negative discriminant in span {i+1} → clamped to 0.")
            disc = 0.0

        H[i + 1] = (axial_next + 0.5 * w * h[i + 1] + np.sqrt(disc)) / denom

    return H, []


def _total_length_error(H0: float, target_total_length: float):
    """
    Compute error = total_length(H0) - target_total_length.
    Returns error, H
    """
    H, _ = _forward_from_H0(H0)
    total_len = float(np.sum(lengths(H)))
    return total_len - target_total_length, H


def solve_horizontal_tensions(
    T_ref: float,
    atol_m: float = 0.01,   # stop when |sum(lengths) - target| <= 1 cm
    max_iters: int = 10000, # higher cap since we're stepping by 1
    step: float = 1.0       # 1 kg per step (integer walk)
):
    """
    Outer solve on H0 (horizontal_tension[0]) using a simple unit-step search.
    - Start at T350
    - If total length is too long (F>0), increase H0 by 1
      If too short (F<0), decrease H0 by 1
    - On first sign change between consecutive integers, pick the better of the two.

    Returns:
      H    : np.ndarray of horizontal tensions per span (length N)
      info : dict with fields (target, total, error, iterations, H0)
    """
    N = len(spans)

    # 1) Target: all spans at T350
    target = float(np.sum(lengths(np.full(N, T_ref, dtype=float))))

    # 2) Start from T350, clamp to (0, 4000)
    H0 = float(np.clip(T_ref, 1e-6, 4000.0))

    # Evaluate at the start
    F, H = _total_length_error(H0, target)
    if abs(F) <= atol_m:
        total = float(np.sum(lengths(H)))
        return H, {"target": target, "total": total, "error": F, "iterations": 0, "H0": H0}

    # Decide initial direction
    # F > 0 → total too long → increase H0 (shortens length)
    # F < 0 → total too short → decrease H0 (lengthens)
    direction = 1.0 if F > 0 else -1.0

    prev_H0, prev_F, prev_H = H0, F, H

    for it in range(1, max_iters + 1):
        # Propose next H0 and clamp to physical bounds
        H0_new = prev_H0 + direction * step
        H0_new = float(np.clip(H0_new, 1e-6, 4000.0))

        # If we can't move further (stuck at bound), return best-so-far
        if H0_new == prev_H0:
            total = float(np.sum(lengths(prev_H)))
            return prev_H, {
                "target": target, "total": total,
                "error": prev_F, "iterations": it, "H0": prev_H0
            }

        F_new, H_new = _total_length_error(H0_new, target)

        # Check tolerance
        if abs(F_new) <= atol_m:
            total = float(np.sum(lengths(H_new)))
            return H_new, {"target": target, "total": total, "error": F_new, "iterations": it, "H0": H0_new}

        # Check for sign change between consecutive integers
        if prev_F * F_new < 0.0:
            # Pick the better of the two neighbors
            if abs(prev_F) <= abs(F_new):
                best_H0, best_F, best_H = prev_H0, prev_F, prev_H
            else:
                best_H0, best_F, best_H = H0_new, F_new, H_new

            total = float(np.sum(lengths(best_H)))
            return best_H, {
                "target": target, "total": total,
                "error": best_F, "iterations": it, "H0": best_H0
            }

        # No sign change: keep walking in the same direction
        prev_H0, prev_F, prev_H = H0_new, F_new, H_new

    # Max iterations reached: return best-so-far
    total = float(np.sum(lengths(prev_H)))
    return prev_H, {"target": target, "total": total, "error": prev_F, "iterations": max_iters, "H0": prev_H0}

################## output ##############

def _format_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted copy for CSV: tensions as integers, sags to 2 decimals."""
    df_exp = df.copy()

    # Integers (no decimals)
    for col in ["T_ref_kg", "H_solution_kg"]:
        df_exp[col] = np.rint(df_exp[col]).astype(int)

    # Two decimals (sags); NaNs are preserved
    for col in ["sag_eq_m", "sag_alt_m", "diorthosi"]:
        df_exp[col] = df_exp[col].round(2)

    # (Optional) make these look clean too
    if "temperature_C" in df_exp:
        df_exp["temperature_C"] = df_exp["temperature_C"].astype(int)
    # If your spans/heights are integers, uncomment:
    # df_exp["span_m"] = np.rint(df_exp["span_m"]).astype(int)
    # df_exp["height_m"] = np.rint(df_exp["height_m"]).astype(int)

    return df_exp


def build_and_export_tables(
    conductor_name: str = "Grosbeak",
    ruling: int = 350,
    atol_m: float = 0.01,
    out_dir: str = "outputs"
):
    """
    For the chosen conductor & ruling span:
      - Solve H per span at each temperature
      - Build 5 DataFrames (one per temperature) with 6 requested columns
      - Save each to CSV in out_dir
    Returns: dict { temperature(float): DataFrame }
    """
    os.makedirs(out_dir, exist_ok=True)

    # Select conductor and set global 'w' for downstream functions
    sel = select_conductor(conductor_name, ruling=ruling)  # uses your dict
    Tvec = sel["Tvec"]                                     # ruling tensions per temperature
    global w
    w = sel["w"]                                           # set weight per horizontal meter

    results = {}

    for idx, T_ref in enumerate(Tvec):
        temperature = float(TEMPS[idx])  # e.g., 0, 10, 20, 30, 40

        # Solve horizontal tensions for this temperature
        H_solution, info = solve_horizontal_tensions(T_ref=T_ref, atol_m=atol_m)

        # Compute equivalent sags (your parabolic sags)
        N = len(spans)
        sag = sags(np.full(N, T_ref, dtype=float))
        sag_alt = sags(H_solution)
        inc = lengths(H_solution)/spans
        dl = lengths(np.full(N, T_ref, dtype=float)) - lengths(H_solution)
        dl_cum = np.cumsum(dl)  

        # Build DataFrame with the 6 requested columns
        df_out = pd.DataFrame({
            "temperature_C": np.full_like(spans, temperature, dtype=float),  # (optional first column/note)
            "span_m": spans,
            "height_m": heights,
            "T_ref_kg": np.full_like(spans, T_ref, dtype=float),             # same value in all rows
            "H_solution_kg": H_solution,
            "sag_eq_m": inc*sag,                                                 # from your sags() function
            "sag_alt_m": inc*sag_alt,
            "diorthosi": dl_cum,                                             
        })

        # Nice-to-have: embed a tiny note in attrs (not written to CSV)
        df_out.attrs["title"] = f"{conductor_name} @ {temperature}°C (ruling {ruling} m)"

        # Save CSV
        fname = f"{conductor_name}_r{ruling}_{int(temperature)}C.csv"
        #df_out.to_csv(os.path.join(out_dir, fname), index=False)
        df_export = _format_for_export(df_out)              # <-- format copy
        df_export.to_csv(os.path.join(out_dir, fname), index=False)
        
        results[temperature] = df_out

    return results

# Example use:
tables = build_and_export_tables("Grosbeak", ruling=350, atol_m=0.001, out_dir="outputs")
#tables[0.0].head()  # DataFrame for 0°C



################# usage   -  comment out when importing ####################

# sel = select_conductor("Grosbeak", ruling=350)
# w = sel["w"]
# Tvec = sel["Tvec"]


# for T_ref in Tvec:
#     H_solution, info = solve_horizontal_tensions(T_ref=T_ref, atol_m=0.001)
#     print("Solved H per span:", H_solution)

# print("Target total length (m):", info["target"])
# print("New total length (m):   ", info["total"])
# print("Error (m):              ", info["error"])
# print("Iterations:             ", info["iterations"])
# print("H0 used (kg):           ", info["H0"])