from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
# from sklearn.isotonic import IsotonicRegression
import sys
import os
from .tensions import sag, Th_from_sag

def evaluate(diag: str, spans):
    """
    Compute interpolated values for the given diagram and span(s).

    Parameters
    ----------
    diag : str or int
        Diagram name / id, e.g. "31185".
    spans : iterable of float
        Span values, e.g. [100, 450, 268.2].

    Returns
    -------
    df_out : pandas.DataFrame
        Index: temperatures (0,10,20,30,40,50[,ICE])
        Columns: the spans, as strings
    """
    diag = str(diag)
    
    # csv_path = os.path.join("data", diag + ".csv")   #### simpler, but can get lost if the script is run from a different directory
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    csv_path = DATA_DIR / f"{diag}.csv"
    
    df = pd.read_csv(csv_path)

    # -------- build interpolators for each temperature --------
    interps = {}
    for i in range(0, df.shape[1], 2):  # iterate by 2
        xcol = df.columns[i]
        ycol = df.columns[i + 1]

        # suffix of the column name (drop X in X_20 etc.)
        suffix = xcol.split("_")[1]

        xy = (
            df[[xcol, ycol]]
            .dropna()
            .sort_values(xcol)
            .drop_duplicates(subset=xcol, keep="first")
        )

        x = xy.iloc[:, 0].to_numpy()
        y = xy.iloc[:, 1].to_numpy()

        interps[suffix] = PchipInterpolator(x, y)
        # interps[suffix] = CubicSpline(x, y)

        # iso = IsotonicRegression(out_of_bounds='clip')
        # interps[suffix] = iso.fit(x, y)

    xs = np.array([float(a) for a in spans], dtype=float)

    ys = {k: np.asarray(interp(xs), dtype=float) for k, interp in interps.items()}

    # -------- fill missing temperatures --------
    if "10" not in ys and "0" in ys and "20" in ys:
        ys["10"] = (ys["0"] + ys["20"]) / 2

    if "30" not in ys and "20" in ys and "40" in ys:
        ys["30"] = (ys["40"] + ys["20"]) / 2

    if "40" in ys and "20" in ys and "50" not in ys:
        ys["50"] = ys["40"] + (ys["40"] - ys["20"]) / 2

    if "ICE" in interps.keys():
        rows = ["0", "10", "20", "30", "40", "50", "ICE"]
    else:
        rows = ["0", "10", "20", "30", "40", "50"]

    val_list = [ys[k] for k in rows]
    stacked = np.vstack(val_list).T

    data = {}
    for i, x in enumerate(xs):
        col_values = stacked[i]
        data[str(x)] = col_values

    df_out = pd.DataFrame(data, index=rows)#.round(3)

    return df_out

def tensions_from_df(df_sag: pd.DataFrame, w: float) -> pd.DataFrame:
    df_tension = df_sag.copy().astype(float)

    for col in df_tension.columns:
        S = float(col)
        df_tension[col] = df_tension[col].apply(lambda sag: Th_from_sag(sag, S, w))

    return df_tension


def tension_evaluate(diag: str, w: float, spans):
    df_sag = evaluate(diag, spans)
    df_tension = tensions_from_df(df_sag, w)
    return df_tension


def dx_calculation(df: pd.DataFrame, dh: list):
    """
    example usage:
    
    1st step
    >>> df1 = evaluate("31185", [100, 450, 268.2])  , where [100, 450, 268.2] are spans
    
    2nd step
    >>> df2 = dx_calculation(df1, [10, 20, 30]) , where [10, 20, 30] are the corresponding height differences for each span
    
    then print df1 and df2 to see the results.
    """
    spans = df.columns.astype(float).values
    dh = np.array(dh, dtype=float)
    
    if len(spans) != len(dh):
        raise ValueError("dh list length must match number of dataframe columns.")

    increase = np.sqrt(1 + (dh / spans)**2)
    return df * increase


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 3:
        print("Usage: python eval.py [diagram] [weight] [span1] [span2] ...")
        sys.exit(1)

    diag = argv[0]
    w = float(argv[1])
    spans = argv[2:]

    df_sag = evaluate(diag, spans).round(3)
    df_tension = tensions_from_df(df_sag, w).round(3)

    print("SAGS")
    print(df_sag.to_string())
    print()
    print("TENSIONS")
    print(df_tension.to_string())

    if diag == "3138" or diag == "???":  ## to-do: add more diagrams here
        print("προσοχή, ειναι τελικό τερματικό, κοίτα το βελοδιάγραμμα,")
        print("εστίασε στην καμπύλη ΠΑΓΟΥ, είναι κάτω από τους 40C")

if __name__ == "__main__":
    main()