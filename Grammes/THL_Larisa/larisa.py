import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.tensions as ts
from scripts.eval import evaluate

from .catenary_dxf_plotter import plot_catenaries_from_file


## usage python -m Grammes.THL_Larisa.larisa

# ============================================================
# USER SETTINGS
# ============================================================

HERE = Path(__file__).resolve().parent

INPUT_CSV = HERE / "larisa2_2nd_submission.csv"
OUTPUT_XLSX = HERE / "outputs" / "larisa2_2nd_submission_processed.xlsx"

# Cardinal default, in kg/m
BASE_WEIGHT = 1.823

# Effective weights / factors used in the original script
# MAX_LOAD_WEIGHT_FACTOR = 2.2662   #  0" ice and 9# wind
ICE_WEIGHT = 2.5          #  1/4" ice - approximation
VARI_WEIGHT_FACTOR = 2.623 # βάρος αγωγού 2η συνθήκη

BASE_TEMP = 0

DIAGRAMS = {
    "term": "31185",
    "BA350": "31187",
    "BA500": "31188",
    "2000": "52740",
    "1000": "31189",
    "700": "31858",
}

LOADS = {
    "S5": 600,
    "S5+8.00": 600,
    "G5": 700,
    "G5+8.00": 700,
    "R5": 800,
    "R5+8.00": 800,
    "R5+18.00": 800,
    "RE5": 1000,
    "RE5+8.00": 1000,
    "T5": 1000,
    "T5+8.00": 1000,
    "T5+18.00": 1000,
    "TE5": 1200,
    "TE5+8.00": 1200,
    "TE5+18.00": 1200,
    "Z5": 800,
    "Z5+8.00": 800,
    "ZE5": 1300,
    "ZE5+8.00": 1300,
    "Z5*": 800,
    "Z5+8.00*": 800,
}

# Hard-coded
BASE_TENSION_OVERRIDES = {      # 0 degrees, initial
    "BA350": 3470,
    "BA500": 3105,
}

TENSION_50_OVERRIDES = {        # 50 degrees, final
    "BA350": 2585,
    "BA500": 2654,
    "term": 2585,
}

LOAD_CASES = [
    {"name": "-10", "temp": -10, "weight": BASE_WEIGHT},
    {"name": "-10_ICE", "temp": -10, "weight": ICE_WEIGHT},
    {"name": "0_ICE", "temp": 0, "weight": ICE_WEIGHT},
]


# ============================================================
# SMALL HELPERS
# ============================================================

def normalize_text_series(s: pd.Series) -> pd.Series:
    """Normalize text values coming from Excel/CSV input."""
    return (
        s.astype(str)
        .str.strip()
        # Greek-looking letters that can silently break dictionary lookups.
        .str.replace("Β", "B", regex=False)
        .str.replace("Α", "A", regex=False)
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either the uploaded uppercase column names or the older lowercase names.
    The rest of the script uses the standardized names.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {
        "tower_type": "type",
        "tower_number": "tower_number",
        "span": "span",
        "span_type": "span_type",
        "suspension_altitude": "suspension_altitude",
        "suspension_height": "suspension_height",
        "height_dif": "height_diff_input",   # kept only for reference; not used
        "height_diff": "height_diff_input",  # kept only for reference; not used
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["type"] = normalize_text_series(df["type"])
    df["span_type"] = normalize_text_series(df["span_type"])
    df["span"] = pd.to_numeric(df["span"], errors="coerce")
    df["suspension_altitude"] = pd.to_numeric(df["suspension_altitude"], errors="coerce")

    return df


def add_backward_forward_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row i describes the FORWARD span from tower i to tower i+1.

    Therefore, at tower i:
      - backward span = span from tower i-1 to tower i = span[i-1]
      - forward span  = span from tower i to tower i+1 = span[i]

    Height differences are derived from suspension_altitude:
      - dh_backward = altitude[i] - altitude[i-1]
      - dh_forward  = altitude[i+1] - altitude[i]

    This removes the need for a manually shifted height_diff input field.
    """
    df = df.copy()

    alt = df["suspension_altitude"]

    df["span_backward"] = df["span"].shift(1)
    df["span_forward"] = df["span"]

    df["dh_backward"] = alt - alt.shift(1)
    df["dh_forward"] = alt.shift(-1) - alt

    return df


def add_max_load(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["base_load_kg"] = df["type"].map(LOADS)

    unknown_types = sorted(df.loc[df["base_load_kg"].isna(), "type"].dropna().unique())
    if unknown_types:
        warnings.warn(f"Unknown tower types in LOADS mapping: {unknown_types}")

    df["max_load_kg"] = df["base_load_kg"] * 2 * BASE_WEIGHT #MAX_LOAD_WEIGHT_FACTOR
    return df


def add_sags_from_diagrams(df: pd.DataFrame, *, row_name: str, output_col: str) -> pd.DataFrame:
    """
    Adds a sag column by calling evaluate() once per span_type instead of once per row.
    row_name is usually "0" or "50".
    """
    df = df.copy()
    df[output_col] = np.nan

    for span_type, group in df.groupby("span_type", dropna=False):
        if span_type not in DIAGRAMS:
            warnings.warn(
                f"No diagram mapping for span_type={span_type!r}. "
                f"Column {output_col!r} will remain NaN for these rows."
            )
            continue

        spans = sorted(group["span"].dropna().unique())
        if len(spans) == 0:
            continue

        df_sags = evaluate(DIAGRAMS[span_type], spans)

        if row_name not in df_sags.index:
            warnings.warn(
                f"Diagram {DIAGRAMS[span_type]!r} has no row {row_name!r}. "
                f"Column {output_col!r} will remain NaN for span_type={span_type!r}."
            )
            continue

        sag_by_span = {float(col): float(df_sags.loc[row_name, col]) for col in df_sags.columns}
        df.loc[group.index, output_col] = group["span"].map(sag_by_span)

    return df


def add_tension_from_sag(df: pd.DataFrame, *, sag_col: str, tension_col: str, weight: float) -> pd.DataFrame:
    df = df.copy()

    df[tension_col] = [
        ts.Th_from_sag(sag, span, weight) if pd.notna(sag) and pd.notna(span) else np.nan
        for sag, span in zip(df[sag_col], df["span"])
    ]

    return df


def apply_tension_overrides(df: pd.DataFrame, *, tension_col: str, overrides: dict[str, float]) -> pd.DataFrame:
    df = df.copy()

    for span_type, value in overrides.items():
        df.loc[df["span_type"] == span_type, tension_col] = value

    return df


def vertical_participation_length(df: pd.DataFrame, *, tension_col: str, weight: float) -> np.ndarray:
    """
    Returns the total loaded horizontal length contributing to the tower vertical load.

    For tower i:
      backward contribution = distance from right support of backward span to its low point
      forward contribution  = distance from left support of forward span to its low point
    """
    backward = ts.distance_lowest_point_l(
        df["span_backward"].to_numpy(),
        df["dh_backward"].to_numpy(),
        df[tension_col].shift(1).to_numpy(),
        weight,
    )

    forward = ts.distance_lowest_point_r(
        df["span_forward"].to_numpy(),
        df["dh_forward"].to_numpy(),
        df[tension_col].to_numpy(),
        weight,
    )

    return backward + forward


def add_vertical_load_columns(
    df: pd.DataFrame,
    *,
    case_name: str,
    tension_col: str,
    weight: float,
) -> pd.DataFrame:
    df = df.copy()

    vertical_col = f"katakoryfo_{case_name}"
    percentage_col = f"load_percentage_{case_name}"

    df[vertical_col] = vertical_participation_length(df, tension_col=tension_col, weight=weight)
    df[percentage_col] = (df[vertical_col] * 2 * weight / df["max_load_kg"] * 100).round(2)

    return df


def add_solved_temperature_case(
    df: pd.DataFrame,
    *,
    case_name: str,
    temp: float,
    weight: float,
    base_tension_col: str = "tensions_0",
) -> pd.DataFrame:
    """Solve new tensions from the base 0°C tension, then calculate vertical load percentage."""
    df = df.copy()

    tension_col = f"tensions_{case_name}"
    solve_H2 = getattr(ts, "solve_for_H2_numeric", ts.solve_for_H2)

    df[tension_col] = [
        solve_H2(span, H1, BASE_TEMP, temp, BASE_WEIGHT, weight)
        if pd.notna(span) and pd.notna(H1)
        else np.nan
        for span, H1 in zip(df["span"], df[base_tension_col])
    ]

    df = add_vertical_load_columns(
        df,
        case_name=case_name,
        tension_col=tension_col,
        weight=weight,
    )

    return df


def add_theoretical_50_case(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = add_sags_from_diagrams(df, row_name="50", output_col="sag_50_theoretical")
    df = add_tension_from_sag(
        df,
        sag_col="sag_50_theoretical",
        tension_col="tensions_50_theoretical",
        weight=BASE_WEIGHT,
    )
    df = apply_tension_overrides(
        df,
        tension_col="tensions_50_theoretical",
        overrides=TENSION_50_OVERRIDES,
    )

    df = add_vertical_load_columns(
        df,
        case_name="50_theoretical",
        tension_col="tensions_50_theoretical",
        weight=BASE_WEIGHT,
    )

    return df


def add_monopleyro_and_vari(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Backward span: tower i is the RIGHT support of span i-1.
    df["monopleyro_backward_50_theoretical"] = ts.monopleyro_right(
        df["span_backward"].to_numpy(),
        df["dh_backward"].to_numpy(),
        df["tensions_50_theoretical"].shift(1).to_numpy(),
        BASE_WEIGHT,
        invalid="zero",
    )

    # Forward span: tower i is the LEFT support of span i.
    df["monopleyro_forward_50_theoretical"] = ts.monopleyro_left(
        df["span_forward"].to_numpy(),
        df["dh_forward"].to_numpy(),
        df["tensions_50_theoretical"].to_numpy(),
        BASE_WEIGHT,
        invalid="zero",
    )

    # Keep shorter aliases if you still want left/right-style columns in Excel.
    df["monopleyro_left_50_theoretical"] = df["monopleyro_backward_50_theoretical"]
    df["monopleyro_right_50_theoretical"] = df["monopleyro_forward_50_theoretical"]

    total_mono = (
        df["monopleyro_backward_50_theoretical"].fillna(0.0)
        + df["monopleyro_forward_50_theoretical"].fillna(0.0)
    )

    eligible = ~df["type"].str.startswith(("R", "G", "S"), na=False)

    reduction_T = df["base_load_kg"] * 0.1 * 0.75
    reduction_Z = df["base_load_kg"] * 0.1 * 65 / 80

    df["vari_Τ"] = 0.0
    df["vari_Ζ"] = 0.0

    df.loc[eligible, "vari_Τ"] = np.maximum(
        0.0,
        (total_mono.loc[eligible] - reduction_T.loc[eligible]) * 2 * VARI_WEIGHT_FACTOR,
    )

    df.loc[eligible, "vari_Ζ"] = np.maximum(
        0.0,
        (total_mono.loc[eligible] - reduction_Z.loc[eligible]) * 2 * VARI_WEIGHT_FACTOR,
    )

    return df


# ============================================================
# MAIN SCRIPT
# ============================================================

def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    df = standardize_columns(df)
    df = add_backward_forward_geometry(df)
    df = add_max_load(df)

    # Base 0°C case from sag diagrams.
    df = add_sags_from_diagrams(df, row_name="0", output_col="sag_0")
    df = add_tension_from_sag(df, sag_col="sag_0", tension_col="tensions_0", weight=BASE_WEIGHT)
    df = apply_tension_overrides(df, tension_col="tensions_0", overrides=BASE_TENSION_OVERRIDES)
    df = add_vertical_load_columns(df, case_name="0", tension_col="tensions_0", weight=BASE_WEIGHT)

    # Temperature / ice cases.
    for case in LOAD_CASES:
        df = add_solved_temperature_case(
            df,
            case_name=case["name"],
            temp=case["temp"],
            weight=case["weight"],
        )

    # 50°C theoretical case and monopleyro / vari checks.
    df = add_theoretical_50_case(df)
    df = add_monopleyro_and_vari(df)

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_XLSX, index=False)

    print(f"Wrote: {OUTPUT_XLSX}")


def main_plot() -> None:

    #########################
    # Optional: plot catenaries 
    #
    #  Possible selections
    #
    #   "0"
    #   "-10"
    #   "-10_ICE"
    #   "0_ICE"
    #   "50_theoretical"
    #
    #########################

    input_path = HERE / "outputs" / "larisa2_2nd_submission_processed.xlsx"
    output_path = HERE / "outputs" / "catenaries.dxf"

    summary = plot_catenaries_from_file(
        input_path,
        output_path=output_path,
        load_case="-10",
    )

if __name__ == "__main__":
    
    #main()
    
    main_plot()