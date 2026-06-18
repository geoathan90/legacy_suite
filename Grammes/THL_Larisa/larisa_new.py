import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.tensions as ts
from scripts.eval import evaluate

from .catenary_dxf_plotter import plot_catenaries_from_file


## usage python -m Grammes.THL_Larisa.larisa_new

# ============================================================
# USER SETTINGS
# ============================================================

HERE = Path(__file__).resolve().parent

INPUT_CSV = HERE / "larisa2_2nd_submission.csv"
OUTPUT_XLSX = HERE / "outputs" / "test.xlsx" #"larisa2_2nd_submission_processed.xlsx"

# Cardinal default, in kg/m
BASE_WEIGHT = 1.823

# Effective weights / factors used in the original script
# MAX_LOAD_WEIGHT_FACTOR = 2.2662   #  0" ice and 9# wind
ICE_WEIGHT = 2.623                      #  1/4" ice - approximation
HEAVY_ICE_WEIGHT = 3.6                  #  1/2" ice - approximation

CROSS_SECTION_AREA = 5.47e-4            # m^2, from 547 mm^2 in the original script
YOUNG_MODULUS_INITIAL = 5.132e9         # kg/m2
YOUNGS_MODULUS_FINAL = 6.8529e9         # kg/m2
THERMAL_EXPANSION = 1.935e-5            # 1/°C

BASE_TEMP = 0

DIAGRAMS = {
    "term": "31185",
    "BA350": "31187",
    "BA500": "31188",
    "2000": "52740",
    "1000": "31189",
    "700": "31858",
}

DIAGRAMS_SW = {
    "term": "31191",
    "BA350": "31192",
    "BA500": "31193",
    "2000": "31194",
    "1000": "31859",
    "700": "31859",
}

LOADS = {                       #### in kg
    "S5": 4220,
    "S5+8.00": 4220,
    "G5": 4890,
    "G5+8.00": 4890,
    "R5": 5570,
    "R5+8.00": 5570,
    "R5+18.00": 5570,
    "RE5": 5570*1000/800,
    "RE5+8.00": 5570*1000/800,
    "T5": 7810,
    "T5+8.00": 7810,
    "T5+18.00": 7810,
    "TE5": 7810,
    "TE5+8.00": 7810,
    "TE5+18.00": 7810,
    "Z5": 6470,
    "Z5+8.00": 6470,
    "ZE5": 6470*1.5,
    "ZE5+8.00": 6470*1.5,
    "Z5*": 6470,
    "Z5+8.00*": 6470,
}

LOADS_SW = {
    "S5": 1000,
    "S5+8.00": 1000,
    "G5": 1170,
    "G5+8.00": 1170,
    "R5": 1340,
    "R5+8.00": 1340,
    "R5+18.00": 1340,
    "RE5": 1340*1000/800,
    "RE5+8.00": 1340*1000/800,
    "T5": 1680,
    "T5+8.00": 1680,
    "T5+18.00": 1680,
    "TE5": 1680*1.2,
    "TE5+8.00": 1680*1.2,
    "TE5+18.00": 1680*1.2,
    "Z5": 1340,
    "Z5+8.00": 1340,
    "ZE5": 1340,
    "ZE5+8.00": 1340,
    "Z5*": 1340,
    "Z5+8.00*": 1340,
}

# Hard-coded
BASE_TENSION_OVERRIDES_INITIAL = {      # 0 degrees, initial
    "BA350": 3470,
    "BA500": 3105,
}

BASE_TENSION_OVERRIDES_FINAL = {      # 0 degrees, final
    "BA350": 3151.2,
    "BA500": 2953.6,
}

TENSION_50_OVERRIDES = {        # 50 degrees, final
    "BA350": 2585,
    "BA500": 2665,  #2654?
    "term": 2585,
}

LOAD_CASES = [
    {"name": "-19_bare", "temp": -19, "weight": BASE_WEIGHT},   
    {"name": "-10_bare", "temp": -10, "weight": BASE_WEIGHT},
    {"name": "-10_ICE", "temp": -10, "weight": ICE_WEIGHT},
    {"name": "0_ICE", "temp": 0, "weight": ICE_WEIGHT},
    {"name": "-19_HEAVY_ICE", "temp": -19, "weight": HEAVY_ICE_WEIGHT},
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


def add_span_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """
        Add backward/forward span geometry around each tower.

        Input convention:
            Each row represents a tower.
            df["span"][i] is the forward span from tower i to tower i+1.
            df["suspension_altitude"][i] is the conductor suspension altitude at tower i.

        Added columns:
            span_backward:
                Span from tower i-1 to tower i.
                This is span[i-1], so the first row is NaN.

            span_forward:
                Span from tower i to tower i+1.
                This is span[i].

            dh_backward:
                Height difference of the backward span:
                altitude[i] - altitude[i-1].

            dh_forward:
                Height difference of the forward span:
                altitude[i+1] - altitude[i].

        Sign convention:
            dh > 0 means the right/forward support is higher than the left/backward support.

        Boundary rows:
            First row has no backward span.
            Last row has no forward dh unless there is a next support row.
    """
    
    alt = df["suspension_altitude"]

    df["span_backward"] = df["span"].shift(1)
    df["span_forward"] = df["span"]

    df["dh_backward"] = alt - alt.shift(1)
    df["dh_forward"] = alt.shift(-1) - alt

    return df

def add_functional_span(df):
    df = df.copy()

    # Start as a direct copy of the original span column
    df["functional_span"] = df["span"].astype(float)

    # Classify towers by first character of the type string
    first_letter = df["type"].astype(str).str.strip().str[0]

    is_strain = first_letter.isin(["T", "Z"])
    is_suspension = first_letter.isin(["S", "R", "G"])

    last_strain_pos = None

    for pos in range(len(df)):
        if is_strain.iloc[pos]:   # 

            if last_strain_pos is not None:
                middle = range(last_strain_pos + 1, pos)

                # Valid segment: at least one suspension tower between two strain towers
                if len(middle) > 0 and is_suspension.iloc[list(middle)].all():

                    # Spans physically belonging to the segment:
                    # from the starting strain tower row up to the row before the ending strain tower
                    segment_rows = df.index[last_strain_pos:pos]
                    segment_spans = df.loc[segment_rows, "span"].astype(float)

                    ruling_span = np.sqrt(
                        (segment_spans**3).sum() / segment_spans.sum()
                    )

                    df.loc[segment_rows, "functional_span"] = ruling_span

            last_strain_pos = pos

        elif not is_suspension.iloc[pos]:
            # Unknown tower category: break the chain
            last_strain_pos = None

    return df


def add_functional_span2(df):
    # Make a copy so that we do not modify the original dataframe in-place.
    # This is optional, but usually safer.
    df = df.copy()

    # functional_span starts out identical to span.
    # Later, only the rows that belong to valid suspension segments
    # will be overwritten with the ruling span of that segment.
    df["functional_span"] = df["span"].astype(float)

    # Extract the first letter of each tower type.
    #
    # Examples:
    # "Z5+8.00"  -> "Z"
    # "R5+18.00" -> "R"
    # "ZE5"      -> "Z"
    # "G5+8.00"  -> "G"
    # "Z5+8.00*" -> "Z"
    #
    # .astype(str) makes sure the values are strings.
    # .str.strip() removes accidental leading/trailing spaces.
    # .str[0] takes the first character.
    first_letter = df["type"].astype(str).str.strip().str[0]

    # Boolean Series:
    # True where the tower is a strain tower, False otherwise.
    #
    # According to your rule:
    # T and Z towers are strain towers.
    is_strain = first_letter.isin(["T", "Z"])

    # Boolean Series:
    # True where the tower is a suspension tower, False otherwise.
    #
    # According to your rule:
    # S, R and G towers are suspension towers.
    is_suspension = first_letter.isin(["S", "R", "G"])

    # This will store the integer position of the most recent strain tower
    # encountered while scanning down the dataframe.
    #
    # At the beginning, we have not found any strain tower yet.
    last_strain_pos = None

    # Go through the dataframe row by row, using integer positions:
    # 0, 1, 2, 3, ...
    for pos in range(len(df)):

        # Check whether the current row is a strain tower.
        #
        # is_strain is a boolean Series.
        # is_strain.iloc[pos] is one boolean value: True or False.
        #
        # So this means:
        # "If the current tower is a strain tower..."
        if is_strain.iloc[pos]:

            # If this is not the first strain tower we have found,
            # then we now have a possible segment:
            #
            # previous strain tower  --->  current strain tower
            #
            # Example:
            # last_strain_pos = 4
            # pos = 7
            #
            # Potential segment:
            # tower 4 -> tower 7
            if last_strain_pos is not None:

                # The towers strictly between the two strain towers.
                #
                # Example:
                # last_strain_pos = 4
                # pos = 7
                #
                # middle = 5, 6
                #
                # These must all be suspension towers for the segment to be valid.
                middle = range(last_strain_pos + 1, pos)

                # The segment is valid if:
                #
                # 1. There is at least one tower between the strain towers.
                #    This avoids treating two adjacent strain towers as a segment.
                #
                # 2. Every intermediate tower is a suspension tower.
                #
                # Example valid:
                # Z - R - R - Z
                #
                # Example invalid:
                # Z - R - T - Z
                #
                # because T is not a suspension tower.
                if len(middle) > 0 and is_suspension.iloc[list(middle)].all():

                    # Now we need the span rows that physically belong
                    # to this strain-to-strain segment.
                    #
                    # Important:
                    # The span value in row i is the span AFTER tower i.
                    #
                    # So if the segment is:
                    #
                    # tower 4 -> tower 5 -> tower 6 -> tower 7
                    #
                    # then the actual spans are:
                    #
                    # row 4: span between tower 4 and tower 5
                    # row 5: span between tower 5 and tower 6
                    # row 6: span between tower 6 and tower 7
                    #
                    # Therefore we use rows 4, 5, 6.
                    # We do NOT use row 7.
                    segment_rows = df.index[last_strain_pos:pos]

                    # Get the actual span lengths for this segment.
                    segment_spans = df.loc[segment_rows, "span"].astype(float)

                    # Calculate the ruling span.
                    #
                    # Standard ruling span formula:
                    #
                    # Lr = sqrt( sum(L_i^3) / sum(L_i) )
                    #
                    # where L_i are the individual span lengths.
                    ruling_span = np.sqrt(
                        (segment_spans**3).sum() / segment_spans.sum()
                    )

                    # Replace the functional_span values for the rows
                    # belonging to this segment.
                    #
                    # The original span column is left untouched.
                    df.loc[segment_rows, "functional_span"] = ruling_span

            # Whether or not the previous pair formed a valid segment,
            # the current strain tower now becomes the latest strain tower.
            #
            # This allows a strain tower to be:
            # - the end of the previous segment
            # - the start of the next segment
            #
            # Example:
            # Z - R - Z - G - Z
            #
            # The middle Z is both:
            # - end of segment 1
            # - start of segment 2
            last_strain_pos = pos

        # If the current tower is not a strain tower,
        # and also not a suspension tower, then it is unknown/unclassified.
        #
        # In that case, we break the chain.
        #
        # Example:
        # Z - R - UNKNOWN - R - Z
        #
        # should not be treated as a valid segment.
        elif not is_suspension.iloc[pos]:
            last_strain_pos = None

    # Return the dataframe with the added functional_span column.
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
# MAIN SCRIPTS
# ============================================================

def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    df = standardize_columns(df)
    df = add_backward_forward_geometry(df)
    df = add_max_load(df)

    # Base 0°C case from sag diagrams.
    df = add_sags_from_diagrams(df, row_name="0", output_col="sag_0")
    df = add_tension_from_sag(df, sag_col="sag_0", tension_col="tensions_0", weight=BASE_WEIGHT)
    df = apply_tension_overrides(df, tension_col="tensions_0", overrides=BASE_TENSION_OVERRIDES_INITIAL)
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

    load_case = "0_ICE"
    #output_name = f"catenaries_{load_case}.dxf"
    
    input_path = HERE / "outputs" / "larisa2_2nd_submission_processed.xlsx"
    output_path = HERE / "outputs" / f"catenaries_{load_case}.dxf"

    summary = plot_catenaries_from_file(
        input_path,
        output_path=output_path,
        load_case=load_case,
    )

def main_test() -> None:

    df = pd.read_csv(INPUT_CSV)

    df = standardize_columns(df)
    df = add_span_geometry(df)
    df = add_functional_span(df)

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_XLSX, index=False)
         

if __name__ == "__main__":
    
    #main()
    
    #main_plot()

    main_test()