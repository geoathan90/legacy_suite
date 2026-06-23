import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.tensions as ts
from scripts.eval import evaluate

from .catenary_dxf_plotter import (
    plot_catenaries_from_file,
    plot_catenary_cases_from_file_to_one_dxf,
)


## usage python -m Grammes.THL_Larisa.larisa_new

# ============================================================
# USER SETTINGS
# ============================================================

HERE = Path(__file__).resolve().parent

INPUT_CSV = HERE / "larisa2_2nd_submission_line_1.csv"
OUTPUT_XLSX = HERE / "outputs" / "test.xlsx" #"larisa2_2nd_submission_processed.xlsx"

# Cardinal default, in kg/m
BASE_WEIGHT = 1.823
BASE_WEIGHT_SW = 0.769

# Effective weights / factors used in the original script
# MAX_LOAD_WEIGHT_FACTOR = 2.2662   #  0" ice and 9# wind
ICE_WEIGHT = 2.623                      #  1/4" ice +k - approximation
HEAVY_ICE_WEIGHT = 4.02                  #  1/2" ice +k - approximation

ICE_WEIGHT_SW = 1.11                   #  1/4" ice - approximation
HEAVY_ICE_WEIGHT_SW = 1.68             #  1/2" ice - approximation

CROSS_SECTION_AREA = 5.47e-4            # m^2, from 547 mm^2 in the original script
YOUNG_MODULUS_INITIAL = 5.132e9         # kg/m2
YOUNG_MODULUS_FINAL = 6.8529e9         # kg/m2
THERMAL_EXPANSION = 1.935e-5            # 1/°C

CROSS_SECTION_AREA_SW = 9.6454e-5       # m^2, from 547 mm^2 in the original script
YOUNG_MODULUS_INITIAL_SW = 19.33e9         # kg/m2
THERMAL_EXPANSION_SW = 1.152e-5            # 1/°C

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
    "BA350": "31191",
    "BA500": "31191",
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
    "ZE5": 7810,
    "ZE5+8.00": 7810,
    "Z5*": 7810,
    "Z5+8.00*": 7810,
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

SHIELD_WIRE_HEIGHT_OFFSETS = {
    "S5": 21.4,
    "S5+8.00": 21.4,
    "G5": 21.4,
    "G5+8.00": 21.4,
    "R5": 22.7,
    "R5+8.00": 22.7,
    "R5+18.00": 22.7,
    "RE5": 22.7,
    "RE5+8.00": 22.7,
    "T5": 25.2,
    "T5+8.00": 25.2,
    "T5+18.00": 25.2,
    "TE5": 25.2,
    "TE5+8.00": 25.2,
    "TE5+18.00": 25.2,
    "Z5": 25.2,
    "Z5+8.00": 25.2,
    "ZE5": 25.2,
    "ZE5+8.00": 25.2,
    "Z5*": 25.2,
    "Z5+8.00*": 25.2,
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

BASE_TENSION_OVERRIDES_SW = {
    "BA350": 1810,
    "BA500": 1520, 
}

TENSION_50_OVERRIDES_SW = {}

LOAD_CASES = [
    {"name": "0_ICE", "temp": 0, "weight": ICE_WEIGHT},  
    {"name": "-10_bare", "temp": -10, "weight": BASE_WEIGHT},
    {"name": "-10_ICE", "temp": -10, "weight": ICE_WEIGHT},
    {"name": "-19_bare", "temp": -19, "weight": BASE_WEIGHT}, 
    {"name": "-19_HEAVY_ICE", "temp": -19, "weight": HEAVY_ICE_WEIGHT},
]

LOAD_CASES_SW = [
    {"name": "0_ICE", "temp": 0, "weight": ICE_WEIGHT_SW},
    {"name": "-10_bare", "temp": -10, "weight": BASE_WEIGHT_SW},
    {"name": "-10_ICE", "temp": -10, "weight": ICE_WEIGHT_SW},
    {"name": "-19_bare", "temp": -19, "weight": BASE_WEIGHT_SW},
    {"name": "-19_HEAVY_ICE", "temp": -19, "weight": HEAVY_ICE_WEIGHT_SW},
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

def _add_functional_span2(df):
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

def add_sag_column_from_diagrams(
    df: pd.DataFrame,
    row_name: str,
    output_col: str,
    span_col: str = "functional_span",
    diagrams: dict[str, str] = DIAGRAMS,
) -> pd.DataFrame:
    """
        Add a sag column from the sag diagrams.

        Each row provides:
            span_type:
                Selects which diagram to use, through the DIAGRAMS dictionary.

            span_col:
                Selects which span value to evaluate on the diagram.
                By default this is "functional_span", not the physical "span".

        Example:
            add_sag_column_from_diagrams_simple(
                df,
                row_name="0",
                output_col="sag_0",
            )

        This will:
            - read row["span_type"]
            - find the corresponding diagram
            - evaluate sag at row["functional_span"]
            - store the result in row["sag_0"]
    """
    sag_values = []

    for _, row in df.iterrows():
        span_type = row["span_type"]
        span_value = row[span_col]

        diagram = diagrams.get(span_type)

        if diagram is None or pd.isna(span_value):
            sag_values.append(np.nan)
            continue

        sag_table = evaluate(diagram, [span_value])

        if row_name not in sag_table.index:
            warnings.warn(
                f"Diagram {diagram!r} for span_type={span_type!r} "
                f"does not contain row {row_name!r}."
            )
            sag_values.append(np.nan)
            continue

        # evaluate() names the output column using str(float(span)).
        col_name = str(float(span_value))

        sag_values.append(float(sag_table.loc[row_name, col_name]))

    df[output_col] = sag_values
    return df

def add_tension_from_sag(
    df: pd.DataFrame,
    sag_col: str,
    tension_col: str,
    weight: float,
    span_col: str = "functional_span",
) -> pd.DataFrame:

    tensions = []

    for sag, span in zip(df[sag_col], df[span_col]):
        if pd.isna(sag) or pd.isna(span):
            tensions.append(np.nan)
        else:
            tensions.append(ts.Th_from_sag(sag, span, weight))

    df[tension_col] = tensions
    return df

def apply_tension_overrides(
    df: pd.DataFrame,
    tension_col: str,
    overrides: dict[str, float],
) -> pd.DataFrame:
    """
        Replace calculated tensions with known/manual values for selected span types.

        Example:
            overrides = {"BA350": 3470, "BA500": 3105}

        This means:
            rows with span_type == "BA350" get tension_col = 3470
            rows with span_type == "BA500" get tension_col = 3105
            all other rows are left unchanged
    """
    for span_type, tension_value in overrides.items():
        matching_rows = (df["span_type"] == span_type)   # boolean Series: True where span_type matches, False otherwise
        df.loc[matching_rows, tension_col] = tension_value

    return df

def apply_shield_wire_altitudes(
    df: pd.DataFrame,
    offsets: dict[str, float] = SHIELD_WIRE_HEIGHT_OFFSETS,
) -> pd.DataFrame:
    """
    Adjust suspension_altitude from phase-conductor attachment level
    to shield-wire attachment level.

    The offset is selected from tower type.
    """
    df = df.copy()

    offset = df["type"].map(offsets)

    unknown_types = sorted(df.loc[offset.isna(), "type"].dropna().unique())
    if unknown_types:
        warnings.warn(f"Unknown tower types in SHIELD_WIRE_HEIGHT_OFFSETS: {unknown_types}")

    df["suspension_altitude"] = df["suspension_altitude"] + offset

    return df

def warn_suspicious_catenary_inputs(
    df: pd.DataFrame,
    *,
    case_name: str,
    tension_col: str,
    weight: float,
) -> None:
    """
    Warn about rows likely to cause numerical overflow or physically suspicious catenary results.
    """
    H = pd.to_numeric(df[tension_col], errors="coerce")
    a = H / weight

    ratio_forward = df["span_forward"] / (2.0 * a)
    ratio_backward = df["span_backward"] / (2.0 * H.shift(1) / weight)

    suspicious = (
        H.isna()
        | (H <= 0.0)
        | (a <= 0.0)
        | (ratio_forward.abs() > 50.0)
        | (ratio_backward.abs() > 50.0)
    )

    bad = df.loc[
        suspicious,
        ["tower_number", "type", "span_type", "span", "functional_span", tension_col]
    ].copy()

    if not bad.empty:
        warnings.warn(
            f"Suspicious catenary inputs in case {case_name!r}, column {tension_col!r}:\n"
            f"{bad.to_string(index=False)}"
        )

def katakoryfa(
    df: pd.DataFrame,
    case_name: str,
    tension_col: str,
    weight: float,
    load_dict: dict[str, float] = LOADS,
    quantity: float = 2.0,
) -> pd.DataFrame:
    """
        Calculate vertical-load participation for a specific load case.

        Input convention:
            Each row i represents tower i.
            df["span"][i] is the physical forward span from tower i to tower i+1.
            df[tension_col][i] is the horizontal tension of that same forward span.

        Geometry convention:
            For tower i:
                backward span = row i-1 span
                forward span  = row i span

            backward tension = df[tension_col].shift(1)
            forward tension  = df[tension_col]

        Added columns:
            backward_meters_<case_name>
            forward_meters_<case_name>
            backward_kg_<case_name>
            forward_kg_<case_name>
            total_vert_meters_<case_name>
            total_vert_kg_<case_name>
            load_percentage_<case_name>

        Notes:
            The physical span geometry remains based on span_backward/span_forward.
            The tension column may have been calculated from functional_span earlier.

            This function does not add helper/capacity columns to the dataframe.
            The tower capacity is looked up internally from load_dict.
    """

    backward_m_col = f"backward_meters_{case_name}"
    forward_m_col = f"forward_meters_{case_name}"

    backward_kg_col = f"backward_kg_{case_name}"
    forward_kg_col = f"forward_kg_{case_name}"

    total_m_col = f"total_vert_meters_{case_name}"
    total_kg_col = f"total_vert_kg_{case_name}"

    percentage_col = f"load_percentage_{case_name}"

    warn_suspicious_catenary_inputs(
        df,
        case_name=case_name,
        tension_col=tension_col,
        weight=weight,
    )

    # Tower i receives backward contribution from span i-1.
    # According to your L/R convention, distance_lowest_point_l() gives
    # the contribution of the backward span up to the examined tower.
    df[backward_m_col] = ts.distance_lowest_point_l(
        df["span_backward"].to_numpy(),
        df["dh_backward"].to_numpy(),
        df[tension_col].shift(1).to_numpy(),
        weight,
    )

    # Tower i receives forward contribution from span i.
    # According to your L/R convention, distance_lowest_point_r() gives
    # the contribution of the forward span up to the examined tower.
    df[forward_m_col] = ts.distance_lowest_point_r(
        df["span_forward"].to_numpy(),
        df["dh_forward"].to_numpy(),
        df[tension_col].to_numpy(),
        weight,
    )

    # Convert contributing lengths to vertical load.
    # The factor 2 preserves the previous script's two-conductor/subconductor logic.
    df[backward_kg_col] = df[backward_m_col] * weight * quantity
    df[forward_kg_col] = df[forward_m_col] * weight * quantity

    df[total_m_col] = df[backward_m_col] + df[forward_m_col]
    df[total_kg_col] = df[backward_kg_col] + df[forward_kg_col]

    # Internal capacity lookup. This does NOT add a capacity column to df.
    tower_capacity_kg = df["type"].map(load_dict)

    unknown_types = sorted(df.loc[tower_capacity_kg.isna(), "type"].dropna().unique())
    if unknown_types:
        warnings.warn(f"Unknown tower types in LOADS mapping: {unknown_types}")

    df[percentage_col] = (df[total_kg_col] / tower_capacity_kg * 100).round(2)

    return df

def add_solved_temperature_case(
    df: pd.DataFrame,
    case_name: str,
    temp: float,
    weight: float,
    base_tension_col: str = "tensions_0",
    span_col: str = "functional_span",
    base_temp: float = BASE_TEMP,
    base_weight: float = BASE_WEIGHT,
    E: float = YOUNG_MODULUS_INITIAL,
    A: float = CROSS_SECTION_AREA,
    alpha: float = THERMAL_EXPANSION,
    load_dict: dict[str, float] = LOADS,
    quantity: float = 2.0,
) -> pd.DataFrame:
    """
        Solve a new temperature / weight condition from an existing base tension.

        This function does two things:

        1. Calculates a new horizontal tension column:
            tensions_<case_name>

        The calculation uses functional_span by default, because this is a
        sag-tension/state-equation calculation.

        2. Calls katakoryfa() to calculate the tower vertical loading columns
        for that same load case.

        Important distinction:
            functional_span is used for the tension-state calculation.
            physical span geometry is still used inside katakoryfa().
    """

    tension_col = f"tensions_{case_name}"

    # Prefer the faster numeric solver if it exists.
    # Fall back to the symbolic/SymPy version otherwise.
    solve_H2 = getattr(ts, "solve_for_H2_numeric", ts.solve_for_H2)

    tensions = []

    for span, H1 in zip(df[span_col], df[base_tension_col]):

        if pd.isna(span) or pd.isna(H1):
            tensions.append(np.nan)
            continue

        H2 = solve_H2(
            span,        # S: functional/ruling span for sag-tension calculation
            H1,          # H1: starting horizontal tension
            E,           # modulus of elasticity
            A,           # conductor cross-sectional area
            alpha,       # thermal expansion coefficient
            base_temp,   # T1
            temp,        # T2
            base_weight, # w1
            weight,      # w2
        )

        tensions.append(H2)

    df[tension_col] = tensions

    df = katakoryfa(
        df,
        case_name=case_name,
        tension_col=tension_col,
        weight=weight,
        load_dict=load_dict,
        quantity=quantity
    )

    return df



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


def build_phase_conductor_results(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    df = add_span_geometry(df)
    df = add_functional_span(df)

    df = add_sag_column_from_diagrams(df, row_name="0", output_col="sag_0")
    df = add_tension_from_sag(df, sag_col="sag_0", tension_col="tensions_0", weight=BASE_WEIGHT)
    df = apply_tension_overrides(df, tension_col="tensions_0", overrides=BASE_TENSION_OVERRIDES_INITIAL)
    df = katakoryfa(df, case_name="0", tension_col="tensions_0", weight=BASE_WEIGHT)

    for case in LOAD_CASES:
        df = add_solved_temperature_case(
            df,
            case_name=case["name"],
            temp=case["temp"],
            weight=case["weight"],
        )

    df = add_sag_column_from_diagrams(df, row_name="50", output_col="sag_50_theoretical")
    df = add_tension_from_sag(df, sag_col="sag_50_theoretical", tension_col="tensions_50_theoretical", weight=BASE_WEIGHT)
    df = apply_tension_overrides(df, tension_col="tensions_50_theoretical", overrides=TENSION_50_OVERRIDES)
    df = katakoryfa(df, case_name="50_theoretical", tension_col="tensions_50_theoretical", weight=BASE_WEIGHT)

    return df

def build_shield_wire_results(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build shield-wire sag/tension and vertical-load results.

    This mirrors the phase-conductor workflow, but uses:
        DIAGRAMS_SW
        LOADS_SW
        shield-wire weights
        shield-wire mechanical properties
        adjusted shield-wire suspension altitude
    """

    df_sw = df_raw.copy()

    # Move from phase-conductor attachment altitude to shield-wire attachment altitude.
    df_sw = apply_shield_wire_altitudes(df_sw)

    # Geometry must be recalculated after the altitude adjustment.
    df_sw = add_span_geometry(df_sw)

    # Functional/ruling span logic is the same as for phase conductors.
    df_sw = add_functional_span(df_sw)

    # ========================================================
    # Base 0°C shield-wire case
    # ========================================================

    df_sw = add_sag_column_from_diagrams(
        df_sw,
        row_name="0",
        output_col="sag_0",
        diagrams=DIAGRAMS_SW,
    )

    df_sw = add_tension_from_sag(
        df_sw,
        sag_col="sag_0",
        tension_col="tensions_0",
        weight=BASE_WEIGHT_SW,
    )

    df_sw = apply_tension_overrides(
        df_sw,
        tension_col="tensions_0",
        overrides=BASE_TENSION_OVERRIDES_SW,
    )

    df_sw = katakoryfa(
        df_sw,
        case_name="0",
        tension_col="tensions_0",
        weight=BASE_WEIGHT_SW,
        load_dict=LOADS_SW,
        quantity=1.0,  # change to 2.0 if LOADS_SW/checking basis expects two shield wires
    )

    # ========================================================
    # Solved shield-wire temperature / ice cases
    # ========================================================

    for case in LOAD_CASES_SW:
        df_sw = add_solved_temperature_case(
            df_sw,
            case_name=case["name"],
            temp=case["temp"],
            weight=case["weight"],
            base_tension_col="tensions_0",
            base_weight=BASE_WEIGHT_SW,
            E=YOUNG_MODULUS_INITIAL_SW,
            A=CROSS_SECTION_AREA_SW,
            alpha=THERMAL_EXPANSION_SW,
            load_dict=LOADS_SW,
            quantity=1.0,  # change to 2.0 if appropriate
        )

    # ========================================================
    # Optional 50°C shield-wire case from diagrams
    # ========================================================

    df_sw = add_sag_column_from_diagrams(
        df_sw,
        row_name="50",
        output_col="sag_50_theoretical",
        diagrams=DIAGRAMS_SW,
    )

    df_sw = add_tension_from_sag(
        df_sw,
        sag_col="sag_50_theoretical",
        tension_col="tensions_50_theoretical",
        weight=BASE_WEIGHT_SW,
    )

    # # If you define TENSION_50_OVERRIDES_SW, use it.
    # # If not, either skip this or set TENSION_50_OVERRIDES_SW = {}.
    # df_sw = apply_tension_overrides(
    #     df_sw,
    #     tension_col="tensions_50_theoretical",
    #     overrides=TENSION_50_OVERRIDES_SW,
    # )

    # df_sw = katakoryfa(
    #     df_sw,
    #     case_name="50_theoretical",
    #     tension_col="tensions_50_theoretical",
    #     weight=BASE_WEIGHT_SW,
    #     load_dict=LOADS_SW,
    #     quantity=1.0,  # change to 2.0 if appropriate
    # )

    return df_sw

# ============================================================
# EXPORT SCRIPTS 
# ============================================================

def export_formatted_excel(df: pd.DataFrame, output_path: Path) -> None:
    """
    Export dataframe to a formatted Excel file.

    This is meant to replace:
        df.to_excel(OUTPUT_XLSX, index=False)
    """
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        sheet_name = "results"
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        header_format = workbook.add_format({
            "bold": True,
            "bg_color": "#D9EAF7",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        })

        default_format = workbook.add_format({})
        meters_format = workbook.add_format({"num_format": "0.00"})
        kg_format = workbook.add_format({"num_format": "0.00"})
        percent_format = workbook.add_format({"num_format": "0.00"})

        default_divider_format = workbook.add_format({"left": 5})
        meters_divider_format = workbook.add_format({"num_format": "0.00", "left": 5})
        kg_divider_format = workbook.add_format({"num_format": "0.00", "left": 5})
        percent_divider_format = workbook.add_format({"num_format": "0.00", "left": 5})

        # Header row.
        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_format)

        worksheet.freeze_panes(1, 0)
        worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

        for col_num, col_name in enumerate(df.columns):
            width = max(12, min(30, len(str(col_name)) + 2))

            starts_case_group = col_name.startswith("tensions_")

            if "meters" in col_name:
                fmt = meters_divider_format if starts_case_group else meters_format
            elif "kg" in col_name or "tensions" in col_name:
                fmt = kg_divider_format if starts_case_group else kg_format
            elif "percentage" in col_name:
                fmt = percent_divider_format if starts_case_group else percent_format
            else:
                fmt = default_divider_format if starts_case_group else default_format

            worksheet.set_column(col_num, col_num, width, fmt)

def format_excel_worksheet(workbook, worksheet, df: pd.DataFrame) -> None:
    """
    Apply the same formatting logic to one worksheet.
    """

    header_format = workbook.add_format({
        "bold": True,
        "bg_color": "#D9EAF7",
        "border": 1,
        "align": "center",
        "valign": "vcenter",
    })

    default_format = workbook.add_format({})
    meters_format = workbook.add_format({"num_format": "0.00"})
    kg_format = workbook.add_format({"num_format": "0.00"})
    percent_format = workbook.add_format({"num_format": "0.00"})

    default_divider_format = workbook.add_format({"left": 5})
    meters_divider_format = workbook.add_format({"num_format": "0.00", "left": 5})
    kg_divider_format = workbook.add_format({"num_format": "0.00", "left": 5})
    percent_divider_format = workbook.add_format({"num_format": "0.00", "left": 5})

    for col_num, col_name in enumerate(df.columns):
        worksheet.write(0, col_num, col_name, header_format)

    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

    for col_num, col_name in enumerate(df.columns):
        width = max(12, min(30, len(str(col_name)) + 2))

        starts_case_group = col_name.startswith("tensions_")

        if "meters" in col_name:
            fmt = meters_divider_format if starts_case_group else meters_format
        elif "kg" in col_name or "tensions" in col_name:
            fmt = kg_divider_format if starts_case_group else kg_format
        elif "percentage" in col_name:
            fmt = percent_divider_format if starts_case_group else percent_format
        else:
            fmt = default_divider_format if starts_case_group else default_format

        worksheet.set_column(col_num, col_num, width, fmt)

def export_formatted_excel_multi_sheet(
    sheets: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """
    Export multiple dataframes to one formatted Excel workbook.

    Example:
        export_formatted_excel_multi_sheet(
            {
                "phase_conductors": df_phase,
                "shield_wires": df_sw,
            },
            OUTPUT_XLSX,
        )
    """
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            format_excel_worksheet(workbook, worksheet, df)

# ============================================================
# MAIN SCRIPTS
# ============================================================

def main_test() -> None:
    df_raw = pd.read_csv(INPUT_CSV)

    df_raw = standardize_columns(df_raw)

    df_phase = build_phase_conductor_results(df_raw)
    df_sw = build_shield_wire_results(df_raw)

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)

    export_formatted_excel_multi_sheet(
        {
            "phase_conductors": df_phase,
            "shield_wires": df_sw,
        },
        OUTPUT_XLSX,
    )
         
def main_plot() -> None:
    """
    Plot all available catenary load cases into one DXF file.

    The processed Excel must already exist.
    Run main_test() first if needed.
    """

    input_path = OUTPUT_XLSX
    output_path = HERE / "outputs" / "catenaries_all_cases.dxf"

    plot_cases = [
        {"name": "0", "weight": BASE_WEIGHT},
        *[
            {"name": case["name"], "weight": case["weight"]}
            for case in LOAD_CASES
        ],
        {"name": "50_theoretical", "weight": BASE_WEIGHT},
    ]

    summary = plot_catenary_cases_from_file_to_one_dxf(
        input_path,
        output_path=output_path,
        cases=plot_cases,
    )

    print(f"Wrote: {output_path}")
    print()
    print("Vertex-location cases by load case:")
    print(
        summary.groupby(["load_case", "case"])
        .size()
        .to_string()
    )



if __name__ == "__main__":
    
    main_test()

    main_plot()
