"""
Automatic DXF plotting of chained catenaries with ezdxf.

Expected input
--------------
A processed CSV/XLSX table containing, at minimum:

    span
    suspension_altitude
    tensions_<load_case>

For example, for load_case="50_theoretical", the script looks for:

    tensions_50_theoretical

The assumed row convention is:

    row i describes the span from tower i to tower i+1

Therefore:

    span[i] is drawn between suspension_altitude[i] and suspension_altitude[i+1]
    tension[i] belongs to that same forward span

This matches the convention of the analysis script after reverting the mistaken
backward/forward span-type shift experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import warnings

import ezdxf
import numpy as np
import pandas as pd


# ============================================================
# DEFAULT SETTINGS
# ============================================================

HERE = Path(__file__).resolve().parent

# This default assumes you first ran your processing script.
INPUT_TABLE = HERE / "outputs" / "larisa2_2nd_submission_processed.xlsx"
OUTPUT_DXF = HERE / "outputs" / "catenaries_50_theoretical.dxf"

# Cardinal default, kg/m. Used for non-ice cases unless overridden.
BASE_WEIGHT = 1.823

# Approximate ice-case effective weight, kg/m.
ICE_WEIGHT = 2.5

# Vertical exaggeration. 10.0 means: 1 real vertical meter is drawn as 10 DXF units.
VERTICAL_EXAGGERATION = 10.0

# DXF units per real meter. Keep 1.0 if your profile drawing is in meters.
UNITS_PER_METER = 1.0

# Number of straight segments per catenary curve. Higher = smoother but heavier DXF.
N_SEGMENTS = 120

# Tick length is in final DXF drawing units, not real meters.
VERTEX_TICK_LENGTH = 10.0
SUPPORT_TICK_LENGTH = 6.0


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class CatenarySpanResult:
    """Small summary object for one plotted catenary span."""

    row_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    span: float
    tension: float
    weight: float
    a: float
    vertex_x: float
    vertex_y: float
    plot_x_start: float
    plot_x_end: float
    case: str


# ============================================================
# INPUT / COLUMN HELPERS
# ============================================================

def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel table based on the file extension."""
    path = Path(path)

    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported input file type: {path.suffix!r}")


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names just enough for this plotting script.

    This lets the function accept either your raw uppercase names or the processed
    lowercase names. It deliberately does not alter the actual calculation values.
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    rename_map = {
        "span": "span",
        "suspension_altitude": "suspension_altitude",
        "tower_number": "tower_number",
        "tower_type": "type",
        "type": "type",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"span", "suspension_altitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for DXF plotting: {sorted(missing)}")

    df["span"] = pd.to_numeric(df["span"], errors="coerce")
    df["suspension_altitude"] = pd.to_numeric(df["suspension_altitude"], errors="coerce")

    return df


def resolve_tension_column(df: pd.DataFrame, load_case: str) -> str:
    """
    Resolve a human-friendly load_case into an actual dataframe column.

    Examples
    --------
    load_case="50_theoretical" -> tensions_50_theoretical
    load_case="-10_ICE"        -> tensions_-10_ICE
    load_case="tensions_0"     -> tensions_0

    The last form is useful if you want to pass the exact column name directly.
    """
    candidates = [
        load_case,
        f"tensions_{load_case}",
        f"tension_{load_case}",
        f"tension_forward_{load_case}",
    ]

    # Column names were lowercased in standardize_column_names().
    lower_to_actual = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.lower()
        if key in lower_to_actual:
            return lower_to_actual[key]

    raise ValueError(
        "Could not find a tension column for load_case="
        f"{load_case!r}. Tried: {candidates}"
    )


def default_weight_for_case(load_case: str) -> float:
    """Use the ice effective weight for ICE cases, otherwise use base conductor weight."""
    return ICE_WEIGHT if "ice" in load_case.lower() else BASE_WEIGHT


# ============================================================
# CATENARY MATH
# ============================================================

def catenary_vertex_x(x1: float, y1: float, x2: float, y2: float, a: float) -> float:
    """
    Analytic x-coordinate of the catenary vertex / lowest point.

    Catenary form:

        y(x) = yv + a * (cosh((x - xv) / a) - 1)

    where:

        a = H / w

    The formula below is the closed-form solution for xv using the two supports.
    It avoids the iterative x0 search used in the LISP routine.
    """
    span = x2 - x1
    dy = y2 - y1

    if span <= 0.0:
        raise ValueError("x2 must be greater than x1.")

    if a <= 0.0:
        raise ValueError("Catenary parameter a=H/w must be positive.")

    denominator = 2.0 * a * math.sinh(span / (2.0 * a))
    xv = (x1 + x2) / 2.0 - a * math.asinh(dy / denominator)
    return xv


def catenary_vertex_y(x1: float, y1: float, xv: float, a: float) -> float:
    """
    Compute the vertex elevation yv after xv is known.

    We force the catenary to pass through support 1:

        y1 = yv + a * (cosh((x1 - xv)/a) - 1)
    """
    return y1 - a * (math.cosh((x1 - xv) / a) - 1.0)


def catenary_y(x: np.ndarray, xv: float, yv: float, a: float) -> np.ndarray:
    """Evaluate y(x) for an array of x coordinates."""
    return yv + a * (np.cosh((x - xv) / a) - 1.0)


def choose_plot_limits(x1: float, y1: float, x2: float, y2: float, xv: float) -> tuple[float, float, str]:
    """
    Decide which part of the catenary to draw.

    Case A: vertex lies inside the physical span
        Draw support-to-support: [x1, x2]

    Case B: vertex lies left of the physical span
        The right support is the higher one. Draw from the vertex to the right
        support: [xv, x2]. This includes the physical span and the extension to
        the mathematical low point.

    Case C: vertex lies right of the physical span
        The left support is the higher one. Draw from the left support to the
        vertex: [x1, xv].

    This implements the behavior you described, rather than the LISP's symmetric
    extension-to-yhigh option.
    """
    tol = 1e-9

    if x1 - tol <= xv <= x2 + tol:
        return x1, x2, "vertex_inside_span"

    if xv < x1:
        if y2 < y1:
            warnings.warn(
                "Vertex is left of span, but right support is not higher. "
                "This is unexpected for a normal catenary; drawing [xv, x2] anyway."
            )
        return xv, x2, "vertex_left_of_span"

    # xv > x2
    if y1 < y2:
        warnings.warn(
            "Vertex is right of span, but left support is not higher. "
            "This is unexpected for a normal catenary; drawing [x1, xv] anyway."
        )
    return x1, xv, "vertex_right_of_span"


def sample_catenary_points(
    *,
    x_start: float,
    x_end: float,
    x1: float,
    x2: float,
    xv: float,
    yv: float,
    a: float,
    n_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample catenary points while forcing important x locations to be included.

    The extra breakpoints ensure that the physical supports and the vertex are
    actual polyline vertices, not merely nearby sampled points.
    """
    lo = min(x_start, x_end)
    hi = max(x_start, x_end)

    breakpoints = [lo, hi]
    for x in (x1, x2, xv):
        if lo - 1e-9 <= x <= hi + 1e-9:
            breakpoints.append(float(x))

    breakpoints = sorted(set(round(x, 10) for x in breakpoints))

    xs_all: list[np.ndarray] = []
    for xa, xb in zip(breakpoints[:-1], breakpoints[1:]):
        length_fraction = abs(xb - xa) / max(abs(hi - lo), 1e-9)
        n_local = max(2, int(math.ceil(n_segments * length_fraction)))
        xs = np.linspace(xa, xb, n_local, endpoint=True)

        # Avoid duplicating the joining point between consecutive segments.
        if xs_all:
            xs = xs[1:]

        xs_all.append(xs)

    x_values = np.concatenate(xs_all) if xs_all else np.array([lo, hi], dtype=float)
    y_values = catenary_y(x_values, xv=xv, yv=yv, a=a)

    return x_values, y_values


# ============================================================
# DXF HELPERS
# ============================================================

def to_dxf_points(
    x_real: np.ndarray,
    y_real: np.ndarray,
    *,
    y_reference: float,
    vertical_exaggeration: float,
    units_per_meter: float,
) -> list[tuple[float, float, float]]:
    """
    Convert real profile coordinates to DXF coordinates.

    The x-coordinate is cumulative chainage in real meters.
    The y-coordinate is relative elevation, vertically exaggerated:

        y_dxf = (y_real - y_reference) * vertical_exaggeration

    This keeps the drawing near the origin instead of placing it around the
    absolute altitude, e.g. 5500 drawing units for 550 m × 10.
    """
    x_dxf = x_real * units_per_meter
    y_dxf = (y_real - y_reference) * vertical_exaggeration * units_per_meter
    return [(float(x), float(y), 0.0) for x, y in zip(x_dxf, y_dxf)]


def to_dxf_point(
    x_real: float,
    y_real: float,
    *,
    y_reference: float,
    vertical_exaggeration: float,
    units_per_meter: float,
) -> tuple[float, float, float]:
    """Scalar version of to_dxf_points()."""
    return (
        float(x_real * units_per_meter),
        float((y_real - y_reference) * vertical_exaggeration * units_per_meter),
        0.0,
    )


def add_vertical_tick(
    msp,
    *,
    x_real: float,
    y_real: float,
    y_reference: float,
    vertical_exaggeration: float,
    units_per_meter: float,
    tick_length: float,
    layer: str,
) -> None:
    """Draw a small vertical tick centered on a real profile point."""
    x_dxf, y_dxf, z_dxf = to_dxf_point(
        x_real,
        y_real,
        y_reference=y_reference,
        vertical_exaggeration=vertical_exaggeration,
        units_per_meter=units_per_meter,
    )
    half = tick_length / 2.0
    msp.add_line(
        (x_dxf, y_dxf - half, z_dxf),
        (x_dxf, y_dxf + half, z_dxf),
        dxfattribs={"layer": layer},
    )


def ensure_layers(doc: ezdxf.EzDxfDocument, layer_names: list[str]) -> None:
    """Create layers if they do not already exist."""
    for layer in layer_names:
        if layer not in doc.layers:
            doc.layers.add(layer)


# ============================================================
# MAIN PLOTTING FUNCTION
# ============================================================

def plot_catenaries_to_dxf(
    df: pd.DataFrame,
    *,
    output_path: str | Path,
    load_case: str = "50_theoretical",
    weight: float | None = None,
    vertical_exaggeration: float = VERTICAL_EXAGGERATION,
    units_per_meter: float = UNITS_PER_METER,
    n_segments: int = N_SEGMENTS,
    vertex_tick_length: float = VERTEX_TICK_LENGTH,
    support_tick_length: float = SUPPORT_TICK_LENGTH,
    draw_vertex_ticks: bool = True,
    draw_support_ticks: bool = False,
    y_reference: float | None = None,
    catenary_layer: str = "CATENARIES",
    #vertex_layer: str = "CATENARY_VERTEX_TICKS",
    support_layer: str = "SUPPORT_TICKS",
) -> pd.DataFrame:
    """
    Draw all available forward-span catenaries to a DXF file.

    Parameters
    ----------
    df:
        Processed dataframe containing spans, suspension altitudes, and tension columns.

    output_path:
        DXF file to create.

    load_case:
        Either a load-case suffix such as "50_theoretical" or the exact tension
        column name such as "tensions_50_theoretical".

    weight:
        Effective conductor weight in kg/m. If omitted, the script uses ICE_WEIGHT
        for load cases containing "ICE" and BASE_WEIGHT otherwise.

    y_reference:
        Real elevation used as the vertical drawing origin. If omitted, the first
        valid suspension altitude is used.

    Returns
    -------
    pandas.DataFrame
        A span-by-span plotting summary, useful for checking which spans had
        vertices inside/outside their physical span.
    """
    df = standardize_column_names(df)
    tension_col = resolve_tension_column(df, load_case)
    weight = default_weight_for_case(load_case) if weight is None else float(weight)

    if weight <= 0.0:
        raise ValueError("weight must be positive.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # x coordinate of tower i = sum of all previous forward spans.
    # row i itself then spans from x_tower[i] to x_tower[i] + span[i].
    spans = df["span"].to_numpy(dtype=float)
    x_tower = np.zeros(len(df), dtype=float)
    if len(df) > 1:
        x_tower[1:] = np.cumsum(spans[:-1])

    alt = df["suspension_altitude"].to_numpy(dtype=float)
    tension = pd.to_numeric(df[tension_col], errors="coerce").to_numpy(dtype=float)

    if y_reference is None:
        valid_altitudes = df["suspension_altitude"].dropna()
        if valid_altitudes.empty:
            raise ValueError("No valid suspension_altitude values found.")
        y_reference = float(valid_altitudes.iloc[0])

    doc = ezdxf.new("R2010")
    #ensure_layers(doc, [catenary_layer, vertex_layer, support_layer])
    ensure_layers(doc, [catenary_layer, support_layer])
    msp = doc.modelspace()

    results: list[CatenarySpanResult] = []

    # Draw support ticks for the listed towers. This is optional but useful when
    # visually checking that the catenaries are chained to the intended supports.
    if draw_support_ticks:
        for i, (x_i, y_i) in enumerate(zip(x_tower, alt)):
            if not np.isfinite(x_i) or not np.isfinite(y_i):
                continue
            add_vertical_tick(
                msp,
                x_real=float(x_i),
                y_real=float(y_i),
                y_reference=float(y_reference),
                vertical_exaggeration=vertical_exaggeration,
                units_per_meter=units_per_meter,
                tick_length=support_tick_length,
                layer=support_layer,
            )

    for i in range(len(df) - 1):
        S = spans[i]
        H = tension[i]
        y1 = alt[i]
        y2 = alt[i + 1]
        x1 = x_tower[i]
        x2 = x1 + S

        if not all(np.isfinite(v) for v in (S, H, y1, y2, x1, x2)):
            warnings.warn(f"Skipping row {i}: span/tension/altitude contains NaN or inf.")
            continue

        if S <= 0.0:
            warnings.warn(f"Skipping row {i}: non-positive span {S!r}.")
            continue

        if H <= 0.0:
            warnings.warn(f"Skipping row {i}: non-positive tension {H!r}.")
            continue

        # Physical catenary parameter. The rest of the calculation assumes the
        # same kgf / kg-per-meter convention as your existing LISP and Python code.
        a = H / weight

        try:
            xv = catenary_vertex_x(float(x1), float(y1), float(x2), float(y2), a)
            yv = catenary_vertex_y(float(x1), float(y1), xv, a)
        except Exception as exc:
            warnings.warn(f"Skipping row {i}: failed to compute catenary vertex: {exc}")
            continue

        x_start, x_end, case = choose_plot_limits(float(x1), float(y1), float(x2), float(y2), xv)

        x_values, y_values = sample_catenary_points(
            x_start=x_start,
            x_end=x_end,
            x1=float(x1),
            x2=float(x2),
            xv=xv,
            yv=yv,
            a=a,
            n_segments=n_segments,
        )

        dxf_points = to_dxf_points(
            x_values,
            y_values,
            y_reference=float(y_reference),
            vertical_exaggeration=vertical_exaggeration,
            units_per_meter=units_per_meter,
        )

        # 3D polyline mirrors the AutoLISP ENTMAKE behavior closely, while still
        # keeping all z coordinates at zero for a 2D profile view.
        msp.add_polyline3d(dxf_points, dxfattribs={"layer": catenary_layer})

        if draw_vertex_ticks:
            add_vertical_tick(
                msp,
                x_real=xv,
                y_real=yv,
                y_reference=float(y_reference),
                vertical_exaggeration=vertical_exaggeration,
                units_per_meter=units_per_meter,
                tick_length=vertex_tick_length,
                layer=catenary_layer,
            )

        results.append(
            CatenarySpanResult(
                row_index=i,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                span=float(S),
                tension=float(H),
                weight=float(weight),
                a=float(a),
                vertex_x=float(xv),
                vertex_y=float(yv),
                plot_x_start=float(x_start),
                plot_x_end=float(x_end),
                case=case,
            )
        )

    doc.saveas(output_path)
    return pd.DataFrame([r.__dict__ for r in results])


def plot_catenaries_from_file(
    input_path: str | Path,
    *,
    output_path: str | Path,
    load_case: str = "50_theoretical",
    weight: float | None = None,
    vertical_exaggeration: float = VERTICAL_EXAGGERATION,
) -> pd.DataFrame:
    """Convenience wrapper: read table -> plot DXF -> return plotting summary."""
    df = read_table(input_path)
    return plot_catenaries_to_dxf(
        df,
        output_path=output_path,
        load_case=load_case,
        weight=weight,
        vertical_exaggeration=vertical_exaggeration,
    )

def plot_catenary_cases_to_one_dxf(
    df: pd.DataFrame,
    *,
    output_path: str | Path,
    cases: list[dict],
    vertical_exaggeration: float = VERTICAL_EXAGGERATION,
    units_per_meter: float = UNITS_PER_METER,
    n_segments: int = N_SEGMENTS,
    vertex_tick_length: float = VERTEX_TICK_LENGTH,
    support_tick_length: float = SUPPORT_TICK_LENGTH,
    draw_vertex_ticks: bool = True,
    draw_support_ticks: bool = False,
    y_reference: float | None = None,
    support_layer: str = "SUPPORT_TICKS",
) -> pd.DataFrame:
    """
    Draw multiple catenary load cases into one DXF file.

    Each case is drawn in its own layer:

        CATENARIES_<case_name>

    Expected cases format:
        cases = [
            {"name": "0", "weight": BASE_WEIGHT},
            {"name": "0_ICE", "weight": ICE_WEIGHT},
            {"name": "-10_bare", "weight": BASE_WEIGHT},
            ...
        ]

    Vertex tick marks are drawn in the same layer as their catenary.
    """

    df = standardize_column_names(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spans = df["span"].to_numpy(dtype=float)

    x_tower = np.zeros(len(df), dtype=float)
    if len(df) > 1:
        x_tower[1:] = np.cumsum(spans[:-1])

    alt = df["suspension_altitude"].to_numpy(dtype=float)

    if y_reference is None:
        valid_altitudes = df["suspension_altitude"].dropna()
        if valid_altitudes.empty:
            raise ValueError("No valid suspension_altitude values found.")
        y_reference = float(valid_altitudes.iloc[0])

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    ensure_layers(doc, [support_layer])

    results = []

    # Draw support ticks only once, because supports do not change by load case.
    if draw_support_ticks:
        for x_i, y_i in zip(x_tower, alt):
            if not np.isfinite(x_i) or not np.isfinite(y_i):
                continue

            add_vertical_tick(
                msp,
                x_real=float(x_i),
                y_real=float(y_i),
                y_reference=float(y_reference),
                vertical_exaggeration=vertical_exaggeration,
                units_per_meter=units_per_meter,
                tick_length=support_tick_length,
                layer=support_layer,
            )

    for case_data in cases:
        load_case = case_data["name"]
        weight = float(case_data["weight"])

        if weight <= 0.0:
            raise ValueError(f"Weight must be positive for load case {load_case!r}.")

        tension_col = resolve_tension_column(df, load_case)
        tension = pd.to_numeric(df[tension_col], errors="coerce").to_numpy(dtype=float)

        catenary_layer = f"CATENARIES_{load_case}"
        ensure_layers(doc, [catenary_layer])

        for i in range(len(df) - 1):
            S = spans[i]
            H = tension[i]
            y1 = alt[i]
            y2 = alt[i + 1]
            x1 = x_tower[i]
            x2 = x1 + S

            if not all(np.isfinite(v) for v in (S, H, y1, y2, x1, x2)):
                warnings.warn(
                    f"Skipping row {i}, case {load_case!r}: "
                    "span/tension/altitude contains NaN or inf."
                )
                continue

            if S <= 0.0:
                warnings.warn(f"Skipping row {i}, case {load_case!r}: non-positive span {S!r}.")
                continue

            if H <= 0.0:
                warnings.warn(f"Skipping row {i}, case {load_case!r}: non-positive tension {H!r}.")
                continue

            a = H / weight

            try:
                xv = catenary_vertex_x(float(x1), float(y1), float(x2), float(y2), a)
                yv = catenary_vertex_y(float(x1), float(y1), xv, a)
            except Exception as exc:
                warnings.warn(
                    f"Skipping row {i}, case {load_case!r}: "
                    f"failed to compute catenary vertex: {exc}"
                )
                continue

            x_start, x_end, vertex_case = choose_plot_limits(
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                xv,
            )

            x_values, y_values = sample_catenary_points(
                x_start=x_start,
                x_end=x_end,
                x1=float(x1),
                x2=float(x2),
                xv=xv,
                yv=yv,
                a=a,
                n_segments=n_segments,
            )

            dxf_points = to_dxf_points(
                x_values,
                y_values,
                y_reference=float(y_reference),
                vertical_exaggeration=vertical_exaggeration,
                units_per_meter=units_per_meter,
            )

            msp.add_polyline3d(
                dxf_points,
                dxfattribs={"layer": catenary_layer},
            )

            # Vertex tick goes in the same layer as the catenary.
            if draw_vertex_ticks:
                add_vertical_tick(
                    msp,
                    x_real=xv,
                    y_real=yv,
                    y_reference=float(y_reference),
                    vertical_exaggeration=vertical_exaggeration,
                    units_per_meter=units_per_meter,
                    tick_length=vertex_tick_length,
                    layer=catenary_layer,
                )

            results.append(
                CatenarySpanResult(
                    row_index=i,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    span=float(S),
                    tension=float(H),
                    weight=float(weight),
                    a=float(a),
                    vertex_x=float(xv),
                    vertex_y=float(yv),
                    plot_x_start=float(x_start),
                    plot_x_end=float(x_end),
                    case=vertex_case,
                ).__dict__ | {"load_case": load_case, "layer": catenary_layer}
            )

    doc.saveas(output_path)

    return pd.DataFrame(results)

def plot_catenary_cases_from_file_to_one_dxf(
    input_path: str | Path,
    *,
    output_path: str | Path,
    cases: list[dict],
    vertical_exaggeration: float = VERTICAL_EXAGGERATION,
) -> pd.DataFrame:
    """
    Convenience wrapper:
        read processed table -> plot all catenary cases into one DXF.
    """
    df = read_table(input_path)

    return plot_catenary_cases_to_one_dxf(
        df,
        output_path=output_path,
        cases=cases,
        vertical_exaggeration=vertical_exaggeration,
    )

# ============================================================
# EXAMPLE DIRECT RUN
# ============================================================

if __name__ == "__main__":
    summary = plot_catenaries_from_file(
        INPUT_TABLE,
        output_path=OUTPUT_DXF,
        load_case="50_theoretical",
        vertical_exaggeration=VERTICAL_EXAGGERATION,
    )

    print(f"Wrote: {OUTPUT_DXF}")
    print()
    print("Vertex-location cases:")
    print(summary["case"].value_counts(dropna=False).to_string())

    # Optional: uncomment if you want a CSV diagnostic table next to the DXF.
    # summary.to_csv(OUTPUT_DXF.with_suffix(".summary.csv"), index=False)
