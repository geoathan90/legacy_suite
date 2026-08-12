"""DXF export for a solved 2D catenary span with insulators.

Place this file inside the ``insulators`` package and import it with::

    from insulators.dxf_export import export_span_to_dxf

Coordinates are written to modelspace as metres. Curves are represented by
lightweight polylines so the file opens consistently in AutoCAD and similar
CAD software.
"""

from pathlib import Path

import ezdxf

from .catenary import (
    catenary,
    catenary_low_point,
    catenary_points,
    catenary_y,
)
from .line import line_between_points
from .sag import catenary_sag_vertical
from .th_solver import Th_for_target_sag


def _point(value, name):
    """Return a two-float tuple and give a useful error for invalid points."""
    try:
        x, y = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly two coordinates.") from exc
    return float(x), float(y)


def _polyline_points(cat, n):
    x_values, y_values = catenary_points(cat, n=n)
    return [(float(x), float(y)) for x, y in zip(x_values, y_values)]


def _add_marker(
    modelspace,
    point,
    radius,
    layer,
    name,
    text_height,
    *,
    annotate,
    offset=None,
):
    """Add a circular point marker and, optionally, its short point ID."""
    x, y = point
    modelspace.add_circle((x, y), radius, dxfattribs={"layer": layer})
    if annotate:
        if offset is None:
            offset = (1.5 * radius, 1.5 * radius)
        modelspace.add_text(
            name,
            dxfattribs={
                "insert": (x + offset[0], y + offset[1]),
                "height": text_height,
                "layer": "ANNOTATIONS",
            },
        )


def _add_point_legend(
    modelspace,
    entries,
    *,
    left,
    top,
    span,
    text_height,
    precision,
):
    """Add an ID/name/coordinate schedule below the exported geometry."""
    row_height = 1.7 * text_height
    columns = (left, left + 0.10 * span, left + 0.52 * span, left + 0.76 * span)
    headers = ("ID", "POINT", "X", "Y")

    for x, value in zip(columns, headers):
        modelspace.add_text(
            value,
            dxfattribs={
                "insert": (x, top),
                "height": text_height,
                "layer": "POINT_LEGEND",
            },
        )

    modelspace.add_line(
        (left, top - 0.35 * text_height),
        (left + span, top - 0.35 * text_height),
        dxfattribs={"layer": "POINT_LEGEND"},
    )

    for row, (point_id, point_name, point) in enumerate(entries, start=1):
        y = top - row * row_height
        x_coord, y_coord = point
        values = (
            point_id,
            point_name,
            f"{x_coord:.{precision}f}",
            f"{y_coord:.{precision}f}",
        )
        for x, value in zip(columns, values):
            modelspace.add_text(
                value,
                dxfattribs={
                    "insert": (x, y),
                    "height": text_height,
                    "layer": "POINT_LEGEND",
                },
            )


def _low_point(cat):
    """Return the catenary vertex as a two-float tuple."""
    x = float(catenary_low_point(cat))
    return x, float(catenary_y(cat, x))


def _sag_points(sag_info):
    """Return the conductor and chord endpoints of a vertical sag segment."""
    x = float(sag_info["x_sag"])
    curve_point = (x, float(sag_info["y_curve"]))
    chord_point = (x, float(sag_info["y_line"]))
    return curve_point, chord_point


def export_span_to_dxf(
    result,
    A,
    B,
    output_path="insulator_span.dxf",
    *,
    n=1000,
    show_supports=True,
    show_attachments=True,
    show_chord=True,
    show_sag=False,
    show_low_point=True,
    show_idealized=False,
    show_idealized_sag=True,
    w=None,
    idealized_sag=None,
    annotate_characteristic_points=True,
    include_point_coordinates=True,
    coordinate_precision=2,
    marker_radius=None,
    text_height=None,
    show_point_legend=True,
    legend_text_height=None,
):
    """Export the solved span geometry to a layered DXF file.

    Parameters
    ----------
    result : dict
        Dictionary returned by ``solve_span_for_target_sag``. It must contain
        ``C``, ``D`` and ``catenary``.
    A, B : tuple[float, float]
        Main left and right support points.
    output_path : str or pathlib.Path
        Destination filename. A ``.dxf`` suffix is added when necessary.
    n : int
        Number of vertices used to approximate each catenary.
    show_supports, show_attachments, show_chord, show_sag : bool
        Toggle the corresponding CAD geometry.
    show_low_point : bool
        Mark the vertex of the mathematical actual catenary. For a steep span,
        this point can lie beyond the finite conductor segment C-D.
    show_idealized : bool
        Export the no-insulator catenary between A and B whose A-B sag matches
        the actual catenary (or ``idealized_sag`` when supplied).
    show_idealized_sag : bool
        Add the idealized curve's vertical sag segment. Its endpoints are
        marked when characteristic-point annotations are enabled. Has an
        effect only when ``show_idealized=True``.
    w : float or None
        Conductor weight per unit length. Required for ``show_idealized=True``.
    idealized_sag : float or None
        Optional explicit target sag for the idealized catenary.
    annotate_characteristic_points : bool
        Label A, B, C, D, the catenary vertex, and both endpoints of each
        calculated sag segment. Idealized points are included when enabled.
    include_point_coordinates : bool
        Retained for backward compatibility. Coordinates are now shown in the
        point legend rather than beside the drawing.
    coordinate_precision : int
        Decimal places used in point-coordinate labels.
    marker_radius, text_height : float or None
        CAD marker and label sizes. Defaults scale with the horizontal span.
    show_point_legend : bool
        Add a point-name and coordinate schedule below the drawing.
    legend_text_height : float or None
        Text height for the schedule. Defaults to ``text_height``.

    Returns
    -------
    pathlib.Path
        The path of the saved DXF file.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    if (
        not isinstance(coordinate_precision, int)
        or not 0 <= coordinate_precision <= 12
    ):
        raise ValueError("coordinate_precision must be an integer from 0 to 12.")

    try:
        cat = result["catenary"]
        C = _point(result["C"], "result['C']")
        D = _point(result["D"], "result['D']")
    except KeyError as exc:
        raise KeyError(
            "result must contain 'C', 'D', and 'catenary' from "
            "solve_span_for_target_sag."
        ) from exc

    A = _point(A, "A")
    B = _point(B, "B")

    span = abs(B[0] - A[0])
    if marker_radius is None:
        marker_radius = max(0.003 * span, 0.05)
    if text_height is None:
        text_height = max(0.012 * span, 0.20)
    if legend_text_height is None:
        legend_text_height = text_height
    if marker_radius <= 0 or text_height <= 0 or legend_text_height <= 0:
        raise ValueError(
            "marker_radius, text_height, and legend_text_height must be positive."
        )

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".dxf":
        output_path = output_path.with_suffix(".dxf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 6  # metres
    modelspace = doc.modelspace()

    layer_specs = {
        "CONDUCTOR": (1, "CONTINUOUS"),
        "INSULATORS": (5, "CONTINUOUS"),
        "SUPPORTS": (2, "CONTINUOUS"),
        "ATTACHMENTS": (4, "CONTINUOUS"),
        "CHORD_AB": (8, "DASHED"),
        "SAG": (6, "DASHED"),
        "CHARACTERISTIC_POINTS": (30, "CONTINUOUS"),
        "IDEALIZED": (3, "DASHED"),
        "IDEALIZED_POINTS": (3, "CONTINUOUS"),
        "ANNOTATIONS": (7, "CONTINUOUS"),
        "POINT_LEGEND": (7, "CONTINUOUS"),
    }
    for name, (color, linetype) in layer_specs.items():
        doc.layers.add(name=name, color=color, linetype=linetype)

    actual_curve_points = _polyline_points(cat, n)
    all_geometry_points = [A, B, C, D, *actual_curve_points]
    legend_entries = []
    modelspace.add_lwpolyline(
        actual_curve_points, dxfattribs={"layer": "CONDUCTOR"}
    )
    modelspace.add_line(A, C, dxfattribs={"layer": "INSULATORS"})
    modelspace.add_line(B, D, dxfattribs={"layer": "INSULATORS"})

    if show_supports:
        _add_marker(
            modelspace,
            A,
            marker_radius,
            "SUPPORTS",
            "A",
            text_height,
            annotate=annotate_characteristic_points,
        )
        legend_entries.append(("A", "Left support", A))
        _add_marker(
            modelspace,
            B,
            marker_radius,
            "SUPPORTS",
            "B",
            text_height,
            annotate=annotate_characteristic_points,
        )
        legend_entries.append(("B", "Right support", B))

    if show_attachments:
        _add_marker(
            modelspace,
            C,
            marker_radius,
            "ATTACHMENTS",
            "C",
            text_height,
            annotate=annotate_characteristic_points,
            offset=(0.5 * text_height, -1.5 * text_height),
        )
        legend_entries.append(("C", "Left attachment", C))
        _add_marker(
            modelspace,
            D,
            marker_radius,
            "ATTACHMENTS",
            "D",
            text_height,
            annotate=annotate_characteristic_points,
            offset=(0.5 * text_height, -1.5 * text_height),
        )
        legend_entries.append(("D", "Right attachment", D))

    line_AB = line_between_points(A, B)
    if show_chord:
        modelspace.add_line(A, B, dxfattribs={"layer": "CHORD_AB"})

    sag_info_actual = None
    if show_sag or show_idealized or annotate_characteristic_points:
        sag_info_actual = catenary_sag_vertical(cat, line_AB)

    if show_low_point:
        _add_marker(
            modelspace,
            _low_point(cat),
            marker_radius,
            "CHARACTERISTIC_POINTS",
            "V",
            text_height,
            annotate=annotate_characteristic_points,
        )
        actual_vertex = _low_point(cat)
        legend_entries.append(("V", "Actual catenary vertex", actual_vertex))
        all_geometry_points.append(actual_vertex)

    actual_sag_curve = None
    actual_sag_chord = None
    if sag_info_actual is not None:
        actual_sag_curve, actual_sag_chord = _sag_points(sag_info_actual)

    if show_sag:
        modelspace.add_line(
            actual_sag_curve,
            actual_sag_chord,
            dxfattribs={"layer": "SAG"},
        )

    if annotate_characteristic_points:
        _add_marker(
            modelspace,
            actual_sag_curve,
            marker_radius,
            "CHARACTERISTIC_POINTS",
            "S",
            text_height,
            annotate=True,
            offset=(0.5 * text_height, -1.5 * text_height),
        )
        _add_marker(
            modelspace,
            actual_sag_chord,
            marker_radius,
            "CHARACTERISTIC_POINTS",
            "P",
            text_height,
            annotate=True,
            offset=(0.5 * text_height, 0.5 * text_height),
        )
        legend_entries.extend(
            [
                ("S", "Actual maximum-sag point", actual_sag_curve),
                ("P", "A-B chord point above S", actual_sag_chord),
            ]
        )
        all_geometry_points.extend((actual_sag_curve, actual_sag_chord))

    if show_idealized:
        if w is None or w <= 0:
            raise ValueError("A positive w is required when show_idealized=True.")
        if idealized_sag is None:
            idealized_sag = float(sag_info_actual["sag_max"])
        Th_ideal = Th_for_target_sag(A, B, w, idealized_sag)
        cat_ideal = catenary(A, B, w, Th_ideal)
        ideal_curve_points = _polyline_points(cat_ideal, n)
        all_geometry_points.extend(ideal_curve_points)
        modelspace.add_lwpolyline(
            ideal_curve_points,
            dxfattribs={"layer": "IDEALIZED"},
        )

        _add_marker(
            modelspace,
            _low_point(cat_ideal),
            marker_radius,
            "IDEALIZED_POINTS",
            "V_i",
            text_height,
            annotate=annotate_characteristic_points,
            offset=(0.5 * text_height, -1.5 * text_height),
        )
        ideal_vertex = _low_point(cat_ideal)
        if annotate_characteristic_points:
            legend_entries.append(("V_i", "Idealized catenary vertex", ideal_vertex))
        all_geometry_points.append(ideal_vertex)

        if show_idealized_sag or annotate_characteristic_points:
            sag_info_ideal = catenary_sag_vertical(cat_ideal, line_AB)
            ideal_sag_curve, ideal_sag_chord = _sag_points(sag_info_ideal)

            if show_idealized_sag:
                modelspace.add_line(
                    ideal_sag_curve,
                    ideal_sag_chord,
                    dxfattribs={"layer": "IDEALIZED"},
                )

            if annotate_characteristic_points:
                _add_marker(
                    modelspace,
                    ideal_sag_curve,
                    marker_radius,
                    "IDEALIZED_POINTS",
                    "S_i",
                    text_height,
                    annotate=True,
                    offset=(0.5 * text_height, -3.0 * text_height),
                )
                _add_marker(
                    modelspace,
                    ideal_sag_chord,
                    marker_radius,
                    "IDEALIZED_POINTS",
                    "P_i",
                    text_height,
                    annotate=True,
                    offset=(0.5 * text_height, 2.0 * text_height),
                )
                legend_entries.extend(
                    [
                        ("S_i", "Idealized maximum-sag point", ideal_sag_curve),
                        ("P_i", "A-B chord point above S_i", ideal_sag_chord),
                    ]
                )
                all_geometry_points.extend((ideal_sag_curve, ideal_sag_chord))

    if annotate_characteristic_points and show_point_legend and legend_entries:
        lowest_y = min(point[1] for point in all_geometry_points)
        legend_top = lowest_y - 20.0 * legend_text_height
        _add_point_legend(
            modelspace,
            legend_entries,
            left=min(A[0], B[0]),
            top=legend_top,
            span=max(span, 20.0 * legend_text_height),
            text_height=legend_text_height,
            precision=coordinate_precision,
        )

    doc.saveas(output_path)
    return output_path
