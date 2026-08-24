from ast import literal_eval
from math import asinh, cosh, hypot, sinh, sqrt
from pathlib import Path

from scripts.eval import evaluate, dx_calculation

HERE = Path(__file__).resolve().parent

def read_lines_input(path):
    with open(path, "r", encoding="utf-8") as f:
        return literal_eval(f.read())


def horizontal_span(line):
    (x1, y1), (x2, y2) = line
    return hypot(x2 - x1, y2 - y1)


def catenary_state(A, B, w, H):
    x1, y1 = A
    x2, y2 = B

    if not (w > 0.0 and H > 0.0 and x1 < x2):
        raise ValueError("Invalid catenary inputs.")

    alpha = H / w
    xm = 0.5 * (x1 + x2)
    half_span = 0.5 * (x2 - x1)
    dy = y2 - y1

    temp = dy / (2.0 * alpha * sinh(half_span / alpha))
    x0 = xm - alpha * asinh(temp)
    y0 = y1 - alpha * cosh((x1 - x0) / alpha)

    return {
        "alpha": alpha,
        "x0": x0,
        "y0": y0,
    }


def catenary_y(cat, x):
    alpha = cat["alpha"]
    x0 = cat["x0"]
    y0 = cat["y0"]
    return alpha * cosh((x - x0) / alpha) + y0


def line_y(A, B, x):
    x1, y1 = A
    x2, y2 = B
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b


def max_vertical_sag(cat, A, B):
    x1, _ = A
    x2, _ = B

    alpha = cat["alpha"]
    x0 = cat["x0"]

    m = (B[1] - A[1]) / (B[0] - A[0])
    x_crit = x0 + alpha * asinh(m)

    candidates = [x1, x2]
    if x1 <= x_crit <= x2:
        candidates.append(x_crit)

    best_sag = None

    for x in candidates:
        sag = line_y(A, B, x) - catenary_y(cat, x)
        if best_sag is None or sag > best_sag:
            best_sag = sag

    return best_sag


def solve_H_for_target_sag(A, B, w, target_sag, tol=1e-10, max_iter=200):
    x1, _ = A
    x2, _ = B
    span = x2 - x1

    if not (w > 0.0 and target_sag > 0.0 and x2 > x1):
        raise ValueError("Invalid sag-solver inputs.")

    H_guess = w * span * span / (8.0 * target_sag)
    H_low = max(1e-9, 0.25 * H_guess)
    H_high = 4.0 * H_guess

    sag_low = max_vertical_sag(catenary_state(A, B, w, H_low), A, B)
    sag_high = max_vertical_sag(catenary_state(A, B, w, H_high), A, B)

    expand_count = 0
    while not (sag_low >= target_sag and sag_high <= target_sag):
        if expand_count > 100:
            raise RuntimeError("Could not bracket the solution for H.")

        if sag_low < target_sag:
            H_low *= 0.5
            sag_low = max_vertical_sag(catenary_state(A, B, w, H_low), A, B)

        if sag_high > target_sag:
            H_high *= 2.0
            sag_high = max_vertical_sag(catenary_state(A, B, w, H_high), A, B)

        expand_count += 1

    for _ in range(max_iter):
        H_mid = 0.5 * (H_low + H_high)
        sag_mid = max_vertical_sag(catenary_state(A, B, w, H_mid), A, B)

        err = sag_mid - target_sag
        if abs(err) < tol:
            break

        if sag_mid > target_sag:
            H_low = H_mid
        else:
            H_high = H_mid

    H_mid = 0.5 * (H_low + H_high)
    return H_mid, catenary_state(A, B, w, H_mid)


def polyline_points_from_line(
    line,
    target_sag,
    w,
    n,
    z_start,
    z_end,
):
    (x1, y1), (x2, y2) = line

    span = hypot(x2 - x1, y2 - y1)
    if span <= 0.0:
        raise ValueError("Input line has zero length.")

    ex = ((x2 - x1) / span, (y2 - y1) / span)

    A = (0.0, z_start)
    B = (span, z_end)

    H, cat = solve_H_for_target_sag(A, B, w, target_sag)

    points = []
    for i in range(n + 1):
        s = span * i / n
        z = catenary_y(cat, s)

        x = x1 + ex[0] * s
        y = y1 + ex[1] * s

        points.append((x, y, z))

    return points, H


def points_to_autocad_text(points, decimals=8):
    return "\n".join(
        f"{x:.{decimals}f},{y:.{decimals}f},{z:.{decimals}f}"
        for x, y, z in points
    )


def write_points_file(path, points, decimals=8):
    with open(path, "w", encoding="utf-8") as f:
        f.write(points_to_autocad_text(points, decimals=decimals))
        f.write("\n")

# with open('output.txt', 'w') as f:
#     for x, y in points:
#         f.write(f"{x:.6f},{y:.6f}\n")

def distance_3d(p, q):
    return sqrt(
        (q[0] - p[0]) ** 2 +
        (q[1] - p[1]) ** 2 +
        (q[2] - p[2]) ** 2
    )


def min_vertex_distance(points1, points2):
    best_p1 = None
    best_p2 = None
    best_d = None

    for p1 in points1:
        for p2 in points2:
            d = distance_3d(p1, p2)
            if best_d is None or d < best_d:
                best_d = d
                best_p1 = p1
                best_p2 = p2

    return best_p1, best_p2, best_d


def build_case(
    input_path,
    diagram_names,
    temperature,
    z_starts,
    z_ends,
    w1,
    w2,
    n1,
    n2,
    sag_overrides=None,
):
    input_path = Path(input_path)
    lines = read_lines_input(input_path)

    temperature = str(temperature) # dx_calculation returns strings as left-side headers

    if len(lines) != 2:
        raise ValueError("Input file must contain exactly 2 lines.")

    if len(diagram_names) != 2:
        raise ValueError("diagram_names must contain exactly 2 entries.")

    if sag_overrides is None:
        sag_overrides = [None, None]

    if len(sag_overrides) != 2:
        raise ValueError("sag_overrides must contain exactly 2 entries.")

    line1 = lines[0]
    line2 = lines[1]

    span1 = horizontal_span(line1)
    span2 = horizontal_span(line2)

    height_diffs = [
        abs(z_ends[0] - z_starts[0]),
        abs(z_ends[1] - z_starts[1]),
    ]

    # Line 1 sag
    if sag_overrides[0] is None:
        df1_line1 = evaluate(diagram_names[0], [span1])
        df2_line1 = dx_calculation(df1_line1, [height_diffs[0]])

        if temperature not in df2_line1.index:
            raise ValueError(f"Temperature {temperature} not found in first dataframe index")

        sag1 = float(df2_line1.loc[temperature].iloc[0])
    else:
        df1_line1 = None
        df2_line1 = None
        sag1 = float(sag_overrides[0])

    # Line 2 sag
    if sag_overrides[1] is None:
        df1_line2 = evaluate(diagram_names[1], [span2])
        df2_line2 = dx_calculation(df1_line2, [height_diffs[1]])

        if temperature not in df2_line2.index:
            raise ValueError(f"Temperature {temperature} not found in second dataframe index")

        sag2 = float(df2_line2.loc[temperature].iloc[0])
    else:
        df1_line2 = None
        df2_line2 = None
        sag2 = float(sag_overrides[1])

    points1, H1 = polyline_points_from_line(
        line1,
        target_sag=sag1,
        w=w1,
        n=n1,
        z_start=z_starts[0],
        z_end=z_ends[0],
    )

    points2, H2 = polyline_points_from_line(
        line2,
        target_sag=sag2,
        w=w2,
        n=n2,
        z_start=z_starts[1],
        z_end=z_ends[1],
    )

    out1 = input_path.with_name("line1.txt")
    out2 = input_path.with_name("line2.txt")

    write_points_file(out1, points1)
    write_points_file(out2, points2)

    p1_min, p2_min, dmin = min_vertex_distance(points1, points2)

    return {
        "span1": span1,
        "span2": span2,
        "height_diff1": height_diffs[0],
        "height_diff2": height_diffs[1],
        "df1_line1": df1_line1,
        "df2_line1": df2_line1,
        "df1_line2": df1_line2,
        "df2_line2": df2_line2,
        "sag1": sag1,
        "sag2": sag2,
        "H1": H1,
        "H2": H2,
        "line1_file": out1,
        "line2_file": out2,
        "min_distance": dmin,
        "closest_point_line1": p1_min,
        "closest_point_line2": p2_min,
    }


def main():

    ## TO-DO: add docstring
    """
    Run one two-line clearance study.

    Expected workflow
    -----------------
    1. Export exactly two 2D lines from AutoCAD to `lines_input.txt`.
        (tool to be used: LINES2PY.LSP)
    2. Place `lines_input.txt` in the same directory as this script.
    3. Edit the study inputs below:
       - conductor weight `w`
       - polyline discretization `N`
       - support heights
       - sag-span diagram names
       - temperature
       - optional `sag_overrides`
    4. Run from the project root with:

           python -m test_env.clearance_from_lines

    What the script does
    --------------------
    - Reads the two input lines.
    - Computes each horizontal span.
    - Obtains target sag for each line either:
      a) from `evaluate()` + `dx_calculation()`, or
      b) from `sag_overrides`.
    - Generates two 3D catenary polylines.
    - Writes AutoCAD-ready point files:
      `line1.txt` and `line2.txt`.
    - Computes the minimum vertex-to-vertex 3D distance between them.

    Notes
    -----
    - `sag_overrides = [None, None]` means both lines use chart-based sag.
    - Example manual override:
          sag_overrides = [2.33, 1.84]
    - Height differences are computed internally as `abs(z_end - z_start)`.
    - Clearance is based on sampled vertices, so larger `N` gives a better estimate.
    """

    w = 1.2436
    N = 500

    gantry_height = 19
    gantry_elevation = 0
    gantry = gantry_height + gantry_elevation 
    
    low_bridge = 19 + 2.91
    mid_bridge = low_bridge + 3.9
    high_bridge = mid_bridge + 3.9
    
    tower_elevation = 0

    adjustment = 0.0     

    low = low_bridge + tower_elevation + adjustment
    mid = mid_bridge + tower_elevation + adjustment
    high = high_bridge + tower_elevation + adjustment

    sag_overrides=[None, None]
    #sag_overrides=[2.69, 2.22]
    sag_overrides=[1.85, 1.78]

    result = build_case(
        input_path= HERE/"lines_input.txt",
        diagram_names=["3289", "3136"],  #31858 = 700, 31189 = 1000
        temperature=30,
        z_starts=[high, mid],
        z_ends=[gantry, gantry],
        w1=w,
        w2=w,
        n1=N,
        n2=N,
        sag_overrides= sag_overrides,
    )

    print(f"span1 = {result['span1']:.6f}")
    print(f"span2 = {result['span2']:.6f}")
    print(f"height_diff1 = {result['height_diff1']:.6f}")
    print(f"height_diff2 = {result['height_diff2']:.6f}")
    print(f"sag1 = {result['sag1']:.6f}")
    print(f"sag2 = {result['sag2']:.6f}")
    print(f"H1 = {result['H1']:.6f}")
    print(f"H2 = {result['H2']:.6f}")
    print(f"line1 written to: {result['line1_file']}")
    print(f"line2 written to: {result['line2_file']}")
    print(f"minimum vertex distance = {result['min_distance']:.6f}")


if __name__ == "__main__":
    main()