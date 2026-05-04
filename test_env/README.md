# clearance_from_lines.py

HOW TO RUN:  python -m test_env.clearance_from_lines

Small utility for generating two 3D conductor polylines from two 2D AutoCAD guide lines, then estimating the minimum clearance between them.

## What this script does

1. Reads `lines_input.txt`, which contains exactly two 2D lines exported from AutoCAD.
2. Computes the horizontal span of each line.
3. For each line, gets a target sag either:
   - from the sag-span charts through `evaluate()` and `dx_calculation()`, or
   - from a manual override.
4. Generates a 3D catenary polyline for each line.
5. Writes the generated points to:
   - `line1.txt`
   - `line2.txt`
6. Computes the minimum **vertex-to-vertex** distance between the two generated polylines.

## File locations

The script assumes that `lines_input.txt` lives in the same directory as `clearance_from_lines.py`.

It also writes `line1.txt` and `line2.txt` to that same directory.

## Expected input format

`lines_input.txt` must contain a Python-literal list with exactly two lines:

```python
[
    ((x1, y1), (x2, y2)),
    ((x3, y3), (x4, y4))
]