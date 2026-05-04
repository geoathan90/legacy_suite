"""
Export όλων των βελοδιαγραμμάτων σε μορφή CSV για Μιχάλη.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from .eval import evaluate


def sample_x_range(xmin, xmax, step=0.01):
    """
    Return x-samples from xmin to xmax with spacing 'step',
    including xmax when possible.
    """
    n = int(np.floor((xmax - xmin) / step)) + 1
    xs = xmin + np.arange(n) * step

    # make sure the exact xmax is included if the step misses it slightly
    if xs[-1] < xmax:
        xs = np.append(xs, xmax)

    return xs


def batch_eval_csvs(
    csv_dir=".",
    step=0.01,
    save_outputs=False,
    output_dir="eval_outputs"
):
    """
    For each CSV in csv_dir:
      - read x-range from the first column
      - sample it every 'step'
      - call evaluate(csv_name, xs)
      - store result in a dict of DataFrames

    Returns:
        results: dict[str, pd.DataFrame]
    """
    csv_dir = Path(csv_dir)
    results = {}

    if save_outputs:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("*.csv"))

    for csv_file in csv_files:
        # read csv
        df_raw = pd.read_csv(csv_file)

        # assume x-values are in the first column
        x_col = df_raw.columns[0]
        x_values = pd.to_numeric(df_raw[x_col], errors="coerce").dropna()

        if x_values.empty:
            print(f"Skipping {csv_file.name}: no valid x-values found.")
            continue

        xmin = x_values.min()
        xmax = x_values.max()

        xs = sample_x_range(xmin, xmax, step=step)

        # IMPORTANT:
        # adjust this line depending on what your evaluate() expects:
        # csv_file.stem  -> "file_name" without .csv
        # csv_file.name  -> "file_name.csv"
        result_df = evaluate(csv_file.stem, xs.tolist())

        results[csv_file.stem] = result_df

        print(
            f"{csv_file.name}: "
            f"x-range = [{xmin:.3f}, {xmax:.3f}], "
            f"samples = {len(xs)}"
        )

        if save_outputs:
            out_file = output_dir / f"{csv_file.stem}_evaluated.csv"
            result_df.to_csv(out_file, index=False)

    return results


results = batch_eval_csvs(
    csv_dir="data",
    step=0.01,
    save_outputs=True,
    output_dir="eval_outputs"
)