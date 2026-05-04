"""
Export όλων των βελοδιαγραμμάτων σε μορφή CSV για Μιχάλη.

This version saves the evaluated output TRANSPOSED.

Purpose
-------
The original export script saves each evaluated DataFrame in its normal,
wide orientation. In some cases, this creates CSV files with too many
columns for Excel to open comfortably.

This script keeps the same evaluation procedure, but saves:

    result_df.T

instead of:

    result_df

This usually helps when the output has many columns but relatively fewer rows.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .eval import evaluate


# Excel worksheet limits.
# These are not strict CSV limits, but they are useful warnings because
# the goal is to open the output in Excel.
EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_COLUMNS = 16_384


def sample_x_range(xmin, xmax, step=0.01):
    """
    Return x-samples from xmin to xmax with spacing 'step',
    including xmax when possible.

    Parameters
    ----------
    xmin : float
        Minimum x-value.
    xmax : float
        Maximum x-value.
    step : float, default=0.01
        Sampling interval.

    Returns
    -------
    np.ndarray
        Array of sampled x-values.
    """
    n = int(np.floor((xmax - xmin) / step)) + 1
    xs = xmin + np.arange(n) * step

    # Make sure the exact xmax is included if the step misses it slightly.
    if xs[-1] < xmax:
        xs = np.append(xs, xmax)

    return xs


def warn_if_excel_limits_exceeded(df, file_name):
    """
    Print a warning if the DataFrame shape exceeds Excel worksheet limits.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that will be saved/opened.
    file_name : str
        Name of the output file, used only for the printed message.
    """
    rows, cols = df.shape

    if rows > EXCEL_MAX_ROWS:
        print(
            f"WARNING: {file_name} has {rows:,} rows. "
            f"Excel supports up to {EXCEL_MAX_ROWS:,} rows."
        )

    if cols > EXCEL_MAX_COLUMNS:
        print(
            f"WARNING: {file_name} has {cols:,} columns. "
            f"Excel supports up to {EXCEL_MAX_COLUMNS:,} columns."
        )


def batch_eval_csvs_transposed(
    csv_dir=".",
    step=0.01,
    save_outputs=False,
    output_dir="eval_outputs_transposed",
):
    """
    For each CSV in csv_dir:
      - read x-range from the first column
      - sample it every 'step'
      - call evaluate(csv_name, xs)
      - store the normal result in a dict of DataFrames
      - optionally save the TRANSPOSED result to CSV

    Parameters
    ----------
    csv_dir : str or pathlib.Path, default="."
        Directory containing the input CSV files.

    step : float, default=0.01
        Sampling interval for the x-range.

    save_outputs : bool, default=False
        If True, save one transposed evaluated CSV per input CSV.

    output_dir : str or pathlib.Path, default="eval_outputs_transposed"
        Directory where transposed output CSV files will be saved.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary mapping each input CSV stem to its normal, non-transposed
        evaluated DataFrame.

        Note:
        The returned DataFrames are NOT transposed. Only the saved CSV files
        are transposed.
    """
    csv_dir = Path(csv_dir)
    results = {}

    if save_outputs:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("*.csv"))

    for csv_file in csv_files:
        df_raw = pd.read_csv(csv_file)

        # Assume x-values are in the first column.
        x_col = df_raw.columns[0]
        x_values = pd.to_numeric(df_raw[x_col], errors="coerce").dropna()

        if x_values.empty:
            print(f"Skipping {csv_file.name}: no valid x-values found.")
            continue

        xmin = x_values.min()
        xmax = x_values.max()

        xs = sample_x_range(xmin, xmax, step=step)

        # IMPORTANT:
        # Adjust this line depending on what evaluate() expects:
        # csv_file.stem -> "file_name" without .csv
        # csv_file.name -> "file_name.csv"
        result_df = evaluate(csv_file.stem, xs.tolist())

        results[csv_file.stem] = result_df

        print(
            f"{csv_file.name}: "
            f"x-range = [{xmin:.3f}, {xmax:.3f}], "
            f"samples = {len(xs):,}, "
            f"normal shape = {result_df.shape[0]:,} rows x {result_df.shape[1]:,} columns"
        )

        if save_outputs:
            transposed_df = result_df.T

            out_file = output_dir / f"{csv_file.stem}_evaluated_transposed.csv"

            print(
                f"Saving transposed output: "
                f"{transposed_df.shape[0]:,} rows x {transposed_df.shape[1]:,} columns"
            )

            warn_if_excel_limits_exceeded(transposed_df, out_file.name)

            # index=True is important here.
            # After transposition, the original column names become the row labels.
            # Saving the index preserves them as the first column in the CSV.
            transposed_df.to_csv(
                out_file,
                index=True,
                index_label="original_column",
            )

    return results


results = batch_eval_csvs_transposed(
    csv_dir="data",
    step=0.01,
    save_outputs=True,
    output_dir="eval_outputs_transposed",
)