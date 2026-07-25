"""Pick the frozen 40-shape subset the policy designs with.

Selects square and rectangular HSS shapes spread uniformly across the
stiffness range, endpoints included — a catalog broad enough that section
choice is a real decision, small enough to fit in a prompt.

Usage:
    uv run python -m scripts.data.hss_subset
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data.hss_filter import detect_line_terminator, load_hss, read_header_line

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATHS = {
    "rectangular": REPO_ROOT / "data" / "HSS_rectangular.csv",
    "square": REPO_ROOT / "data" / "HSS_square.csv",
}
OUTPUT_PATH = REPO_ROOT / "data" / "HSS_subset.csv"
NUM_PER_SHAPE = 20


def select_uniform_by_ix(df: pd.DataFrame, num_shapes: int) -> pd.DataFrame:
    """Select rows spread uniformly across the catalog sorted by strong-axis Ix.

    Assumptions:
        1. Keys on the imperial ``Ix`` column — the first occurrence, since
           pandas suffixes the duplicate metric block as ``Ix.1``.
        2. Uses linear interpolation over the sorted positions
           (``np.linspace``) rather than a modulo stride, so both the
           smallest and largest sections are always included.

    Args:
        df: an HSS shape-family catalog loaded as strings
        num_shapes: number of rows to select

    Returns:
        pd.DataFrame: the selected rows in ascending-Ix order, with all
            fields preserved as their original strings
    """
    ix = pd.to_numeric(df["Ix"], errors="coerce")
    sorted_df = df.iloc[ix.to_numpy().argsort()]
    positions = np.linspace(0, len(sorted_df) - 1, num_shapes).round().astype(int)
    return sorted_df.iloc[positions]


def write_selection(
    subset: pd.DataFrame,
    header_line: str,
    line_terminator: str,
    out_path: Path,
) -> int:
    """Write the selected rows to a CSV, reusing the original header verbatim.

    Assumptions:
        1. The original header line is written as raw text and the body via
           ``to_csv(header=False)`` so the output is byte-faithful to the
           source table (the data has no embedded commas or quotes).
        2. The source line terminator is reproduced so CRLF-encoded catalogs
           round trip unchanged.

    Args:
        subset: the selected rows loaded as strings
        header_line: the raw source header line to prepend
        line_terminator: the line ending to write (``"\\r\\n"`` or ``"\\n"``)
        out_path: destination path for the subset CSV

    Returns:
        int: the number of data rows written
    """
    body = subset.to_csv(header=False, index=False, lineterminator=line_terminator)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(header_line + line_terminator)
        handle.write(body)
    return len(subset)


def main() -> None:
    """Write 20 rectangular + 20 square uniformly Ix-spaced HSS shapes to a CSV.

    Assumptions:
        1. The per-shape catalogs were split from the same ``HSS.csv`` by
           ``scripts.data.hss_filter``, so their header lines are identical; the
           script fails fast if the frozen catalogs ever diverge.

    Args: None

    Returns: None
    """
    reference_path = INPUT_PATHS["rectangular"]
    header_line = read_header_line(reference_path)
    line_terminator = detect_line_terminator(reference_path)

    subsets = {}
    for shape, path in INPUT_PATHS.items():
        if read_header_line(path) != header_line:
            raise ValueError(
                f"header of {path.name} differs from {reference_path.name}"
            )
        subsets[shape] = select_uniform_by_ix(load_hss(path), NUM_PER_SHAPE)

    combined = pd.concat(subsets.values(), ignore_index=True)
    ix = pd.to_numeric(combined["Ix"], errors="coerce")
    combined = combined.iloc[ix.to_numpy().argsort()]
    count = write_selection(combined, header_line, line_terminator, OUTPUT_PATH)

    per_shape = "  ".join(f"{shape}={len(sub)}" for shape, sub in subsets.items())
    print(
        f"wrote {count} shapes ({per_shape})  Ix=[{ix.min():g}, {ix.max():g}]  "
        f"-> {OUTPUT_PATH.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
