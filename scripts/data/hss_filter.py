"""Split the AISC HSS catalog by cross-section shape.

First stage of building the frozen catalog: partitions data/HSS.csv into
square, rectangular, and round CSVs byte-faithfully, so every later stage
works from exact, untouched AISC data.

Usage:
    uv run python -m scripts.data.hss_filter
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = REPO_ROOT / "data" / "HSS.csv"
OUTPUT_PATHS = {
    "square": REPO_ROOT / "data" / "HSS_square.csv",
    "rectangular": REPO_ROOT / "data" / "HSS_rectangular.csv",
    "round": REPO_ROOT / "data" / "HSS_round.csv",
}


def detect_line_terminator(path: Path) -> str:
    """Detect whether the file's lines end with CRLF or LF.

    Assumptions:
        1. The line ending of the first line is representative of the whole
           file, so the whole file is not scanned.

    Args:
        path: path to the CSV file to inspect

    Returns:
        str: ``"\\r\\n"`` if the first line ends with a carriage return + line
            feed, otherwise ``"\\n"``
    """
    with path.open("rb") as handle:
        first = handle.readline()
    return "\r\n" if first.endswith(b"\r\n") else "\n"


def read_header_line(path: Path) -> str:
    """Return the first line of a file as text, without its line terminator.

    Assumptions:
        1. The CSV header contains duplicate column names (imperial + metric
           blocks) and an unnamed blank column, so it is captured as raw text
           rather than reconstructed from parsed column labels.

    Args:
        path: path to the CSV file to read the header from

    Returns:
        str: the first line of the file with any trailing CR/LF stripped
    """
    with path.open("r", encoding="utf-8") as handle:
        return handle.readline().rstrip("\r\n")


def load_hss(path: Path) -> pd.DataFrame:
    """Load the HSS catalog with every field preserved as its original string.

    Assumptions:
        1. Reads with ``dtype=str`` and ``keep_default_na=False`` so numeric
           formatting, the em-dash placeholder, and empty fields survive a
           read/write round trip unchanged.

    Args:
        path: path to the ``HSS.csv`` catalog

    Returns:
        pd.DataFrame: the catalog with all values as raw strings
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def classify(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build boolean row masks partitioning the catalog by cross-section shape.

    Assumptions:
        1. Keys on the imperial ``Ht`` (height), ``B`` (width) columns — the
           first occurrences, since pandas suffixes the duplicate metric block.
        2. Round sections are defined as the complement of square and
           rectangular, guaranteeing the three masks partition every row with
           no silent drops.

    Args:
        df: the HSS catalog loaded as strings

    Returns:
        dict: mapping of ``"square"``, ``"rectangular"``, ``"round"`` to the
            boolean row mask selecting that shape
    """
    ht = pd.to_numeric(df["Ht"], errors="coerce")
    b = pd.to_numeric(df["B"], errors="coerce")

    both_present = ht.notna() & b.notna()
    is_square = both_present & np.isclose(ht, b)
    is_rectangular = both_present & ~np.isclose(ht, b)
    is_round = ~(is_square | is_rectangular)

    return {
        "square": is_square,
        "rectangular": is_rectangular,
        "round": is_round,
    }


def write_subset(
    df: pd.DataFrame,
    mask: pd.Series,
    header_line: str,
    line_terminator: str,
    out_path: Path,
) -> int:
    """Write the masked rows to a CSV, reusing the original header verbatim.

    Assumptions:
        1. The original header line is written as raw text and the body via
           ``to_csv(header=False)`` so the output is byte-faithful to the
           source table (the data has no embedded commas or quotes).
        2. The source line terminator is reproduced so CRLF-encoded catalogs
           round trip unchanged.

    Args:
        df: the full HSS catalog loaded as strings
        mask: boolean row mask selecting the subset to write
        header_line: the raw source header line to prepend
        line_terminator: the line ending to write (``"\\r\\n"`` or ``"\\n"``)
        out_path: destination path for the filtered CSV

    Returns:
        int: the number of data rows written
    """
    subset = df[mask]
    body = subset.to_csv(header=False, index=False, lineterminator=line_terminator)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(header_line + line_terminator)
        handle.write(body)
    return len(subset)


def main() -> None:
    """Split ``data/HSS.csv`` into square, rectangular, and round CSVs.

    Args: None

    Returns: None
    """
    header_line = read_header_line(INPUT_PATH)
    line_terminator = detect_line_terminator(INPUT_PATH)
    df = load_hss(INPUT_PATH)
    masks = classify(df)

    counts = {
        shape: write_subset(
            df, masks[shape], header_line, line_terminator, OUTPUT_PATHS[shape]
        )
        for shape in OUTPUT_PATHS
    }

    summary = "  ".join(f"{shape}={count}" for shape, count in counts.items())
    print(f"{summary}  total={sum(counts.values())}")


if __name__ == "__main__":
    main()
