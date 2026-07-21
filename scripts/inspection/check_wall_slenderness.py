"""Report which frozen-catalog HSS sections have slender walls.

`trussRL.capacity.compression_capacity_kip` computes E3 flexural buckling on
gross section properties and applies the Chapter E7 effective-area reduction to
sections whose walls are slender. This script reports which sections take that
E7 path: for each section it compares the wall width-to-thickness ratios
against the AISC Table B4.1a Case 6 limit. It is informational and saves
nothing.

Usage:
    uv run python -m scripts.inspection.check_wall_slenderness
"""

import math

from rich.console import Console
from rich.table import Table

from trussRL.capacity import E_KSI, FY_KSI
from trussRL.catalog import Section, list_sections


def lambda_r_wall() -> float:
    """Nonslender wall width-to-thickness limit for rectangular HSS.

    Assumptions:
        1. AISC 360 Table B4.1a Case 6 — walls of rectangular HSS in uniform
           axial compression — gives lambda_r = 1.40 * sqrt(E / Fy). Round HSS
           (a different D/t limit) is absent from the frozen catalog, so this
           single rectangular limit covers every section.

    Args: None

    Returns:
        float: the nonslender wall slenderness limit lambda_r
    """
    return 1.40 * math.sqrt(E_KSI / FY_KSI)


def wall_slenderness(section: Section) -> tuple[float, float, float]:
    """Read a section's wall width-to-thickness ratios and their governing value.

    Args:
        section: catalog section carrying the AISC design-thickness ratios
            ``b/tdes`` and ``h/tdes`` in its ``properties`` mapping

    Returns:
        tuple: (b_over_t, h_over_t, governing) where governing is the larger of
            the two ratios, since a wall is slender if either ratio exceeds
            lambda_r
    """
    b_over_t = section.properties["b/tdes"]
    h_over_t = section.properties["h/tdes"]
    return b_over_t, h_over_t, max(b_over_t, h_over_t)


def main() -> None:
    """Print a per-section wall-slenderness table and an overall verdict.

    Args: None

    Returns: None
    """
    console = Console()
    limit = lambda_r_wall()

    rows = []
    for section in list_sections():
        b_over_t, h_over_t, governing = wall_slenderness(section)
        rows.append((section.designation, b_over_t, h_over_t, governing))
    rows.sort(key=lambda row: row[3], reverse=True)

    table = Table(title="HSS frozen-catalog wall slenderness (AISC Table B4.1a Case 6)")
    table.add_column("Designation", style="cyan", no_wrap=True)
    table.add_column("b/tdes", justify="right")
    table.add_column("h/tdes", justify="right")
    table.add_column("governing", justify="right")
    table.add_column("λr", justify="right")
    table.add_column("status", justify="left")

    offenders = []
    for designation, b_over_t, h_over_t, governing in rows:
        nonslender = governing <= limit
        if nonslender:
            status = "[green]✓ nonslender[/green]"
        else:
            status = "[yellow]slender → E7[/yellow]"
            offenders.append(designation)
        table.add_row(
            designation,
            f"{b_over_t:.2f}",
            f"{h_over_t:.2f}",
            f"{governing:.2f}",
            f"{limit:.2f}",
            status,
        )

    console.print(
        f"Fy = {FY_KSI:.0f} ksi   E = {E_KSI:.0f} ksi   λr = 1.40·√(E/Fy) ≈ {limit:.2f}"
    )
    console.print(table)

    total = len(rows)
    n_nonslender = total - len(offenders)
    console.print(f"{n_nonslender}/{total} sections have nonslender walls")
    if offenders:
        console.print(
            f"[bold yellow]E7 path[/bold yellow]: slender walls in "
            f"{', '.join(offenders)}; capacity applies the Chapter E7 "
            f"effective-area reduction to these sections."
        )
    else:
        console.print(
            "[bold green]All nonslender[/bold green]: every catalog section "
            "uses the full gross area; the Chapter E7 path is never taken."
        )


if __name__ == "__main__":
    main()
