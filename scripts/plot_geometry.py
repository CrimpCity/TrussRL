"""Visually check the expander.

CLI to plot nodes/members of input design to visually verify the generated geometry.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import typer
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from trussRL.expander import TrussGeometry, expand

app = typer.Typer(add_completion=False)

# Fixed categorical order so a group keeps its color across typologies
# (color follows the entity, never the series count).
GROUP_COLORS = {
    "bottom_chord": "#2a78d6",
    "top_chord": "#1baf7a",
    "diagonals": "#c98500",
    "verticals": "#008300",
}
INK = "#0b0b0b"
SURFACE = "#fcfcfb"


def draw_geometry(ax: Axes, geometry: TrussGeometry) -> list[Line2D]:
    """Draw members, nodes, and support markers onto an axes.

    Args:
        ax: matplotlib axes to draw into geometry: expanded truss geometry to render

    Returns:
        list: legend handles for the member groups present plus the two support markers
    """
    nodes = {node.id: node for node in geometry.nodes}
    groups_present: list[str] = []
    for member in geometry.members:
        node_i = nodes[member.node_i]
        node_j = nodes[member.node_j]
        ax.plot(
            [node_i.x_ft, node_j.x_ft],
            [node_i.y_ft, node_j.y_ft],
            color=GROUP_COLORS[member.group],
            linewidth=2,
            solid_capstyle="round",
            zorder=2,
        )
        if member.group not in groups_present:
            groups_present.append(member.group)

    ax.scatter(
        [node.x_ft for node in geometry.nodes],
        [node.y_ft for node in geometry.nodes],
        s=18,
        color=INK,
        zorder=3,
    )

    pin = nodes[geometry.pin_node]
    roller = nodes[geometry.roller_node]
    ax.plot([pin.x_ft], [pin.y_ft], marker="^", markersize=13, color=INK, zorder=4)
    ax.plot(
        [roller.x_ft], [roller.y_ft], marker="o", markersize=11, color=INK, zorder=4
    )

    handles = [
        Line2D([0], [0], color=GROUP_COLORS[group], linewidth=2, label=group)
        for group in GROUP_COLORS
        if group in groups_present
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=INK,
            markersize=10,
            label="pin",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=INK,
            markersize=9,
            label="roller",
        )
    )
    return handles


@app.command()
def plot(
    span: float = typer.Option(96.0, help="Span in feet."),
    n_bays: int = typer.Option(8, help="Number of bottom-chord bays."),
    depth: float = typer.Option(8.0, help="Truss depth in feet."),
    truss_type: str = typer.Option("warren", help="Typology to expand."),
    save: Path | None = typer.Option(
        None, help="Write a PNG to this path instead of showing a window."
    ),
) -> None:
    """Expand a design and plot its geometry for visual verification.

    Args:
        span: span in feet
        n_bays: number of bottom-chord bays
        depth: truss depth in feet
        truss_type: typology name understood by the expander
        save: optional output PNG path; if omitted, an interactive window opens

    Returns: None
    """
    geometry = expand(truss_type, span, n_bays, depth)

    # Equal aspect means the axes height tracks depth/span; size the figure
    # to fit it plus fixed room for the title, x-label, and bottom legend.
    fig_height = min(6.0, max(2.5, 11.0 * depth / span + 2.0))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    handles = draw_geometry(ax, geometry)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        labelcolor=INK,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x (ft)", color=INK)
    ax.set_ylabel("y (ft)", color=INK)
    ax.set_title(
        f"{truss_type} — span {span:g} ft, {n_bays} bays "
        f"(bay {geometry.bay_width_ft:g} ft), depth {depth:g} ft — "
        f"{geometry.n_nodes} nodes, {len(geometry.members)} members",
        color=INK,
        fontsize=11,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK)
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))

    if save is None:
        plt.show()
    else:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        typer.echo(f"saved {save}")


if __name__ == "__main__":
    app()
