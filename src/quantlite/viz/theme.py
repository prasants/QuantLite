"""Stephen Few-inspired visualisation theme for matplotlib.

Design principles: maximum data-ink ratio, no chartjunk, muted palette,
direct labels over legends, no 3D, no gradients, horizontal gridlines
only in light grey, small multiples over busy charts.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "FEW_PALETTE",
    "apply_few_theme",
    "few_figure",
    "direct_label",
    "sparkline",
    "bullet_graph",
]

FEW_PALETTE: dict[str, str] = {
    "primary": "#4E79A7",
    "secondary": "#F28E2B",
    "negative": "#E15759",
    "positive": "#59A14F",
    "neutral": "#76B7B2",
    "grey_dark": "#4E4E4E",
    "grey_mid": "#999999",
    "grey_light": "#E8E8E8",
    "bg": "#FFFFFF",
}

_FEW_CYCLE = [
    FEW_PALETTE["primary"],
    FEW_PALETTE["secondary"],
    FEW_PALETTE["positive"],
    FEW_PALETTE["negative"],
    FEW_PALETTE["neutral"],
    FEW_PALETTE["grey_mid"],
]


def apply_few_theme() -> None:
    """Apply the Stephen Few theme globally to matplotlib.

    Sets rcParams for clean, high-data-ink-ratio charts: no top/right
    spines, horizontal gridlines only, muted colour cycle, and
    legible font sizes.
    """
    mpl.rcParams.update({
        # Colours
        "axes.facecolor": FEW_PALETTE["bg"],
        "figure.facecolor": FEW_PALETTE["bg"],
        "axes.edgecolor": FEW_PALETTE["grey_mid"],
        "axes.labelcolor": FEW_PALETTE["grey_dark"],
        "text.color": FEW_PALETTE["grey_dark"],
        "xtick.color": FEW_PALETTE["grey_mid"],
        "ytick.color": FEW_PALETTE["grey_mid"],
        # Grid: horizontal only, light
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": FEW_PALETTE["grey_light"],
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Fonts
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        # Colour cycle
        "axes.prop_cycle": mpl.cycler(color=_FEW_CYCLE),
        # Legend
        "legend.frameon": False,
        "legend.fontsize": 10,
        # Figure
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def few_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes | list[Axes]]:
    """Create a figure with the Few theme applied.

    Convenience wrapper around ``plt.subplots`` that ensures the
    theme is active and returns properly sized figures.

    Args:
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        figsize: Figure size in inches. Defaults to sensible sizes
            based on subplot count.
        **kwargs: Passed to ``plt.subplots``.

    Returns:
        Tuple of (Figure, Axes or array of Axes).
    """
    apply_few_theme()

    if figsize is None:
        figsize = (4.5 * ncols + 1, 3.5 * nrows + 0.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    fig.tight_layout(pad=2.0)
    return fig, axes


def direct_label(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    colour: str | None = None,
    fontsize: int = 10,
    ha: str = "left",
    va: str = "center",
) -> None:
    """Add a direct label to a data point (preferred over legends).

    Args:
        ax: Matplotlib Axes.
        x: X position.
        y: Y position.
        text: Label text.
        colour: Text colour. Defaults to ``grey_dark``.
        fontsize: Font size.
        ha: Horizontal alignment.
        va: Vertical alignment.
    """
    if colour is None:
        colour = FEW_PALETTE["grey_dark"]
    ax.annotate(
        text,
        xy=(x, y),
        fontsize=fontsize,
        color=colour,
        ha=ha,
        va=va,
    )


def sparkline(
    ax: Axes,
    values: list[float] | Any,
    colour: str | None = None,
) -> None:
    """Draw a minimal sparkline on the given axes.

    Removes all spines, ticks, and labels for maximum data-ink ratio.

    Args:
        ax: Matplotlib Axes.
        values: Sequence of numeric values.
        colour: Line colour. Defaults to ``primary``.
    """
    if colour is None:
        colour = FEW_PALETTE["primary"]

    ax.plot(values, color=colour, linewidth=1.2)
    ax.set_xlim(0, len(values) - 1)
    ax.axis("off")

    # Mark the final value
    ax.plot(len(values) - 1, values[-1], "o", color=colour, markersize=3)


def bullet_graph(
    ax: Axes,
    value: float,
    target: float,
    ranges: list[float],
    label: str = "",
    colour: str | None = None,
) -> None:
    """Draw a horizontal bullet graph (Stephen Few's design).

    Args:
        ax: Matplotlib Axes.
        value: The actual metric value.
        target: The target/benchmark value.
        ranges: List of 3 floats defining [poor, satisfactory, good]
            thresholds.
        label: Y-axis label.
        colour: Bar colour. Defaults to ``grey_dark``.
    """
    if colour is None:
        colour = FEW_PALETTE["grey_dark"]

    greys = ["#DDDDDD", "#BBBBBB", "#999999"]
    max_range = max(ranges)

    # Background ranges
    for i, r in enumerate(sorted(ranges, reverse=True)):
        ax.barh(0, r, height=0.6, color=greys[i], align="center")

    # Value bar
    ax.barh(0, value, height=0.25, color=colour, align="center")

    # Target marker
    ax.axvline(x=target, color=FEW_PALETTE["grey_dark"], linewidth=2, ymin=0.2, ymax=0.8)

    ax.set_xlim(0, max_range * 1.05)
    ax.set_yticks([0])
    ax.set_yticklabels([label])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
