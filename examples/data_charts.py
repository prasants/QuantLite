"""Data architecture diagram for the data documentation.

Generates a clean box-and-arrow diagram of the unified fetch interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.viz.theme import FEW_PALETTE, apply_few_theme

apply_few_theme()

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_aspect("equal")
ax.axis("off")
# Remove grid for diagram
ax.grid(False)

fig.suptitle(
    "Unified Data Architecture",
    fontsize=14,
    fontweight="bold",
    y=0.95,
)


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    colour: str,
    text_colour: str = "white",
) -> None:
    """Draw a rounded rectangle with centred text."""
    box = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.15",
        facecolor=colour,
        edgecolor="none",
        alpha=0.9,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=text_colour,
    )


def draw_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    """Draw an arrow between two points."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={
            "arrowstyle": "->",
            "color": FEW_PALETTE["grey_mid"],
            "lw": 1.5,
        },
    )


# Sources (left column)
sources = [
    ("Yahoo Finance", 1.8, 5.0, FEW_PALETTE["primary"]),
    ("CCXT", 1.8, 3.8, FEW_PALETTE["secondary"]),
    ("FRED", 1.8, 2.6, FEW_PALETTE["positive"]),
    ("CSV / Parquet", 1.8, 1.4, FEW_PALETTE["neutral"]),
]

for label, x, y, colour in sources:
    draw_box(ax, x, y, 2.4, 0.7, label, colour)
    draw_arrow(ax, x + 1.3, y, 4.3, 3.2)

# Central fetch() box
draw_box(ax, 5.2, 3.2, 1.8, 1.0, "fetch()", FEW_PALETTE["grey_dark"])

# Output (right)
draw_arrow(ax, 6.2, 3.2, 7.5, 3.2)
draw_box(
    ax,
    8.5,
    3.2,
    2.2,
    1.0,
    "Standardised\nDataFrame",
    FEW_PALETTE["primary"],
)

# Subtitle below the output box
ax.text(
    8.5,
    2.4,
    "DatetimeIndex, OHLCV,\nmetadata, no NaN rows",
    ha="center",
    va="top",
    fontsize=8,
    color=FEW_PALETTE["grey_mid"],
    style="italic",
)

fig.savefig(OUT / "data_sources.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'data_sources.png'}")
