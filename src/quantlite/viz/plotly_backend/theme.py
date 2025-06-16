"""Stephen Few-inspired Plotly theme.

Mirrors the matplotlib theme exactly: maximum data-ink ratio, muted palette,
horizontal gridlines only, no chartjunk, no Plotly chrome.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

__all__ = [
    "FEW_PALETTE",
    "FEW_COLORWAY",
    "FEW_TEMPLATE",
    "apply_few_theme_plotly",
    "few_figure",
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

FEW_COLORWAY: list[str] = [
    FEW_PALETTE["primary"],
    FEW_PALETTE["secondary"],
    FEW_PALETTE["positive"],
    FEW_PALETTE["negative"],
    FEW_PALETTE["neutral"],
    FEW_PALETTE["grey_mid"],
]

FEW_TEMPLATE: go.layout.Template = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Arial, sans-serif", size=11, color=FEW_PALETTE["grey_dark"]),
        plot_bgcolor=FEW_PALETTE["bg"],
        paper_bgcolor=FEW_PALETTE["bg"],
        colorway=FEW_COLORWAY,
        xaxis=dict(
            showgrid=False,
            linecolor=FEW_PALETTE["grey_mid"],
            linewidth=1,
            ticks="outside",
            tickcolor=FEW_PALETTE["grey_mid"],
            tickfont=dict(color=FEW_PALETTE["grey_mid"]),
            mirror=False,
            showline=True,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=FEW_PALETTE["grey_light"],
            gridwidth=0.8,
            linecolor=FEW_PALETTE["grey_mid"],
            linewidth=1,
            ticks="outside",
            tickcolor=FEW_PALETTE["grey_mid"],
            tickfont=dict(color=FEW_PALETTE["grey_mid"]),
            mirror=False,
            showline=True,
            zeroline=False,
        ),
        title=dict(
            font=dict(size=13, color=FEW_PALETTE["grey_dark"]),
            x=0.0,
            xanchor="left",
        ),
        showlegend=False,
        hovermode="closest",
        margin=dict(l=60, r=20, t=50, b=50),
    ),
)


def apply_few_theme_plotly() -> None:
    """Register and set the Few theme as the default Plotly template.

    Call once at import time or before creating figures.
    """
    pio.templates["few"] = FEW_TEMPLATE
    pio.templates.default = "few"


def few_figure(
    title: str = "",
    width: int | None = None,
    height: int | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Create a Plotly figure with the Few theme applied.

    Args:
        title: Figure title.
        width: Figure width in pixels.
        height: Figure height in pixels.
        **kwargs: Passed to ``go.Figure``.

    Returns:
        A themed ``go.Figure``.
    """
    layout = dict(
        template=FEW_TEMPLATE,
        title=title,
    )
    if width is not None:
        layout["width"] = width
    if height is not None:
        layout["height"] = height

    fig = go.Figure(layout=layout, **kwargs)
    # Strip Plotly chrome
    fig.update_layout(
        dragmode=False,
        modebar_remove=[
            "zoom", "pan", "select", "lasso", "zoomIn", "zoomOut",
            "autoScale", "resetScale", "toImage",
        ],
    )
    return fig
