"""Regime visualisation charts (Plotly backend).

Regime timelines, transition matrices, conditional distributions,
and changepoint detection, following Stephen Few's principles.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import FEW_PALETTE, FEW_TEMPLATE, few_figure

__all__ = [
    "plot_regime_timeline",
    "plot_transition_matrix",
    "plot_regime_distributions",
    "plot_changepoints",
]

_REGIME_COLOURS = [
    FEW_PALETTE["negative"],
    FEW_PALETTE["neutral"],
    FEW_PALETTE["primary"],
    FEW_PALETTE["positive"],
    FEW_PALETTE["secondary"],
]

_REGIME_COLOURS_RGBA = [
    "rgba(225, 87, 89, 0.25)",
    "rgba(118, 183, 178, 0.25)",
    "rgba(78, 121, 167, 0.25)",
    "rgba(89, 161, 79, 0.25)",
    "rgba(242, 142, 43, 0.25)",
]


def plot_regime_timeline(
    returns: np.ndarray | pd.Series,
    regimes: np.ndarray,
    changepoints: list[int] | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Price series with regime-coloured background bands.

    Args:
        returns: Simple periodic returns.
        regimes: Array of integer regime labels.
        changepoints: Optional list of change-point indices.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    arr = np.asarray(returns, dtype=float)
    reg = np.asarray(regimes, dtype=int)
    cum = np.cumprod(1 + arr)

    fig = few_figure(title="Returns with Regime Timeline", width=width, height=height)

    # Cumulative return line
    fig.add_trace(go.Scatter(
        x=list(range(len(cum))), y=cum,
        mode="lines",
        line=dict(color=FEW_PALETTE["grey_dark"], width=1.2),
        hovertemplate="Period: %{x}<br>Cum Return: %{y:.4f}<extra></extra>",
    ))

    # Regime background bands (contiguous segments, batch for speed)
    shapes: list[dict[str, Any]] = []
    annots: list[dict[str, Any]] = []
    i = 0
    while i < len(reg):
        r = int(reg[i])
        j = i
        while j < len(reg) and reg[j] == r:
            j += 1
        colour = _REGIME_COLOURS_RGBA[r % len(_REGIME_COLOURS_RGBA)]
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=i - 0.5, x1=j - 0.5, y0=0, y1=1,
            fillcolor=colour, line_width=0, layer="below",
        ))
        if (j - i) > len(reg) * 0.05:
            annots.append(dict(
                x=(i + j) / 2, y=1.0, xref="x", yref="paper",
                text=f"R{r}", showarrow=False,
                font=dict(size=8, color=_REGIME_COLOURS[r % len(_REGIME_COLOURS)]),
            ))
        i = j
    fig.update_layout(shapes=shapes, annotations=annots)

    # Changepoints
    if changepoints:
        for cp in changepoints:
            fig.add_vline(
                x=cp, line_color=FEW_PALETTE["grey_mid"],
                line_width=0.7, line_dash="dot", opacity=0.6,
            )

    fig.update_layout(xaxis_title="Period", yaxis_title="Cumulative Return")
    return fig


def plot_transition_matrix(
    model: Any,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Annotated heatmap of transition probabilities.

    Args:
        model: A ``RegimeModel`` with ``transition_matrix`` and
            ``n_regimes`` attributes.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    trans = model.transition_matrix
    n = model.n_regimes
    labels = [f"R{i}" for i in range(n)]

    text = [[f"{trans[i, j]:.3f}" for j in range(n)] for i in range(n)]

    fig = few_figure(
        title="Transition Probabilities",
        width=width or max(400, n * 80),
        height=height or max(350, n * 70),
    )

    fig.add_trace(go.Heatmap(
        z=trans, x=labels, y=labels,
        zmin=0, zmax=1,
        colorscale=[[0.0, "#FFFFFF"], [1.0, FEW_PALETTE["primary"]]],
        text=text, texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="From %{y} to %{x}: %{z:.3f}<extra></extra>",
        colorbar=dict(title="P", thickness=10, len=0.8),
    ))

    fig.update_layout(
        xaxis_title="To Regime",
        yaxis_title="From Regime",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_regime_distributions(
    returns: np.ndarray | pd.Series,
    regimes: np.ndarray,
    bins: int = 40,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Overlaid histograms per regime.

    Args:
        returns: Simple periodic returns.
        regimes: Array of regime labels.
        bins: Number of histogram bins.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    arr = np.asarray(returns, dtype=float)
    reg = np.asarray(regimes, dtype=int)
    unique = sorted(np.unique(reg))
    n_regimes = len(unique)

    fig = make_subplots(
        rows=1, cols=n_regimes,
        subplot_titles=[f"Regime {r}" for r in unique],
        shared_yaxes=True,
    )
    fig.update_layout(
        template=FEW_TEMPLATE,
        title="Return Distributions by Regime",
        width=width or max(600, n_regimes * 300),
        height=height or 350,
        showlegend=False,
        dragmode=False,
    )

    x_min, x_max = float(arr.min()), float(arr.max())

    for i, r in enumerate(unique, 1):
        r_data = arr[reg == r]
        colour = _REGIME_COLOURS[r % len(_REGIME_COLOURS)]
        mu = float(np.mean(r_data))
        sigma = float(np.std(r_data, ddof=1)) if len(r_data) > 1 else 0.0

        fig.add_trace(go.Histogram(
            x=r_data, nbinsx=bins, histnorm="probability density",
            marker_color=colour, opacity=0.7,
            showlegend=False,
            hovertemplate="Return: %{x:.4f}<br>Density: %{y:.2f}<extra></extra>",
        ), row=1, col=i)

        # Use paper-relative coords for annotation positioning
        x_frac = (i - 0.5) / n_regimes + 0.45 / n_regimes
        fig.add_annotation(
            x=x_frac, y=0.95, xref="paper", yref="paper",
            text=f"mean: {mu:.4f}<br>vol: {sigma:.4f}<br>n={len(r_data)}",
            showarrow=False, font=dict(size=8, color=FEW_PALETTE["grey_dark"]),
            xanchor="right", yanchor="top",
        )
        fig.update_xaxes(range=[x_min, x_max], row=1, col=i)

    return fig


def plot_changepoints(
    returns: np.ndarray | pd.Series,
    changepoints: list[int],
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Time series with detected changepoint lines.

    Args:
        returns: Simple periodic returns.
        changepoints: List of change-point indices.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    arr = np.asarray(returns, dtype=float)
    cum = np.cumprod(1 + arr)

    fig = few_figure(title="Changepoint Detection", width=width, height=height)

    fig.add_trace(go.Scatter(
        x=list(range(len(cum))), y=cum,
        mode="lines",
        line=dict(color=FEW_PALETTE["primary"], width=1.5),
        hovertemplate="Period: %{x}<br>Cum Return: %{y:.4f}<extra></extra>",
    ))

    for cp in changepoints:
        fig.add_vline(
            x=cp, line_color=FEW_PALETTE["negative"],
            line_width=1.5, line_dash="dash",
            annotation_text=f"CP {cp}",
            annotation_position="top",
            annotation_font=dict(size=8, color=FEW_PALETTE["negative"]),
        )

    fig.update_layout(xaxis_title="Period", yaxis_title="Cumulative Return")
    return fig
