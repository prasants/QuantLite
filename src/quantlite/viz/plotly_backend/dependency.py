"""Dependency visualisation charts (Plotly backend).

Copula contours, correlation matrices, dynamics, and stress comparisons,
all following Stephen Few's principles.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, rankdata

from .theme import FEW_PALETTE, FEW_TEMPLATE, few_figure

__all__ = [
    "plot_copula_contour",
    "plot_correlation_matrix",
    "plot_correlation_dynamics",
    "plot_stress_correlation",
]


def plot_copula_contour(
    copula: Any,
    data: np.ndarray,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Contour plot for a fitted copula with Gaussian reference.

    Args:
        copula: A fitted copula with ``simulate`` and ``tail_dependence``
            methods.
        data: Original data array of shape ``(n, 2)``.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    from ...dependency.copulas import GaussianCopula

    n = data.shape[0]
    u = np.column_stack([
        rankdata(data[:, j], method="ordinal") / (n + 1)
        for j in range(2)
    ])

    fig = few_figure(
        title=f"{type(copula).__name__} vs Gaussian (grey)",
        width=width or 600, height=height or 550,
    )

    # Scatter of uniform margins
    fig.add_trace(go.Scatter(
        x=u[:, 0], y=u[:, 1], mode="markers",
        marker=dict(color=FEW_PALETTE["grey_mid"], size=2, opacity=0.3),
        showlegend=False,
        hoverinfo="skip",
    ))

    grid_x = np.linspace(0.01, 0.99, 50)
    grid_y = np.linspace(0.01, 0.99, 50)
    X, Y = np.meshgrid(grid_x, grid_y)

    # Gaussian copula reference
    gauss = GaussianCopula()
    gauss.fit(data)
    gauss_samples = gauss.simulate(5000, rng_seed=42)
    try:
        g_kde = gaussian_kde(gauss_samples.T)
        Z_gauss = g_kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        fig.add_trace(go.Contour(
            x=grid_x, y=grid_y, z=Z_gauss,
            showscale=False, ncontours=5,
            contours_coloring="none",
            line=dict(color=FEW_PALETTE["grey_mid"], width=0.8),
            opacity=0.6,
            showlegend=False,
            hoverinfo="skip",
        ))
    except Exception:
        pass

    # Fitted copula contour
    cop_samples = copula.simulate(5000, rng_seed=42)
    try:
        c_kde = gaussian_kde(cop_samples.T)
        Z_cop = c_kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        fig.add_trace(go.Contour(
            x=grid_x, y=grid_y, z=Z_cop,
            showscale=False, ncontours=5,
            contours_coloring="none",
            line=dict(color=FEW_PALETTE["primary"], width=1.2),
            showlegend=False,
            hoverinfo="skip",
        ))
    except Exception:
        pass

    # Tail dependence annotations
    td = copula.tail_dependence()
    fig.add_annotation(
        x=0.05, y=0.05, xref="paper", yref="paper",
        text=f"Lower tail dep: {td['lower']:.3f}",
        showarrow=False, font=dict(size=9, color=FEW_PALETTE["negative"]),
    )
    fig.add_annotation(
        x=0.95, y=0.95, xref="paper", yref="paper",
        text=f"Upper tail dep: {td['upper']:.3f}",
        showarrow=False, font=dict(size=9, color=FEW_PALETTE["positive"]),
    )

    fig.update_layout(
        xaxis_title="U1 (uniform margin)",
        yaxis_title="U2 (uniform margin)",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )
    return fig


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Interactive correlation heatmap with hover values.

    Args:
        corr_matrix: Correlation DataFrame.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    n = len(corr_matrix)
    labels = list(corr_matrix.columns)

    # Annotated text
    text = [[f"{corr_matrix.values[i, j]:.2f}" for j in range(n)] for i in range(n)]

    fig = few_figure(
        title="Correlation Matrix",
        width=width or max(500, n * 60),
        height=height or max(450, n * 55),
    )

    fig.add_trace(go.Heatmap(
        z=corr_matrix.values,
        x=labels, y=labels,
        zmin=-1, zmax=1,
        colorscale=[
            [0.0, FEW_PALETTE["primary"]],
            [0.5, "#FFFFFF"],
            [1.0, FEW_PALETTE["negative"]],
        ],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=max(7, 11 - n // 3)),
        hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Correlation", thickness=12, len=0.8),
    ))

    fig.update_layout(
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_correlation_dynamics(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    window: int = 60,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Rolling correlation line chart.

    Args:
        x: First return series.
        y: Second return series.
        window: Rolling window size.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    sx = pd.Series(np.asarray(x, dtype=float))
    sy = pd.Series(np.asarray(y, dtype=float))
    rolling = sx.rolling(window).corr(sy)
    overall = float(sx.corr(sy))

    fig = few_figure(
        title=f"Rolling {window}-Period Correlation",
        width=width, height=height,
    )

    fig.add_trace(go.Scatter(
        x=list(range(len(rolling))), y=rolling,
        mode="lines",
        line=dict(color=FEW_PALETTE["primary"], width=1.5),
        hovertemplate="Period: %{x}<br>Correlation: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_color=FEW_PALETTE["grey_mid"], line_width=0.8, line_dash="dash")
    fig.add_hline(
        y=overall, line_color=FEW_PALETTE["secondary"],
        line_width=1, line_dash="dot",
        annotation_text=f"Overall: {overall:.3f}",
        annotation_position="top left",
        annotation_font=dict(size=9, color=FEW_PALETTE["secondary"]),
    )

    fig.update_layout(
        xaxis_title="Period",
        yaxis_title="Correlation",
        yaxis_range=[-1.05, 1.05],
    )
    return fig


def plot_stress_correlation(
    calm_corr: pd.DataFrame,
    stress_corr: pd.DataFrame,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Side-by-side calm vs stress correlation heatmaps.

    Args:
        calm_corr: Correlation matrix during calm periods.
        stress_corr: Correlation matrix during stress periods.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    n = len(calm_corr)
    labels = list(calm_corr.columns)
    colorscale = [
        [0.0, FEW_PALETTE["primary"]],
        [0.5, "#FFFFFF"],
        [1.0, FEW_PALETTE["negative"]],
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Calm Period", "Stress Period"],
        horizontal_spacing=0.12,
    )
    fig.update_layout(
        template=FEW_TEMPLATE,
        title="Correlation: Calm vs Stress",
        width=width or max(800, n * 100),
        height=height or max(450, n * 55),
        showlegend=False,
        dragmode=False,
    )

    for col_idx, (corr, subtitle) in enumerate(
        [(calm_corr, "Calm"), (stress_corr, "Stress")], 1
    ):
        text = [[f"{corr.values[i, j]:.2f}" for j in range(n)] for i in range(n)]
        fig.add_trace(go.Heatmap(
            z=corr.values, x=labels, y=labels,
            zmin=-1, zmax=1, colorscale=colorscale,
            text=text, texttemplate="%{text}",
            textfont=dict(size=max(7, 10 - n // 3)),
            showscale=(col_idx == 2),
            colorbar=dict(title="Corr", thickness=10, len=0.8) if col_idx == 2 else None,
            hovertemplate=f"{subtitle}: " + "%{x} vs %{y}: %{z:.3f}<extra></extra>",
        ), row=1, col=col_idx)
        fig.update_yaxes(autorange="reversed", row=1, col=col_idx)

    return fig
