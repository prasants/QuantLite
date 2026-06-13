"""Portfolio visualisation charts (Plotly backend).

Efficient frontiers, weight comparisons, monthly returns heatmaps,
and HRP dendrograms, following Stephen Few's principles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage

from .theme import FEW_PALETTE, FEW_TEMPLATE, few_figure

__all__ = [
    "plot_efficient_frontier",
    "plot_weight_comparison",
    "plot_monthly_returns",
    "plot_hrp_dendrogram",
]


def plot_efficient_frontier(
    returns_df: pd.DataFrame,
    n_portfolios: int = 2000,
    risk_free_rate: float = 0.0,
    freq: int = 252,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Scatter with Sharpe colouring and hover for weights.

    Args:
        returns_df: Asset returns DataFrame.
        n_portfolios: Number of random portfolios to simulate.
        risk_free_rate: Annualised risk-free rate.
        freq: Periods per year.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    from ...portfolio.optimisation import max_sharpe_weights, min_variance_weights

    n_assets = returns_df.shape[1]
    asset_names = list(returns_df.columns)
    mu = returns_df.mean().values
    cov = returns_df.cov().values

    rng = np.random.default_rng(42)
    rand_rets: list[float] = []
    rand_vols: list[float] = []
    rand_sharpes: list[float] = []
    rand_weights: list[str] = []

    for _ in range(n_portfolios):
        w = rng.random(n_assets)
        w = w / w.sum()
        ret = float((1 + w @ mu) ** freq - 1)
        vol = float(np.sqrt(w @ cov @ w) * np.sqrt(freq))
        sharpe = (ret - risk_free_rate) / vol if vol > 1e-10 else 0.0
        rand_rets.append(ret)
        rand_vols.append(vol)
        rand_sharpes.append(sharpe)
        wt_str = ", ".join(f"{asset_names[j]}: {w[j]:.1%}" for j in range(n_assets))
        rand_weights.append(wt_str)

    fig = few_figure(title="Efficient Frontier", width=width, height=height)

    fig.add_trace(go.Scatter(
        x=rand_vols, y=rand_rets,
        mode="markers",
        marker=dict(
            color=rand_sharpes,
            colorscale=[[0, FEW_PALETTE["grey_light"]], [1, FEW_PALETTE["primary"]]],
            size=5, opacity=0.6,
            colorbar=dict(title="Sharpe", thickness=10, len=0.6),
        ),
        customdata=rand_weights,
        hovertemplate=(
            "Vol: %{x:.3f}<br>Return: %{y:.3f}<br>"
            "Sharpe: %{marker.color:.2f}<br>%{customdata}<extra></extra>"
        ),
    ))

    # Min variance
    mv = min_variance_weights(returns_df, freq=freq)
    fig.add_trace(go.Scatter(
        x=[mv.expected_risk], y=[mv.expected_return],
        mode="markers+text",
        marker=dict(color=FEW_PALETTE["positive"], size=12, symbol="circle"),
        text=["Min Variance"], textposition="top right",
        textfont=dict(size=9, color=FEW_PALETTE["positive"]),
    ))

    # Max Sharpe
    ms = max_sharpe_weights(returns_df, risk_free_rate=risk_free_rate, freq=freq)
    fig.add_trace(go.Scatter(
        x=[ms.expected_risk], y=[ms.expected_return],
        mode="markers+text",
        marker=dict(color=FEW_PALETTE["secondary"], size=12, symbol="diamond"),
        text=[f"Max Sharpe ({ms.sharpe:.2f})"], textposition="top right",
        textfont=dict(size=9, color=FEW_PALETTE["secondary"]),
    ))

    fig.update_layout(
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Return",
    )
    return fig


def plot_weight_comparison(
    weights_dict: dict[str, dict[str, float]],
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Grouped bar chart comparing portfolio weights across methods.

    Args:
        weights_dict: Dict mapping method name to asset-weight dict.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    methods = list(weights_dict.keys())
    all_assets: list[str] = []
    for w in weights_dict.values():
        for a in w:
            if a not in all_assets:
                all_assets.append(a)

    colours = [
        FEW_PALETTE["primary"], FEW_PALETTE["secondary"],
        FEW_PALETTE["positive"], FEW_PALETTE["negative"],
        FEW_PALETTE["neutral"], FEW_PALETTE["grey_mid"],
    ]

    fig = few_figure(title="Weight Comparison", width=width, height=height)

    for i, method in enumerate(methods):
        w = weights_dict[method]
        vals = [w.get(a, 0.0) for a in all_assets]
        fig.add_trace(go.Bar(
            x=all_assets, y=vals, name=method,
            marker_color=colours[i % len(colours)],
            text=[f"{v:.1%}" for v in vals],
            textposition="outside",
            textfont=dict(size=8),
        ))

    fig.update_layout(
        barmode="group",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Asset",
        yaxis_title="Weight",
    )
    return fig


def plot_monthly_returns(
    monthly_table: pd.DataFrame,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Monthly returns heatmap with hover values.

    Args:
        monthly_table: DataFrame with years as index, months as columns,
            and return values as data.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    if monthly_table.empty:
        fig = few_figure(title="Monthly Returns")
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="Insufficient data", showarrow=False,
        )
        return fig

    data = monthly_table.values
    vmax = float(np.nanmax(np.abs(data))) if not np.all(np.isnan(data)) else 0.1

    text = [
        [f"{data[i, j]:.1%}" if not np.isnan(data[i, j]) else ""
         for j in range(data.shape[1])]
        for i in range(data.shape[0])
    ]

    fig = few_figure(
        title="Monthly Returns",
        width=width or max(600, monthly_table.shape[1] * 55),
        height=height or max(300, monthly_table.shape[0] * 35 + 80),
    )

    fig.add_trace(go.Heatmap(
        z=data,
        x=[str(c) for c in monthly_table.columns],
        y=[str(i) for i in monthly_table.index],
        zmin=-vmax, zmax=vmax,
        colorscale="RdBu_r",
        text=text, texttemplate="%{text}",
        textfont=dict(size=8),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>",
        colorbar=dict(title="Return", thickness=10, len=0.8),
    ))

    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def plot_hrp_dendrogram(
    returns_df: pd.DataFrame,
    method: str = "ward",
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """HRP dendrogram using scipy linkage and plotly figure_factory.

    Args:
        returns_df: Asset returns DataFrame.
        method: Linkage method (e.g. ``ward``, ``single``, ``complete``).
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    corr = returns_df.corr()
    dist = np.sqrt(0.5 * (1 - corr))
    # Convert distance matrix to condensed form
    n = len(dist)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(dist.iloc[i, j])
    condensed_arr = np.array(condensed)

    linkage(condensed_arr, method=method)
    labels = list(returns_df.columns)

    fig = ff.create_dendrogram(
        X=dist.values,
        labels=labels,
        linkagefun=lambda x: linkage(x, method=method),
        color_threshold=0.7,
    )

    fig.update_layout(
        template=FEW_TEMPLATE,
        title="HRP Dendrogram",
        width=width or max(600, n * 50),
        height=height or 400,
        xaxis_title="Asset",
        yaxis_title="Distance",
        showlegend=False,
        dragmode=False,
    )

    # Apply Few colour to all dendrogram lines
    for trace in fig.data:
        trace.update(line=dict(color=FEW_PALETTE["primary"], width=1.5))

    return fig
