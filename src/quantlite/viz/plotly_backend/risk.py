"""Risk visualisation charts (Plotly backend).

Plotly equivalents of every chart in ``quantlite.viz.risk``, following
Stephen Few's principles identically to the matplotlib theme.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from ...core.types import GPDFit
from ...risk.metrics import cvar, max_drawdown_duration, value_at_risk
from .theme import FEW_PALETTE, FEW_TEMPLATE, few_figure

__all__ = [
    "plot_var_comparison",
    "plot_return_distribution",
    "plot_drawdown",
    "plot_qq",
    "plot_gpd_fit",
    "plot_return_level",
    "plot_hill",
    "plot_risk_bullet",
]


def plot_var_comparison(
    returns: np.ndarray | Any,
    alpha: float = 0.05,
    methods: list[str] | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Grouped bar chart comparing VaR estimation methods.

    Args:
        returns: Simple periodic returns.
        alpha: Significance level.
        methods: VaR method names. Defaults to historical, parametric,
            and Cornish-Fisher.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]

    if methods is None:
        methods = ["historical", "parametric", "cornish-fisher"]

    var_values: list[float] = []
    cvar_values: list[float] = []
    for m in methods:
        var_values.append(abs(value_at_risk(arr, alpha=alpha, method=m)))
        cvar_values.append(abs(cvar(arr, alpha=alpha)))

    fig = few_figure(title="VaR Comparison by Method", width=width, height=height)
    fig.add_trace(go.Bar(
        x=methods, y=var_values, name="VaR",
        marker_color=FEW_PALETTE["primary"],
        text=[f"{v:.4f}" for v in var_values],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=methods, y=cvar_values, name="CVaR",
        marker_color=FEW_PALETTE["negative"],
        text=[f"{v:.4f}" for v in cvar_values],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Method",
        yaxis_title="Loss (absolute)",
    )
    return fig


def plot_return_distribution(
    returns: np.ndarray | Any,
    gpd_fit: GPDFit | None = None,
    bins: int = 80,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Histogram with KDE, normal overlay, and fat-tail highlighting.

    Args:
        returns: Simple periodic returns.
        gpd_fit: Optional fitted GPD for tail overlay.
        bins: Number of histogram bins.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]

    fig = few_figure(title="Return Distribution with Tail Analysis", width=width, height=height)

    # Histogram
    fig.add_trace(go.Histogram(
        x=arr, nbinsx=bins, histnorm="probability density",
        marker_color=FEW_PALETTE["grey_light"],
        marker_line_color=FEW_PALETTE["grey_mid"],
        marker_line_width=0.5,
        opacity=0.6,
        showlegend=False,
    ))

    # Normal overlay
    x_range = np.linspace(float(arr.min()), float(arr.max()), 300)
    mu, sigma = float(np.mean(arr)), float(np.std(arr, ddof=1))
    normal_pdf = stats.norm.pdf(x_range, mu, sigma)
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_pdf, mode="lines",
        line=dict(color=FEW_PALETTE["grey_mid"], width=1.5, dash="dash"),
        name="Normal",
    ))

    # GPD tail overlay
    if gpd_fit is not None:
        threshold = -gpd_fit.threshold
        tail_x = x_range[x_range < threshold]
        if len(tail_x) > 0:
            losses = -tail_x - gpd_fit.threshold
            tail_pdf = stats.genpareto.pdf(losses, gpd_fit.shape, scale=gpd_fit.scale)
            zeta = gpd_fit.n_exceedances / gpd_fit.n_total
            fig.add_trace(go.Scatter(
                x=tail_x, y=tail_pdf * zeta, mode="lines",
                line=dict(color=FEW_PALETTE["primary"], width=2),
                name="GPD tail",
            ))

    # VaR and CVaR lines
    if len(arr) >= 2:
        var_95 = value_at_risk(arr, alpha=0.05)
        cvar_95 = cvar(arr, alpha=0.05)
        fig.add_vline(x=var_95, line_color=FEW_PALETTE["secondary"], line_width=1.5,
                      annotation_text=f"VaR 95%: {var_95:.4f}",
                      annotation_position="top left")
        fig.add_vline(x=cvar_95, line_color=FEW_PALETTE["negative"], line_width=1.5,
                      annotation_text=f"CVaR 95%: {cvar_95:.4f}",
                      annotation_position="top left")

    fig.update_layout(
        xaxis_title="Return",
        yaxis_title="Density",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_drawdown(
    returns: np.ndarray | Any,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Area chart of drawdowns with max drawdown highlighted.

    Args:
        returns: Simple periodic returns.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]

    cum = np.cumprod(1 + arr)
    roll_max = np.maximum.accumulate(cum)
    drawdowns = (cum - roll_max) / roll_max

    fig = few_figure(title="Underwater Chart", width=width, height=height)
    fig.add_trace(go.Scatter(
        x=list(range(len(drawdowns))), y=drawdowns,
        fill="tozeroy",
        fillcolor="rgba(225, 87, 89, 0.35)",
        line=dict(color=FEW_PALETTE["negative"], width=1),
        hovertemplate="Period: %{x}<br>Drawdown: %{y:.2%}<extra></extra>",
    ))

    # Annotate max drawdown
    dd_info = max_drawdown_duration(arr)
    fig.add_annotation(
        x=dd_info.end_idx, y=dd_info.max_drawdown,
        text=f"Max DD: {dd_info.max_drawdown:.2%}<br>Duration: {dd_info.duration} periods",
        showarrow=True, arrowhead=2, arrowcolor=FEW_PALETTE["grey_mid"],
        font=dict(size=9, color=FEW_PALETTE["grey_dark"]),
    )

    fig.update_layout(xaxis_title="Period", yaxis_title="Drawdown")
    return fig


def plot_qq(
    returns: np.ndarray | Any,
    dist: str = "norm",
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """QQ plot with reference line.

    Args:
        returns: Simple periodic returns.
        dist: Theoretical distribution name (scipy convention).
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]

    theoretical, sample = stats.probplot(arr, dist=dist)[:2]

    fig = few_figure(title=f"QQ Plot ({dist})", width=width, height=height)
    fig.add_trace(go.Scatter(
        x=theoretical[0], y=theoretical[1],
        mode="markers",
        marker=dict(color=FEW_PALETTE["primary"], size=4, opacity=0.6),
        hovertemplate="Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>",
    ))

    # Reference line
    x_min, x_max = float(theoretical[0].min()), float(theoretical[0].max())
    slope, intercept = sample[0], sample[1]
    fig.add_trace(go.Scatter(
        x=[x_min, x_max],
        y=[slope * x_min + intercept, slope * x_max + intercept],
        mode="lines",
        line=dict(color=FEW_PALETTE["negative"], width=1.5, dash="dash"),
    ))

    fig.update_layout(xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
    return fig


def plot_gpd_fit(
    returns: np.ndarray | Any,
    gpd_fit: GPDFit,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """GPD tail fit vs empirical distribution.

    Args:
        returns: Simple periodic returns.
        gpd_fit: Fitted GPD from ``fit_gpd``.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    losses = -arr
    exceedances = losses[losses > gpd_fit.threshold] - gpd_fit.threshold
    exceedances_sorted = np.sort(exceedances)

    n_exc = len(exceedances_sorted)
    empirical_sf = np.arange(n_exc, 0, -1) / n_exc

    fitted_sf = stats.genpareto.sf(exceedances_sorted, gpd_fit.shape, scale=gpd_fit.scale)

    fig = few_figure(title="GPD Tail Fit", width=width, height=height)
    fig.add_trace(go.Scatter(
        x=exceedances_sorted, y=empirical_sf,
        mode="markers",
        marker=dict(color=FEW_PALETTE["grey_mid"], size=4),
        name="Empirical",
    ))
    fig.add_trace(go.Scatter(
        x=exceedances_sorted, y=fitted_sf,
        mode="lines",
        line=dict(color=FEW_PALETTE["primary"], width=2),
        name="GPD fit",
    ))

    fig.update_layout(
        xaxis_title="Excess over threshold",
        yaxis_title="Survival probability",
        yaxis_type="log",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_return_level(
    gpd_fit: GPDFit,
    max_period: int = 10000,
    n_points: int = 50,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Return level plot with confidence bands.

    Args:
        gpd_fit: Fitted GPD from ``fit_gpd``.
        max_period: Maximum return period to plot.
        n_points: Number of points on the curve.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    from ...risk.evt import return_level as calc_return_level

    periods = np.logspace(1, np.log10(max_period), n_points)
    levels = [calc_return_level(gpd_fit, rp) for rp in periods]

    se_factor = gpd_fit.scale / np.sqrt(gpd_fit.n_exceedances)
    upper = [lv + 1.96 * se_factor for lv in levels]
    lower = [lv - 1.96 * se_factor for lv in levels]

    fig = few_figure(title="Return Level Plot", width=width, height=height)

    # Confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([periods, periods[::-1]]).tolist(),
        y=(upper + lower[::-1]),
        fill="toself",
        fillcolor="rgba(78, 121, 167, 0.15)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=periods.tolist(), y=levels,
        mode="lines",
        line=dict(color=FEW_PALETTE["primary"], width=2),
        hovertemplate="Period: %{x:.0f}<br>Level: %{y:.4f}<extra></extra>",
    ))

    # Key return levels
    for rp_label in [100, 1000, 5000]:
        if rp_label <= max_period:
            rl = calc_return_level(gpd_fit, rp_label)
            fig.add_trace(go.Scatter(
                x=[rp_label], y=[rl],
                mode="markers+text",
                marker=dict(color=FEW_PALETTE["negative"], size=7),
                text=[f"1-in-{rp_label}: {rl:.4f}"],
                textposition="top right",
                textfont=dict(size=9, color=FEW_PALETTE["negative"]),
                showlegend=False,
            ))

    fig.update_layout(
        xaxis_title="Return period (observations)",
        yaxis_title="Estimated loss",
        xaxis_type="log",
    )
    return fig


def plot_hill(
    returns: np.ndarray | Any,
    max_k: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Hill plot for tail index estimation.

    Args:
        returns: Simple periodic returns.
        max_k: Maximum number of order statistics. Defaults to n // 4.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    losses = np.sort(-arr)[::-1]  # descending

    n = len(losses)
    if max_k is None:
        max_k = max(n // 4, 10)

    k_values = list(range(2, min(max_k + 1, n)))
    hill_estimates: list[float] = []
    for k in k_values:
        log_ratios = np.log(losses[:k]) - np.log(losses[k])
        hill_estimates.append(float(np.mean(log_ratios)))

    fig = few_figure(title="Hill Plot (Tail Index)", width=width, height=height)
    fig.add_trace(go.Scatter(
        x=k_values, y=hill_estimates,
        mode="lines",
        line=dict(color=FEW_PALETTE["primary"], width=1.5),
        hovertemplate="k: %{x}<br>ξ estimate: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Number of order statistics (k)",
        yaxis_title="Hill estimator (ξ)",
    )
    return fig


def plot_risk_bullet(
    metrics: dict[str, dict[str, float]],
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Bullet graphs for risk metrics (Sortino, Calmar, Omega, etc.).

    Args:
        metrics: Dict mapping metric name to ``{"value": float,
            "target": float, "ranges": [poor, satisfactory, good]}``.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A Plotly ``Figure``.
    """
    names = list(metrics.keys())
    n = len(names)

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.15 / max(n, 1),
    )
    fig.update_layout(
        template=FEW_TEMPLATE,
        title="Risk Metrics",
        height=height or max(200, n * 80),
        width=width,
        showlegend=False,
        dragmode=False,
    )

    greys = ["#DDDDDD", "#BBBBBB", "#999999"]

    for i, name in enumerate(names, 1):
        m = metrics[name]
        value = m["value"]
        target = m["target"]
        ranges = sorted(m["ranges"], reverse=True)

        # Background ranges
        for j, r in enumerate(ranges):
            fig.add_trace(go.Bar(
                x=[r], y=[name], orientation="h",
                marker_color=greys[j], showlegend=False,
                width=0.6,
            ), row=i, col=1)

        # Value bar
        fig.add_trace(go.Bar(
            x=[value], y=[name], orientation="h",
            marker_color=FEW_PALETTE["grey_dark"],
            width=0.25, showlegend=False,
        ), row=i, col=1)

        # Target marker
        fig.add_shape(
            type="line", x0=target, x1=target, y0=-0.3, y1=0.3,
            line=dict(color=FEW_PALETTE["grey_dark"], width=2),
            row=i, col=1,
        )

        fig.update_xaxes(range=[0, max(ranges) * 1.05], row=i, col=1)
        fig.update_yaxes(showticklabels=True, row=i, col=1)

    return fig
