"""Individual report sections for tearsheet generation.

Each function accepts a backtest result (or its components) and returns
an HTML string fragment. Sections follow Stephen Few's visual design
principles: minimal chartjunk, high data-ink ratio, and purposeful colour.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import pandas as pd

from ..metrics import (
    annualised_return,
    annualised_volatility,
    max_drawdown,
    sharpe_ratio,
)
from ..risk.metrics import calmar_ratio, cvar, sortino_ratio, value_at_risk

__all__ = [
    "executive_summary_section",
    "risk_section",
    "drawdown_section",
    "monthly_returns_section",
    "rolling_section",
    "regime_section",
    "factor_section",
    "stress_section",
]


def _returns_from_portfolio(portfolio_value: pd.Series) -> np.ndarray:
    """Compute simple returns from a portfolio value series.

    Args:
        portfolio_value: Time series of portfolio values.

    Returns:
        Array of simple returns (length n-1).
    """
    vals = np.asarray(portfolio_value, dtype=float)
    if len(vals) < 2:
        return np.array([], dtype=float)
    return vals[1:] / vals[:-1] - 1


def _fmt_pct(value: float) -> str:
    """Format a float as a percentage string.

    Args:
        value: Value to format (e.g. 0.15 becomes '15.00%').

    Returns:
        Formatted percentage string.
    """
    if np.isnan(value) or np.isinf(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def _fmt_ratio(value: float) -> str:
    """Format a ratio to two decimal places.

    Args:
        value: Ratio value.

    Returns:
        Formatted string.
    """
    if np.isnan(value) or np.isinf(value):
        return "N/A"
    return f"{value:.2f}"


def _matplotlib_to_base64(fig: Any) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string.

    Args:
        fig: A matplotlib Figure object.

    Returns:
        Base64-encoded PNG data URI.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    return f"data:image/png;base64,{encoded}"


def _make_chart_img(fig: Any) -> str:
    """Create an HTML img tag from a matplotlib figure.

    Args:
        fig: A matplotlib Figure object.

    Returns:
        HTML img element string.
    """
    import matplotlib.pyplot as plt

    src = _matplotlib_to_base64(fig)
    plt.close(fig)
    return f'<img src="{src}" style="max-width:100%;height:auto;" />'


def executive_summary_section(
    portfolio_value: pd.Series,
    title: str = "Executive Summary",
) -> str:
    """Generate the executive summary section with key performance metrics.

    Args:
        portfolio_value: Time series of portfolio values.
        title: Section heading.

    Returns:
        HTML string for the executive summary section.
    """
    returns = _returns_from_portfolio(portfolio_value)
    if len(returns) == 0:
        return f"<h2>{title}</h2><p>Insufficient data for metrics.</p>"

    total_ret = float((1 + returns).prod() - 1)
    ann_ret = annualised_return(returns)
    ann_vol = annualised_volatility(returns)
    sr = sharpe_ratio(returns)
    sort_r = sortino_ratio(returns)
    cal_r = calmar_ratio(returns)
    mdd = max_drawdown(returns)
    var_95 = value_at_risk(returns, alpha=0.05)
    cvar_95 = cvar(returns, alpha=0.05)

    rows = [
        ("Total Return", _fmt_pct(total_ret)),
        ("Annualised Return", _fmt_pct(ann_ret)),
        ("Annualised Volatility", _fmt_pct(ann_vol)),
        ("Sharpe Ratio", _fmt_ratio(sr)),
        ("Sortino Ratio", _fmt_ratio(sort_r)),
        ("Calmar Ratio", _fmt_ratio(cal_r)),
        ("Maximum Drawdown", _fmt_pct(mdd)),
        ("VaR (95%)", _fmt_pct(var_95)),
        ("CVaR (95%)", _fmt_pct(cvar_95)),
    ]

    table_rows = "\n".join(
        f"<tr><td>{name}</td><td>{val}</td></tr>" for name, val in rows
    )

    return f"""<div class="section" id="summary">
<h2>{title}</h2>
<table class="metrics-table">
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>"""


def risk_section(
    portfolio_value: pd.Series,
    title: str = "Risk Metrics",
) -> str:
    """Generate the risk metrics section with distribution chart.

    Args:
        portfolio_value: Time series of portfolio values.
        title: Section heading.

    Returns:
        HTML string for the risk section.
    """
    import matplotlib.pyplot as plt

    returns = _returns_from_portfolio(portfolio_value)
    if len(returns) == 0:
        return f"<h2>{title}</h2><p>Insufficient data.</p>"

    var_95 = value_at_risk(returns, alpha=0.05)
    cvar_95 = cvar(returns, alpha=0.05)
    var_99 = value_at_risk(returns, alpha=0.01)
    cvar_99 = cvar(returns, alpha=0.01)

    # Distribution chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(returns, bins=50, color="#4a7c9b", edgecolor="white",
            alpha=0.8, density=True)
    ax.axvline(var_95, color="#c44e52", linestyle="--", linewidth=1.5,
               label=f"VaR 95% ({_fmt_pct(var_95)})")
    ax.axvline(cvar_95, color="#dd8452", linestyle="--", linewidth=1.5,
               label=f"CVaR 95% ({_fmt_pct(cvar_95)})")
    ax.set_xlabel("Return", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Return Distribution", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    chart_html = _make_chart_img(fig)

    # VaR/CVaR comparison table
    table = f"""<table class="metrics-table">
<thead><tr><th>Level</th><th>VaR</th><th>CVaR</th></tr></thead>
<tbody>
<tr><td>95%</td><td>{_fmt_pct(var_95)}</td><td>{_fmt_pct(cvar_95)}</td></tr>
<tr><td>99%</td><td>{_fmt_pct(var_99)}</td><td>{_fmt_pct(cvar_99)}</td></tr>
</tbody>
</table>"""

    return f"""<div class="section" id="risk">
<h2>{title}</h2>
{table}
{chart_html}
</div>"""


def _find_top_drawdowns(
    portfolio_value: pd.Series,
    n: int = 5,
) -> list[dict[str, Any]]:
    """Find the top N drawdowns by depth.

    Args:
        portfolio_value: Time series of portfolio values.
        n: Number of drawdowns to return.

    Returns:
        List of dicts with keys: depth, start, end, duration, recovery.
    """
    vals = np.asarray(portfolio_value, dtype=float)
    if len(vals) < 2:
        return []

    cummax = np.maximum.accumulate(vals)
    dd = vals / cummax - 1

    drawdowns: list[dict[str, Any]] = []
    in_dd = False
    start = 0

    for i in range(len(dd)):
        if dd[i] < 0 and not in_dd:
            in_dd = True
            start = i
        elif dd[i] >= 0 and in_dd:
            in_dd = False
            trough_idx = int(start + np.argmin(dd[start:i]))
            drawdowns.append({
                "depth": float(dd[trough_idx]),
                "start": start,
                "end": i,
                "duration": i - start,
                "recovery": i - trough_idx,
            })

    if in_dd:
        trough_idx = int(start + np.argmin(dd[start:]))
        drawdowns.append({
            "depth": float(dd[trough_idx]),
            "start": start,
            "end": len(dd) - 1,
            "duration": len(dd) - start,
            "recovery": None,
        })

    drawdowns.sort(key=lambda x: x["depth"])
    return drawdowns[:n]


def drawdown_section(
    portfolio_value: pd.Series,
    title: str = "Drawdown Analysis",
) -> str:
    """Generate the drawdown analysis section with underwater chart.

    Args:
        portfolio_value: Time series of portfolio values.
        title: Section heading.

    Returns:
        HTML string for the drawdown section.
    """
    import matplotlib.pyplot as plt

    vals = np.asarray(portfolio_value, dtype=float)
    if len(vals) < 2:
        return f"<h2>{title}</h2><p>Insufficient data.</p>"

    cummax = np.maximum.accumulate(vals)
    dd = vals / cummax - 1

    # Underwater chart
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.fill_between(range(len(dd)), dd, 0, color="#c44e52", alpha=0.6)
    ax.plot(dd, color="#c44e52", linewidth=0.8)
    ax.set_ylabel("Drawdown", fontsize=10)
    ax.set_title("Underwater Chart", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    chart_html = _make_chart_img(fig)

    # Top drawdowns table
    top_dds = _find_top_drawdowns(portfolio_value)
    dd_rows = ""
    for i, d in enumerate(top_dds, 1):
        rec = str(d["recovery"]) + " periods" if d["recovery"] is not None else "Ongoing"
        dd_rows += (
            f"<tr><td>{i}</td><td>{_fmt_pct(d['depth'])}</td>"
            f"<td>{d['duration']} periods</td><td>{rec}</td></tr>\n"
        )

    table = f"""<table class="metrics-table">
<thead><tr><th>#</th><th>Depth</th><th>Duration</th><th>Recovery</th></tr></thead>
<tbody>
{dd_rows}
</tbody>
</table>"""

    return f"""<div class="section" id="drawdown">
<h2>{title}</h2>
{chart_html}
<h3>Top Drawdowns</h3>
{table}
</div>"""


def monthly_returns_section(
    portfolio_value: pd.Series,
    title: str = "Monthly Returns",
) -> str:
    """Generate a monthly returns heatmap section.

    Args:
        portfolio_value: Time series of portfolio values with a DatetimeIndex.
        title: Section heading.

    Returns:
        HTML string for the monthly returns section.
    """
    import matplotlib.pyplot as plt

    if not isinstance(portfolio_value.index, pd.DatetimeIndex):
        return f'<div class="section" id="monthly"><h2>{title}</h2><p>DatetimeIndex required for monthly returns.</p></div>'

    # Resample to monthly returns
    monthly = portfolio_value.resample("ME").last().pct_change().dropna()
    if len(monthly) == 0:
        return f'<div class="section" id="monthly"><h2>{title}</h2><p>Insufficient data.</p></div>'

    # Build pivot table
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = monthly_df.pivot_table(
        index="year", columns="month", values="return", aggfunc="first"
    )
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(10, max(2, len(pivot) * 0.5)))
    data = pivot.values * 100
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto",
                   vmin=-np.nanmax(np.abs(data)),
                   vmax=np.nanmax(np.abs(data)))

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=8, color="black" if abs(val) < np.nanmax(np.abs(data)) * 0.6 else "white")

    ax.set_title("Monthly Returns (%)", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    chart_html = _make_chart_img(fig)

    return f"""<div class="section" id="monthly">
<h2>{title}</h2>
{chart_html}
</div>"""


def rolling_section(
    portfolio_value: pd.Series,
    window: int = 63,
    title: str = "Rolling Statistics",
) -> str:
    """Generate rolling statistics section (Sharpe, volatility).

    Args:
        portfolio_value: Time series of portfolio values.
        window: Rolling window size in periods.
        title: Section heading.

    Returns:
        HTML string for the rolling statistics section.
    """
    import matplotlib.pyplot as plt

    returns = _returns_from_portfolio(portfolio_value)
    if len(returns) < window:
        return f'<div class="section" id="rolling"><h2>{title}</h2><p>Insufficient data for rolling window of {window}.</p></div>'

    ret_series = pd.Series(returns)

    rolling_vol = ret_series.rolling(window).std() * np.sqrt(252)
    rolling_mean = ret_series.rolling(window).mean() * 252
    rolling_sharpe = rolling_mean / rolling_vol

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(rolling_sharpe.values, color="#4a7c9b", linewidth=1)
    axes[0].axhline(0, color="grey", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel("Sharpe Ratio", fontsize=10)
    axes[0].set_title(f"Rolling {window}-Period Sharpe Ratio", fontsize=11)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(rolling_vol.values, color="#dd8452", linewidth=1)
    axes[1].set_ylabel("Volatility (ann.)", fontsize=10)
    axes[1].set_title(f"Rolling {window}-Period Volatility", fontsize=11)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    chart_html = _make_chart_img(fig)

    return f"""<div class="section" id="rolling">
<h2>{title}</h2>
{chart_html}
</div>"""


def regime_section(
    portfolio_value: pd.Series,
    regime_labels: np.ndarray | None = None,
    title: str = "Regime Analysis",
) -> str:
    """Generate regime analysis section with per-regime metrics.

    Args:
        portfolio_value: Time series of portfolio values.
        regime_labels: Array of regime labels per period (same length as returns).
        title: Section heading.

    Returns:
        HTML string for the regime analysis section.
    """
    if regime_labels is None:
        return f'<div class="section" id="regimes"><h2>{title}</h2><p>No regime labels provided. Pass regime_labels to enable this section.</p></div>'

    returns = _returns_from_portfolio(portfolio_value)
    labels = np.asarray(regime_labels)

    # Trim to match returns length
    if len(labels) > len(returns):
        labels = labels[:len(returns)]
    elif len(labels) < len(returns):
        returns = returns[:len(labels)]

    unique_regimes = np.unique(labels)

    rows = ""
    for regime in unique_regimes:
        mask = labels == regime
        r = returns[mask]
        if len(r) == 0:
            continue
        rows += (
            f"<tr><td>Regime {regime}</td>"
            f"<td>{len(r)}</td>"
            f"<td>{_fmt_pct(float(np.mean(r) * 252))}</td>"
            f"<td>{_fmt_pct(float(np.std(r, ddof=1) * np.sqrt(252)))}</td>"
            f"<td>{_fmt_ratio(float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252)) if np.std(r, ddof=1) > 0 else float('nan'))}</td>"
            f"</tr>\n"
        )

    table = f"""<table class="metrics-table">
<thead><tr><th>Regime</th><th>Periods</th><th>Ann. Return</th><th>Ann. Vol</th><th>Sharpe</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>"""

    return f"""<div class="section" id="regimes">
<h2>{title}</h2>
{table}
</div>"""


def factor_section(title: str = "Factor Exposure") -> str:
    """Generate a placeholder factor exposure section.

    Args:
        title: Section heading.

    Returns:
        HTML string for the factor section placeholder.
    """
    return f"""<div class="section" id="factors">
<h2>{title}</h2>
<p class="placeholder">Factor analysis available in v0.8.</p>
</div>"""


def stress_section(title: str = "Stress Tests") -> str:
    """Generate a placeholder stress test section.

    Args:
        title: Section heading.

    Returns:
        HTML string for the stress test section placeholder.
    """
    return f"""<div class="section" id="stress">
<h2>{title}</h2>
<p class="placeholder">Stress testing scenarios available in v0.6.</p>
</div>"""
