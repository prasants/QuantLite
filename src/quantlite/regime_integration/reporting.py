"""Regime-aware reporting: tearsheets, attribution, and comparison tables.

Provides functions to generate comprehensive regime-breakdown reports
including equity curves, drawdowns, risk metrics, and performance
attribution per regime.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .risk import regime_risk_summary

__all__ = [
    "regime_tearsheet",
    "regime_performance_attribution",
    "regime_comparison_table",
]


def _max_drawdown(equity: np.ndarray) -> float:
    """Compute maximum drawdown from an equity curve."""
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    return float(np.min(dd))


def regime_tearsheet(
    returns: np.ndarray | pd.Series | list[float],
    regimes: np.ndarray | pd.Series | list[Any],
    benchmark: np.ndarray | pd.Series | None = None,
    initial_capital: float = 10000.0,
) -> dict[str, Any]:
    """Full tearsheet broken down by regime.

    Generates equity curves, drawdowns, risk metrics, and time spent
    in each regime.

    Args:
        returns: Simple periodic returns.
        regimes: Regime labels, same length as *returns*.
        benchmark: Optional benchmark returns for comparison.
        initial_capital: Starting value for the equity curve.

    Returns:
        Dictionary containing:
        - ``"equity_curve"``: pandas Series of portfolio values.
        - ``"drawdowns"``: pandas Series of drawdown percentages.
        - ``"regime_metrics"``: per-regime risk summary from
          :func:`regime_risk_summary`.
        - ``"time_in_regime"``: dict of regime to fraction of time.
        - ``"overall_metrics"``: dict of aggregate performance metrics.
        - ``"benchmark_metrics"``: benchmark stats if provided.
    """
    arr = np.asarray(returns, dtype=float)
    reg = np.asarray(regimes)
    if len(arr) != len(reg):
        raise ValueError("returns and regimes must have the same length")

    # Equity curve
    equity = initial_capital * np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / np.where(peak > 0, peak, 1.0)

    # Index
    idx = returns.index if isinstance(returns, pd.Series) else pd.RangeIndex(len(arr))
    equity_series = pd.Series(equity, index=idx)
    dd_series = pd.Series(drawdowns, index=idx)

    # Regime metrics
    regime_metrics = regime_risk_summary(arr, reg)

    # Time in each regime
    unique, counts = np.unique(reg, return_counts=True)
    total = len(reg)
    time_in_regime = {str(u): float(c) / total for u, c in zip(unique, counts)}

    # Overall metrics
    total_return = float(equity[-1] / initial_capital - 1.0) if len(equity) > 0 else 0.0
    ann_return = float((1.0 + total_return) ** (252.0 / max(len(arr), 1)) - 1.0)
    ann_vol = float(np.std(arr, ddof=1) * np.sqrt(252)) if len(arr) > 1 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    max_dd = _max_drawdown(equity) if len(equity) > 0 else 0.0

    overall = {
        "total_return": total_return,
        "annualised_return": ann_return,
        "annualised_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "n_observations": len(arr),
    }

    result: dict[str, Any] = {
        "equity_curve": equity_series,
        "drawdowns": dd_series,
        "regime_metrics": regime_metrics,
        "time_in_regime": time_in_regime,
        "overall_metrics": overall,
    }

    # Benchmark
    if benchmark is not None:
        bench_arr = np.asarray(benchmark, dtype=float)
        bench_equity = initial_capital * np.cumprod(1.0 + bench_arr)
        bench_total = (
            float(bench_equity[-1] / initial_capital - 1.0)
            if len(bench_equity) > 0 else 0.0
        )
        bench_ann = float((1.0 + bench_total) ** (252.0 / max(len(bench_arr), 1)) - 1.0)
        bench_vol = float(np.std(bench_arr, ddof=1) * np.sqrt(252)) if len(bench_arr) > 1 else 0.0
        result["benchmark_metrics"] = {
            "total_return": bench_total,
            "annualised_return": bench_ann,
            "annualised_volatility": bench_vol,
            "sharpe_ratio": bench_ann / bench_vol if bench_vol > 0 else 0.0,
            "max_drawdown": _max_drawdown(bench_equity) if len(bench_equity) > 0 else 0.0,
        }

    return result


def regime_performance_attribution(
    returns: np.ndarray | pd.Series | list[float],
    regimes: np.ndarray | pd.Series | list[Any],
) -> dict[str, dict[str, float]]:
    """Attribute total return to each regime period.

    For each regime, computes the cumulative return contributed,
    the fraction of total return, and the average per-period return.

    Args:
        returns: Simple periodic returns.
        regimes: Regime labels, same length as *returns*.

    Returns:
        Dictionary mapping regime labels (as strings) to a dict with
        keys ``"cumulative_return"``, ``"contribution_pct"``,
        ``"mean_return"``, ``"count"``.
    """
    arr = np.asarray(returns, dtype=float)
    reg = np.asarray(regimes)
    if len(arr) != len(reg):
        raise ValueError("returns and regimes must have the same length")

    total_cum = float(np.sum(arr))
    result: dict[str, dict[str, float]] = {}

    for label in np.unique(reg):
        mask = reg == label
        regime_returns = arr[mask]
        cum = float(np.sum(regime_returns))
        pct = (cum / total_cum * 100) if abs(total_cum) > 1e-12 else 0.0
        result[str(label)] = {
            "cumulative_return": cum,
            "contribution_pct": pct,
            "mean_return": float(np.mean(regime_returns)),
            "count": float(len(regime_returns)),
        }

    return result


def regime_comparison_table(
    returns: np.ndarray | pd.Series | list[float],
    regimes: np.ndarray | pd.Series | list[Any],
    alpha: float = 0.05,
) -> str:
    """Generate a markdown table comparing metrics across regimes.

    Args:
        returns: Simple periodic returns.
        regimes: Regime labels, same length as *returns*.
        alpha: Significance level for VaR and CVaR.

    Returns:
        Markdown-formatted table string.
    """
    summary = regime_risk_summary(returns, regimes, alpha=alpha)

    header = (
        "| Regime | Count | Ann. Vol "
        f"| VaR ({1 - alpha:.0%}) | CVaR ({1 - alpha:.0%}) "
        "| Skewness | Kurtosis |"
    )
    separator = "|--------|-------|----------|------------|-------------|----------|----------|"
    rows = [header, separator]

    for regime, metrics in sorted(summary.items()):
        row = "| {} | {:,.0f} | {:.2%} | {:.4f} | {:.4f} | {:.2f} | {:.2f} |".format(
            regime,
            metrics["count"],
            metrics["volatility"],
            metrics["var"],
            metrics["cvar"],
            metrics["skewness"],
            metrics["kurtosis"],
        )
        rows.append(row)

    return "\n".join(rows)
