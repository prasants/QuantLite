"""Post-backtest analysis: performance summaries, monthly tables, rolling metrics.

All functions accept a ``BacktestResult`` and return DataFrames or dicts
suitable for display and further analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "performance_summary",
    "monthly_returns_table",
    "rolling_metrics",
    "trade_analysis",
    "regime_attribution",
]


def performance_summary(result: Any) -> pd.DataFrame:
    """Produce a comprehensive performance metrics table.

    Args:
        result: A ``BacktestResult`` from the backtesting engine.

    Returns:
        Single-column DataFrame with metric names as the index.
    """
    m = result.metrics
    rows = {
        "Initial Capital": m.get("initial_capital", 0),
        "Final Value": m.get("final_value", 0),
        "Total Return": m.get("total_return", 0),
        "Annualised Return": m.get("annualised_return", 0),
        "Annualised Volatility": m.get("annualised_volatility", 0),
        "Sharpe Ratio": m.get("sharpe_ratio", 0),
        "Sortino Ratio": m.get("sortino_ratio", 0),
        "Calmar Ratio": m.get("calmar_ratio", 0),
        "Max Drawdown": m.get("max_drawdown", 0),
        "Max DD Duration": m.get("max_drawdown_duration", 0),
        "VaR (95%)": m.get("var_95", 0),
        "CVaR (95%)": m.get("cvar_95", 0),
        "Omega Ratio": m.get("omega_ratio", 0),
        "Tail Ratio": m.get("tail_ratio", 0),
        "Skewness": m.get("skewness", 0),
        "Kurtosis": m.get("kurtosis", 0),
        "Total Trades": m.get("n_trades", 0),
    }
    return pd.DataFrame({"Value": rows})


def monthly_returns_table(result: Any) -> pd.DataFrame:
    """Compute a month-by-year returns table.

    Args:
        result: A ``BacktestResult`` from the backtesting engine.

    Returns:
        DataFrame with years as rows (index) and months 1-12 as columns.
        Values are total returns for each month. Missing months are NaN.
    """
    pv = result.portfolio_value
    # Compute monthly returns from portfolio value
    monthly_pv = pv.resample("ME").last()
    monthly_rets = monthly_pv.pct_change()

    # Build year x month table
    if len(monthly_rets) == 0:
        return pd.DataFrame()

    table_data: dict[int, dict[int, float]] = {}
    for date, ret in monthly_rets.items():
        year = date.year
        month = date.month
        if year not in table_data:
            table_data[year] = {}
        table_data[year][month] = ret

    df = pd.DataFrame(table_data).T
    # Ensure columns 1-12
    for m in range(1, 13):
        if m not in df.columns:
            df[m] = np.nan
    df = df[list(range(1, 13))]
    df.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df.index.name = "Year"
    return df


def rolling_metrics(
    result: Any,
    window: int = 63,
    freq: int = 252,
) -> pd.DataFrame:
    """Compute rolling Sharpe, volatility, and drawdown.

    Args:
        result: A ``BacktestResult`` from the backtesting engine.
        window: Rolling window in periods.
        freq: Periods per year for annualisation.

    Returns:
        DataFrame with columns: rolling_sharpe, rolling_vol, rolling_drawdown.
    """
    pv = result.portfolio_value
    rets = pv.pct_change().fillna(0.0)

    rolling_vol = rets.rolling(window).std() * np.sqrt(freq)
    rolling_mean = rets.rolling(window).mean() * freq
    rolling_sharpe = rolling_mean / rolling_vol.replace(0, np.nan)

    # Rolling drawdown
    rolling_dd = pd.Series(0.0, index=pv.index)
    for i in range(window, len(pv)):
        window_pv = pv.iloc[i - window:i + 1]
        peak = window_pv.cummax()
        dd = (window_pv - peak) / peak
        rolling_dd.iloc[i] = dd.iloc[-1]

    return pd.DataFrame({
        "rolling_sharpe": rolling_sharpe,
        "rolling_vol": rolling_vol,
        "rolling_drawdown": rolling_dd,
    })


def trade_analysis(result: Any) -> dict[str, float]:
    """Analyse trade-level statistics.

    Args:
        result: A ``BacktestResult`` from the backtesting engine.

    Returns:
        Dict with win_rate, avg_win, avg_loss, profit_factor,
        avg_trade_cost, and total_cost.
    """
    trades = result.trades
    if not trades:
        return {
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "avg_trade_cost": 0.0,
            "total_cost": 0.0,
        }

    # Compute P&L per trade from weight changes
    pnls: list[float] = []
    costs: list[float] = []
    for trade in trades:
        delta_w = trade["new_weight"] - trade["old_weight"]
        pnls.append(delta_w)  # proxy: positive delta = going long
        costs.append(trade.get("cost", 0.0))

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate = len(wins) / len(pnls) if pnls else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    return {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_trade_cost": float(np.mean(costs)),
        "total_cost": float(np.sum(costs)),
    }


def regime_attribution(result: Any) -> pd.DataFrame:
    """Attribute portfolio P&L to market regimes.

    Args:
        result: A ``BacktestResult`` with regime_labels populated.

    Returns:
        DataFrame indexed by regime label with columns: total_return,
        mean_return, volatility, n_periods, pct_time.
    """
    if result.regime_labels is None:
        return pd.DataFrame()

    pv = result.portfolio_value
    rets = pv.pct_change().fillna(0.0)
    labels = np.asarray(result.regime_labels)

    if len(labels) != len(rets):
        # Trim to match
        min_len = min(len(labels), len(rets))
        labels = labels[:min_len]
        rets = rets.iloc[:min_len]

    unique_regimes = sorted(set(labels))
    rows: list[dict[str, float]] = []

    for regime in unique_regimes:
        mask = labels == regime
        regime_rets = rets.values[mask]
        n = int(mask.sum())
        total_ret = float(np.prod(1 + regime_rets) - 1) if n > 0 else 0.0
        mean_ret = float(np.mean(regime_rets)) if n > 0 else 0.0
        vol = float(np.std(regime_rets, ddof=1)) if n > 1 else 0.0

        rows.append({
            "regime": int(regime),
            "total_return": total_ret,
            "mean_return": mean_ret,
            "volatility": vol,
            "n_periods": float(n),
            "pct_time": float(n) / len(labels) if len(labels) > 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("regime")
    return df
