"""Basic visualisation utilities for time series and equity curves."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "plot_time_series",
    "plot_ohlc",
    "plot_return_distribution",
    "plot_equity_curve",
    "plot_multiple_equity_curves",
]


def plot_time_series(
    data: pd.Series | pd.DataFrame,
    title: str = "Time Series",
    indicators: dict[str, pd.Series] | None = None,
    figsize: tuple[float, float] = (10, 5),
    grid: bool = True,
) -> None:
    """Plot one or more time series.

    Args:
        data: Series or DataFrame to plot.
        title: Chart title.
        indicators: Optional dict of overlay series.
        figsize: Figure size in inches.
        grid: Whether to show gridlines.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame("Main Series")

    plt.figure(figsize=figsize)
    for col in data.columns:
        plt.plot(data.index, data[col], label=col)

    if indicators:
        for label, series in indicators.items():
            plt.plot(series.index, series.values, label=label, linestyle="--")

    plt.title(title)
    plt.xlabel("Date / Index")
    plt.ylabel("Value")
    if grid:
        plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


def plot_ohlc(
    df: pd.DataFrame,
    title: str = "Candlestick Chart",
    type: str = "candle",
    volume: bool = True,
    style: str = "yahoo",
) -> None:
    """Plot an OHLC candlestick chart using mplfinance.

    Args:
        df: DataFrame with Open, High, Low, Close columns and DatetimeIndex.
        title: Chart title.
        type: Chart type (``"candle"``, ``"ohlc"``, etc.).
        volume: Whether to show volume subplot.
        style: mplfinance style name.

    Raises:
        ValueError: If required columns are missing.
    """
    if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        raise ValueError("DataFrame must have columns: Open, High, Low, Close")

    mpf.plot(
        df,
        type=type,
        volume=volume if "Volume" in df.columns else False,
        style=style,
        title=title,
    )


def plot_return_distribution(
    returns: pd.Series | list[float] | np.ndarray,
    title: str = "Return Distribution",
    bins: int = 50,
    figsize: tuple[float, float] = (8, 4),
) -> None:
    """Plot a histogram of returns with a KDE overlay.

    Args:
        returns: Return series.
        title: Chart title.
        bins: Number of histogram bins.
        figsize: Figure size.
    """
    returns = pd.Series(returns).dropna()
    plt.figure(figsize=figsize)
    plt.hist(returns, bins=bins, density=True, alpha=0.6, color="g", label="Histogram")
    returns.plot(kind="kde", label="KDE", color="black")
    plt.title(title)
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_equity_curve(
    equity_series: pd.Series | list[float],
    drawdowns: bool = False,
    figsize: tuple[float, float] = (10, 5),
) -> None:
    """Plot an equity curve with optional drawdown shading.

    Args:
        equity_series: Portfolio value series.
        drawdowns: Whether to shade drawdown regions.
        figsize: Figure size.
    """
    if not isinstance(equity_series, pd.Series):
        equity_series = pd.Series(equity_series)

    plt.figure(figsize=figsize)
    plt.plot(equity_series.index, equity_series.values, label="Equity Curve")
    plt.title("Equity Curve")
    plt.xlabel("Date / Index")
    plt.ylabel("Value")
    plt.grid(alpha=0.3)

    if drawdowns:
        roll_max = equity_series.cummax()
        dd = (equity_series - roll_max) / roll_max
        dd_mask = dd < 0
        plt.fill_between(
            equity_series.index, equity_series.values, roll_max,
            where=dd_mask, color="red", alpha=0.2, label="Drawdown",
        )

    plt.legend()
    plt.show()


def plot_multiple_equity_curves(
    curves_dict: dict[str, pd.Series],
    title: str = "Multiple Equity Curves",
    rolling_sharpe: bool = False,
    window: int = 30,
    figsize: tuple[float, float] = (10, 5),
) -> None:
    """Plot multiple equity curves, optionally with rolling Sharpe subplots.

    Args:
        curves_dict: Mapping of label to equity Series.
        title: Chart title.
        rolling_sharpe: Whether to add a rolling Sharpe subplot.
        window: Rolling window size for Sharpe calculation.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)

    n_subplots = 2 if rolling_sharpe else 1
    ax1 = plt.subplot(n_subplots, 1, 1)
    ax1.set_title(title)

    for label, eq in curves_dict.items():
        eq = eq.sort_index()
        ax1.plot(eq.index, eq.values, label=label)

    ax1.set_xlabel("Date / Index")
    ax1.set_ylabel("Equity Value")
    ax1.grid(alpha=0.3)
    ax1.legend()

    if rolling_sharpe:
        ax2 = plt.subplot(n_subplots, 1, 2)
        ax2.set_title("Rolling Sharpe")
        for label, eq in curves_dict.items():
            eq = eq.sort_index()
            daily_returns = eq.pct_change().dropna()
            roll_sharpe = _rolling_sharpe(daily_returns, window=window)
            ax2.plot(roll_sharpe.index, roll_sharpe.values, label=f"{label} RS")
        ax2.set_xlabel("Date / Index")
        ax2.set_ylabel("Sharpe")
        ax2.grid(alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    plt.show()


def _rolling_sharpe(returns: pd.Series, window: int = 30) -> pd.Series:
    """Compute rolling Sharpe ratio (assumes 0% risk-free rate)."""
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    return roll_mean / (roll_std + 1e-9)
