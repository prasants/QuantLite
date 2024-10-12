import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy.stats import norm

def plot_time_series(data, title="Time Series", indicators=None, figsize=(10, 5), grid=True):
    # (Same as before)
    if isinstance(data, pd.Series):
        data = data.to_frame("Main Series")

    plt.figure(figsize=figsize)
    for col in data.columns:
        plt.plot(data.index, data[col], label=col)

    if indicators:
        for label, series in indicators.items():
            plt.plot(series.index, series.values, label=label, linestyle='--')

    plt.title(title)
    plt.xlabel("Date / Index")
    plt.ylabel("Value")
    if grid:
        plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

def plot_ohlc(df, title="Candlestick Chart", type="candle", volume=True, style="yahoo"):
    # (Same as before)
    if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        raise ValueError("DataFrame must have columns: Open, High, Low, Close")

    mpf.plot(
        df, 
        type=type,
        volume=volume if "Volume" in df.columns else False,
        style=style,
        title=title
    )

def plot_return_distribution(returns, title="Return Distribution", bins=50, figsize=(8, 4)):
    # (Same as before)
    returns = pd.Series(returns).dropna()
    plt.figure(figsize=figsize)
    plt.hist(returns, bins=bins, density=True, alpha=0.6, color='g', label='Histogram')
    returns.plot(kind='kde', label='KDE', color='black')
    plt.title(title)
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_equity_curve(equity_series, drawdowns=False, figsize=(10,5)):
    # (Same as before)
    if not isinstance(equity_series, pd.Series):
        equity_series = pd.Series(equity_series)

    plt.figure(figsize=figsize)
    plt.plot(equity_series.index, equity_series.values, label='Equity Curve')
    plt.title("Equity Curve")
    plt.xlabel("Date / Index")
    plt.ylabel("Value")
    plt.grid(alpha=0.3)

    if drawdowns:
        roll_max = equity_series.cummax()
        dd = (equity_series - roll_max) / roll_max
        dd_mask = dd < 0
        plt.fill_between(equity_series.index, equity_series.values, roll_max, where=dd_mask, color='red', alpha=0.2, label='Drawdown')

    plt.legend()
    plt.show()

def plot_multiple_equity_curves(
    curves_dict,
    title="Multiple Equity Curves",
    rolling_sharpe=False,
    window=30,
    figsize=(10,5)
):
    """
    Plots multiple equity curves on the same figure. 
    Optionally shows their rolling Sharpe ratio in a sub-plot.

    Parameters
    ----------
    curves_dict : dict
        { "Label A": pd.Series, "Label B": pd.Series, ... }
        Each value should be an equity curve (e.g., from a backtest).
    rolling_sharpe : bool
        If True, create a second subplot showing rolling Sharpe for each curve.
    window : int
        Rolling window size (in days) for Sharpe.
    """
    # Convert all to same date index (if possible)
    plt.figure(figsize=figsize)

    n_subplots = 2 if rolling_sharpe else 1
    ax1 = plt.subplot(n_subplots,1,1)
    ax1.set_title(title)

    for label, eq in curves_dict.items():
        eq = eq.sort_index()
        ax1.plot(eq.index, eq.values, label=label)

    ax1.set_xlabel("Date / Index")
    ax1.set_ylabel("Equity Value")
    ax1.grid(alpha=0.3)
    ax1.legend()

    if rolling_sharpe:
        ax2 = plt.subplot(n_subplots,1,2)
        ax2.set_title("Rolling Sharpe")
        for label, eq in curves_dict.items():
            eq = eq.sort_index()
            # Compute daily returns
            daily_returns = eq.pct_change().dropna()
            roll_sharpe = _rolling_sharpe(daily_returns, window=window)
            ax2.plot(roll_sharpe.index, roll_sharpe.values, label=f"{label} RS")

        ax2.set_xlabel("Date / Index")
        ax2.set_ylabel("Sharpe")
        ax2.grid(alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    plt.show()

def _rolling_sharpe(returns, window=30):
    """
    Computes a rolling Sharpe ratio over a given window,
    assuming a 0% risk-free rate for simplicity.
    """
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    # avoid division by zero
    sharpe = roll_mean / (roll_std + 1e-9)
    return sharpe
