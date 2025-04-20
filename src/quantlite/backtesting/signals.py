"""Signal generation helpers for backtesting strategies.

Provides momentum, mean reversion, trend following, volatility targeting,
and regime filtering signals. All functions return DataFrames or Series
aligned with the input data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "momentum_signal",
    "mean_reversion_signal",
    "volatility_targeting",
    "trend_following",
    "regime_filter",
]


def momentum_signal(
    prices: pd.DataFrame | pd.Series,
    lookback: int = 60,
) -> pd.DataFrame | pd.Series:
    """Compute normalised momentum score per asset.

    Momentum is the percentage return over the lookback window,
    cross-sectionally normalised to z-scores when multiple assets
    are provided.

    Args:
        prices: Price data (DataFrame for multi-asset, Series for single).
        lookback: Number of periods for momentum calculation.

    Returns:
        Momentum signal, same shape as input. NaN for insufficient history.
    """
    raw_mom = prices.pct_change(lookback)

    if isinstance(raw_mom, pd.DataFrame) and raw_mom.shape[1] > 1:
        # Cross-sectional z-score
        mean = raw_mom.mean(axis=1)
        std = raw_mom.std(axis=1)
        std = std.replace(0, np.nan)
        return raw_mom.sub(mean, axis=0).div(std, axis=0)

    return raw_mom


def mean_reversion_signal(
    prices: pd.DataFrame | pd.Series,
    lookback: int = 20,
    z_threshold: float = 1.5,
) -> pd.DataFrame | pd.Series:
    """Z-score based mean reversion signal.

    Returns +1 when price is ``z_threshold`` standard deviations below
    the rolling mean, -1 when above, and 0 otherwise.

    Args:
        prices: Price data.
        lookback: Rolling window for mean and standard deviation.
        z_threshold: Number of standard deviations for signal activation.

    Returns:
        Signal values in {-1, 0, +1}, same shape as input.
    """
    rolling_mean = prices.rolling(lookback).mean()
    rolling_std = prices.rolling(lookback).std()

    z_score = (prices - rolling_mean) / rolling_std.replace(0, np.nan)

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns) if isinstance(prices, pd.DataFrame) else pd.Series(0.0, index=prices.index)

    if isinstance(z_score, pd.DataFrame):
        signal = signal.copy()
        signal[z_score < -z_threshold] = 1.0
        signal[z_score > z_threshold] = -1.0
    else:
        signal = signal.copy()
        signal[z_score < -z_threshold] = 1.0
        signal[z_score > z_threshold] = -1.0

    return signal


def volatility_targeting(
    returns: pd.DataFrame | pd.Series,
    target_vol: float = 0.10,
    lookback: int = 60,
    freq: int = 252,
) -> pd.DataFrame | pd.Series:
    """Scale exposure to target a specific annualised volatility.

    Computes rolling realised volatility and returns a scalar multiplier
    that would bring portfolio volatility to the target level.

    Args:
        returns: Return series (periodic).
        target_vol: Target annualised volatility.
        lookback: Rolling window for volatility estimation.
        freq: Periods per year for annualisation.

    Returns:
        Exposure scalar per period. Values > 1 indicate levering up;
        values < 1 indicate deleveraging.
    """
    rolling_vol = returns.rolling(lookback).std() * np.sqrt(freq)

    if isinstance(rolling_vol, pd.DataFrame):
        rolling_vol = rolling_vol.replace(0, np.nan)
    else:
        rolling_vol = rolling_vol.replace(0, np.nan)

    scalar = target_vol / rolling_vol
    # Cap leverage at 3x to prevent extreme scaling
    scalar = scalar.clip(upper=3.0) if isinstance(scalar, pd.DataFrame) else scalar.clip(upper=3.0)

    return scalar


def trend_following(
    prices: pd.DataFrame | pd.Series,
    fast_window: int = 20,
    slow_window: int = 60,
) -> pd.DataFrame | pd.Series:
    """Dual moving average crossover trend following signal.

    Returns +1 when the fast MA is above the slow MA (uptrend),
    -1 when below (downtrend).

    Args:
        prices: Price data.
        fast_window: Short moving average window.
        slow_window: Long moving average window.

    Returns:
        Signal in {-1, +1}, same shape as input. NaN until enough history.
    """
    fast_ma = prices.rolling(fast_window).mean()
    slow_ma = prices.rolling(slow_window).mean()

    signal = (fast_ma > slow_ma).astype(float) * 2 - 1
    # NaN where either MA is NaN
    mask = fast_ma.isna() | slow_ma.isna()
    if isinstance(signal, pd.DataFrame):
        signal[mask] = np.nan
    else:
        signal[mask] = np.nan

    return signal


def regime_filter(
    signal: pd.DataFrame | pd.Series,
    regimes: np.ndarray | pd.Series,
    allowed_regimes: list[int] | set[int],
) -> pd.DataFrame | pd.Series:
    """Zero out signal when in disallowed regimes.

    Args:
        signal: Trading signal to filter.
        regimes: Regime labels aligned with the signal index.
        allowed_regimes: Set of regime labels where trading is permitted.

    Returns:
        Filtered signal: original values in allowed regimes, zero elsewhere.
    """
    regime_arr = np.asarray(regimes)
    mask = np.isin(regime_arr, list(allowed_regimes))

    filtered = signal.copy()
    if isinstance(filtered, pd.DataFrame):
        filtered.loc[~mask] = 0.0
    else:
        filtered.loc[~mask] = 0.0

    return filtered
