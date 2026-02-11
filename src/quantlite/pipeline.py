"""Dream API: high-level pipeline for the common quant workflow.

Provides a minimal, composable API that chains data fetching, regime
detection, portfolio construction, backtesting, and tearsheet generation
into a few intuitive function calls.

Example::

    import quantlite as ql

    data = ql.fetch(["AAPL", "BTC-USD", "GLD", "TLT"], period="5y")
    regimes = ql.detect_regimes(data)
    weights = ql.construct_portfolio(data, regime_aware=True, regimes=regimes)
    result = ql.backtest(data, weights)
    ql.tearsheet(result, regimes=regimes, save="portfolio.html")
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "fetch",
    "detect_regimes",
    "construct_portfolio",
    "backtest",
    "tearsheet",
]


def fetch(
    tickers: str | list[str],
    period: str = "5y",
    source: str = "yahoo",
    **kwargs: Any,
) -> pd.DataFrame:
    """Unified data fetching that returns a returns DataFrame.

    Fetches price data for one or more tickers and computes simple
    returns. For multiple tickers, returns a DataFrame with one column
    per ticker.

    Args:
        tickers: Single ticker string or list of ticker strings.
        period: Lookback period (e.g. ``"5y"``, ``"1y"``, ``"6mo"``).
            Forwarded to the data source.
        source: Data source name (default ``"yahoo"``).
        **kwargs: Extra parameters forwarded to
            :func:`quantlite.data.fetch`.

    Returns:
        DataFrame of simple returns with a DatetimeIndex.

    Raises:
        ImportError: If the required data source dependency is missing.
    """
    from .data import fetch as _data_fetch

    if isinstance(tickers, str):
        tickers = [tickers]

    result = _data_fetch(tickers, source=source, period=period, **kwargs)

    # result is a dict of {ticker: DataFrame}
    close_frames = {}
    for ticker, df in result.items():
        if "close" in df.columns:
            close_frames[ticker] = df["close"]
        elif "Close" in df.columns:
            close_frames[ticker] = df["Close"]
        else:
            # Use last column as price
            close_frames[ticker] = df.iloc[:, -1]

    prices = pd.DataFrame(close_frames)
    returns = prices.pct_change().dropna()
    return returns


def detect_regimes(
    returns_df: pd.DataFrame,
    method: str = "hmm",
    n_regimes: int = 3,
    rng_seed: int | None = None,
    signal: str = "mean",
) -> np.ndarray:
    """Detect market regimes in a returns DataFrame.

    Wraps the HMM regime detection module. For multi-asset DataFrames,
    the ``signal`` parameter controls how returns are reduced to a
    univariate series before fitting:

    * ``"mean"`` — arithmetic mean across assets (default).
    * ``"pca"``  — first principal component, preserving multivariate
      correlation structure.
    * ``"min"``  — worst-performing asset each period (most
      conservative signal).

    Args:
        returns_df: DataFrame of simple returns.
        method: Detection method. Currently only ``"hmm"`` is supported.
        n_regimes: Number of regimes to detect.
        rng_seed: Random seed for reproducibility.
        signal: Signal extraction method: ``"mean"``, ``"pca"``, or
            ``"min"``.

    Returns:
        Array of integer regime labels, same length as *returns_df*.

    Raises:
        ValueError: On unsupported method or signal.
        ImportError: If hmmlearn is not installed.
    """
    if method != "hmm":
        raise ValueError(f"Unsupported method: {method}. Use 'hmm'.")

    if signal not in ("mean", "pca", "min"):
        raise ValueError(f"Unsupported signal: {signal}. Use 'mean', 'pca', or 'min'.")

    from .regimes.hmm import fit_regime_model

    if isinstance(returns_df, pd.DataFrame) and returns_df.shape[1] > 1:
        if signal == "mean":
            sig = returns_df.mean(axis=1)
        elif signal == "pca":
            from numpy.linalg import eigh

            centered = returns_df.values - returns_df.values.mean(axis=0)
            cov = np.cov(centered, rowvar=False)
            eigenvalues, eigenvectors = eigh(cov)
            # First PC = eigenvector with largest eigenvalue (last from eigh)
            pc1 = centered @ eigenvectors[:, -1]
            sig = pd.Series(pc1, index=returns_df.index)
        elif signal == "min":
            sig = returns_df.min(axis=1)
    elif isinstance(returns_df, pd.DataFrame):
        sig = returns_df.iloc[:, 0]
    else:
        sig = returns_df

    model = fit_regime_model(sig, n_regimes=n_regimes, rng_seed=rng_seed)
    return model.regime_labels


def construct_portfolio(
    returns_df: pd.DataFrame,
    method: str = "hrp",
    regime_aware: bool = True,
    regimes: np.ndarray | None = None,
    defensive_tilt: float = 0.3,
) -> dict[str, float]:
    """Construct portfolio weights with optional regime awareness.

    Args:
        returns_df: DataFrame of simple returns (assets as columns).
        method: Weight method: ``"hrp"``, ``"min_variance"``, or
            ``"equal_weight"``.
        regime_aware: Whether to apply regime-aware defensive tilting.
        regimes: Regime labels (required if *regime_aware* is ``True``).
        defensive_tilt: Tilt magnitude for crisis regimes.

    Returns:
        Dictionary mapping asset names to weights summing to 1.0.

    Raises:
        ValueError: If *regime_aware* is True but *regimes* is None.
    """
    if regime_aware:
        if regimes is None:
            raise ValueError("regimes must be provided when regime_aware=True")
        from .regime_integration.portfolio import regime_aware_weights
        return regime_aware_weights(
            returns_df, regimes, method=method, defensive_tilt=defensive_tilt,
        )

    n = len(returns_df.columns)
    assets = list(returns_df.columns)

    if method == "equal_weight":
        return {a: 1.0 / n for a in assets}
    elif method == "min_variance":
        cov = returns_df.cov().values
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        ones = np.ones(n)
        w = inv_cov @ ones / (ones @ inv_cov @ ones)
        w = np.maximum(w, 0)
        w = w / w.sum()
        return dict(zip(assets, w.tolist()))
    elif method == "hrp":
        from .dependency.clustering import hrp_weights
        return dict(hrp_weights(returns_df))
    else:
        raise ValueError(f"Unknown method: {method}")


def backtest(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    rebalance: str = "monthly",
    initial_capital: float = 10000.0,
) -> dict[str, Any]:
    """Run a simple portfolio backtest with fixed weights.

    Computes an equity curve by applying the given weights to the
    returns DataFrame, rebalancing at the specified frequency.

    Args:
        returns_df: DataFrame of simple returns (assets as columns).
        weights: Dictionary mapping asset names to target weights.
        rebalance: Rebalance frequency: ``"daily"``, ``"weekly"``,
            ``"monthly"``.
        initial_capital: Starting portfolio value.

    Returns:
        Dictionary with keys:
        - ``"equity_curve"``: pandas Series of portfolio values.
        - ``"returns"``: pandas Series of portfolio returns.
        - ``"total_return"``: total percentage return.
        - ``"annualised_return"``: annualised return.
        - ``"annualised_volatility"``: annualised volatility.
        - ``"sharpe_ratio"``: Sharpe ratio.
        - ``"max_drawdown"``: maximum drawdown.
    """
    assets = list(returns_df.columns)
    w = np.array([weights.get(a, 0.0) for a in assets])

    # Portfolio returns
    port_returns = (returns_df.values * w).sum(axis=1)
    equity = initial_capital * np.cumprod(1.0 + port_returns)

    equity_series = pd.Series(equity, index=returns_df.index)
    returns_series = pd.Series(port_returns, index=returns_df.index)

    total_ret = float(equity[-1] / initial_capital - 1.0)
    n = len(port_returns)
    ann_ret = float((1.0 + total_ret) ** (252.0 / max(n, 1)) - 1.0)
    ann_vol = float(np.std(port_returns, ddof=1) * np.sqrt(252)) if n > 1 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.min(dd))

    return {
        "equity_curve": equity_series,
        "returns": returns_series,
        "total_return": total_ret,
        "annualised_return": ann_ret,
        "annualised_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "weights": weights,
    }


def tearsheet(
    backtest_result: dict[str, Any],
    regimes: np.ndarray | None = None,
    save: str | None = None,
) -> dict[str, Any]:
    """Generate a tearsheet from backtest results.

    Wraps the regime-aware tearsheet when regimes are provided,
    otherwise produces a basic performance summary.

    Args:
        backtest_result: Output from :func:`backtest`.
        regimes: Optional regime labels for regime-breakdown reporting.
        save: Optional file path. If provided, saves a text summary
            to this path.

    Returns:
        Dictionary containing tearsheet data (equity curves, metrics,
        and optionally regime breakdowns).
    """
    returns = backtest_result["returns"]

    if regimes is not None:
        from .regime_integration.reporting import (
            regime_comparison_table,
            regime_performance_attribution,
            regime_tearsheet,
        )

        ts = regime_tearsheet(returns.values, regimes)
        ts["attribution"] = regime_performance_attribution(returns.values, regimes)
        ts["comparison_table"] = regime_comparison_table(returns.values, regimes)
    else:
        equity = backtest_result["equity_curve"]
        ts = {
            "equity_curve": equity,
            "overall_metrics": {
                "total_return": backtest_result["total_return"],
                "annualised_return": backtest_result["annualised_return"],
                "annualised_volatility": backtest_result["annualised_volatility"],
                "sharpe_ratio": backtest_result["sharpe_ratio"],
                "max_drawdown": backtest_result["max_drawdown"],
            },
        }

    if save is not None:
        _save_tearsheet(ts, save)

    return ts


def _save_tearsheet(ts: dict[str, Any], path: str) -> None:
    """Save tearsheet summary to a text file."""
    lines = ["QuantLite Portfolio Tearsheet", "=" * 40, ""]

    metrics = ts.get("overall_metrics", {})
    for key, val in metrics.items():
        if isinstance(val, float):
            lines.append(f"{key}: {val:.4f}")
        else:
            lines.append(f"{key}: {val}")

    if "comparison_table" in ts:
        lines.append("")
        lines.append("Regime Comparison")
        lines.append("-" * 40)
        lines.append(ts["comparison_table"])

    if "attribution" in ts:
        lines.append("")
        lines.append("Performance Attribution")
        lines.append("-" * 40)
        for regime, data in ts["attribution"].items():
            lines.append(f"  Regime {regime}: {data['contribution_pct']:.1f}% of total return")

    with open(path, "w") as f:
        f.write("\n".join(lines))
