"""Dataclass return types used across QuantLite.

All structured results are returned as frozen dataclasses for
immutability, dot-access, and clear ``repr`` output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "BacktestResult",
    "GreeksResult",
    "ReturnMoments",
    "DrawdownInfo",
    "GPDFit",
    "GEVFit",
    "HillEstimate",
    "TailRiskSummary",
]


@dataclass(frozen=True)
class BacktestResult:
    """Result container for a single-asset backtest."""

    portfolio_value: pd.Series
    positions: pd.Series
    trades: list[tuple[object, str, int, float, float]]
    final_value: float

    def __repr__(self) -> str:
        n_trades = len(self.trades)
        return (
            f"BacktestResult(final_value={self.final_value:,.2f}, "
            f"trades={n_trades}, bars={len(self.portfolio_value)})"
        )


@dataclass(frozen=True)
class GreeksResult:
    """Black-Scholes Greeks for a European option."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def __repr__(self) -> str:
        return (
            f"Greeks(delta={self.delta:.4f}, gamma={self.gamma:.4f}, "
            f"vega={self.vega:.4f}, theta={self.theta:.4f}, rho={self.rho:.4f})"
        )


@dataclass(frozen=True)
class ReturnMoments:
    """Descriptive statistics for a return series."""

    mean: float
    volatility: float
    skewness: float
    kurtosis: float

    def __repr__(self) -> str:
        return (
            f"ReturnMoments(mean={self.mean:.6f}, vol={self.volatility:.6f}, "
            f"skew={self.skewness:.4f}, kurt={self.kurtosis:.4f})"
        )


@dataclass(frozen=True)
class DrawdownInfo:
    """Maximum drawdown with duration information."""

    max_drawdown: float
    duration: int
    start_idx: int
    end_idx: int

    def __repr__(self) -> str:
        return (
            f"DrawdownInfo(max_dd={self.max_drawdown:.4f}, "
            f"duration={self.duration} periods)"
        )


@dataclass(frozen=True)
class GPDFit:
    """Fitted Generalised Pareto Distribution parameters."""

    shape: float  # xi (tail index)
    scale: float  # sigma
    threshold: float
    n_exceedances: int
    n_total: int

    def __repr__(self) -> str:
        return (
            f"GPDFit(shape={self.shape:.4f}, scale={self.scale:.4f}, "
            f"threshold={self.threshold:.4f}, exceedances={self.n_exceedances})"
        )


@dataclass(frozen=True)
class GEVFit:
    """Fitted Generalised Extreme Value Distribution parameters."""

    shape: float  # xi
    loc: float  # mu
    scale: float  # sigma

    def __repr__(self) -> str:
        return (
            f"GEVFit(shape={self.shape:.4f}, loc={self.loc:.4f}, "
            f"scale={self.scale:.4f})"
        )


@dataclass(frozen=True)
class HillEstimate:
    """Hill estimator result for tail index."""

    tail_index: float  # alpha
    k: int  # number of order statistics used

    def __repr__(self) -> str:
        return f"HillEstimate(alpha={self.tail_index:.4f}, k={self.k})"


@dataclass(frozen=True)
class TailRiskSummary:
    """Comprehensive tail risk analysis."""

    gpd_fit: GPDFit
    hill_estimate: HillEstimate
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    return_level_100: float  # 1-in-100 period loss
    excess_kurtosis: float

    def __repr__(self) -> str:
        return (
            f"TailRiskSummary(VaR95={self.var_95:.4f}, VaR99={self.var_99:.4f}, "
            f"CVaR99={self.cvar_99:.4f}, tail_idx={self.hill_estimate.tail_index:.4f})"
        )
