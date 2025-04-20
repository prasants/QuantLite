"""Backtesting: multi-asset engine, signals, and post-hoc analysis."""

from .analysis import (
    monthly_returns_table,
    performance_summary,
    regime_attribution,
    rolling_metrics,
    trade_analysis,
)
from .engine import (
    BacktestConfig,
    BacktestContext,
    BacktestResult,
    RiskLimits,
    SlippageModel,
    run_backtest,
)
from .signals import (
    mean_reversion_signal,
    momentum_signal,
    regime_filter,
    trend_following,
    volatility_targeting,
)

__all__ = [
    "BacktestConfig",
    "BacktestContext",
    "BacktestResult",
    "RiskLimits",
    "SlippageModel",
    "mean_reversion_signal",
    "momentum_signal",
    "monthly_returns_table",
    "performance_summary",
    "regime_attribution",
    "regime_filter",
    "rolling_metrics",
    "run_backtest",
    "trade_analysis",
    "trend_following",
    "volatility_targeting",
]
