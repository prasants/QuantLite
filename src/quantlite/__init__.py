"""QuantLite: a fat-tail-native quantitative finance toolkit.

Provides stochastic process generators, option and bond pricing,
risk metrics, extreme value theory, fat-tailed distributions,
portfolio optimisation, multi-asset backtesting, and
Stephen Few-inspired visualisation.

Submodules and public names are imported lazily (PEP 562): accessing
``quantlite.run_backtest`` or ``quantlite.score`` imports the backing module
on first use. This keeps light entry points light -- importing
``quantlite.score`` does not pull in pandas or matplotlib -- while preserving
the full ``from quantlite import X`` and ``quantlite.X`` public API.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "1.7.1"

# Public names mapped to the module that defines them, for lazy loading.
_ATTR_SOURCES = {
    # Data generation
    "geometric_brownian_motion": "quantlite.data_generation",
    "correlated_gbm": "quantlite.data_generation",
    "ornstein_uhlenbeck": "quantlite.data_generation",
    "merton_jump_diffusion": "quantlite.data_generation",
    # Instruments
    "black_scholes_call": "quantlite.instruments.option_pricing",
    "black_scholes_put": "quantlite.instruments.option_pricing",
    "black_scholes_greeks": "quantlite.instruments.option_pricing",
    "bond_price": "quantlite.instruments.bond_pricing",
    "bond_yield_to_maturity": "quantlite.instruments.bond_pricing",
    # Metrics
    "annualised_return": "quantlite.metrics",
    "annualised_volatility": "quantlite.metrics",
    "sharpe_ratio": "quantlite.metrics",
    "max_drawdown": "quantlite.metrics",
    # Backtesting
    "run_backtest": "quantlite.backtesting",
    "BacktestConfig": "quantlite.backtesting",
    "BacktestContext": "quantlite.backtesting",
    "BacktestResult": "quantlite.backtesting",
    "RiskLimits": "quantlite.backtesting",
    "SlippageModel": "quantlite.backtesting",
    # Visualisation
    "plot_time_series": "quantlite.visualisation",
    # Dream API (pipeline)
    "fetch": "quantlite.pipeline",
    "detect_regimes": "quantlite.pipeline",
    "construct_portfolio": "quantlite.pipeline",
    "backtest": "quantlite.pipeline",
    "tearsheet": "quantlite.pipeline",
    # Streaming & alerts (v1.1)
    "AlertManager": "quantlite.alerts",
    "AlertRule": "quantlite.alerts",
    "Alert": "quantlite.alerts",
    "PriceStream": "quantlite.data.stream",
    "PriceTick": "quantlite.data.stream",
    "create_stream": "quantlite.data.stream",
    "stream": "quantlite.data.stream",
    "OnlineRegimeDetector": "quantlite.regimes.online",
    "RegimeUpdate": "quantlite.regimes.online",
}

# Names whose attribute on the source module differs from the public name.
_ALIASES = {"stream": "create_stream"}

__all__ = [
    # Data generation
    "geometric_brownian_motion",
    "correlated_gbm",
    "ornstein_uhlenbeck",
    "merton_jump_diffusion",
    # Instruments
    "black_scholes_call",
    "black_scholes_put",
    "black_scholes_greeks",
    "bond_price",
    "bond_yield_to_maturity",
    # Metrics
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "max_drawdown",
    # Backtesting
    "run_backtest",
    "BacktestConfig",
    "BacktestContext",
    "BacktestResult",
    "RiskLimits",
    "SlippageModel",
    # Visualisation
    "plot_time_series",
    # Ergodicity economics
    "ergodicity",
    # Antifragility framework
    "antifragile",
    # Scenario engine
    "scenarios",
    # Strategy forensics
    "forensics",
    # Overfitting detection
    "overfit",
    # Resampled backtesting
    "resample",
    # Contagion metrics
    "contagion",
    # Network risk
    "network",
    # Diversification analysis
    "diversification",
    # Crypto-native risk
    "crypto",
    # Fat-tail Monte Carlo simulation
    "simulation",
    # Regime-aware integration
    "regime_integration",
    # Dream API (pipeline)
    "fetch",
    "detect_regimes",
    "construct_portfolio",
    "backtest",
    "tearsheet",
    # Streaming & alerts (v1.1)
    "AlertManager",
    "AlertRule",
    "Alert",
    "PriceStream",
    "PriceTick",
    "create_stream",
    "stream",
    "OnlineRegimeDetector",
    "RegimeUpdate",
    # Benchmarking (v1.5)
    "benchmark",
    # QuantLite Score (v1.6)
    "score",
]


def __getattr__(name: str) -> Any:
    """Lazily import a public name or submodule on first access (PEP 562).

    Public names listed in ``__all__`` resolve to their defining module; any
    other name is tried as a submodule, preserving the eager package's
    behaviour of exposing ``quantlite.<submodule>`` without an explicit
    import.
    """
    source = _ATTR_SOURCES.get(name)
    if source is not None:
        module = importlib.import_module(source)
        value = getattr(module, _ALIASES.get(name, name))
        globals()[name] = value
        return value
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """List the lazily exported names for tab completion and introspection."""
    return sorted(__all__)
