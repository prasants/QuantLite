"""QuantLite: a fat-tail-native quantitative finance toolkit.

Provides stochastic process generators, option and bond pricing,
risk metrics, extreme value theory, fat-tailed distributions,
portfolio optimisation, multi-asset backtesting, and
Stephen Few-inspired visualisation.
"""

__version__ = "0.2.0"

from .backtesting import (
    BacktestConfig,
    BacktestContext,
    BacktestResult,
    RiskLimits,
    SlippageModel,
    run_backtest,
)
from .data_generation import (
    correlated_gbm,
    geometric_brownian_motion,
    merton_jump_diffusion,
    ornstein_uhlenbeck,
)
from .instruments.bond_pricing import bond_price, bond_yield_to_maturity
from .instruments.option_pricing import (
    black_scholes_call,
    black_scholes_greeks,
    black_scholes_put,
)
from .metrics import annualised_return, annualised_volatility, max_drawdown, sharpe_ratio
from .visualisation import plot_time_series

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
]

from . import antifragile, ergodicity, scenarios  # noqa: E402
