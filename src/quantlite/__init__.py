__version__ = "0.1.0"

from .metrics import (
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    max_drawdown
)

from .backtesting import run_backtest

from .data_generation import (
    geometric_brownian_motion,
    ornstein_uhlenbeck
)

from .visualisation import (
    plot_time_series  
)

from .instruments.bond_pricing import (
    bond_price,
    bond_yield_to_maturity
)

from .instruments.option_pricing import (
    black_scholes_call,
    black_scholes_put
)
