# QuantLite

QuantLite is a feature-rich Python library for quantitative finance. It spans everything from **data generation** and **Monte Carlo** to **exotic options pricing**, **enhanced backtesting**, and **advanced visualisations**. Whether you’re exploring novel strategies or needing robust analytics, QuantLite aims to cover all your bases.

## Table of Contents

*   [Installation](#installation)
*   [Modules Overview](#modules-overview)
    *   [Data Generation](#data-generation)
    *   [Instruments (Bond, Vanilla Options, Exotic Options)](#instruments)
    *   [Monte Carlo](#monte-carlo)
    *   [Backtesting](#backtesting)
    *   [Visualisation](#visualisation)
*   [Usage Examples & Synergy](#usage-examples--synergy)
    *   [1. Data Generation + Backtesting + Visualisation](#1-data-generation--backtesting--visualisation)
    *   [2. Monte Carlo + Backtesting + Visualisation](#2-monte-carlo--backtesting--visualisation)
    *   [3. Exotic Options Pricing](#3-exotic-options-pricing)
*   [Roadmap](#roadmap)
*   [Licence](#license)
*   [Contact/Support](#contact)

## Installation

QuantLite is available on [PyPI](https://pypi.org/project/quantlite/). Install it simply by:

```bash
pip install quantlite
```

(Ensure you’re using Python 3.8+.)
## Modules Overview

### 1. Data Generation
Location: `quantlite.data_generation`
* `geometric_brownian_motion`: Single-asset GBM path generation.
* `correlated_gbm`: Multi-asset correlated path generation using covariance matrices.
* `ornstein_uhlenbeck`: Mean-reverting process.
* `merton_jump_diffusion`: GBM with Poisson jump arrivals (Merton’s model).

#### Example:
```python
import quantlite.data_generation as qd
prices = qd.geometric_brownian_motion(S0=100, mu=0.05, sigma=0.2, steps=252)
print(prices[:10])  # First 10 days
```
### 2. Instruments (Bond, Vanilla Options, Exotic Options)
Location: `quantlite.instruments`

* Bond Pricing (bond_pricing.py): bond_price, bond_yield_to_maturity, bond_duration, etc.
* Vanilla Options (option_pricing.py): black_scholes_call, black_scholes_put, plus Greeks.
* Exotic Options (exotic_options.py): barrier_option_knock_out, asian_option_arithmetic

Example:
```python
from quantlite.instruments.option_pricing import black_scholes_call
call_val = black_scholes_call(S=100, K=95, T=1, r=0.01, sigma=0.2)
print("Vanilla call option price:", call_val)
```

### 3. Monte Carlo
Location: `quantlite.monte_carlo`

* run_monte_carlo_sims: Single-asset multi-sim approach with different random "modes."
* multi_asset_correlated_sim: Direct correlated multi-asset simulation (similar to * data_generation.correlated_gbm, but designed for scenario testing).

Example:

```python
import pandas as pd
from quantlite.monte_carlo import run_monte_carlo_sims

price_data = pd.Series([100, 101, 99, 102], index=[1,2,3,4])
def always_buy(idx, series):
    return 1

mc_results = run_monte_carlo_sims(price_data, always_buy, n_sims=5)
for i, res in enumerate(mc_results):
    print("Sim", i, "final value:", res["final_value"])
```
### 4. Backtesting
Location: `quantlite.backtesting`

A robust function `run_backtest` with partial capital, short-selling toggles, and transaction cost modelling.

Example:
```python
def run_backtest(
    price_data,
    signal_function,
    initial_capital=10_000.0,
    fee=0.0,
    partial_capital=False,
    capital_fraction=1.0,
    allow_short=True,
    per_share_cost=0.0
):
```

### 5. Visualisation
Location: `quantlite.visualisation`

* `plot_time_series`: Basic line chart with optional indicators.
* `plot_ohlc`: Candlesticks or OHLC bars via mplfinance.
* `plot_return_distribution`: Histogram + KDE for returns.
* `plot_equity_curve`: Equity curve with optional drawdown shading.
* `plot_multiple_equity_curves`: Compare multiple strategies and optionally show rolling Sharpe.

Example:

```python
from quantlite.visualisation import plot_equity_curve
# Suppose we have a backtest result with result["portfolio_value"]
plot_equity_curve(result["portfolio_value"], drawdowns=True)
```
## Usage Examples & Synergy
Below are extended examples to show how modules can be combined.

### 1. Data Generation + Backtesting + Visualisation

```python
import quantlite.data_generation as qd
from quantlite.backtesting import run_backtest
from quantlite.visualisation import plot_equity_curve
import pandas as pd

# 1. Create synthetic price data
prices_array = qd.merton_jump_diffusion(S0=100, mu=0.06, sigma=0.25, steps=252, rng_seed=42)
prices_series = pd.Series(prices_array, index=range(253))

# 2. Simple signal: Buy if today's price < yesterday's
def naive_signal(idx, series):
    if idx == 0:
        return 0
    return 1 if series.iloc[idx] < series.iloc[idx-1] else 0

# 3. Run backtest with partial capital
result = run_backtest(prices_series, naive_signal, fee=1.0, partial_capital=True, capital_fraction=0.5)

# 4. Visualise
plot_equity_curve(result["portfolio_value"], drawdowns=True)
print("Final portfolio value:", result["final_value"])
```

### 2. Monte Carlo + Backtesting + Visualisation
```python
import pandas as pd
from quantlite.monte_carlo import run_monte_carlo_sims
from quantlite.visualisation import plot_multiple_equity_curves

prices = pd.Series([100, 101, 102, 103, 99, 98, 101, 102], index=range(8))
def always_long(idx, series):
    return 1

# Multiple sims
results = run_monte_carlo_sims(prices, always_long, n_sims=3, mode="replace")
curves = {}
for i, res in enumerate(results):
    curves[f"Sim {i}"] = res["portfolio_value"]

plot_multiple_equity_curves(curves_dict=curves, rolling_sharpe=True)
```

### 3. Exotic Options Pricing
```python
from quantlite.instruments.exotic_options import barrier_option_knock_out, asian_option_arithmetic

barrier_val = barrier_option_knock_out(
    S0=120, K=100, H=90, T=1.0, r=0.01, sigma=0.2,
    option_type="call", barrier_type="down-and-out",
    steps=252, sims=10000, rng_seed=42
)
print("Knock-out barrier call value:", barrier_val)

asian_val = asian_option_arithmetic(
    S0=120, K=100, T=1.0, r=0.01, sigma=0.2,
    option_type="call", steps=252, sims=10000, rng_seed=42
)
print("Arithmetic average Asian call value:", asian_val)
```


## Roadmap

1. More Data Generation: 
    * Stochastic volatility (Heston model)
    * Regime-switching
2. Deeper Monte Carlo: 
    * Correlated jumps
    * Advanced param sweeps
    * Multi-factor models
3. Backtesting Enhancements: 
    * Multi-asset portfolio rebalancing
    * Advanced slippage
    * Partial fills.
4. More Exotic Instruments: 
    * Up-and-out barrier
    * Lookback options
5. Interactive Visualisation:
    * Plotly or Bokeh integration 
    * Auto-report generation.

## Licence
This project is distributed under the MIT License.
See the LICENSE file for the full text.

## Contact/Support
**Need help or want to contribute?**  
> Please open an issue on our [GitHub repo](https://github.com/prasants/quantlite).
    