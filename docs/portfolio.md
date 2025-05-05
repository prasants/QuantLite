# Portfolio Optimisation, Rebalancing, and Backtesting

## Portfolio Optimisation (`quantlite.portfolio.optimisation`)

All optimisation functions return a `PortfolioWeights` dataclass with `weights`, `method`, `expected_return`, `expected_risk`, and `sharpe` attributes.

### Efficient Frontier

Random portfolios coloured by Sharpe ratio, with minimum variance and maximum Sharpe portfolios highlighted. Individual assets are marked for reference:

![Efficient Frontier](images/efficient_frontier.png)

### HRP Dendrogram and Weights

Hierarchical Risk Parity clusters assets by correlation distance, then allocates inversely proportional to cluster variance:

![HRP Dendrogram](images/hrp_dendrogram.png)

### Weight Comparison Across Methods

Four optimisation methods produce notably different allocations. Risk parity and HRP tend to be more diversified than mean-variance:

![Weight Comparison](images/weight_comparison.png)

### Monthly Returns Heatmap

A visual summary of portfolio performance by month and year, making it easy to spot seasonal patterns and drawdown clusters:

![Monthly Returns](images/monthly_returns_heatmap.png)

### Backtest Tearsheet

Strategy vs benchmark equity curve, underwater chart, monthly returns, and rolling Sharpe ratio:

![Equity Curve](images/equity_curve.png)

![Backtest Drawdown](images/backtest_drawdown.png)

![Rolling Sharpe](images/rolling_sharpe.png)

### Mean-Variance (Markowitz)

```python
from quantlite.portfolio.optimisation import mean_variance_weights, min_variance_weights, max_sharpe_weights
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
returns_df = pd.DataFrame({
    "US_Equity": rng.normal(0.0004, 0.012, 504),
    "Intl_Equity": rng.normal(0.0003, 0.014, 504),
    "Govt_Bonds": rng.normal(0.00015, 0.004, 504),
    "Corp_Bonds": rng.normal(0.0002, 0.006, 504),
    "Gold": rng.normal(0.0001, 0.010, 504),
})

# Target 8% annualised return
mv = mean_variance_weights(returns_df, target_return=0.08, long_only=True)
print(f"Expected return: {mv.expected_return:.2%}")
print(f"Expected risk:   {mv.expected_risk:.2%}")
for asset, w in mv.weights.items():
    print(f"  {asset}: {w:.2%}")

# Minimum variance portfolio
min_var = min_variance_weights(returns_df)

# Maximum Sharpe ratio (tangency) portfolio
max_sr = max_sharpe_weights(returns_df, risk_free_rate=0.04)
print(f"Max Sharpe: {max_sr.sharpe:.2f}")
```

### CVaR Optimisation

Minimises Conditional Value at Risk instead of variance, penalising tail losses rather than all deviations:

```python
from quantlite.portfolio.optimisation import mean_cvar_weights

cvar_port = mean_cvar_weights(returns_df, alpha=0.05, long_only=True)
print(f"CVaR (risk measure): {cvar_port.expected_risk:.5f}")
print(f"Sharpe ratio:        {cvar_port.sharpe:.2f}")
```

### Risk Parity

Each asset contributes equally to total portfolio risk:

```python
from quantlite.portfolio.optimisation import risk_parity_weights

rp = risk_parity_weights(returns_df)
for asset, w in rp.weights.items():
    print(f"  {asset}: {w:.2%}")
```

### Hierarchical Risk Parity

The convenience wrapper around `dependency.clustering.hrp_weights`:

```python
from quantlite.portfolio.optimisation import hrp_weights

hrp = hrp_weights(returns_df)
print(f"HRP Sharpe: {hrp.sharpe:.2f}")
```

### Black-Litterman

Combine market equilibrium returns with subjective views:

```python
from quantlite.portfolio.optimisation import black_litterman

market_caps = {
    "US_Equity": 40e12, "Intl_Equity": 25e12,
    "Govt_Bonds": 30e12, "Corp_Bonds": 10e12, "Gold": 5e12,
}

# Views: US equity will return 10%, Gold will return 8%
views = {"US_Equity": 0.10, "Gold": 0.08}
confidences = {"US_Equity": 0.7, "Gold": 0.5}

posterior_returns, posterior_cov = black_litterman(
    returns_df, market_caps, views, confidences,
    tau=0.05, risk_aversion=2.5,
)
print("Posterior expected returns:")
print(posterior_returns)
```

### Kelly Criterion

Position sizing for a single strategy:

```python
from quantlite.portfolio.optimisation import kelly_criterion, half_kelly

# From return statistics
kelly = kelly_criterion(returns_df["US_Equity"].values)
hk = half_kelly(returns_df["US_Equity"].values)
print(f"Full Kelly: {kelly:.2f}")
print(f"Half Kelly: {hk:.2f}")

# From win probability and win/loss ratio
kelly_discrete = kelly_criterion([], win_prob=0.55, win_loss_ratio=1.5)
print(f"Kelly (discrete): {kelly_discrete:.2f}")
```

Half-Kelly produces approximately 75% of the growth rate with significantly lower drawdowns.

## Rebalancing Strategies (`quantlite.portfolio.rebalancing`)

All strategies return a `RebalanceResult` with `portfolio_returns`, `weights_over_time`, `rebalance_dates`, `turnover`, and `n_rebalances`.

### Calendar Rebalancing

```python
from quantlite.portfolio.rebalancing import rebalance_calendar
import pandas as pd

# Add DatetimeIndex
returns_df.index = pd.bdate_range("2022-01-03", periods=len(returns_df))

def equal_weight(df):
    n = df.shape[1]
    return {col: 1.0 / n for col in df.columns}

monthly = rebalance_calendar(returns_df, equal_weight, freq="monthly")
quarterly = rebalance_calendar(returns_df, equal_weight, freq="quarterly")
print(f"Monthly: {monthly.n_rebalances} rebalances, turnover={monthly.turnover:.2f}")
print(f"Quarterly: {quarterly.n_rebalances} rebalances, turnover={quarterly.turnover:.2f}")
```

### Threshold Rebalancing

Rebalance only when any weight drifts beyond a tolerance from its target:

```python
from quantlite.portfolio.rebalancing import rebalance_threshold

result = rebalance_threshold(returns_df, equal_weight, threshold=0.05)
print(f"Threshold rebalances: {result.n_rebalances}")
print(f"Turnover: {result.turnover:.2f}")
```

### Tactical (Regime-Triggered) Rebalancing

Rebalance when the market regime changes, enabling tactical allocation shifts:

```python
from quantlite.portfolio.rebalancing import rebalance_tactical
import numpy as np

# Simulate regime labels (0=calm, 1=crisis)
regime_labels = np.random.default_rng(42).choice([0, 1], size=len(returns_df), p=[0.8, 0.2])

result = rebalance_tactical(returns_df, equal_weight, regime_labels)
print(f"Regime-triggered rebalances: {result.n_rebalances}")
```

## Backtesting Engine (`quantlite.backtesting.engine`)

The production backtesting engine supports multi-asset portfolios, configurable slippage, transaction fees, risk limits with circuit breakers, and regime-aware allocation.

### Configuration

```python
from quantlite.backtesting.engine import (
    BacktestConfig, SlippageModel, RiskLimits,
)

config = BacktestConfig(
    initial_capital=1_000_000,
    fractional_shares=True,
    slippage_model=SlippageModel(kind="fixed", spread_bps=5),
    risk_limits=RiskLimits(
        max_drawdown=-0.20,        # circuit breaker at 20% drawdown
        max_position_pct=0.40,     # no single position > 40%
        max_leverage=1.0,          # no leverage
        daily_loss_limit=-0.05,    # halt on 5% daily loss
    ),
    rebalance_freq="weekly",
    fee_per_trade_pct=0.001,       # 10bps per trade
)
```

### Running a Backtest

The allocation function receives a `BacktestContext` with current prices, historical returns, current weights, drawdown, regime, and portfolio value:

```python
from quantlite.backtesting.engine import run_backtest, BacktestContext
from quantlite.data_generation import correlated_gbm
import numpy as np
import pandas as pd

# Generate price data
cov = np.array([[0.04, 0.01], [0.01, 0.02]])
prices_df = correlated_gbm(
    S0_list=[100, 50], mu_list=[0.08, 0.05],
    cov_matrix=cov, steps=504, rng_seed=42, return_as="dataframe",
)
prices_df.index = pd.bdate_range("2022-01-03", periods=len(prices_df))
prices_df.columns = ["Equities", "Bonds"]

def simple_allocation(ctx: BacktestContext) -> dict[str, float]:
    """60/40 allocation that reduces equity exposure during drawdowns."""
    eq_weight = 0.6
    if ctx.current_drawdown < -0.10:
        eq_weight = 0.3  # defensive shift
    return {"Equities": eq_weight, "Bonds": 1.0 - eq_weight}

result = run_backtest(prices_df, simple_allocation, config)
print(f"Final value: {result.portfolio_value.iloc[-1]:,.0f}")
print(f"Total trades: {len(result.trades)}")
print(f"Max drawdown: {result.metrics.get('max_drawdown', 0):.2%}")
```

### Signal Generation

Built-in signal helpers for common strategies:

```python
from quantlite.backtesting.signals import (
    momentum_signal, mean_reversion_signal,
    volatility_targeting, trend_following, regime_filter,
)

# Cross-sectional momentum
mom = momentum_signal(prices_df, lookback=60)

# Z-score mean reversion
mr = mean_reversion_signal(prices_df, lookback=20, z_threshold=1.5)

# Dual MA trend following
trend = trend_following(prices_df, fast_window=20, slow_window=60)

# Volatility targeting at 10% annualised
vol_scale = volatility_targeting(prices_df.pct_change().dropna(), target_vol=0.10)
```

### Post-Backtest Analysis

```python
from quantlite.backtesting.analysis import (
    performance_summary, monthly_returns_table,
    rolling_metrics, trade_analysis, regime_attribution,
)

# Full performance table
summary = performance_summary(result)
print(summary)

# Monthly returns heatmap data
monthly = monthly_returns_table(result)
print(monthly)

# Rolling Sharpe, volatility, drawdown
rolling = rolling_metrics(result, window=63)

# Trade-level statistics
trades = trade_analysis(result)
print(f"Win rate: {trades['win_rate']:.1%}")
print(f"Total cost: {trades['total_cost']:.2f}")
```

### Regime-Aware Backtesting

```python
import numpy as np

# Provide regime labels to the backtest
regime_labels = np.random.default_rng(42).choice([0, 1], size=len(prices_df), p=[0.85, 0.15])

result = run_backtest(prices_df, simple_allocation, config, regime_labels=regime_labels)

# Attribution by regime
attr = regime_attribution(result)
print(attr)
```

## Monte Carlo Simulation (`quantlite.monte_carlo`)

Run multiple backtests on perturbed or synthetic price paths:

```python
from quantlite.monte_carlo import run_monte_carlo_sims

def simple_signal(idx, prices):
    if idx < 20:
        return 0
    return 1 if prices.iloc[idx] > prices.iloc[idx-20:idx].mean() else -1

results = run_monte_carlo_sims(
    prices_df["Equities"], simple_signal,
    n_sims=100, noise_scale=0.01, mode="perturb", rng_seed=42,
)

final_values = [r["final_value"] for r in results]
print(f"Mean final value: {np.mean(final_values):,.0f}")
print(f"5th percentile:   {np.percentile(final_values, 5):,.0f}")
print(f"95th percentile:  {np.percentile(final_values, 95):,.0f}")
```
