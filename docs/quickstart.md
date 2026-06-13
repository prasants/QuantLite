# Quickstart: 10-Minute Tour

This walkthrough hits the key modules in QuantLite. By the end, you will have generated synthetic data, measured tail risk, detected regimes, built a portfolio, and backtested it honestly.

## 1. Generate Realistic Market Data

QuantLite includes stochastic process generators with fat-tailed dynamics:

```python
import quantlite as ql
import numpy as np

# Jump-diffusion process (realistic equity returns)
prices = ql.merton_jump_diffusion(
    s0=100, mu=0.08, sigma=0.20,
    lam=5, jump_mean=-0.02, jump_std=0.04,
    T=5.0, n_steps=1260, seed=42,
)

# Correlated multi-asset GBM
multi_prices = ql.correlated_gbm(
    s0=[100, 50, 30],
    mu=[0.08, 0.06, 0.04],
    sigma=[0.20, 0.15, 0.10],
    corr=[[1.0, 0.6, -0.2],
          [0.6, 1.0, 0.1],
          [-0.2, 0.1, 1.0]],
    T=5.0, n_steps=1260, seed=42,
)
```

## 2. Measure Tail Risk

Standard risk metrics lie about tail risk. QuantLite uses extreme value theory by default:

```python
from quantlite.risk import var, cvar, tail_index

returns = np.diff(np.log(prices))

# Value at Risk (95th and 99th percentile)
var_95 = var(returns, level=0.95)
var_99 = var(returns, level=0.99)

# Conditional VaR (Expected Shortfall)
cvar_99 = cvar(returns, level=0.99)

# Tail index (lower = fatter tails; equity markets typically 2-4)
alpha = tail_index(returns)
print(f"Tail index: {alpha:.2f}")
```

## 3. Detect Market Regimes

Markets oscillate between calm, volatile, and crisis states. QuantLite detects these with Hidden Markov Models:

```python
from quantlite.regimes import fit_regime_model, decode_regimes

model = fit_regime_model(returns, n_regimes=3)
labels = decode_regimes(model, returns)

# Each observation is now labelled 0, 1, or 2
# Typically: 0 = low vol, 1 = medium vol, 2 = crisis
print(f"Regime counts: {np.bincount(labels)}")
```

## 4. Fit Fat-Tailed Distributions

Never assume normality. Fit a Generalised Pareto Distribution to the tails:

```python
from quantlite.distributions import fit_gpd

# Fit GPD to the left tail (losses)
gpd_params = fit_gpd(-returns, threshold_quantile=0.95)
print(f"Shape (xi): {gpd_params.shape:.3f}")
print(f"Scale (sigma): {gpd_params.scale:.4f}")
```

A positive shape parameter confirms fat tails. Values above 0.2 indicate very heavy tails.

## 5. Build and Optimise a Portfolio

```python
from quantlite.portfolio import optimise_portfolio

import pandas as pd

# Using the correlated multi-asset data
ret_df = pd.DataFrame(
    np.diff(np.log(multi_prices), axis=0),
    columns=["Equity", "Bond", "Gold"],
)

weights = optimise_portfolio(
    ret_df,
    method="min_cvar",  # Minimise Conditional VaR, not variance
    target_return=0.06,
)
print(f"Optimal weights: {weights}")
```

## 6. Backtest Honestly

QuantLite's backtesting engine includes realistic transaction costs, slippage, and risk limits:

```python
config = ql.BacktestConfig(
    initial_capital=1_000_000,
    slippage=ql.SlippageModel(bps=5),
    risk_limits=ql.RiskLimits(
        max_drawdown=0.15,
        max_position_pct=0.40,
    ),
)

result = ql.run_backtest(
    returns=ret_df,
    weights=weights,
    config=config,
)

print(f"Sharpe: {result.sharpe:.2f}")
print(f"Max drawdown: {result.max_drawdown:.1%}")
print(f"Final value: £{result.final_value:,.0f}")
```

## 7. Check for Overfitting

Did your backtest just get lucky? The deflated Sharpe ratio tells you:

```python
from quantlite.overfit import deflated_sharpe_ratio

dsr = deflated_sharpe_ratio(
    observed_sharpe=result.sharpe,
    n_trials=50,       # How many strategies did you try?
    n_observations=len(ret_df),
)
print(f"Deflated Sharpe p-value: {dsr.p_value:.4f}")
# p < 0.05 means the Sharpe is likely genuine
```

## 8. Stream Live Data and Detect Regimes Online

```python
import asyncio

async def live_monitoring():
    detector = ql.OnlineRegimeDetector(n_regimes=3)
    detector.fit(returns)  # Pre-train on historical data

    alerts = ql.AlertManager()
    alerts.add_rule("BTC-USD", condition="regime_change",
                    callback=lambda a: print(f"🚨 {a.message}"))

    stream = ql.stream(["BTC-USD"], exchange="binance")
    async for tick in stream:
        result = detector.update(tick.price)
        alerts.check("BTC-USD", regime=result.regime,
                     previous_regime=result.regime if not result.regime_changed
                     else (result.regime + 1) % 3)

asyncio.run(live_monitoring())
```

## What Next?

- **Deep dive into any module** — every section in the sidebar covers one capability in detail
- **API Reference** — see [API Reference](api.md) for every public function
- **Architecture** — see [Architecture](architecture.md) for design philosophy and contributing guidelines
