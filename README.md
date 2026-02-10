# QuantLite

**A fat-tail-native quantitative finance toolkit for Python.**

[![PyPI version](https://img.shields.io/pypi/v/quantlite)](https://pypi.org/project/quantlite/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

QuantLite is what pandas is to data manipulation, but for fat-tail quant finance. It provides honest modelling tools for markets that bite: extreme value theory, copula dependence structures, regime detection, fat-tailed distributions, and risk metrics that do not assume returns are Gaussian.

Most quantitative finance libraries treat fat tails as an afterthought. QuantLite treats them as the starting point.

## Key Differentiators

- **Fat-tail-native**: Every risk metric, simulation, and optimisation accounts for non-Gaussian behaviour
- **Extreme Value Theory**: GPD, GEV, Hill estimator, and Peaks Over Threshold for rigorous tail modelling
- **Copula dependence**: Five copula families with tail dependence analysis, not just Pearson correlation
- **Regime detection**: Hidden Markov Models and Bayesian changepoint detection for structural breaks
- **Stephen Few visualisations**: Publication-quality charts following Few's principles of maximum data-ink ratio
- **Production backtesting**: Multi-asset engine with circuit breakers, slippage models, and regime-aware allocation

## What QuantLite Looks Like

Every chart follows Stephen Few's principles: maximum data-ink ratio, muted palette, direct labels, no chartjunk.

### Fat Tails vs Gaussian

Where the Gaussian distribution underestimates tail risk, QuantLite's EVT and Student-t fitting reveal the true shape:

![Return Distribution](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/return_distribution_fat_tails.png)

### Regime Detection

Hidden Markov Models automatically identify bull, bear, and crisis regimes in price series:

![Regime Timeline](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/regime_timeline.png)

### Portfolio Construction

Efficient frontier with individual assets, minimum variance, and maximum Sharpe portfolios:

![Efficient Frontier](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/efficient_frontier.png)

### Copula Dependency Structures

Five copula families fitted to the same data, showing how each captures different tail behaviour:

![Copula Contours](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/copula_contours.png)

### Correlation Breakdown During Crisis

Rolling correlation spikes during stress periods, revealing the well-documented diversification failure:

![Rolling Correlation](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/rolling_correlation.png)

### Backtest Tearsheet

Strategy equity curve with benchmark comparison:

![Equity Curve](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/equity_curve.png)

> See the `examples/` directory for the scripts that generate all of these charts.

## Installation

```bash
pip install quantlite
```

For development:

```bash
pip install quantlite[dev]
```

Optional dependency for HMM regime detection:

```bash
pip install hmmlearn
```

## Quickstart

```python
import numpy as np
import quantlite as ql
from quantlite.distributions.fat_tails import student_t_process
from quantlite.risk.metrics import value_at_risk, cvar, return_moments
from quantlite.risk.evt import tail_risk_summary

# Generate fat-tailed returns (nu=4 gives realistic equity tail behaviour)
returns = student_t_process(nu=4.0, mu=0.0003, sigma=0.012, n_steps=2520, rng_seed=42)

# Standard risk metrics
var_95 = value_at_risk(returns, alpha=0.05)
cvar_95 = cvar(returns, alpha=0.05)
moments = return_moments(returns)

print(f"VaR (95%):         {var_95:.4f}")
print(f"CVaR (95%):        {cvar_95:.4f}")
print(f"Excess kurtosis:   {moments.kurtosis:.2f}")

# Full tail risk analysis with EVT
summary = tail_risk_summary(returns)
print(f"Hill tail index:   {summary.hill_estimate.tail_index:.2f}")
print(f"GPD shape (xi):    {summary.gpd_fit.shape:.4f}")
print(f"1-in-100 loss:     {summary.return_level_100:.4f}")
```

## Modules

### Risk Metrics

Classical and tail-aware risk measures: VaR (historical, parametric, Cornish-Fisher), CVaR, Sortino ratio, Calmar ratio, Omega ratio, tail ratio, and maximum drawdown with duration tracking.

```python
from quantlite.risk.metrics import (
    value_at_risk, cvar, sortino_ratio, calmar_ratio,
    omega_ratio, tail_ratio, max_drawdown_duration, return_moments,
)

# Cornish-Fisher VaR accounts for skewness and kurtosis
cf_var = value_at_risk(returns, alpha=0.01, method="cornish-fisher")
hist_var = value_at_risk(returns, alpha=0.01, method="historical")
print(f"CF VaR (99%):   {cf_var:.4f}")
print(f"Hist VaR (99%): {hist_var:.4f}")

# Drawdown analysis
dd = max_drawdown_duration(returns)
print(f"Max drawdown: {dd.max_drawdown:.2%}, duration: {dd.duration} periods")
```

![VaR/CVaR Comparison](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/var_cvar_comparison.png)

![Drawdown Chart](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/drawdown_chart.png)

[Detailed documentation: docs/risk.md](docs/risk.md)

### Extreme Value Theory

Rigorous tail modelling via the Generalised Pareto Distribution, Generalised Extreme Value distribution, Hill estimator, and Peaks Over Threshold method.

```python
from quantlite.risk.evt import fit_gpd, hill_estimator, return_level, tail_risk_summary

# Fit GPD to the tail exceedances
gpd = fit_gpd(returns)
print(f"GPD shape (xi): {gpd.shape:.4f}")
print(f"GPD scale:      {gpd.scale:.4f}")
print(f"Exceedances:    {gpd.n_exceedances} / {gpd.n_total}")

# Estimate the 1-in-1000-day loss
rl = return_level(gpd, return_period=1000)
print(f"1-in-1000-day loss: {rl:.4f}")

# Hill estimator for tail index
hill = hill_estimator(returns)
print(f"Tail index (alpha): {hill.tail_index:.2f}")
```

![GPD Tail Fit](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/gpd_tail_fit.png)

![Return Level Plot](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/return_level_plot.png)

[Detailed documentation: docs/evt.md](docs/evt.md)

### Fat-Tailed Distributions

Generate realistic return series from Student-t, Levy stable, regime-switching GBM, and Kou's double-exponential jump-diffusion models.

```python
from quantlite.distributions.fat_tails import (
    student_t_process, levy_stable_process,
    regime_switching_gbm, kou_double_exponential_jump, RegimeParams,
)
import numpy as np

# Student-t returns (nu=4 matches typical equity behaviour)
t_returns = student_t_process(nu=4.0, mu=0.0003, sigma=0.012, n_steps=1260, rng_seed=42)

# Levy stable (alpha < 2 gives infinite variance)
stable_returns = levy_stable_process(alpha=1.7, beta=-0.1, sigma=0.008, n_steps=1260, rng_seed=42)

# Regime-switching GBM (calm and crisis regimes)
calm = RegimeParams(mu=0.08, sigma=0.15)
crisis = RegimeParams(mu=-0.20, sigma=0.40)
transition = np.array([[0.98, 0.02], [0.05, 0.95]])
prices, regimes = regime_switching_gbm(
    [calm, crisis], transition, n_steps=2520, rng_seed=42
)

# Kou's double-exponential jump-diffusion
prices_kou = kou_double_exponential_jump(
    S0=100, mu=0.05, sigma=0.2, lam=1.0, p=0.4,
    eta1=10, eta2=5, n_steps=252, rng_seed=42,
)
```

[Detailed documentation: docs/risk.md](docs/risk.md)

### Copulas

Five copula families for modelling dependence beyond linear correlation: Gaussian, Student-t, Clayton, Gumbel, and Frank. Each provides fitting, simulation, log-likelihood, and analytical tail dependence coefficients.

```python
from quantlite.dependency.copulas import (
    StudentTCopula, ClaytonCopula, select_best_copula,
)
import numpy as np

# Simulate correlated fat-tailed returns
rng = np.random.default_rng(42)
z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=1000)
data = np.column_stack([z[:, 0] * 0.02, z[:, 1] * 0.015])

# Fit Student-t copula (captures tail dependence)
cop = StudentTCopula()
cop.fit(data)
td = cop.tail_dependence()
print(f"Student-t copula: rho={cop.rho:.3f}, nu={cop.nu:.1f}")
print(f"Lower tail dependence: {td['lower']:.3f}")
print(f"Upper tail dependence: {td['upper']:.3f}")

# Automatic model selection by AIC
best = select_best_copula(data)
print(f"Best copula: {best.name} (AIC={best.aic:.1f})")
```

![Tail Dependence Comparison](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/tail_dependence_comparison.png)

[Detailed documentation: docs/copulas.md](docs/copulas.md)

### Correlation Analysis

Rolling, exponentially-weighted, stress-conditional, and rank-based correlation measures.

```python
from quantlite.dependency.correlation import (
    rolling_correlation, exponential_weighted_correlation,
    stress_correlation, correlation_breakdown_test,
)
import pandas as pd
import numpy as np

# Simulate two equity return series
rng = np.random.default_rng(42)
n = 504
rets = pd.DataFrame({
    "Equities": rng.normal(0.0003, 0.012, n),
    "Bonds": rng.normal(0.0001, 0.005, n),
    "Commodities": rng.normal(0.0002, 0.015, n),
})

# EWMA correlation (more responsive to regime changes)
ewma_corr = exponential_weighted_correlation(rets["Equities"], rets["Bonds"], halflife=30)

# Stress correlation (correlation during drawdowns)
stress_corr = stress_correlation(rets, threshold_percentile=10)
print("Stress-period correlation matrix:")
print(stress_corr)

# Test for correlation breakdown
test = correlation_breakdown_test(rets)
print(f"Calm corr: {test['calm_corr']:.3f}, Stress corr: {test['stress_corr']:.3f}")
print(f"p-value: {test['p_value']:.4f}")
```

### Hierarchical Risk Parity

Lopez de Prado's HRP method produces diversified, stable portfolio weights without covariance matrix inversion.

```python
from quantlite.dependency.clustering import hrp_weights
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
returns_df = pd.DataFrame({
    "US_Equity": rng.normal(0.0003, 0.012, 504),
    "EU_Equity": rng.normal(0.0002, 0.014, 504),
    "Govt_Bonds": rng.normal(0.0001, 0.004, 504),
    "Gold": rng.normal(0.0001, 0.010, 504),
    "REITs": rng.normal(0.0002, 0.013, 504),
})

weights = hrp_weights(returns_df)
for asset, w in weights.items():
    print(f"  {asset}: {w:.2%}")
```

[Detailed documentation: docs/copulas.md](docs/copulas.md)

### Regime Detection

Hidden Markov Models and Bayesian changepoint detection for identifying structural breaks in market behaviour.

```python
from quantlite.regimes.hmm import fit_regime_model, select_n_regimes
from quantlite.regimes.changepoint import detect_changepoints
from quantlite.regimes.conditional import conditional_metrics, regime_aware_var
from quantlite.distributions.fat_tails import student_t_process

# Generate a return series with embedded regime structure
returns = student_t_process(nu=4, mu=0.0003, sigma=0.015, n_steps=1260, rng_seed=42)

# Fit a 2-regime HMM (requires hmmlearn)
model = fit_regime_model(returns, n_regimes=2, rng_seed=42)
print(f"Regime means: {model.means}")
print(f"Regime variances: {model.variances}")
print(f"Stationary distribution: {model.stationary_distribution}")

# Conditional risk metrics per regime
cond = conditional_metrics(returns, model.regime_labels)
for regime, metrics in cond.items():
    print(f"Regime {regime}: mean={metrics['mean']:.5f}, vol={metrics['volatility']:.5f}")

# Regime-aware VaR
rvar = regime_aware_var(returns, model.regime_labels, alpha=0.05)
print(f"Regime-aware VaR (95%): {rvar:.4f}")

# Bayesian changepoint detection (no hmmlearn required)
cps = detect_changepoints(returns, method="bayesian", penalty=50)
for cp in cps:
    print(f"  Changepoint at index {cp.index}, confidence={cp.confidence:.2f}, {cp.direction}")
```

![Transition Matrix](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/transition_matrix.png)

![Changepoint Detection](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/changepoint_detection.png)

[Detailed documentation: docs/regimes.md](docs/regimes.md)

### Portfolio Optimisation

Six optimisation methods: Markowitz mean-variance, minimum variance, CVaR optimisation, risk parity, HRP, and maximum Sharpe. Plus Black-Litterman and Kelly criterion.

```python
from quantlite.portfolio.optimisation import (
    mean_variance_weights, mean_cvar_weights, risk_parity_weights,
    hrp_weights, max_sharpe_weights, black_litterman, kelly_criterion,
)
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

# CVaR-optimised portfolio (minimises expected shortfall)
cvar_port = mean_cvar_weights(returns_df, alpha=0.05)
print(f"CVaR portfolio: return={cvar_port.expected_return:.2%}, risk={cvar_port.expected_risk:.4f}")
for asset, w in cvar_port.weights.items():
    print(f"  {asset}: {w:.2%}")

# Risk parity
rp = risk_parity_weights(returns_df)
print(f"\nRisk parity Sharpe: {rp.sharpe:.2f}")

# Kelly criterion for position sizing
from quantlite.distributions.fat_tails import student_t_process
strat_returns = student_t_process(nu=5, mu=0.001, sigma=0.02, n_steps=252, rng_seed=42)
kelly_f = kelly_criterion(strat_returns)
print(f"Full Kelly fraction: {kelly_f:.2f}")
```

![HRP Dendrogram](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/hrp_dendrogram.png)

![Weight Comparison](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/weight_comparison.png)

[Detailed documentation: docs/portfolio.md](docs/portfolio.md)

### Rebalancing Strategies

Calendar-based, threshold-triggered, and regime-tactical rebalancing with full turnover tracking.

```python
from quantlite.portfolio.rebalancing import (
    rebalance_calendar, rebalance_threshold, rebalance_tactical,
)

# Monthly calendar rebalance with equal weights
def equal_weight(df):
    n = df.shape[1]
    return {col: 1.0 / n for col in df.columns}

result = rebalance_calendar(returns_df, equal_weight, freq="monthly")
print(f"Rebalances: {result.n_rebalances}, turnover: {result.turnover:.2f}")

# Threshold rebalance (rebalance when any weight drifts >5% from target)
result_thresh = rebalance_threshold(returns_df, equal_weight, threshold=0.05)
print(f"Threshold rebalances: {result_thresh.n_rebalances}")
```

[Detailed documentation: docs/portfolio.md](docs/portfolio.md)

### Reports

Generate professional tearsheet reports from backtest results with one line:

```python
from quantlite.report import tearsheet

tearsheet(result, save="portfolio_report.html")
```

Includes executive summary, risk metrics, drawdown analysis, monthly returns heatmap, rolling statistics, and regime analysis. Outputs HTML with interactive charts or PDF for print.

[Detailed documentation: docs/reports.md](docs/reports.md)

### Backtesting Engine

Multi-asset production backtesting with configurable slippage, transaction costs, risk limits, and circuit breakers.

```python
from quantlite.backtesting.engine import (
    run_backtest, BacktestConfig, BacktestContext, RiskLimits, SlippageModel,
)
from quantlite.backtesting.analysis import performance_summary, trade_analysis
from quantlite.backtesting.signals import momentum_signal, volatility_targeting
from quantlite.data_generation import correlated_gbm
import numpy as np
import pandas as pd

# Generate multi-asset price data
cov = np.array([[0.04, 0.01, 0.005], [0.01, 0.03, 0.002], [0.005, 0.002, 0.01]])
prices_df = correlated_gbm(
    S0_list=[100, 50, 200], mu_list=[0.08, 0.06, 0.04],
    cov_matrix=cov, steps=504, rng_seed=42, return_as="dataframe",
)
prices_df.index = pd.bdate_range("2022-01-03", periods=len(prices_df))
prices_df.columns = ["Equities", "Bonds", "Commodities"]

# Momentum-based allocation with risk limits
def momentum_allocation(ctx: BacktestContext) -> dict[str, float]:
    if len(ctx.historical_returns) < 60:
        n = len(ctx.current_prices)
        return {a: 1.0 / n for a in ctx.current_prices.index}
    mom = momentum_signal(ctx.historical_returns.cumsum() + 100, lookback=60)
    latest = mom.iloc[-1]
    w = latest.clip(lower=0)
    total = w.sum()
    if total > 0:
        w = w / total
    return w.to_dict()

config = BacktestConfig(
    initial_capital=1_000_000,
    slippage_model=SlippageModel(kind="fixed", spread_bps=5),
    risk_limits=RiskLimits(max_drawdown=-0.15, max_position_pct=0.5),
    rebalance_freq="weekly",
    fee_per_trade_pct=0.001,
)

result = run_backtest(prices_df, momentum_allocation, config)
print(performance_summary(result))
```

[Detailed documentation: docs/portfolio.md](docs/portfolio.md)

### Data Connectors

A unified `fetch()` function that pulls OHLCV data from Yahoo Finance, crypto exchanges (CCXT), FRED, or local files through a single interface.

```python
from quantlite.data import fetch

# Equities from Yahoo Finance
df = fetch("AAPL", period="5y")
data = fetch(["AAPL", "MSFT", "GOOG"], period="5y")

# Crypto from any CCXT-supported exchange
df = fetch("BTC/USDT", source="ccxt", exchange="binance")

# Macroeconomic data from FRED
df = fetch("DGS10", source="fred")

# Local CSV or Parquet files
df = fetch("prices.csv", source="local")

# Mix sources in one call
data = fetch({"AAPL": "yahoo", "BTC/USDT": {"source": "ccxt", "exchange": "binance"}})
```

Install only the sources you need:

```bash
pip install quantlite[yahoo]    # Yahoo Finance
pip install quantlite[crypto]   # Cryptocurrency exchanges
pip install quantlite[fred]     # FRED macroeconomic data
pip install quantlite[all]      # All data sources
```

Custom data sources can be registered via the plugin architecture:

```python
from quantlite.data import DataSource, register_source

@register_source("my_api")
class MySource(DataSource):
    def fetch(self, symbol, **kwargs):
        ...
    def supported_symbols(self):
        return None
```

[Detailed documentation: docs/data.md](docs/data.md)

### Data Generation

Stochastic process simulators: Geometric Brownian Motion, correlated multi-asset GBM, Ornstein-Uhlenbeck mean-reversion, and Merton jump-diffusion.

```python
from quantlite.data_generation import (
    geometric_brownian_motion, correlated_gbm,
    ornstein_uhlenbeck, merton_jump_diffusion,
)

# Single asset with jumps
prices = merton_jump_diffusion(
    S0=100, mu=0.05, sigma=0.2, lamb=0.5,
    jump_mean=-0.02, jump_std=0.08, steps=252, rng_seed=42,
)

# Mean-reverting interest rate process
rates = ornstein_uhlenbeck(
    x0=0.03, theta=0.5, mu=0.04, sigma=0.01, steps=252, rng_seed=42,
)
```

### Instruments

Black-Scholes option pricing with Greeks, bond pricing with duration and yield-to-maturity, and Monte Carlo pricing for exotic options (barrier, Asian).

```python
from quantlite.instruments.option_pricing import black_scholes_call, black_scholes_greeks
from quantlite.instruments.bond_pricing import bond_price, bond_duration
from quantlite.instruments.exotic_options import barrier_option_knock_out, asian_option_arithmetic

# European call option
call = black_scholes_call(S=100, K=105, T=0.5, r=0.05, sigma=0.2)
greeks = black_scholes_greeks(S=100, K=105, T=0.5, r=0.05, sigma=0.2)
print(f"Call price: {call:.2f}, Delta: {greeks.delta:.4f}, Gamma: {greeks.gamma:.4f}")

# Coupon bond
price = bond_price(face_value=1000, coupon_rate=0.05, market_rate=0.04, maturity=10)
dur = bond_duration(face_value=1000, coupon_rate=0.05, market_rate=0.04, maturity=10)
print(f"Bond price: {price:.2f}, Duration: {dur:.2f} years")

# Down-and-out barrier option
barrier_price = barrier_option_knock_out(S0=100, K=105, H=85, T=1, r=0.05, sigma=0.2, sims=50000)
print(f"Barrier option: {barrier_price:.2f}")
```

### Visualisation

Stephen Few-inspired charts with maximum data-ink ratio, muted palette, direct labels, and no chartjunk. Covers risk dashboards, copula contours, regime timelines, efficient frontiers, and correlation heatmaps.

```python
from quantlite.viz.theme import apply_few_theme, FEW_PALETTE, few_figure, bullet_graph
from quantlite.viz.risk import plot_tail_distribution, plot_risk_dashboard, plot_drawdown
from quantlite.viz.dependency import plot_copula_contour, plot_correlation_matrix
from quantlite.viz.regimes import plot_regime_timeline, plot_regime_summary
from quantlite.viz.portfolio import plot_efficient_frontier, plot_weights_over_time

# Apply the Few theme globally
apply_few_theme()

# Single-page risk dashboard
fig, axes = plot_risk_dashboard(returns)

# Tail distribution with GPD overlay
from quantlite.risk.evt import fit_gpd
gpd = fit_gpd(returns)
fig, ax = plot_tail_distribution(returns, gpd_fit=gpd)

# Efficient frontier
fig, ax = plot_efficient_frontier(returns_df)
```

[Detailed documentation: docs/visualisation.md](docs/visualisation.md)

### Interactive Visualisation (Plotly)

Every matplotlib chart has an interactive Plotly equivalent. Install with `pip install quantlite[plotly]`.

```python
# Option 1: Import directly from the Plotly backend
from quantlite.viz.plotly_backend.risk import plot_var_comparison
fig = plot_var_comparison(returns)
fig.show()

# Option 2: Use the backend parameter on existing functions
from quantlite.viz.risk import plot_drawdown
fig = plot_drawdown(returns, backend="plotly")
fig.show()
```

Same Stephen Few theme, same muted palette, but with hover info, zoom, and native Jupyter rendering. See [docs/interactive_viz.md](docs/interactive_viz.md) for the full chart reference.

## Module Reference

| Module | Description |
|--------|-------------|
| `quantlite.data` | Unified data connectors: Yahoo Finance, CCXT, FRED, local files, plugin registry, caching |
| `quantlite.risk.metrics` | VaR, CVaR, Sortino, Calmar, Omega, tail ratio, drawdowns |
| `quantlite.risk.evt` | GPD, GEV, Hill estimator, POT, return levels |
| `quantlite.distributions.fat_tails` | Student-t, Levy stable, regime-switching GBM, Kou jump-diffusion |
| `quantlite.dependency.copulas` | Gaussian, Student-t, Clayton, Gumbel, Frank copulas |
| `quantlite.dependency.correlation` | Rolling, EWMA, stress, rank correlation |
| `quantlite.dependency.clustering` | HRP (Hierarchical Risk Parity) |
| `quantlite.regimes.hmm` | Hidden Markov Model regime detection |
| `quantlite.regimes.changepoint` | CUSUM and Bayesian changepoint detection |
| `quantlite.regimes.conditional` | Regime-conditional risk metrics and VaR |
| `quantlite.portfolio.optimisation` | Mean-variance, CVaR, risk parity, HRP, Black-Litterman, Kelly |
| `quantlite.portfolio.rebalancing` | Calendar, threshold, and tactical rebalancing |
| `quantlite.backtesting.engine` | Multi-asset backtesting with circuit breakers |
| `quantlite.backtesting.signals` | Momentum, mean reversion, trend following, vol targeting |
| `quantlite.backtesting.analysis` | Performance summaries, monthly tables, regime attribution |
| `quantlite.data_generation` | GBM, correlated GBM, OU, Merton jump-diffusion |
| `quantlite.instruments` | Black-Scholes, bonds, barrier and Asian options |
| `quantlite.viz` | Stephen Few-themed risk, dependency, regime, and portfolio charts |
| `quantlite.metrics` | Basic annualised return, volatility, Sharpe, max drawdown |
| `quantlite.monte_carlo` | Monte Carlo simulation harness |
| `quantlite.contagion` | CoVaR, Delta CoVaR, MES, Granger causality, causal networks |
| `quantlite.network` | Correlation networks, eigenvector centrality, cascade simulation, community detection |
| `quantlite.diversification` | ENB, entropy diversification, tail diversification, diversification ratio, Herfindahl index |
| `quantlite.crypto.stablecoin` | Depeg probability, peg deviation tracking, recovery time, reserve risk scoring |
| `quantlite.crypto.exchange` | Exchange concentration (HHI), wallet risk, proof of reserves, liquidity risk, slippage |
| `quantlite.crypto.onchain` | Wallet exposure, TVL tracking, DeFi dependency graphs, smart contract risk scoring |
| `quantlite.factors.classical` | Fama-French three/five-factor, Carhart four-factor, factor attribution, factor summary |
| `quantlite.factors.custom` | CustomFactor, significance testing, correlation matrix, factor portfolios, decay analysis |
| `quantlite.factors.tail_risk` | CVaR decomposition, regime factor exposure, crowding score, tail factor beta |
| `quantlite.simulation.evt_simulation` | EVT tail simulation, parametric tail simulation, historical bootstrap EVT, scenario fan |
| `quantlite.simulation.copula_mc` | Gaussian copula MC, t-copula MC, stress correlation MC, joint tail probability |
| `quantlite.simulation.regime_mc` | Regime-switching simulation, stress test scenarios, reverse stress test, simulation summary |

## v0.9: Fat-Tail Monte Carlo

Three simulation families that go beyond naive Monte Carlo. Standard MC assumes returns are Gaussian and independent. QuantLite's fat-tail MC uses EVT for realistic tails, copulas for joint dependence, and regime switching for structural breaks.

### EVT Tail Simulation

Generate scenarios with GPD-fitted tails that respect the true shape of financial returns.

```python
import numpy as np
from quantlite.simulation import evt_tail_simulation, scenario_fan

rng = np.random.default_rng(42)
returns = np.concatenate([
    rng.normal(0.0003, 0.01, 900),
    rng.standard_t(3, 100) * 0.03,
])

# EVT-based scenario generation
simulated = evt_tail_simulation(returns, n_scenarios=20000, seed=42)
print(f"Historical 1st pctl: {np.percentile(returns, 1):.4f}")
print(f"Simulated 1st pctl:  {np.percentile(simulated, 1):.4f}")

# Fan chart across multiple horizons
fan = scenario_fan(returns, horizons=[1, 5, 21, 63, 252])
for h in fan["horizons"]:
    p5, p95 = fan["fans"][h]["5"], fan["fans"][h]["95"]
    print(f"  {h:>3}d: [{p5:+.2%}, {p95:+.2%}]")
```

![EVT Tail Simulation](docs/images/evt_tail_simulation.png)

![Scenario Fan](docs/images/scenario_fan.png)

### Copula Monte Carlo

Multivariate simulation that preserves fat-tailed marginals and captures tail dependence.

```python
import numpy as np
from quantlite.simulation import t_copula_mc, joint_tail_probability

rng = np.random.default_rng(42)
fund_a = np.concatenate([rng.normal(0.0005, 0.012, 900), rng.standard_t(3, 100) * 0.03])
fund_b = np.concatenate([rng.normal(0.0003, 0.015, 900), rng.standard_t(4, 100) * 0.025])
corr = np.array([[1.0, 0.6], [0.6, 1.0]])

# t-copula captures tail dependence that Gaussian copula misses
simulated = t_copula_mc([fund_a, fund_b], corr, df=4, n_scenarios=50000)

result = joint_tail_probability(simulated, thresholds=[-0.03, -0.03])
print(f"Joint crash probability: {result['joint_probability']:.4f}")
print(f"Marginal probabilities:  {result['marginal_probabilities']}")
```

![Copula Comparison](docs/images/copula_comparison.png)

![Stressed Correlations](docs/images/stressed_correlations.png)

### Regime-Switching Simulation

Paths that switch between calm, volatile, and crisis regimes via a Markov chain.

```python
import numpy as np
from quantlite.simulation import (
    regime_switching_simulation,
    reverse_stress_test,
    simulation_summary,
)

regimes = [
    {"mu": 0.0004, "sigma": 0.008},   # Calm
    {"mu": 0.0001, "sigma": 0.020},   # Volatile
    {"mu": -0.002, "sigma": 0.035},   # Crisis
]
transition = np.array([
    [0.95, 0.04, 0.01],
    [0.10, 0.80, 0.10],
    [0.05, 0.15, 0.80],
])

sim = regime_switching_simulation(regimes, transition, n_steps=252, n_scenarios=5000)
stats = simulation_summary(sim["returns"])
print(f"VaR 95%: {stats['var']['95%']:.2%}")
print(f"CVaR 95%: {stats['cvar']['95%']:.2%}")
print(f"P(loss > 20%): {stats['probability_of_ruin']['20%']:.2%}")
```

![Regime Simulation](docs/images/regime_simulation.png)

![Reverse Stress Test](docs/images/reverse_stress_test.png)

See [docs/simulation_evt.md](docs/simulation_evt.md), [docs/simulation_copula.md](docs/simulation_copula.md), and [docs/simulation_regime.md](docs/simulation_regime.md) for the full API reference.

## v0.8: Factor Models

Three modules for comprehensive factor analysis: classical academic models, custom factor tools, and tail-risk-aware factor decomposition.

### Classical Factor Models

Decompose returns into systematic factor exposures and genuine alpha.

```python
from quantlite.factors import fama_french_three, factor_summary

# Fama-French three-factor regression
result = fama_french_three(fund_returns, market, smb, hml)
print(f"Alpha: {result['alpha']:.5f} (t={result['t_stats']['alpha']:.2f})")
print(f"Market beta: {result['betas']['market']:.3f}")
print(f"R-squared: {result['r_squared']:.3f}")
```

![Factor Betas](docs/images/factor_betas.png)

### Custom Factor Tools

Build, test, and analyse proprietary factors.

```python
from quantlite.factors import CustomFactor, factor_portfolio, factor_decay

# Test factor decay
decay = factor_decay(returns, momentum_signal, max_lag=20)
print(f"Half-life: {decay['half_life']:.1f} periods")

# Build long-short portfolios
result = factor_portfolio(stock_returns, factor_values, n_quantiles=5)
print(f"Long-short spread: {result['spread'] * 252:.2%} annualised")
```

![Factor Quintiles](docs/images/factor_quintiles.png)

### Tail Risk Factor Analysis

Understand how factor exposures behave in the tails and across regimes.

```python
from quantlite.factors import tail_factor_beta, factor_crowding_score

# Tail betas: how exposures amplify in crises
result = tail_factor_beta(returns, [market, value], ["Market", "Value"])
for name in ["Market", "Value"]:
    print(f"{name}: full={result['full_betas'][name]:.2f}, "
          f"tail={result['tail_betas'][name]:.2f}")

# Factor crowding detection
crowd = factor_crowding_score([value_rets, momentum_rets])
print(f"Crowding score: {crowd['current_score']:.3f}")
```

![Factor Crowding](docs/images/factor_crowding.png)

See [docs/factors_classical.md](docs/factors_classical.md), [docs/factors_custom.md](docs/factors_custom.md), and [docs/factors_tail_risk.md](docs/factors_tail_risk.md) for the full API reference.

## v0.4: The Taleb Stack

Three new modules bringing Nassim Taleb's key ideas into quantitative practice: ergodicity economics, antifragility measurement, and scenario stress testing.

### Ergodicity Economics

The ensemble average lies. The time average tells the truth. Measure the gap and find the optimal Kelly leverage.

```python
from quantlite.ergodicity import time_average, ensemble_average, ergodicity_gap

returns = [0.50, -0.40, 0.50, -0.40, 0.50, -0.40]
print(f"Ensemble average: {ensemble_average(returns):+.4f}")   # +0.0500 (looks great)
print(f"Time average:     {time_average(returns):+.4f}")        # -0.0513 (you go broke)
print(f"Ergodicity gap:   {ergodicity_gap(returns):+.4f}")      # +0.1013 (the lie)
```

![Leverage vs Growth](https://raw.githubusercontent.com/prasants/QuantLite/main/examples/ergodicity_leverage.png)

[Documentation: docs/ergodicity.md](docs/ergodicity.md)

### Antifragility Measurement

Quantify whether your portfolio gains from disorder or breaks under it.

```python
from quantlite.antifragile import antifragility_score

fragile_score = antifragility_score(short_vol_returns)     # Negative: fragile
robust_score = antifragility_score(index_returns)           # Near zero: robust
antifragile_score_ = antifragility_score(long_vol_returns)  # Positive: antifragile
```

![Antifragility Scores](https://raw.githubusercontent.com/prasants/QuantLite/main/examples/antifragility_scores.png)

[Documentation: docs/antifragility.md](docs/antifragility.md)

### Scenario Stress Testing

Build crisis scenarios with a fluent API and stress-test your portfolio against historical and hypothetical crises.

```python
from quantlite.scenarios import stress_test, SCENARIO_LIBRARY

weights = {"SPX": 0.30, "BTC": 0.15, "ETH": 0.10, "BONDS_10Y": 0.30, "GLD": 0.15}
result = stress_test(weights, SCENARIO_LIBRARY["2008 GFC"])
print(f"Portfolio impact: {result['portfolio_impact']:.2%}")
print(f"Survives: {result['survival']}")
```

![Scenario Heatmap](https://raw.githubusercontent.com/prasants/QuantLite/main/examples/scenario_heatmap.png)

[Documentation: docs/scenarios.md](docs/scenarios.md)

### New Module Reference

| Module | Description |
|--------|-------------|
| `quantlite.ergodicity` | Time-average vs ensemble-average growth, Kelly fraction, leverage effect |
| `quantlite.antifragile` | Antifragility score, convexity, Fourth Quadrant, barbell allocation, Lindy, skin in the game |
| `quantlite.scenarios` | Composable scenario engine, pre-built crisis library, fragility heatmap, shock propagation |

## v0.5: Honest Backtesting

Three new modules that answer the question every quant should ask but rarely does: "Is this backtest result real, or did I just get lucky?"

### Strategy Forensics

Most reported Sharpe ratios are inflated by multiple testing. The `quantlite.forensics` module implements Lopez de Prado's Deflated Sharpe Ratio framework to separate signal from noise.

```python
from quantlite.forensics import deflated_sharpe_ratio

# You tried 50 strategies and the best had Sharpe 1.8
dsr = deflated_sharpe_ratio(observed_sharpe=1.8, n_trials=50, n_obs=252)
print(f"Probability Sharpe is genuine: {dsr:.2%}")
```

![Deflated Sharpe](docs/images/deflated_sharpe.png)

![Minimum Track Record](docs/images/min_track_record.png)

![Signal Decay](docs/images/signal_decay.png)

[Documentation: docs/forensics.md](docs/forensics.md)

### Overfitting Detection

The more strategies you test, the more likely your best one is a fluke. Detect and quantify backtest overfitting with CSCV, walk-forward validation, and the TrialTracker.

```python
from quantlite.overfit import TrialTracker

with TrialTracker("momentum_search") as tracker:
    for lookback in [10, 20, 40, 60, 120]:
        tracker.log(params={"lookback": lookback}, sharpe=1.2, returns=returns)
    print(f"Overfitting probability: {tracker.overfitting_probability():.2%}")
```

![Overfitting Trials](docs/images/overfitting_trials.png)

![Walk-Forward Validation](docs/images/walk_forward.png)

[Documentation: docs/overfitting.md](docs/overfitting.md)

### Resampled Backtesting

A single backtest is a single draw from a distribution. Bootstrap methods build honest confidence intervals around any performance metric.

```python
from quantlite.resample import bootstrap_sharpe_distribution

result = bootstrap_sharpe_distribution(returns, n_samples=2000, seed=42)
print(f"Sharpe: {result['point_estimate']:.2f}")
print(f"95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
```

![Sharpe Distribution](docs/images/sharpe_distribution.png)

![Drawdown CI](docs/images/drawdown_ci.png)

![Block vs Stationary](docs/images/block_vs_stationary.png)

[Documentation: docs/resampling.md](docs/resampling.md)

### New Module Reference

| Module | Description |
|--------|-------------|
| `quantlite.forensics` | Deflated Sharpe Ratio, Probabilistic Sharpe, haircut adjustments, minimum track record, signal decay |
| `quantlite.overfit` | CSCV/PBO, TrialTracker, multiple testing correction, walk-forward validation |
| `quantlite.resample` | Block and stationary bootstrap, confidence intervals for Sharpe, drawdown, and custom metrics |

## v0.6: Systemic Risk and Diversification

Three new modules for understanding how risk propagates through portfolios and markets: contagion analysis, financial network modelling, and diversification diagnostics.

### Contagion and Systemic Risk

When one asset crashes, how much does it drag down the rest? CoVaR and Marginal Expected Shortfall measure the contagion channels that standard correlation misses.

```python
from quantlite.contagion import covar, systemic_risk_contributions

result = covar(market_returns, bank_returns, alpha=0.05)
print(f"Unconditional VaR: {result['var_b']:.4f}")
print(f"CoVaR (market stress): {result['covar']:.4f}")

contributions = systemic_risk_contributions(returns_df, alpha=0.05)
for asset, mes in contributions.items():
    print(f"{asset:15s} MES: {mes:+.4f}")
```

![CoVaR Comparison](docs/images/covar_comparison.png)

![Systemic Risk Contributions](docs/images/systemic_risk_contributions.png)

![Granger Causality](docs/images/granger_causality.png)

[Documentation: docs/contagion.md](docs/contagion.md)

### Financial Network Analysis

Model markets as networks. Find the most systemically important assets, simulate shock cascades, and detect communities of co-moving assets.

```python
from quantlite.network import network_summary, cascade_simulation

summary = network_summary(returns_df, threshold=0.4)
for i, name in enumerate(summary["nodes"]):
    print(f"{name:10s} centrality: {summary['centrality'][i]:.4f}")

cascade = cascade_simulation(
    summary["network"]["adjacency_matrix"],
    shock_node=0,
    shock_magnitude=-0.50,
)
```

![Correlation Network](docs/images/correlation_network.png)

![Cascade Simulation](docs/images/cascade_simulation.png)

![Eigenvector Centrality](docs/images/eigenvector_centrality.png)

[Documentation: docs/network.md](docs/network.md)

### Diversification Analysis

You think you are diversified. These six metrics tell you whether that is actually true, and whether your diversification survives crises.

```python
from quantlite.diversification import (
    effective_number_of_bets,
    tail_diversification,
)

enb = effective_number_of_bets(weights, cov)
print(f"Assets: 10, Effective bets: {enb:.1f}")

td = tail_diversification(returns_df, weights, alpha=0.05)
print(f"Normal diversification: {td['normal_diversification']:.3f}")
print(f"Tail diversification:   {td['tail_diversification']:.3f}")
```

![Diversification Metrics](docs/images/diversification_metrics.png)

![Tail Diversification](docs/images/tail_diversification.png)

![Marginal Tail Risk](docs/images/marginal_tail_risk.png)

[Documentation: docs/diversification.md](docs/diversification.md)

### New Module Reference

| Module | Description |
|--------|-------------|
| `quantlite.contagion` | CoVaR, Delta CoVaR, Marginal Expected Shortfall, Granger causality, causal networks |
| `quantlite.network` | Correlation networks, eigenvector centrality, cascade simulation, community detection |
| `quantlite.diversification` | Effective Number of Bets, entropy diversification, tail diversification, diversification ratio, Herfindahl index |

## v1.0: The Dream API

QuantLite v1.0 introduces the **Dream API**, a five-function pipeline that chains the entire quant workflow:

```python
import quantlite as ql

data = ql.fetch(["AAPL", "BTC-USD", "GLD", "TLT"], period="5y")
regimes = ql.detect_regimes(data, n_regimes=3)
weights = ql.construct_portfolio(data, regime_aware=True, regimes=regimes)
result = ql.backtest(data, weights)
ql.tearsheet(result, regimes=regimes, save="portfolio.txt")
```

### Regime-Aware Portfolio Construction

Weights automatically tilt defensive during crisis regimes, increasing allocations to bonds and gold while reducing equity exposure:

![Pipeline Equity Curve](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/pipeline_equity_curve.png)

### Regime Risk Analysis

VaR, CVaR, volatility, skewness, and kurtosis computed separately for each market regime:

![Regime Risk Summary](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/regime_risk_summary.png)

### Regime-Filtered Backtesting

Different weight sets applied per regime, with full performance attribution:

![Regime Filtered Backtest](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/regime_filtered_backtest.png)

### v1.0 Module Reference

| Module | Description |
|--------|-------------|
| `quantlite.pipeline` | Dream API: `fetch`, `detect_regimes`, `construct_portfolio`, `backtest`, `tearsheet` |
| `quantlite.regime_integration` | Regime-conditional risk, defensive portfolio tilting, filtered backtesting, tearsheets |
| `quantlite.regimes` | HMM regime detection, Bayesian changepoint detection, conditional metrics |
| `quantlite.portfolio` | Markowitz, CVaR, risk parity, HRP, Black-Litterman, Kelly optimisation |
| `quantlite.backtesting` | Multi-asset engine with circuit breakers, slippage, regime-aware signals |
| `quantlite.risk` | VaR, CVaR, Sortino, Calmar, omega ratio, tail ratio, drawdown analysis |
| `quantlite.data` | Unified fetching: Yahoo Finance, CCXT, FRED, local files |
| `quantlite.distributions` | Student-t, stable, GPD, GEV fitting and simulation |
| `quantlite.simulation` | Fat-tail Monte Carlo: EVT-based, copula-based engines |
| `quantlite.viz` | Stephen Few-inspired charts: regimes, portfolios, risk dashboards |
| `quantlite.factors` | Classical factors, custom factors, tail risk factors |
| `quantlite.ergodicity` | Time-average vs ensemble-average growth, Kelly sizing |
| `quantlite.antifragile` | Barbell metrics, convexity scores, stress testing |
| `quantlite.scenarios` | Historical, hypothetical, and Monte Carlo scenario analysis |
| `quantlite.forensics` | Overfitting detection, data snooping tests, walk-forward analysis |
| `quantlite.contagion` | CoVaR, Delta CoVaR, MES, Granger causality |
| `quantlite.network` | Correlation networks, centrality, cascade simulation |
| `quantlite.diversification` | Effective Number of Bets, entropy, tail diversification |
| `quantlite.crypto` | On-chain risk, stablecoin depeg, exchange risk scoring |

## Design Philosophy

1. **Fat tails are the default.** Gaussian assumptions are explicitly opt-in, never implicit.
2. **Typed return values.** Every function returns frozen dataclasses with clear attributes, not opaque dicts.
3. **Composable modules.** Risk metrics feed into portfolio optimisation which feeds into backtesting. Each layer works independently.
4. **Honest modelling.** If a method has known limitations (e.g., Gaussian copula has zero tail dependence), the docstring says so.
5. **Reproducible.** Every stochastic function accepts `rng_seed` for deterministic output.

## Requirements

- Python >= 3.10
- numpy >= 1.24
- pandas >= 2.0
- scipy >= 1.10
- matplotlib >= 3.7
- mplfinance

Optional: `hmmlearn` for HMM regime detection. `yfinance` for Yahoo data. `ccxt` for crypto. `fredapi` for FRED.

## Contributing

Contributions are welcome. Please ensure:

1. All new functions have type hints and docstrings
2. Tests pass: `pytest`
3. Code is formatted: `ruff check` and `ruff format`
4. British spellings in all documentation

## License

MIT License. See [LICENSE](LICENSE) for details.

## Links

- [PyPI](https://pypi.org/project/quantlite/)
- [GitHub](https://github.com/prasants/QuantLite)
- [Issue Tracker](https://github.com/prasants/QuantLite/issues)
