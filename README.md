# QuantLite

[![PyPI version](https://img.shields.io/pypi/v/quantlite)](https://pypi.org/project/quantlite/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A fat-tail-native quantitative finance toolkit for Python.**

Most quantitative finance libraries bolt fat tails on as an afterthought, if they bother at all. They fit Gaussians, compute VaR with normal assumptions, and hope the tails never bite. The tails always bite.

QuantLite starts from the other end. Every distribution is fat-tailed by default. Every risk metric accounts for extremes. Every backtest ships with an honesty check. The result is a toolkit that models markets as they actually behave, not as textbooks wish they would.

```python
import quantlite as ql

data = ql.fetch(["AAPL", "BTC-USD", "GLD", "TLT"], period="5y")
regimes = ql.detect_regimes(data, n_regimes=3)
weights = ql.construct_portfolio(data, regime_aware=True, regimes=regimes)
result = ql.backtest(data, weights)
ql.tearsheet(result, regimes=regimes, save="portfolio.html")
```

Five lines. Fetch data, detect market regimes, build a regime-aware portfolio, backtest it, and generate a full tearsheet. That is the Dream API.

---

## Why QuantLite

- **Fat tails are the default, not an afterthought.** Student-t, Lévy stable, GPD, and GEV distributions are first-class citizens. Gaussian is explicitly opt-in, never implicit.
- **Operationalises Taleb, Peters, and Lopez de Prado.** Ergodicity economics, antifragility scoring, the Fourth Quadrant map, Deflated Sharpe Ratios, and CSCV overfitting detection are built in, not bolted on.
- **Every backtest comes with an honesty check.** Bootstrap confidence intervals, multiple-testing corrections, and walk-forward validation ensure you know whether your Sharpe ratio is genuine or a statistical artefact.
- **Every chart follows Stephen Few's principles.** Maximum data-ink ratio, muted palette, direct labels, no chartjunk. Publication-ready by default.

---

## Visual Showcase

### Fat Tails vs Gaussian

Where the Gaussian underestimates tail risk, EVT and Student-t fitting reveal the true shape of returns. The difference is where fortunes are lost.

![Return Distribution](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/return_distribution_fat_tails.png)

### Regime Timeline

Hidden Markov Models automatically identify bull, bear, and crisis regimes. Your portfolio should know which world it is living in.

![Regime Timeline](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/regime_timeline.png)

### Ergodicity Gap

The ensemble average says you are making money. The time average says you are going broke. The gap between them is the most important number in finance that nobody computes.

![Ergodicity Gap](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/ergodicity_gap.png)

### Fourth Quadrant Map

Taleb's Fourth Quadrant: where payoffs are extreme and distributions are unknown. Know which quadrant your portfolio lives in before the market tells you.

![Fourth Quadrant Map](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/fourth_quadrant_map.png)

### Deflated Sharpe Ratio

You tested 50 strategies and picked the best. The Deflated Sharpe Ratio tells you the probability that your winner is genuine, not a multiple-testing artefact.

![Deflated Sharpe](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/deflated_sharpe.png)

### Scenario Stress Heatmap

How does your portfolio fare under the GFC, COVID, the taper tantrum, and a dozen other crises? One glance.

![Scenario Heatmap](https://raw.githubusercontent.com/prasants/QuantLite/main/examples/scenario_heatmap.png)

### Copula Contours

Five copula families fitted to the same data. Gaussian copula says tail dependence is zero. Student-t and Clayton disagree. They are right.

![Copula Contours](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/copula_contours.png)

### Pipeline Equity Curve

The Dream API in action: regime-aware portfolio construction with automatic defensive tilting during crisis periods.

![Pipeline Equity Curve](https://raw.githubusercontent.com/prasants/QuantLite/main/docs/images/pipeline_equity_curve.png)

---

## Installation

```bash
pip install quantlite
```

Install only the data sources you need:

```bash
pip install quantlite[yahoo]    # Yahoo Finance
pip install quantlite[crypto]   # Cryptocurrency exchanges (CCXT)
pip install quantlite[fred]     # FRED macroeconomic data
pip install quantlite[plotly]   # Interactive Plotly charts
pip install quantlite[all]      # Everything
```

Optional: `hmmlearn` for HMM regime detection.

---

## Quickstart

### The Dream API

```python
import quantlite as ql

# Fetch → detect regimes → build portfolio → backtest → report
data = ql.fetch(["AAPL", "BTC-USD", "GLD", "TLT"], period="5y")
regimes = ql.detect_regimes(data, n_regimes=3)
weights = ql.construct_portfolio(data, regime_aware=True, regimes=regimes)
result = ql.backtest(data, weights)
ql.tearsheet(result, regimes=regimes, save="portfolio.html")
```

### Fat-Tail Risk Analysis

```python
from quantlite.distributions.fat_tails import student_t_process
from quantlite.risk.metrics import value_at_risk, cvar, return_moments
from quantlite.risk.evt import tail_risk_summary

# Generate fat-tailed returns (nu=4 gives realistic equity tail behaviour)
returns = student_t_process(nu=4.0, mu=0.0003, sigma=0.012, n_steps=2520, rng_seed=42)

# Cornish-Fisher VaR accounts for skewness and kurtosis
var_99 = value_at_risk(returns, alpha=0.01, method="cornish-fisher")
cvar_99 = cvar(returns, alpha=0.01)
moments = return_moments(returns)

print(f"VaR (99%):       {var_99:.4f}")
print(f"CVaR (99%):      {cvar_99:.4f}")
print(f"Excess kurtosis: {moments.kurtosis:.2f}")

# Full EVT tail analysis
summary = tail_risk_summary(returns)
print(f"Hill tail index: {summary.hill_estimate.tail_index:.2f}")
print(f"GPD shape (xi):  {summary.gpd_fit.shape:.4f}")
print(f"1-in-100 loss:   {summary.return_level_100:.4f}")
```

### Backtest Forensics

```python
from quantlite.forensics import deflated_sharpe_ratio
from quantlite.resample import bootstrap_sharpe_distribution

# You tried 50 strategies and the best had Sharpe 1.8.
# Is it real?
dsr = deflated_sharpe_ratio(observed_sharpe=1.8, n_trials=50, n_obs=252)
print(f"Probability Sharpe is genuine: {dsr:.2%}")

# Bootstrap confidence interval on the Sharpe ratio
result = bootstrap_sharpe_distribution(returns, n_samples=2000, seed=42)
print(f"Sharpe: {result['point_estimate']:.2f}")
print(f"95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
```

---

## Module Reference

### Core Risk

| Module | Description |
|--------|-------------|
| `quantlite.risk.metrics` | VaR (historical, parametric, Cornish-Fisher), CVaR, Sortino, Calmar, Omega, tail ratio, drawdowns |
| `quantlite.risk.evt` | GPD, GEV, Hill estimator, Peaks Over Threshold, return levels |
| `quantlite.distributions.fat_tails` | Student-t, Lévy stable, regime-switching GBM, Kou jump-diffusion |
| `quantlite.metrics` | Annualised return, volatility, Sharpe, max drawdown |

### Dependency and Portfolio

| Module | Description |
|--------|-------------|
| `quantlite.dependency.copulas` | Gaussian, Student-t, Clayton, Gumbel, Frank copulas with tail dependence |
| `quantlite.dependency.correlation` | Rolling, EWMA, stress, rank correlation |
| `quantlite.dependency.clustering` | Hierarchical Risk Parity |
| `quantlite.portfolio.optimisation` | Mean-variance, CVaR, risk parity, HRP, Black-Litterman, Kelly |
| `quantlite.portfolio.rebalancing` | Calendar, threshold, and tactical rebalancing |

### Backtesting

| Module | Description |
|--------|-------------|
| `quantlite.backtesting.engine` | Multi-asset backtesting with circuit breakers and slippage |
| `quantlite.backtesting.signals` | Momentum, mean reversion, trend following, volatility targeting |
| `quantlite.backtesting.analysis` | Performance summaries, monthly tables, regime attribution |

### Data

| Module | Description |
|--------|-------------|
| `quantlite.data` | Unified connectors: Yahoo Finance, CCXT, FRED, local files, plugin registry |
| `quantlite.data_generation` | GBM, correlated GBM, Ornstein-Uhlenbeck, Merton jump-diffusion |

### Taleb Stack

| Module | Description |
|--------|-------------|
| `quantlite.ergodicity` | Time-average vs ensemble-average growth, Kelly fraction, leverage effect |
| `quantlite.antifragile` | Antifragility score, convexity, Fourth Quadrant, barbell allocation, Lindy |
| `quantlite.scenarios` | Composable scenario engine, pre-built crisis library, shock propagation |

### Honest Backtesting

| Module | Description |
|--------|-------------|
| `quantlite.forensics` | Deflated Sharpe Ratio, Probabilistic Sharpe, haircut adjustments, minimum track record |
| `quantlite.overfit` | CSCV/PBO, TrialTracker, multiple testing correction, walk-forward validation |
| `quantlite.resample` | Block and stationary bootstrap, confidence intervals for Sharpe and drawdown |

### Systemic Risk

| Module | Description |
|--------|-------------|
| `quantlite.contagion` | CoVaR, Delta CoVaR, Marginal Expected Shortfall, Granger causality |
| `quantlite.network` | Correlation networks, eigenvector centrality, cascade simulation, community detection |
| `quantlite.diversification` | Effective Number of Bets, entropy diversification, tail diversification |

### Crypto

| Module | Description |
|--------|-------------|
| `quantlite.crypto.stablecoin` | Depeg probability, peg deviation tracking, reserve risk scoring |
| `quantlite.crypto.exchange` | Exchange concentration (HHI), wallet risk, proof of reserves, slippage |
| `quantlite.crypto.onchain` | Wallet exposure, TVL tracking, DeFi dependency graphs, smart contract risk |

### Simulation

| Module | Description |
|--------|-------------|
| `quantlite.simulation.evt_simulation` | EVT tail simulation, parametric tail simulation, scenario fan |
| `quantlite.simulation.copula_mc` | Gaussian copula MC, t-copula MC, stress correlation MC |
| `quantlite.simulation.regime_mc` | Regime-switching simulation, reverse stress test |
| `quantlite.monte_carlo` | Monte Carlo simulation harness |

### Factor Models

| Module | Description |
|--------|-------------|
| `quantlite.factors.classical` | Fama-French three/five-factor, Carhart four-factor, factor attribution |
| `quantlite.factors.custom` | Custom factor construction, significance testing, decay analysis |
| `quantlite.factors.tail_risk` | CVaR decomposition, regime factor exposure, crowding score |

### Regime Integration and Pipeline

| Module | Description |
|--------|-------------|
| `quantlite.regimes.hmm` | Hidden Markov Model regime detection |
| `quantlite.regimes.changepoint` | CUSUM and Bayesian changepoint detection |
| `quantlite.regimes.conditional` | Regime-conditional risk metrics and VaR |
| `quantlite.regime_integration` | Defensive tilting, filtered backtesting, regime tearsheets |
| `quantlite.pipeline` | Dream API: `fetch`, `detect_regimes`, `construct_portfolio`, `backtest`, `tearsheet` |

### Other

| Module | Description |
|--------|-------------|
| `quantlite.instruments` | Black-Scholes, bonds, barrier and Asian options |
| `quantlite.viz` | Stephen Few-themed charts: risk dashboards, copula contours, regime timelines |
| `quantlite.report` | HTML/PDF tearsheet generation |

---

## Design Philosophy

1. **Fat tails are the default.** Gaussian assumptions are explicitly opt-in, never implicit.
2. **Typed return values.** Every function returns frozen dataclasses with clear attributes, not opaque dicts.
3. **Composable modules.** Risk metrics feed into portfolio optimisation which feeds into backtesting. Each layer works independently.
4. **Honest modelling.** If a method has known limitations (e.g., Gaussian copula has zero tail dependence), the docstring says so.
5. **Reproducible.** Every stochastic function accepts `rng_seed` for deterministic output.

---

## Documentation

Full documentation lives in the [`docs/`](docs/) directory:

| Document | Topic |
|----------|-------|
| [risk.md](docs/risk.md) | Risk metrics and EVT |
| [copulas.md](docs/copulas.md) | Copula dependency structures |
| [regimes.md](docs/regimes.md) | Regime detection |
| [portfolio.md](docs/portfolio.md) | Portfolio optimisation and rebalancing |
| [data.md](docs/data.md) | Data connectors |
| [visualisation.md](docs/visualisation.md) | Stephen Few-themed charts |
| [interactive_viz.md](docs/interactive_viz.md) | Plotly interactive charts |
| [ergodicity.md](docs/ergodicity.md) | Ergodicity economics |
| [antifragility.md](docs/antifragility.md) | Antifragility measurement |
| [scenarios.md](docs/scenarios.md) | Scenario stress testing |
| [forensics.md](docs/forensics.md) | Deflated Sharpe and strategy forensics |
| [overfitting.md](docs/overfitting.md) | Overfitting detection |
| [resampling.md](docs/resampling.md) | Bootstrap resampling |
| [contagion.md](docs/contagion.md) | Systemic risk and contagion |
| [network.md](docs/network.md) | Financial network analysis |
| [diversification.md](docs/diversification.md) | Diversification diagnostics |
| [factors_classical.md](docs/factors_classical.md) | Classical factor models |
| [factors_custom.md](docs/factors_custom.md) | Custom factor tools |
| [factors_tail_risk.md](docs/factors_tail_risk.md) | Tail risk factor analysis |
| [simulation_evt.md](docs/simulation_evt.md) | EVT simulation |
| [simulation_copula.md](docs/simulation_copula.md) | Copula Monte Carlo |
| [simulation_regime.md](docs/simulation_regime.md) | Regime-switching simulation |
| [regime_integration.md](docs/regime_integration.md) | Regime-aware pipelines |
| [pipeline.md](docs/pipeline.md) | Dream API reference |
| [reports.md](docs/reports.md) | Tearsheet generation |
| [stablecoin_risk.md](docs/stablecoin_risk.md) | Stablecoin risk |
| [exchange_risk.md](docs/exchange_risk.md) | Exchange risk |
| [onchain_risk.md](docs/onchain_risk.md) | On-chain risk |
| [architecture.md](docs/architecture.md) | Library architecture |

---

## Contributing

Contributions are welcome. Please ensure:

1. All new functions have type hints and docstrings
2. Tests pass: `pytest`
3. Code is formatted: `ruff check` and `ruff format`
4. British spellings in all documentation

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Links

- [PyPI](https://pypi.org/project/quantlite/)
- [GitHub](https://github.com/prasants/QuantLite)
- [Issue Tracker](https://github.com/prasants/QuantLite/issues)
- [Changelog](CHANGELOG.md)
