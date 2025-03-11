# QuantLite

A fat-tail-native quantitative finance toolkit for Python.

QuantLite provides stochastic process generators, option and bond pricing, risk metrics, extreme value theory, fat-tailed distributions, and clean visualisation. It is designed for practitioners who take tail risk seriously: every simulation supports non-Gaussian dynamics, every risk metric goes beyond mean-variance, and every chart follows Stephen Few's principles of maximum data-ink ratio.

## Installation

```bash
pip install quantlite
```

For development:

```bash
pip install quantlite[dev]
```

## Quick Start

### Risk Metrics

```python
from quantlite.risk.metrics import value_at_risk, cvar, return_moments

returns = [...]  # your daily returns

var_95 = value_at_risk(returns, alpha=0.05, method="cornish-fisher")
es_95 = cvar(returns, alpha=0.05)
moments = return_moments(returns)
print(moments)  # ReturnMoments(mean=..., vol=..., skew=..., kurt=...)
```

### Extreme Value Theory

```python
from quantlite.risk.evt import fit_gpd, return_level, tail_risk_summary

gpd = fit_gpd(returns)
loss_100 = return_level(gpd, return_period=25000)  # 1-in-100-year daily loss
summary = tail_risk_summary(returns)
print(summary)
```

### Fat-Tailed Distributions

```python
from quantlite.distributions.fat_tails import (
    student_t_process,
    regime_switching_gbm,
    kou_double_exponential_jump,
    RegimeParams,
)
import numpy as np

# Student-t returns (power-law tails)
rets = student_t_process(nu=4, sigma=0.02, n_steps=1000, rng_seed=42)

# Regime-switching GBM
calm = RegimeParams(mu=0.08, sigma=0.12)
crisis = RegimeParams(mu=-0.15, sigma=0.45)
trans = np.array([[0.98, 0.02], [0.10, 0.90]])
prices, regimes = regime_switching_gbm([calm, crisis], trans, n_steps=2520)

# Kou's double-exponential jump diffusion
prices = kou_double_exponential_jump(S0=100, lam=2.0, rng_seed=42)
```

### Visualisation (Stephen Few Theme)

```python
from quantlite.viz.theme import apply_few_theme, few_figure
from quantlite.viz.risk import plot_tail_distribution, plot_risk_dashboard

apply_few_theme()
fig, ax = plot_tail_distribution(returns, gpd_fit=gpd)
fig, axes = plot_risk_dashboard(returns)
```

### Option Pricing

```python
from quantlite import black_scholes_call, black_scholes_greeks

price = black_scholes_call(S=100, K=95, T=1.0, r=0.05, sigma=0.2)
greeks = black_scholes_greeks(S=100, K=95, T=1.0, r=0.05, sigma=0.2)
print(greeks)  # Greeks(delta=..., gamma=..., vega=..., theta=..., rho=...)
```

### Monte Carlo Simulation

```python
from quantlite import geometric_brownian_motion, merton_jump_diffusion

gbm_path = geometric_brownian_motion(S0=100, sigma=0.3, steps=252, rng_seed=42)
mjd_path = merton_jump_diffusion(S0=100, lamb=1.0, jump_std=0.1, rng_seed=42)
```

### Copulas and Dependency Modelling

```python
from quantlite.dependency.copulas import StudentTCopula, select_best_copula
from quantlite.dependency.clustering import hrp_weights

# Fit a Student-t copula (captures tail dependence)
cop = StudentTCopula()
cop.fit(bivariate_data)
print(cop.tail_dependence())  # {'lower': 0.23, 'upper': 0.23}

# Automatic copula selection by AIC
best = select_best_copula(bivariate_data)
print(best)  # CopulaFitResult(name='Student-t', AIC=..., BIC=...)

# Hierarchical Risk Parity allocation
weights = hrp_weights(returns_df)
print(weights)  # OrderedDict([('SPY', 0.32), ('TLT', 0.28), ...])
```

### Regime Detection

```python
from quantlite.regimes.hmm import fit_regime_model, select_n_regimes
from quantlite.regimes.changepoint import detect_changepoints
from quantlite.regimes.conditional import conditional_metrics, regime_aware_var

# Fit a 2-regime Hidden Markov Model
model = fit_regime_model(returns, n_regimes=2, rng_seed=42)
print(model.means)        # [-0.002, 0.001] (crisis vs calm)
print(model.transition_matrix)

# Automatic regime count selection by BIC
best_model = select_n_regimes(returns, max_regimes=4)

# Change-point detection (CUSUM or Bayesian)
cps = detect_changepoints(returns, method="bayesian", penalty=100)

# Regime-conditional risk
metrics = conditional_metrics(returns, model.regime_labels)
var = regime_aware_var(returns, model.regime_labels, alpha=0.05)
```

## Features

**Risk Metrics:** VaR (historical, parametric, and Cornish-Fisher), CVaR/Expected Shortfall, Sortino ratio, Calmar ratio, max drawdown with duration, Omega ratio, tail ratio, and return moments.

**Extreme Value Theory:** Generalised Pareto Distribution fitting, Generalised Extreme Value fitting, Hill tail index estimator, Peaks Over Threshold method, return level estimation, and comprehensive tail risk summaries.

**Fat-Tailed Distributions:** Student-t process, Levy alpha-stable process, Markov regime-switching GBM, and Kou's double-exponential jump diffusion.

**Copulas and Dependency:** Gaussian, Student-t, Clayton, Gumbel, and Frank copulas with fitting, simulation, log-likelihood, and analytical tail dependence coefficients. Automatic model selection by AIC/BIC. Rolling, EWMA, and stress-conditional correlation. Rank correlation (Spearman, Kendall). Correlation breakdown testing.

**Clustering and Allocation:** Hierarchical Risk Parity (Lopez de Prado) with correlation-distance clustering and recursive bisection. Quasi-diagonalisation for visual cluster analysis.

**Regime Detection:** Gaussian Hidden Markov Models via hmmlearn with automatic regime count selection by BIC. CUSUM and Bayesian online change-point detection (Adams and MacKay, 2007). Regime-conditional risk metrics, transition risk analysis, and regime-aware VaR.

**Instruments:** Black-Scholes pricing and Greeks, bond pricing with duration and YTM, vectorised Monte Carlo for barrier and Asian options.

**Data Generation:** GBM, correlated multi-asset GBM, Ornstein-Uhlenbeck, and Merton jump diffusion. All using NumPy's modern Generator API for thread-safe reproducibility.

**Visualisation:** Stephen Few-inspired theme with muted palette, direct labels, bullet graphs, sparklines, tail distribution plots, return level charts, drawdown charts, risk dashboards, copula contour plots, correlation heatmaps, stress correlation comparisons, regime timelines, regime distribution small multiples, transition matrices, and composite regime summaries.

## Design Principles

- **Fat tails by default.** Gaussian is available but never assumed. Every risk metric, every simulation, every test accounts for the reality that markets bite.
- **Modern Python.** Type hints throughout, dataclass return types, NumPy Generator API, Python 3.10+.
- **Vectorised.** Monte Carlo simulations use NumPy broadcasting, not Python loops.
- **Honest charts.** Following Stephen Few: maximum data-ink ratio, no chartjunk, horizontal gridlines only, direct labels over legends.

## Requirements

- Python >= 3.10
- NumPy, pandas, SciPy, matplotlib, mplfinance

## Licence

MIT. See [LICENSE](LICENSE) for details.

## Author

Prasant Sudhakaran
