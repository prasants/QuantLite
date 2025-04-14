# Risk Metrics and Extreme Value Theory

## Risk Metrics (`quantlite.risk.metrics`)

QuantLite's risk metrics module provides tail-aware measures that go beyond standard volatility. All functions accept NumPy arrays, pandas Series, or plain Python lists of simple returns.

### Value at Risk

Three VaR methods, each with different assumptions about the return distribution:

```python
from quantlite.risk.metrics import value_at_risk
from quantlite.distributions.fat_tails import student_t_process

returns = student_t_process(nu=4, mu=0.0003, sigma=0.012, n_steps=2520, rng_seed=42)

# Historical VaR: non-parametric, uses the empirical quantile
hist_var = value_at_risk(returns, alpha=0.05, method="historical")

# Parametric VaR: assumes normality (use with caution on fat-tailed data)
param_var = value_at_risk(returns, alpha=0.05, method="parametric")

# Cornish-Fisher VaR: adjusts for skewness and excess kurtosis
cf_var = value_at_risk(returns, alpha=0.05, method="cornish-fisher")

print(f"Historical:    {hist_var:.5f}")
print(f"Parametric:    {param_var:.5f}")
print(f"Cornish-Fisher:{cf_var:.5f}")
```

The Cornish-Fisher expansion corrects the Gaussian quantile using the third and fourth moments:

z_CF = z + (z² - 1)S/6 + (z³ - 3z)K/24 - (2z³ - 5z)S²/36

where S is skewness and K is excess kurtosis. For fat-tailed distributions (K > 0), this produces a more conservative VaR than the parametric method.

### Conditional VaR (Expected Shortfall)

CVaR measures the average loss in the worst α fraction of outcomes, providing a coherent risk measure that satisfies subadditivity:

```python
from quantlite.risk.metrics import cvar

cvar_95 = cvar(returns, alpha=0.05)
cvar_99 = cvar(returns, alpha=0.01)
print(f"CVaR 95%: {cvar_95:.5f}")
print(f"CVaR 99%: {cvar_99:.5f}")
```

### Performance Ratios

```python
from quantlite.risk.metrics import sortino_ratio, calmar_ratio, omega_ratio, tail_ratio

# Sortino: uses downside deviation only (penalises losses, not upside volatility)
sortino = sortino_ratio(returns, risk_free_rate=0.04, freq=252)

# Calmar: annualised return / max drawdown (measures pain-adjusted performance)
calmar = calmar_ratio(returns, freq=252)

# Omega: probability-weighted gains over losses (captures the full distribution)
omega = omega_ratio(returns, threshold=0.0)

# Tail ratio: right tail magnitude / left tail magnitude (asymmetry measure)
tr = tail_ratio(returns, alpha=0.05)

print(f"Sortino: {sortino:.2f}")
print(f"Calmar:  {calmar:.2f}")
print(f"Omega:   {omega:.2f}")
print(f"Tail ratio: {tr:.2f}")
```

### Drawdown Analysis

```python
from quantlite.risk.metrics import max_drawdown_duration

dd = max_drawdown_duration(returns)
print(f"Max drawdown:  {dd.max_drawdown:.2%}")
print(f"Duration:      {dd.duration} periods")
print(f"Peak index:    {dd.start_idx}")
print(f"Trough index:  {dd.end_idx}")
```

### Return Moments

```python
from quantlite.risk.metrics import return_moments

moments = return_moments(returns)
print(f"Mean:     {moments.mean:.6f}")
print(f"Vol:      {moments.volatility:.6f}")
print(f"Skewness: {moments.skewness:.4f}")
print(f"Kurtosis: {moments.kurtosis:.4f} (excess)")
```

Excess kurtosis > 0 indicates fatter tails than the Gaussian distribution. Typical equity return series exhibit excess kurtosis in the range of 3 to 10.

## Extreme Value Theory (`quantlite.risk.evt`)

EVT provides a mathematically rigorous framework for modelling the tails of distributions. While standard risk metrics extrapolate from the body of the distribution, EVT fits models directly to the tail, yielding more reliable estimates of extreme quantiles.

### Generalised Pareto Distribution

The GPD models exceedances over a high threshold. The Pickands-Balkema-de Haan theorem guarantees that, for a sufficiently high threshold, the excess distribution converges to a GPD regardless of the parent distribution.

```python
from quantlite.risk.evt import fit_gpd, return_level

gpd = fit_gpd(returns)
print(f"Shape (xi):    {gpd.shape:.4f}")
print(f"Scale (sigma): {gpd.scale:.4f}")
print(f"Threshold:     {gpd.threshold:.4f}")
print(f"Exceedances:   {gpd.n_exceedances} / {gpd.n_total}")

# Return levels: the loss expected to be exceeded once per N observations
for period in [100, 250, 1000, 5000]:
    rl = return_level(gpd, return_period=period)
    print(f"  1-in-{period}: {rl:.4f}")
```

The shape parameter xi determines the tail type:
- xi > 0: Heavy tail (Frechet domain). Common for financial returns.
- xi = 0: Exponential tail (Gumbel domain).
- xi < 0: Bounded tail (Weibull domain). Rare in finance.

### Generalised Extreme Value Distribution

The GEV models block maxima (e.g., the worst monthly loss over many years):

```python
from quantlite.risk.evt import fit_gev
import numpy as np

# Compute monthly block maxima of losses
losses = -returns
n_months = len(losses) // 21
block_maxima = np.array([
    losses[i*21:(i+1)*21].max() for i in range(n_months)
])

gev = fit_gev(block_maxima)
print(f"GEV shape: {gev.shape:.4f}")
print(f"GEV loc:   {gev.loc:.4f}")
print(f"GEV scale: {gev.scale:.4f}")
```

### Hill Estimator

The Hill estimator provides a non-parametric estimate of the tail index for heavy-tailed distributions:

```python
from quantlite.risk.evt import hill_estimator

hill = hill_estimator(returns)
print(f"Tail index (alpha): {hill.tail_index:.2f}")
print(f"Order statistics used (k): {hill.k}")
```

A tail index of alpha means the tail probability decays as x^(-alpha). Lower values indicate heavier tails. Typical equity indices have alpha between 3 and 5.

### Peaks Over Threshold

The POT method combines threshold selection with GPD fitting:

```python
from quantlite.risk.evt import peaks_over_threshold

exceedances, gpd_fit = peaks_over_threshold(returns, threshold=None)
print(f"Found {len(exceedances)} exceedances")
print(f"GPD fit: {gpd_fit}")
```

### Comprehensive Tail Risk Summary

```python
from quantlite.risk.evt import tail_risk_summary

summary = tail_risk_summary(returns)
print(f"VaR 95%:           {summary.var_95:.4f}")
print(f"VaR 99%:           {summary.var_99:.4f}")
print(f"CVaR 95%:          {summary.cvar_95:.4f}")
print(f"CVaR 99%:          {summary.cvar_99:.4f}")
print(f"Hill tail index:   {summary.hill_estimate.tail_index:.2f}")
print(f"GPD shape:         {summary.gpd_fit.shape:.4f}")
print(f"1-in-100 loss:     {summary.return_level_100:.4f}")
print(f"Excess kurtosis:   {summary.excess_kurtosis:.2f}")
```

## Fat-Tailed Distributions (`quantlite.distributions.fat_tails`)

### Student-t Process

The Student-t distribution is the workhorse of fat-tailed modelling. With degrees of freedom nu, the tail index equals nu. The distribution is scaled so that variance equals sigma²:

```python
from quantlite.distributions.fat_tails import student_t_process

# nu=4: realistic equity tails (excess kurtosis ~ 6)
returns = student_t_process(nu=4.0, mu=0.0003, sigma=0.012, n_steps=2520, rng_seed=42)

# nu=3: very fat tails (infinite kurtosis)
heavy = student_t_process(nu=3.0, mu=0.0, sigma=0.015, n_steps=2520, rng_seed=42)
```

### Levy Stable Process

For truly heavy tails where even variance may be infinite:

```python
from quantlite.distributions.fat_tails import levy_stable_process

# alpha=1.7: heavy tails, finite mean but infinite variance
# beta=-0.1: slight negative skew
returns = levy_stable_process(alpha=1.7, beta=-0.1, mu=0.0, sigma=0.008, n_steps=1260, rng_seed=42)
```

### Regime-Switching GBM

A Markov-modulated geometric Brownian motion where drift and volatility switch between regimes:

```python
from quantlite.distributions.fat_tails import regime_switching_gbm, RegimeParams
import numpy as np

calm = RegimeParams(mu=0.08, sigma=0.15)     # bull market
volatile = RegimeParams(mu=0.02, sigma=0.25)  # uncertain
crisis = RegimeParams(mu=-0.30, sigma=0.45)   # crash

transition = np.array([
    [0.96, 0.03, 0.01],  # calm -> calm/volatile/crisis
    [0.05, 0.90, 0.05],  # volatile -> calm/volatile/crisis
    [0.10, 0.10, 0.80],  # crisis -> calm/volatile/crisis
])

prices, regimes = regime_switching_gbm(
    [calm, volatile, crisis], transition, n_steps=5040, rng_seed=42,
)
```

### Kou's Double-Exponential Jump-Diffusion

Kou's model uses asymmetric double-exponential jumps, producing a better fit to the leptokurtic, asymmetric returns observed empirically than Merton's Gaussian-jump model:

```python
from quantlite.distributions.fat_tails import kou_double_exponential_jump

prices = kou_double_exponential_jump(
    S0=100, mu=0.05, sigma=0.2,
    lam=1.0,     # 1 expected jump per year
    p=0.4,       # 40% probability jump is upward
    eta1=10,     # upward jump mean = 1/10 = 10%
    eta2=5,      # downward jump mean = 1/5 = 20%
    n_steps=252, rng_seed=42,
)
```

The asymmetry (eta2 < eta1) captures the empirical observation that downward jumps tend to be larger than upward ones.
