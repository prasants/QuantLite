# Strategy Forensics

## Overview

Most reported Sharpe ratios are lies.

Not because anyone is deliberately dishonest, but because the standard Sharpe ratio calculation ignores three inconvenient truths: **multiple testing** (you tried 100 strategies and reported the winner), **non-normality** (returns are skewed and fat-tailed), and **insufficient track records** (two years of data proves nothing).

The `quantlite.forensics` module implements Lopez de Prado's framework for honest backtesting. It provides five tools that every quantitative analyst should run before trusting a backtest result:

1. **Deflated Sharpe Ratio** corrects for the number of strategies tested
2. **Probabilistic Sharpe Ratio** accounts for estimation error and non-normality
3. **Haircut Sharpe Ratio** applies conservative adjustments
4. **Minimum Track Record Length** tells you how much data you actually need
5. **Signal Decay** measures how quickly alpha disappears

## API Reference

### `deflated_sharpe_ratio`

```python
deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float
```

Adjusts the observed Sharpe ratio for the number of strategies tried (multiple testing bias). Returns the probability that the best strategy's Sharpe is genuine after accounting for how many strategies were tested.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `observed_sharpe` | float | Best observed Sharpe ratio among all trials |
| `n_trials` | int | Number of independent strategy trials conducted |
| `n_obs` | int | Number of return observations per trial |
| `skewness` | float | Skewness of returns (default 0) |
| `kurtosis` | float | Kurtosis of returns (default 3, i.e. normal) |

**Returns:** Probability in [0, 1] that the observed Sharpe is genuine.

**Example:**

```python
from quantlite.forensics import deflated_sharpe_ratio

# You tried 50 strategies and the best had Sharpe 1.8
dsr = deflated_sharpe_ratio(
    observed_sharpe=1.8,
    n_trials=50,
    n_obs=252,
)
print(f"Probability Sharpe is genuine: {dsr:.2%}")
# If this is below 95%, your "best" strategy is likely noise.
```

![Deflated Sharpe](images/deflated_sharpe.png)

### `probabilistic_sharpe_ratio`

```python
probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float
```

Computes the probability that the true Sharpe ratio exceeds a given benchmark, accounting for estimation error and non-normality of returns.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `observed_sharpe` | float | Observed (in-sample) Sharpe ratio |
| `benchmark_sharpe` | float | Benchmark Sharpe ratio to test against |
| `n_obs` | int | Number of return observations |
| `skewness` | float | Skewness of returns (default 0) |
| `kurtosis` | float | Kurtosis of returns (default 3) |

**Returns:** Probability in [0, 1] that the true Sharpe exceeds the benchmark.

**Example:**

```python
from quantlite.forensics import probabilistic_sharpe_ratio

# Is this Sharpe genuinely above 0.5?
psr = probabilistic_sharpe_ratio(
    observed_sharpe=1.2,
    benchmark_sharpe=0.5,
    n_obs=504,
    skewness=-0.3,
    kurtosis=4.5,
)
print(f"P(true Sharpe > 0.5): {psr:.2%}")
```

### `haircut_sharpe_ratio`

```python
haircut_sharpe_ratio(
    observed_sharpe: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    method: str = "holm",
) -> float
```

Applies a conservative adjustment to the observed Sharpe ratio based on the chosen correction method. More aggressive methods yield larger haircuts.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `observed_sharpe` | float | Observed Sharpe ratio |
| `n_obs` | int | Number of return observations |
| `skewness` | float | Skewness of returns (default 0) |
| `kurtosis` | float | Kurtosis of returns (default 3) |
| `method` | str | Correction: `"bonferroni"`, `"holm"`, or `"bhy"` |

**Returns:** Adjusted (haircutted) Sharpe ratio.

**Example:**

```python
from quantlite.forensics import haircut_sharpe_ratio

observed = 2.0
for method in ["bonferroni", "holm", "bhy"]:
    adjusted = haircut_sharpe_ratio(observed, n_obs=252, method=method)
    print(f"{method:>12}: {observed:.2f} -> {adjusted:.2f}")
```

### `min_track_record_length`

```python
min_track_record_length(
    observed_sharpe: float,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float
```

Computes the minimum number of observations required for the observed Sharpe ratio to be statistically significant at the given confidence level.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `observed_sharpe` | float | Observed Sharpe ratio |
| `benchmark_sharpe` | float | Benchmark to test against (default 0) |
| `confidence` | float | Confidence level (default 0.95) |
| `skewness` | float | Skewness of returns (default 0) |
| `kurtosis` | float | Kurtosis of returns (default 3) |

**Returns:** Minimum number of observations (float).

**Example:**

```python
from quantlite.forensics import min_track_record_length

# How many days of data do you need to trust a Sharpe of 1.5?
n_min = min_track_record_length(observed_sharpe=1.5)
years = n_min / 252
print(f"Minimum track record: {n_min:.0f} observations ({years:.1f} years)")
```

![Minimum Track Record](images/min_track_record.png)

### `signal_decay`

```python
signal_decay(
    returns: array-like,
    signal: array-like,
    lags: sequence of int | None = None,
) -> dict
```

Analyses how alpha decays over time by computing the correlation between a trading signal and forward returns at increasing lags.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `returns` | array-like | Array of asset returns |
| `signal` | array-like | Array of signal values, same length as returns |
| `lags` | sequence of int | Lag values to test (default [1, 2, 3, 5, 10, 20]) |

**Returns:** Dictionary with keys:
- `half_life`: estimated half-life in periods (float or None)
- `decay_curve`: list of (lag, correlation) tuples
- `r_squared_curve`: list of (lag, r_squared) tuples

**Example:**

```python
import numpy as np
from quantlite.forensics import signal_decay

rng = np.random.default_rng(42)
returns = rng.normal(0.0005, 0.01, 500)
signal = np.roll(returns, 1) + rng.normal(0, 0.005, 500)

result = signal_decay(returns, signal)
print(f"Half-life: {result['half_life']:.1f} periods")
for lag, corr in result["decay_curve"]:
    print(f"  Lag {lag:2d}: correlation = {corr:.4f}")
```

![Signal Decay](images/signal_decay.png)

## Interpretation Guide

### Deflated Sharpe Ratio

| DSR Value | Interpretation |
|-----------|----------------|
| > 0.95 | Strong evidence the Sharpe is genuine |
| 0.80 - 0.95 | Moderate evidence; proceed with caution |
| 0.50 - 0.80 | Weak evidence; likely inflated by multiple testing |
| < 0.50 | The "best" strategy is almost certainly noise |

### When to Trust a Track Record

A track record deserves trust when **all** of the following hold:

1. **Sufficient length.** The track record exceeds `min_track_record_length` at 95% confidence.
2. **Deflation-robust.** The DSR remains above 0.95 after accounting for all strategies tested.
3. **Non-normality adjusted.** The PSR uses realistic skewness and kurtosis, not Gaussian defaults.
4. **Signal persistence.** `signal_decay` shows the alpha half-life exceeds the rebalancing frequency.

If any of these fail, the reported Sharpe is not evidence of skill.
