# Overfitting Detection

## Overview

The more strategies you test, the more likely your best one is a fluke.

This is not a philosophical concern. It is a mathematical certainty. If you test 100 random strategies on the same data, the best one will have a Sharpe ratio of roughly 2.0 purely by chance. The `quantlite.overfit` module provides rigorous tools to detect, quantify, and defend against backtest overfitting:

1. **TrialTracker** logs every strategy you test and estimates overfitting probability
2. **Probability of Backtest Overfitting (PBO)** via Combinatorially Symmetric Cross-Validation (CSCV)
3. **Multiple testing correction** adjusts p-values for the number of hypotheses tested
4. **Minimum backtest length** tells you how much data you need
5. **Walk-forward validation** tests strategies on genuinely out-of-sample data

## TrialTracker

The `TrialTracker` context manager logs every backtest trial and provides overfitting diagnostics. Use it to wrap any strategy search process.

```python
import numpy as np
from quantlite.overfit import TrialTracker

rng = np.random.default_rng(42)

with TrialTracker("momentum_search") as tracker:
    for lookback in [10, 20, 40, 60, 120]:
        returns = rng.normal(0.0003, 0.01, 500)  # simulated strategy returns
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        tracker.log(
            params={"lookback": lookback},
            sharpe=sharpe,
            returns=returns,
        )

    best = tracker.best_trial
    pbo = tracker.overfitting_probability()
    print(f"Best Sharpe: {best['sharpe']:.2f} (lookback={best['params']['lookback']})")
    print(f"Overfitting probability: {pbo:.2%}")
```

### TrialTracker API

**`TrialTracker(name: str = "unnamed")`**

- `log(params=None, sharpe=0.0, returns=None)` - Log a single trial
- `overfitting_probability(n_splits=10)` - Estimate PBO via CSCV (if returns available) or heuristic
- `best_trial` - Property returning the trial with highest Sharpe

## Probability of Backtest Overfitting (CSCV)

```python
probability_of_backtest_overfitting(
    trial_returns: array-like,  # shape (n_trials, n_obs)
    n_splits: int = 10,
) -> dict
```

Implements Combinatorially Symmetric Cross-Validation. Splits the data into S temporal subsets, forms all C(S, S/2) train/test combinations, and computes the rank correlation between in-sample and out-of-sample performance. PBO is the fraction of combinations where rank correlation is negative.

**Returns:** Dictionary with keys:
- `pbo`: Probability of Backtest Overfitting in [0, 1]
- `rank_correlations`: list of rank correlations per combination
- `n_combinations`: number of combinations evaluated

**Example:**

```python
import numpy as np
from quantlite.overfit import probability_of_backtest_overfitting

rng = np.random.default_rng(42)
# 20 random strategies, 1000 observations each
trial_returns = rng.normal(0, 0.01, (20, 1000))

result = probability_of_backtest_overfitting(trial_returns, n_splits=10)
print(f"PBO: {result['pbo']:.2%}")
print(f"Combinations tested: {result['n_combinations']}")
```

![Overfitting Trials](images/overfitting_trials.png)

## Multiple Testing Correction

```python
multiple_testing_correction(
    p_values: array-like,
    method: str = "bhy",
) -> numpy.ndarray
```

Adjusts p-values for the number of hypotheses tested. Supports three methods:

| Method | Description | Conservatism |
|--------|-------------|--------------|
| `"bonferroni"` | Multiply by number of tests | Most conservative |
| `"holm"` | Step-down procedure | Moderate |
| `"bhy"` | Benjamini-Hochberg-Yekutieli | Least conservative |

**Example:**

```python
from quantlite.overfit import multiple_testing_correction

raw_p = [0.01, 0.04, 0.03, 0.15, 0.005]
adjusted = multiple_testing_correction(raw_p, method="bhy")
print("Raw p-values:     ", [f"{p:.3f}" for p in raw_p])
print("Adjusted p-values:", [f"{p:.3f}" for p in adjusted])
```

## Minimum Backtest Length

```python
min_backtest_length(
    sharpe: float,
    confidence: float = 0.95,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float
```

Computes the minimum number of observations needed for a given Sharpe ratio to be statistically significant.

**Example:**

```python
from quantlite.overfit import min_backtest_length

n_min = min_backtest_length(sharpe=1.0, confidence=0.95)
print(f"Need at least {n_min:.0f} observations ({n_min/252:.1f} years)")
```

## Walk-Forward Validation

```python
walk_forward_validate(
    returns: array-like,
    strategy_fn: callable,
    window: int,
    step: int,
    expanding: bool = False,
) -> dict
```

Iterates through the return series using a rolling (or expanding) window, calling `strategy_fn` on the training window to obtain weights, then evaluating performance on the subsequent test window.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `returns` | array-like | 1D array of asset returns |
| `strategy_fn` | callable | Takes training returns, returns weight(s) for test period |
| `window` | int | Training window size |
| `step` | int | Test window size (step forward per fold) |
| `expanding` | bool | If True, use expanding window (default False) |

**Returns:** Dictionary with keys:
- `folds`: list of per-fold dicts with train/test indices, test_return, weight
- `aggregate_return`: total return across all test folds
- `n_folds`: number of folds evaluated
- `mean_fold_return`: mean return per fold

**Example:**

```python
import numpy as np
from quantlite.overfit import walk_forward_validate

rng = np.random.default_rng(42)
returns = rng.normal(0.0003, 0.01, 1000)

def momentum_strategy(train_returns):
    """Go long if recent trend is positive, else flat."""
    return 1.0 if np.mean(train_returns) > 0 else 0.0

result = walk_forward_validate(
    returns, momentum_strategy, window=200, step=50,
)
print(f"Folds: {result['n_folds']}")
print(f"Mean fold return: {result['mean_fold_return']:.4f}")
```

![Walk-Forward Validation](images/walk_forward.png)

## Interpretation Guide

### PBO Values

| PBO | Interpretation |
|-----|----------------|
| < 0.10 | Low overfitting risk; strategy selection appears robust |
| 0.10 - 0.30 | Moderate risk; further validation recommended |
| 0.30 - 0.50 | High risk; in-sample performance is likely inflated |
| > 0.50 | The best in-sample strategy is more likely to underperform than outperform out-of-sample |

### Red Flags

Watch for these warning signs:

- **PBO > 0.50**: Your strategy selection process is worse than random. The "best" strategy in-sample is more likely to be the worst out-of-sample.
- **Many trials, few observations**: Testing 100 strategies on 2 years of daily data is almost guaranteed to produce a spurious winner.
- **Walk-forward folds mostly negative**: If out-of-sample performance is consistently poor across folds, the in-sample result is not predictive.
- **Rank correlations mostly negative**: CSCV rank correlations below zero across most combinations indicate severe overfitting.
