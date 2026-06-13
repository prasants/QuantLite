# Contagion and Systemic Risk

## Overview

Diversification fails precisely when you need it most.

During calm markets, assets appear independent. During crises, correlations spike towards one and losses cascade through the financial system like dominoes. The 2008 Global Financial Crisis demonstrated this brutally: assets that seemed uncorrelated in normal times became perfectly correlated in the tail. Traditional risk models, built on unconditional statistics, completely missed this.

The `quantlite.contagion` module provides tools to measure, visualise, and stress-test systemic risk. It answers questions that standard correlation analysis cannot:

1. **CoVaR** measures the VaR of one asset conditional on another being in distress
2. **Delta CoVaR** isolates the marginal contribution of each asset to system-wide risk
3. **Marginal Expected Shortfall** captures each asset's behaviour during system-wide tail events
4. **Systemic Risk Contributions** ranks every asset by its contribution to system-wide crashes
5. **Granger Causality** tests whether one asset's returns predict another's (lead-lag relationships)
6. **Causal Network** builds a directed graph of information flow across the entire asset universe

## API Reference

### `covar`

```python
covar(
    returns_a: array-like,
    returns_b: array-like,
    alpha: float = 0.05,
    method: str = "quantile",
) -> dict
```

Compute CoVaR: the Value at Risk of asset B conditional on asset A being at its own VaR level. This reveals how much worse B's tail risk becomes when A is in distress.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `returns_a` | array-like | Return series for the conditioning asset |
| `returns_b` | array-like | Return series for the target asset |
| `alpha` | float | Significance level for VaR (default 0.05) |
| `method` | str | `"quantile"` for conditional quantile, `"regression"` for OLS approximation |

**Returns:** Dictionary with keys:

| Key | Description |
|-----|-------------|
| `covar` | VaR of B conditional on A being at its VaR |
| `var_a` | Unconditional VaR of A |
| `var_b` | Unconditional VaR of B |
| `delta_covar` | Difference between stressed and median CoVaR |

**Interpretation:**

| CoVaR vs VaR | Meaning |
|--------------|---------|
| CoVaR much worse than VaR | Strong contagion channel from A to B |
| CoVaR close to VaR | Little dependence in the tail |
| CoVaR better than VaR | Hedging effect (rare, typically safe havens) |

**Example:**

```python
import numpy as np
import pandas as pd
from quantlite.contagion import covar

rng = np.random.default_rng(42)
n = 2000
market = rng.normal(0.0003, 0.012, n)
financials = 0.7 * market + rng.normal(0.0001, 0.015, n)

result = covar(market, financials, alpha=0.05)
print(f"VaR(Financials):  {result['var_b']:.4f}")
print(f"CoVaR(Fin|Mkt):   {result['covar']:.4f}")
print(f"Delta CoVaR:      {result['delta_covar']:.4f}")
# CoVaR is substantially worse than unconditional VaR,
# confirming the contagion channel from market to financials.
```

![CoVaR Comparison](images/covar_comparison.png)

### `delta_covar`

```python
delta_covar(
    returns_a: array-like,
    returns_b: array-like,
    alpha: float = 0.05,
) -> float
```

Compute the marginal contribution of asset A to asset B's tail risk. This is the difference between CoVaR when A is in distress and CoVaR when A is at its median. A large negative value indicates that A's distress significantly worsens B's tail risk.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `returns_a` | array-like | Return series for the conditioning asset |
| `returns_b` | array-like | Return series for the target asset |
| `alpha` | float | Significance level (default 0.05) |

**Returns:** Delta CoVaR value (float). More negative means stronger contagion.

**Example:**

```python
from quantlite.contagion import delta_covar

# How much does a bank's distress worsen the index?
delta = delta_covar(bank_returns, index_returns, alpha=0.05)
print(f"Delta CoVaR: {delta:.4f}")
# A value of -0.03 means the index's 5% VaR worsens by 3%
# when the bank is in distress versus at its median.
```

### `marginal_expected_shortfall`

```python
marginal_expected_shortfall(
    returns_system: array-like,
    returns_asset: array-like,
    alpha: float = 0.05,
) -> float
```

Average return of the asset on days when the system is in its worst alpha-percentile. This directly measures each asset's contribution to system-wide tail risk. Unlike CoVaR (which conditions on individual asset stress), MES conditions on system-wide stress.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `returns_system` | array-like | System-level return series (e.g., equal-weighted portfolio) |
| `returns_asset` | array-like | Individual asset return series |
| `alpha` | float | Tail percentile (default 0.05) |

**Returns:** MES value (float). Typically negative; more negative means higher contribution to systemic risk.

**Interpretation:**

| MES Value | Meaning |
|-----------|---------|
| Strongly negative | Asset amplifies system crashes (e.g., leveraged financials) |
| Mildly negative | Asset participates in crashes but does not drive them |
| Near zero | Asset is largely independent of system-wide stress |
| Positive | Asset acts as a hedge during crises (e.g., gold, government bonds) |

**Example:**

```python
from quantlite.contagion import marginal_expected_shortfall

# System = equal-weighted portfolio of all assets
system_returns = returns_df.mean(axis=1).values

for col in returns_df.columns:
    mes = marginal_expected_shortfall(system_returns, returns_df[col].values)
    print(f"{col:15s} MES: {mes:+.4f}")
```

![Systemic Risk Contributions](images/systemic_risk_contributions.png)

### `systemic_risk_contributions`

```python
systemic_risk_contributions(
    returns_df: pandas.DataFrame,
    alpha: float = 0.05,
) -> dict
```

Convenience function that computes MES for every asset relative to an equal-weighted system portfolio. Returns results sorted by contribution (most systemically important first).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `returns_df` | DataFrame | Asset returns, one column per asset |
| `alpha` | float | Tail percentile (default 0.05) |

**Returns:** Dictionary mapping asset names to MES values, sorted most negative first.

**Example:**

```python
import pandas as pd
from quantlite.contagion import systemic_risk_contributions

returns_df = pd.DataFrame({
    "Financials": financials,
    "Tech": tech,
    "Energy": energy,
    "Gold": gold,
    "Bonds": bonds,
})

contributions = systemic_risk_contributions(returns_df, alpha=0.05)
for asset, mes in contributions.items():
    print(f"{asset:15s} {mes:+.4f}")
# Most negative = most systemically dangerous
```

### `granger_causality`

```python
granger_causality(
    returns_a: array-like,
    returns_b: array-like,
    max_lag: int = 5,
) -> dict
```

Test Granger causality in both directions between two return series. Uses OLS regression to test whether lagged values of A predict B beyond B's own lags, and vice versa. The optimal lag is selected by AIC.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `returns_a` | array-like | First return series |
| `returns_b` | array-like | Second return series |
| `max_lag` | int | Maximum number of lags to test (default 5) |

**Returns:** Dictionary with keys `a_to_b` and `b_to_a`, each containing:

| Key | Description |
|-----|-------------|
| `f_statistic` | F-test statistic |
| `p_value` | P-value for the null hypothesis of no Granger causality |
| `optimal_lag` | Lag selected by AIC |
| `direction` | Direction label (`"a_to_b"` or `"b_to_a"`) |

**Interpretation:**

| p-value | Meaning |
|---------|---------|
| < 0.01 | Strong evidence of Granger causality |
| 0.01 to 0.05 | Moderate evidence |
| 0.05 to 0.10 | Weak evidence |
| > 0.10 | No significant evidence |

**Example:**

```python
from quantlite.contagion import granger_causality

result = granger_causality(spy_returns, eem_returns, max_lag=5)
print(f"SPY -> EEM: p={result['a_to_b']['p_value']:.4f}, lag={result['a_to_b']['optimal_lag']}")
print(f"EEM -> SPY: p={result['b_to_a']['p_value']:.4f}, lag={result['b_to_a']['optimal_lag']}")
# If SPY -> EEM is significant but EEM -> SPY is not,
# US equities lead emerging markets (as commonly observed).
```

![Granger Causality](images/granger_causality.png)

### `causal_network`

```python
causal_network(
    returns_df: pandas.DataFrame,
    max_lag: int = 5,
    significance: float = 0.05,
) -> dict
```

Build a directed causal graph from pairwise Granger causality tests across all assets. Each significant relationship becomes a directed edge in the network.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `returns_df` | DataFrame | Asset returns, one column per asset |
| `max_lag` | int | Maximum number of lags to test (default 5) |
| `significance` | float | P-value threshold for including an edge (default 0.05) |

**Returns:** Dictionary with:

| Key | Description |
|-----|-------------|
| `edges` | List of `(source, target, p_value, lag)` tuples |
| `adjacency_matrix` | NumPy array (1.0 where significant, 0.0 otherwise) |
| `nodes` | List of column names |

**Example:**

```python
from quantlite.contagion import causal_network

network = causal_network(returns_df, max_lag=5, significance=0.05)
print(f"Significant causal links: {len(network['edges'])}")
for src, tgt, p, lag in network["edges"]:
    print(f"  {src} -> {tgt} (p={p:.4f}, lag={lag})")
```

## Practical Guidance

### When to Use Each Metric

| Question | Use |
|----------|-----|
| "How much worse is my VaR when X crashes?" | `covar` |
| "Which asset contributes most to system tail risk?" | `systemic_risk_contributions` |
| "Does asset A lead asset B?" | `granger_causality` |
| "What does the information flow network look like?" | `causal_network` |
| "How much does one asset's distress worsen another's?" | `delta_covar` |

### Data Requirements

- **Minimum observations:** 250 for CoVaR, 500+ for reliable Granger causality
- **Frequency:** Daily returns work best; weekly returns reduce power substantially
- **Stationarity:** All functions assume stationary return series. Use returns, not prices.
- **Missing data:** Align all series to the same dates before passing to these functions

### Limitations

- Granger causality tests linear predictability only; it does not imply true economic causation
- CoVaR with the quantile method can be noisy with fewer than 500 observations
- The regression method for CoVaR assumes a linear relationship in the tails, which may not hold
- Causal network construction is O(n^2) in the number of assets; consider pruning large universes
