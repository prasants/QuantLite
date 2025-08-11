# Antifragility

## Overview

Fragile things break under stress. Robust things resist it. **Antifragile things get stronger.**

Nassim Taleb introduced this concept to describe systems that benefit from volatility, randomness, and disorder. In portfolio terms: a fragile position loses disproportionately from shocks, a robust one is indifferent, and an antifragile one gains more from upside shocks than it loses from downside shocks of equal magnitude.

The `quantlite.antifragile` module provides six tools for measuring where your portfolio sits on the fragile-antifragile spectrum, detecting when you are in Taleb's dangerous Fourth Quadrant, constructing barbell allocations, estimating Lindy survival, and quantifying principal-agent alignment.

## API Reference

### `antifragility_score(returns)`

Measure antifragility via payoff convexity. Compares the average gain from positive shocks (above median) to the average loss from negative shocks (below median).

```python
from quantlite.antifragile import antifragility_score

# Antifragile: gains more from upside than loses from downside
score = antifragility_score([0.10, -0.02, 0.15, -0.03, 0.20, -0.01])
print(f"Antifragility score: {score:.4f}")  # Positive = antifragile
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `returns` | array-like | Simple period returns |

**Returns:** `float` — Score > 0 indicates antifragility, < 0 indicates fragility, near 0 indicates robustness.

**Interpretation guide:**

| Score range | Classification | Meaning |
|-------------|---------------|---------|
| > 0.5 | Strongly antifragile | Significant convex payoff |
| 0.1 to 0.5 | Mildly antifragile | Slight upside bias |
| -0.1 to 0.1 | Robust | Symmetric payoff |
| -0.5 to -0.1 | Mildly fragile | Slight downside bias |
| < -0.5 | Strongly fragile | Concave payoff, avoid |

---

### `convexity_score(returns, shocks)`

Measure payoff curvature by fitting a second-order polynomial to (shock, return) pairs. Positive curvature means convex payoff (antifragile); negative means concave (fragile).

```python
import numpy as np
from quantlite.antifragile import convexity_score

shocks = np.linspace(-0.10, 0.10, 50)
# Convex payoff: quadratic with positive curvature
returns = 0.5 * shocks + 2.0 * shocks**2 + np.random.default_rng(42).normal(0, 0.001, 50)
score = convexity_score(returns, shocks)
print(f"Convexity score: {score:.4f}")  # Should be close to 2.0
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `returns` | array-like | Observed returns (dependent variable) |
| `shocks` | array-like | Shock magnitudes (independent variable) |

**Returns:** `float` — Second-order polynomial coefficient. Positive = convex, negative = concave.

---

### `fourth_quadrant(returns)`

Detect if returns fall in Taleb's Fourth Quadrant, where fat tails meet complex payoffs, making statistical models dangerous and unreliable.

```python
from quantlite.antifragile import fourth_quadrant

result = fourth_quadrant(returns)
print(f"Excess kurtosis: {result['kurtosis']:.2f}")
print(f"Fat-tailed: {result['fat_tailed']}")
print(f"Fourth Quadrant: {result['fourth_quadrant']}")
print(result['warning'])
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `returns` | array-like | Simple period returns |

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `kurtosis` | float | Excess kurtosis (normal = 0) |
| `fat_tailed` | bool | True if excess kurtosis > 1 |
| `payoff_nonlinearity` | float | Ratio of tail impact to body impact |
| `fourth_quadrant` | bool | True if fat-tailed AND nonlinear payoff |
| `warning` | str | Human-readable advisory |

---

### `barbell_allocation(conservative_returns, aggressive_returns, conservative_pct=0.9)`

Compute barbell allocation metrics. The barbell strategy allocates most capital to hyperconservative assets and a small fraction to hyperaggressive ones, with nothing in the mediocre middle.

```python
import numpy as np
from quantlite.antifragile import barbell_allocation

rng = np.random.default_rng(42)
bonds = rng.normal(0.0001, 0.002, 252)       # Conservative: low vol
crypto = rng.standard_t(3, 252) * 0.03       # Aggressive: fat tails

result = barbell_allocation(bonds, crypto, conservative_pct=0.90)
print(f"Blended geometric return: {result['blended_geometric']:.6f}")
print(f"Maximum single-period loss: {result['max_loss']:.4f}")
print(f"Upside capture (top 10%): {result['upside_capture']:.4f}")
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `conservative_returns` | array-like | — | Returns from the conservative leg |
| `aggressive_returns` | array-like | — | Returns from the aggressive leg |
| `conservative_pct` | float | 0.9 | Fraction allocated to conservative leg |

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `conservative_pct` | float | Conservative allocation |
| `aggressive_pct` | float | Aggressive allocation |
| `blended_arithmetic` | float | Arithmetic mean of blended returns |
| `blended_geometric` | float | Geometric mean of blended returns |
| `max_loss` | float | Worst single-period loss |
| `upside_capture` | float | Mean of top 10% of blended returns |

---

### `lindy_estimate(age, confidence=0.95)`

Estimate remaining life expectancy using the Lindy effect. For non-perishable entities (ideas, technologies, institutions), expected remaining lifespan is proportional to current age.

```python
from quantlite.antifragile import lindy_estimate

# Bitcoin: ~15 years old
btc = lindy_estimate(age=15)
print(f"Expected remaining life: {btc['expected_remaining']:.0f} years")
print(f"Total expected life: {btc['total_expected']:.0f} years")

# US Dollar: ~230 years old
usd = lindy_estimate(age=230)
print(f"USD expected remaining: {usd['expected_remaining']:.0f} years")
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `age` | float | — | Current age of the entity (any consistent unit) |
| `confidence` | float | 0.95 | Confidence level for the survival bound |

**Returns:** `dict` with keys `age`, `expected_remaining`, `lower_bound`, `total_expected`.

---

### `skin_in_game_score(manager_returns, fund_returns)`

Measure principal-agent alignment via payoff asymmetry. A good score means the manager shares the pain when the fund loses.

```python
import numpy as np
from quantlite.antifragile import skin_in_game_score

rng = np.random.default_rng(42)
fund = rng.normal(0.001, 0.02, 252)
# Good manager: returns track the fund
aligned_mgr = fund * 0.8 + rng.normal(0, 0.002, 252)

result = skin_in_game_score(aligned_mgr, fund)
print(f"Alignment: {result['alignment']:.3f}")
print(f"Downside sharing: {result['downside_sharing']:.3f}")
print(f"Composite score: {result['score']:.3f}")
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `manager_returns` | array-like | Returns experienced by the manager |
| `fund_returns` | array-like | Returns experienced by fund investors |

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `alignment` | float | Correlation between manager and fund |
| `downside_sharing` | float | Ratio of manager downside to fund downside |
| `upside_asymmetry` | float | Manager upside capture relative to fund |
| `score` | float | Composite score, 0 to 1. 1.0 = perfect alignment |

## Interpretation Guide

The antifragility framework is not about predicting the future. It is about positioning so that uncertainty works in your favour rather than against you.

**Fragile positions** have concave payoffs: they gain a little from small positive moves but lose catastrophically from large negative ones. Short volatility strategies, highly leveraged positions, and concentrated bets in the Fourth Quadrant are fragile.

**Robust positions** have linear payoffs: gains and losses are roughly symmetric. A diversified index fund is approximately robust.

**Antifragile positions** have convex payoffs: they lose a little from small negative moves but gain disproportionately from large positive ones. Long option positions, barbell allocations, and venture-style portfolios can be antifragile.

The key insight: you do not need to predict which shocks will occur. You only need to ensure your portfolio benefits from shocks of any kind. That is the antifragile position.
