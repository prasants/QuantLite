# Ergodicity Economics

## Overview

"What happens to the average is not what happens to you."

Traditional finance assumes **ergodicity**: that the average outcome across many people at one moment equals the average outcome for one person across many moments. This assumption is wrong for nearly every real-world wealth process.

Ole Peters and Nassim Taleb have shown that this distinction is not academic; it is the difference between strategies that look profitable on paper and strategies that actually grow your wealth over time. The ensemble average (arithmetic mean) tells you what happens to the average investor. The time average (geometric mean) tells you what happens to *you*.

Consider a simple coin flip: heads gives +50%, tails gives -40%. The ensemble average is +5% per flip. Looks great. But the time average is negative: after many flips, almost every individual path converges to ruin. The ensemble average is a mirage created by a handful of astronomically lucky paths.

The `quantlite.ergodicity` module provides tools to quantify this gap, compute optimal leverage via the Kelly criterion, and visualise how leverage destroys time-average growth through volatility drag.

## API Reference

### `time_average(returns)`

Compute the time-average (geometric mean) growth rate. This is the growth rate experienced by a single individual over many periods: the only rate that matters for wealth dynamics.

```python
from quantlite.ergodicity import time_average

g = time_average([0.10, -0.05, 0.08, -0.12, 0.06])
print(f"Time-average growth: {g:.4f}")
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `returns` | array-like | Simple period returns (e.g. 0.05 for 5%) |

**Returns:** `float` — Geometric mean growth rate per period.

---

### `ensemble_average(returns)`

Compute the ensemble-average (arithmetic mean) growth rate. This is what textbooks report and what misleads: the average across many parallel realisations at a single point in time.

```python
from quantlite.ergodicity import ensemble_average

ea = ensemble_average([0.10, -0.05, 0.08, -0.12, 0.06])
print(f"Ensemble average: {ea:.4f}")
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `returns` | array-like | Simple period returns |

**Returns:** `float` — Arithmetic mean return per period.

---

### `ergodicity_gap(returns)`

Compute the gap between ensemble and time averages. A large positive gap means the strategy looks good on paper (ensemble average) but is dangerous for any individual over time.

```python
from quantlite.ergodicity import ergodicity_gap

gap = ergodicity_gap([0.50, -0.40, 0.50, -0.40])
print(f"Ergodicity gap: {gap:.4f}")
# Positive gap = non-ergodic danger
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `returns` | array-like | Simple period returns |

**Returns:** `float` — `ensemble_average - time_average`. Positive means non-ergodic danger.

---

### `kelly_fraction(returns, risk_free=0.0)`

Compute the optimal Kelly fraction for geometric growth. The Kelly criterion maximises the expected logarithmic growth rate, finding the leverage level where time-average growth peaks.

Uses a numerical grid search over [-0.5, 3.0] for robustness with empirical return distributions.

```python
from quantlite.ergodicity import kelly_fraction

returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03]
f = kelly_fraction(returns)
print(f"Optimal Kelly fraction: {f:.2f}")
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `returns` | array-like | — | Simple period returns |
| `risk_free` | float | 0.0 | Risk-free rate per period |

**Returns:** `float` — Optimal fraction of capital to deploy. Can be < 0 (short) or > 1 (leveraged).

---

### `leverage_effect(returns, leverages=None)`

Show how leverage affects time-average growth rate. Demonstrates why 2x or 3x leverage can destroy wealth even when the underlying has positive expected returns: volatility drag compounds geometrically.

```python
from quantlite.ergodicity import leverage_effect

returns = [0.02, -0.01, 0.03, -0.015, 0.01]
result = leverage_effect(returns, leverages=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
for lev, growth in result.items():
    print(f"  {lev:.1f}x leverage: {growth:.4f} time-average growth")
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `returns` | array-like | — | Simple period returns of the underlying |
| `leverages` | list of float | `[1, 2, 3, 5]` | Leverage multiples to evaluate |

**Returns:** `dict[float, float]` — Mapping of leverage multiple to time-average growth rate. Returns -1.0 for any leverage that would wipe out capital.

---

### `geometric_mean_dominance(returns_a, returns_b)`

Test whether strategy A dominates strategy B in geometric mean.

```python
from quantlite.ergodicity import geometric_mean_dominance

result = geometric_mean_dominance(
    returns_a=[0.02, 0.01, -0.005, 0.015],
    returns_b=[0.03, -0.02, 0.04, -0.03],
)
print(f"Dominant strategy: {result['dominant']}")
print(f"Margin: {result['margin']:.6f}")
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `returns_a` | array-like | Simple period returns for strategy A |
| `returns_b` | array-like | Simple period returns for strategy B |

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `g_mean_a` | float | Geometric mean growth of A |
| `g_mean_b` | float | Geometric mean growth of B |
| `dominant` | str | `'A'`, `'B'`, or `'neither'` |
| `margin` | float | Absolute difference in geometric means |

## Conceptual Example: The Ergodicity Gap

```python
import numpy as np
from quantlite.ergodicity import time_average, ensemble_average, ergodicity_gap

# Simulate a volatile strategy: +50% or -40% each period
rng = np.random.default_rng(42)
returns = rng.choice([0.50, -0.40], size=1000)

ta = time_average(returns)
ea = ensemble_average(returns)
gap = ergodicity_gap(returns)

print(f"Ensemble average (what the brochure says): {ea:+.4f}")
print(f"Time average (what you actually get):      {ta:+.4f}")
print(f"Ergodicity gap:                             {gap:+.4f}")
```

Output:

```
Ensemble average (what the brochure says): +0.0510
Time average (what you actually get):      -0.0513
Ergodicity gap:                             +0.1023
```

The strategy has a positive expected return but negative geometric growth. Over time, you go broke while the brochure shows profits.

## Common Misunderstandings

**"The geometric mean is just a more conservative estimate."**
No. The geometric mean is the *correct* measure for multiplicative processes like wealth growth. The arithmetic mean answers a different question entirely (the ensemble question, not the time question).

**"Diversification solves the ergodicity problem."**
Diversification helps by reducing volatility, which narrows the ergodicity gap. But it does not eliminate the fundamental non-ergodicity of multiplicative processes. A diversified portfolio of ergodicity-violating strategies is still non-ergodic.

**"Kelly is too aggressive; use half-Kelly."**
This is actually good advice, but for the wrong reason. The Kelly fraction is optimal given perfect knowledge of the return distribution. Since we never have perfect knowledge, fractional Kelly provides a margin of safety against estimation error. The right framing: Kelly is the *ceiling*, not the target.

**"High expected returns compensate for high volatility."**
This is precisely the ergodicity illusion. A strategy with 20% expected return and 40% volatility can have a negative time-average growth rate. The ensemble average (expected return) is irrelevant to your individual trajectory.
