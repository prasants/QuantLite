# Scenario Stress Testing

## Overview

Stress testing answers a simple question: "What happens to my portfolio when things go wrong?"

The `quantlite.scenarios` module provides a composable scenario engine with a fluent API for building custom crisis scenarios, a library of pre-built historical scenarios, and tools for fragility analysis and shock propagation modelling.

## The Scenario Fluent API

Build crisis scenarios by chaining method calls:

```python
from quantlite.scenarios import Scenario

scenario = (
    Scenario("China property crisis")
    .shock("CNY", -0.15)
    .shock("BTC", -0.40)
    .shock("HANG_SENG", -0.30)
    .correlations(spike_to=0.85)
    .duration(days=30)
)
```

### `Scenario(name)`

Create a new scenario with a descriptive name.

### `.shock(asset, magnitude)`

Add a shock to a specific asset. Magnitude is expressed as a decimal (e.g. -0.40 for a 40% drop, +0.10 for a 10% rally).

### `.correlations(spike_to)`

Set the correlation spike level during the scenario. During crises, correlations typically spike to 0.7-0.95, destroying diversification benefits.

### `.duration(days)`

Set the scenario duration in trading days.

### Chaining

All methods return the `Scenario` instance, allowing unlimited chaining:

```python
scenario = (
    Scenario("Custom liquidity crisis")
    .shock("HY_CREDIT", -0.20)
    .shock("EM_EQUITY", -0.35)
    .shock("GLD", +0.08)
    .correlations(spike_to=0.80)
    .duration(days=45)
)
```

## Pre-built Scenario Library

The `SCENARIO_LIBRARY` contains five historical and hypothetical scenarios ready to use:

| Name | Key shocks | Correlation spike | Duration |
|------|-----------|-------------------|----------|
| **2008 GFC** | SPX -55%, HY_CREDIT -30%, COMMODITIES -40% | 0.90 | 250 days |
| **2020 COVID** | SPX -34%, OIL -65%, BTC -50% | 0.85 | 30 days |
| **2022 Luna/FTX** | BTC -65%, ETH -70%, SOL -95% | 0.92 | 60 days |
| **USDT depeg** | USDT -10%, BTC -25%, ETH -30% | 0.80 | 14 days |
| **rates +200bps** | BONDS_10Y -15%, SPX -20%, GROWTH -30% | 0.70 | 90 days |

```python
from quantlite.scenarios import SCENARIO_LIBRARY

for name, scenario in SCENARIO_LIBRARY.items():
    print(scenario)
```

## Functions

### `stress_test(weights, scenario, returns=None)`

Apply a scenario to a portfolio and return impact metrics.

```python
from quantlite.scenarios import stress_test, SCENARIO_LIBRARY

weights = {"SPX": 0.40, "BTC": 0.20, "BONDS_10Y": 0.30, "GLD": 0.10}
result = stress_test(weights, SCENARIO_LIBRARY["2008 GFC"])

print(f"Portfolio impact: {result['portfolio_impact']:.2%}")
print(f"Worst asset: {result['worst_asset']}")
print(f"Survives: {result['survival']}")
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `weights` | dict | — | Asset names to portfolio weights |
| `scenario` | Scenario | — | The crisis scenario to apply |
| `returns` | dict of array-like | None | Historical returns per asset (for volatility scaling) |

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `scenario_name` | str | Name of the applied scenario |
| `portfolio_impact` | float | Total weighted shock |
| `asset_impacts` | dict | Per-asset weighted impact |
| `worst_asset` | str | Most negatively impacted asset |
| `best_asset` | str | Least impacted (or positively impacted) asset |
| `survival` | bool | Whether portfolio value remains above zero |

---

### `fragility_heatmap(weights, scenarios, returns=None)`

Compute a fragility heatmap across multiple scenarios. Shows the impact on each position under each scenario.

```python
from quantlite.scenarios import fragility_heatmap, SCENARIO_LIBRARY

weights = {"SPX": 0.30, "BTC": 0.20, "ETH": 0.10, "BONDS_10Y": 0.25, "GLD": 0.15}
scenarios = list(SCENARIO_LIBRARY.values())

heatmap = fragility_heatmap(weights, scenarios)
for scenario_name, impacts in heatmap.items():
    print(f"\n{scenario_name}:")
    for asset, impact in impacts.items():
        print(f"  {asset}: {impact:+.4f}")
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `weights` | dict | — | Portfolio weights per asset |
| `scenarios` | list of Scenario | — | Scenarios to test against |
| `returns` | dict of array-like | None | Historical returns per asset |

**Returns:** `dict[str, dict[str, float]]` — Nested mapping: scenario name to asset to impact value.

---

### `shock_propagation(returns, initial_shock, correlation_matrix=None)`

Model how an initial shock cascades through correlated assets. Uses empirical correlations (or a supplied matrix) to estimate second-order effects.

```python
import numpy as np
from quantlite.scenarios import shock_propagation

rng = np.random.default_rng(42)
returns = {
    "SPX": rng.normal(0.0003, 0.012, 500),
    "BTC": rng.normal(0.0005, 0.035, 500),
    "GLD": rng.normal(0.0001, 0.008, 500),
}

propagated = shock_propagation(returns, initial_shock={"BTC": -0.50})
for asset, shock in propagated.items():
    print(f"  {asset}: {shock:+.4f}")
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `returns` | dict of array-like | — | Historical returns per asset |
| `initial_shock` | dict | — | Asset to shock magnitude for the initial event |
| `correlation_matrix` | ndarray | None | Pre-computed correlation matrix (estimated from returns if None) |

**Returns:** `dict[str, float]` — Asset to propagated shock magnitude after cascade.

## Building Custom Scenarios

Combine the fluent API with `stress_test` for bespoke analysis:

```python
from quantlite.scenarios import Scenario, stress_test

# Build a scenario for your specific risk concerns
tariff_war = (
    Scenario("US-China tariff escalation")
    .shock("SPX", -0.15)
    .shock("HANG_SENG", -0.25)
    .shock("USD_CNY", +0.05)
    .shock("GLD", +0.10)
    .correlations(spike_to=0.75)
    .duration(days=60)
)

weights = {"SPX": 0.35, "HANG_SENG": 0.15, "GLD": 0.20, "BONDS_10Y": 0.30}
result = stress_test(weights, tariff_war)
print(f"Portfolio impact: {result['portfolio_impact']:.2%}")
```

The scenario engine is deliberately simple. It applies direct shocks and weighted aggregation without black-box simulation, so you can reason about the results transparently. For Monte Carlo approaches, combine with `quantlite.monte_carlo`.
