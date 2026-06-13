# Architecture

QuantLite is organised as a layered toolkit where higher-level modules compose lower-level primitives. This document describes the module dependency structure, design principles, and extension points.

## Module Dependency Diagram

```
                        ┌─────────────┐
                        │  pipeline   │   Dream API (top-level entry point)
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌─────▼──────┐  ┌──────▼───────┐
    │ regime_        │  │ backtesting │  │   report     │
    │ integration    │  │             │  │              │
    └───┬───┬───┬────┘  └──────┬──────┘  └──────────────┘
        │   │   │              │
        │   │   └──────────────┤
        │   │                  │
  ┌─────▼─┐ │  ┌──────────────▼──────────────┐
  │regimes│ │  │        portfolio             │
  │ (HMM, │ │  │ (optimisation, rebalancing)  │
  │change-│ │  └──────────────┬───────────────┘
  │point) │ │                 │
  └───────┘ │        ┌────────┼────────┐
            │        │        │        │
       ┌────▼───┐  ┌─▼──┐  ┌─▼────┐ ┌─▼──────────┐
       │  risk  │  │data│  │core  │ │ dependency  │
       │metrics │  │    │  │types │ │ (clustering)│
       └────────┘  └────┘  └──────┘ └─────────────┘

  ┌──────────────────────────────────────────────────┐
  │              Independent modules                  │
  │  distributions, ergodicity, antifragile,          │
  │  scenarios, forensics, overfit, resample,         │
  │  contagion, network, diversification, crypto,     │
  │  simulation, factors, viz                         │
  └──────────────────────────────────────────────────┘
```

## Design Principles

1. **Fat-tail-native.** Every risk metric, simulation, and optimisation accounts for non-Gaussian behaviour. No function silently assumes normality.

2. **Composable primitives.** Low-level functions (VaR, CVaR, HMM fitting) are usable independently. Higher-level modules (pipeline, regime_integration) compose them for convenience.

3. **Dataclass results.** Functions return frozen dataclasses or plain dictionaries rather than opaque objects. This makes results easy to inspect, serialise, and test.

4. **Optional dependencies.** Heavy dependencies (hmmlearn, yfinance, ccxt) are optional. Functions that need them raise a helpful ImportError with install instructions.

5. **No global state.** Functions are pure where possible. Random seeds are explicit parameters, not global settings.

6. **Test everything.** Every public function has at least one test. Edge cases (empty inputs, single observations, degenerate matrices) are covered.

## Key Modules

| Module | Purpose | Key exports |
|--------|---------|-------------|
| `core` | Shared types and utilities | `DrawdownInfo`, `ReturnMoments` |
| `data` | Unified data fetching | `fetch()`, data source registry |
| `risk` | Risk metrics | `value_at_risk()`, `cvar()`, `sortino_ratio()` |
| `regimes` | Regime detection | `fit_regime_model()`, `RegimeModel` |
| `portfolio` | Portfolio construction | `hrp_weights()`, `min_variance_weights()` |
| `backtesting` | Multi-asset backtester | `run_backtest()`, `BacktestResult` |
| `regime_integration` | Regime-aware analytics | `regime_aware_weights()`, `regime_tearsheet()` |
| `pipeline` | Dream API | `fetch()`, `detect_regimes()`, `backtest()` |
| `distributions` | Fat-tailed distributions | Student-t, stable, GPD fitting |
| `simulation` | Monte Carlo engines | EVT-based, copula-based simulation |
| `viz` | Visualisation | Regime plots, portfolio charts, risk dashboards |

## Extension Points

### Adding a new data source

1. Create a class inheriting from `data.base.DataSource`.
2. Implement the `fetch(symbol, **kwargs)` method.
3. Decorate with `@register_source("name")`.
4. Import the module in `data/__init__.py` to trigger registration.

### Adding a new portfolio method

1. Add the function to `portfolio/optimisation.py`.
2. Return a `PortfolioWeights` dataclass.
3. Export from `portfolio/__init__.py`.
4. Add support in `pipeline.construct_portfolio()` if desired.

### Adding a new regime detection method

1. Add the module under `regimes/`.
2. Return a result compatible with the `RegimeModel` interface.
3. Add a method branch in `pipeline.detect_regimes()`.

### Adding new visualisations

1. Add functions to the appropriate `viz/` submodule.
2. Follow Stephen Few chart standards (see CONTRIBUTING.md).
3. Add example usage in `examples/`.
