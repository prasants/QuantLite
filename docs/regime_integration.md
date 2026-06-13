# Regime-Aware Integration

The `regime_integration` module bridges regime detection with portfolio construction, risk analytics, and reporting. Rather than treating regimes as a standalone analysis, this module makes regime awareness a first-class input to every stage of the investment workflow.

## Overview

| Submodule | Purpose |
|-----------|---------|
| `risk` | VaR, CVaR, and summary statistics broken down by regime |
| `portfolio` | Regime-aware weight construction and filtered backtesting |
| `reporting` | Tearsheets, attribution, and comparison tables by regime |

## Risk Analytics

### `regime_conditional_var(returns, regimes, alpha=0.05)`

Compute historical Value at Risk separately for each regime.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `returns` | array-like | required | Simple periodic returns |
| `regimes` | array-like | required | Regime labels (same length) |
| `alpha` | float | 0.05 | Significance level (0.05 = 95% VaR) |

**Returns:** `dict[str, float]` mapping regime labels to VaR values.

```python
from quantlite.regime_integration import regime_conditional_var

var_by_regime = regime_conditional_var(returns, regimes, alpha=0.05)
# {'0': -0.0312, '1': -0.0187, '2': -0.0098}
```

**Interpretation:** Regime 0 (crisis) has a 95% VaR of -3.12%, meaning on 5% of crisis days, losses exceed 3.12%. Compare with regime 2 (bull) at just -0.98%.

### `regime_conditional_cvar(returns, regimes, alpha=0.05)`

Same interface as `regime_conditional_var`, but returns Expected Shortfall (mean of losses beyond VaR).

### `regime_risk_summary(returns, regimes, alpha=0.05)`

One-call summary returning VaR, CVaR, annualised volatility, skewness, and kurtosis per regime plus overall.

| Return key | Description |
|------------|-------------|
| `var` | Historical VaR at the given alpha |
| `cvar` | Conditional VaR (Expected Shortfall) |
| `volatility` | Annualised volatility |
| `skewness` | Return distribution skewness |
| `kurtosis` | Excess kurtosis |
| `count` | Number of observations in this regime |

```python
from quantlite.regime_integration import regime_risk_summary

summary = regime_risk_summary(returns, regimes)
for regime, metrics in summary.items():
    print(f"Regime {regime}: vol={metrics['volatility']:.2%}")
```

### `regime_transition_risk(transition_matrix, current_regime)`

Estimates the probability of transitioning to a worse regime within 1, 5, and 21 trading days.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `transition_matrix` | np.ndarray | required | Row-stochastic transition matrix |
| `current_regime` | int | required | Current regime index |
| `worse_regimes` | list[int] or None | None | Indices considered worse (default: lower indices) |

```python
from quantlite.regime_integration import regime_transition_risk

risk = regime_transition_risk(model.transition_matrix, current_regime=2)
# {'1_step': 0.15, '5_step': 0.42, '21_step': 0.68}
```

## Portfolio Construction

### `regime_aware_weights(returns_df, regimes, method="hrp", defensive_tilt=0.3)`

Computes portfolio weights that automatically tilt towards defensive assets when the current regime is crisis.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `returns_df` | DataFrame | required | Asset returns (columns = assets) |
| `regimes` | array-like | required | Regime labels |
| `method` | str | "hrp" | "hrp", "min_variance", or "equal_weight" |
| `defensive_tilt` | float | 0.3 | How aggressively to tilt during crisis (0 to 1) |
| `defensive_assets` | list[str] or None | None | Assets to treat as defensive |

**Defensive asset detection:** If `defensive_assets` is None, assets with names containing "GLD", "TLT", "bond", "gold", "treasury", "IEF", or "SHY" are automatically identified.

### `regime_rebalance_signals(regimes, lookback=5)`

Detects regime transitions confirmed by consecutive observations to avoid whipsaw.

Returns a list of signal dictionaries:
```python
[{"index": 210, "from_regime": 2, "to_regime": 1}, ...]
```

### `regime_filtered_backtest(returns_df, weights_by_regime, regimes)`

Backtest with different weight sets applied per regime.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `returns_df` | DataFrame | required | Asset returns |
| `weights_by_regime` | dict | required | `{regime_label: {asset: weight}}` |
| `regimes` | array-like | required | Regime labels |
| `rebalance` | str | "monthly" | "daily", "weekly", or "monthly" |
| `initial_capital` | float | 10000.0 | Starting value |

**Returns:** Dictionary with `equity_curve` (Series), `regime_attribution` (dict), `total_return` (float), and `weights_history` (list).

## Reporting

### `regime_tearsheet(returns, regimes, benchmark=None)`

Full tearsheet with equity curves, drawdowns, risk metrics, and time spent in each regime.

### `regime_performance_attribution(returns, regimes)`

Attributes total return to each regime period, showing cumulative return, percentage contribution, and mean per-period return.

### `regime_comparison_table(returns, regimes)`

Generates a markdown table comparing VaR, CVaR, volatility, skewness, and kurtosis across all regimes.

```
| Regime | Count | Ann. Vol | VaR (95%) | CVaR (95%) | Skewness | Kurtosis |
|--------|-------|----------|-----------|------------|----------|----------|
| 0      | 50    | 63.49%   | -0.0654   | -0.0812    | -0.31    | 0.12     |
| 1      | 100   | 39.69%   | -0.0412   | -0.0523    | -0.15    | 0.08     |
| 2      | 200   | 15.87%   | -0.0165   | -0.0198    | 0.02     | -0.04    |
| overall| 350   | 31.50%   | -0.0312   | -0.0456    | -0.45    | 1.23     |
```
