# Dream API (Pipeline)

The `pipeline` module provides QuantLite's highest-level API: five functions that chain together the entire quant workflow from data fetching to tearsheet generation.

## Quick Start

```python
import quantlite as ql

# Fetch returns for a multi-asset portfolio
data = ql.fetch(["AAPL", "BTC-USD", "GLD", "TLT"], period="5y")

# Detect market regimes
regimes = ql.detect_regimes(data, n_regimes=3)

# Construct regime-aware portfolio weights
weights = ql.construct_portfolio(data, regime_aware=True, regimes=regimes)

# Backtest the strategy
result = ql.backtest(data, weights)

# Generate a full tearsheet
ql.tearsheet(result, regimes=regimes, save="portfolio.txt")
```

## API Reference

### `fetch(tickers, period="5y", source="yahoo")`

Fetch price data and compute simple returns.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tickers` | str or list[str] | required | Ticker symbols |
| `period` | str | "5y" | Lookback period (e.g. "1y", "5y", "6mo") |
| `source` | str | "yahoo" | Data source name |

**Returns:** `pd.DataFrame` of simple returns with DatetimeIndex, one column per ticker.

**Note:** Requires `yfinance` for Yahoo data. Install with `pip install quantlite[yahoo]`.

### `detect_regimes(returns_df, method="hmm", n_regimes=3)`

Detect market regimes using Hidden Markov Models.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `returns_df` | DataFrame | required | Returns (single or multi-asset) |
| `method` | str | "hmm" | Detection method (currently only "hmm") |
| `n_regimes` | int | 3 | Number of regimes to detect |
| `rng_seed` | int or None | None | Random seed for reproducibility |

**Returns:** `np.ndarray` of integer regime labels (0 = worst regime, sorted by mean return).

For multi-asset DataFrames, the mean return across assets is used as the input signal.

**Requires:** `hmmlearn`. Install with `pip install hmmlearn`.

### `construct_portfolio(returns_df, method="hrp", regime_aware=True, regimes=None)`

Construct portfolio weights with optional regime-aware defensive tilting.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `returns_df` | DataFrame | required | Asset returns (columns = assets) |
| `method` | str | "hrp" | "hrp", "min_variance", or "equal_weight" |
| `regime_aware` | bool | True | Apply defensive tilting in crisis regimes |
| `regimes` | np.ndarray or None | None | Regime labels (required if regime_aware) |
| `defensive_tilt` | float | 0.3 | Tilt magnitude for crisis regimes |

**Returns:** `dict[str, float]` mapping asset names to weights summing to 1.0.

### `backtest(returns_df, weights, rebalance="monthly")`

Run a portfolio backtest with fixed target weights.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `returns_df` | DataFrame | required | Asset returns |
| `weights` | dict[str, float] | required | Target weights |
| `rebalance` | str | "monthly" | "daily", "weekly", or "monthly" |
| `initial_capital` | float | 10000.0 | Starting portfolio value |

**Returns:** Dictionary containing:

| Key | Type | Description |
|-----|------|-------------|
| `equity_curve` | pd.Series | Portfolio value over time |
| `returns` | pd.Series | Portfolio returns per period |
| `total_return` | float | Total percentage return |
| `annualised_return` | float | Annualised return |
| `annualised_volatility` | float | Annualised volatility |
| `sharpe_ratio` | float | Sharpe ratio |
| `max_drawdown` | float | Maximum drawdown (negative) |
| `weights` | dict | The input weights |

### `tearsheet(backtest_result, regimes=None, save=None)`

Generate a tearsheet from backtest results.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backtest_result` | dict | required | Output from `backtest()` |
| `regimes` | np.ndarray or None | None | Regime labels for breakdown |
| `save` | str or None | None | File path to save summary |

When `regimes` is provided, the tearsheet includes:
- Equity curves and drawdowns
- Per-regime risk metrics (VaR, CVaR, volatility, skewness, kurtosis)
- Performance attribution by regime
- Markdown comparison table

When `save` is provided, a text summary is written to the specified path.

## Design Notes

The pipeline module is intentionally thin. Each function is a convenience wrapper around QuantLite's lower-level modules:

- `fetch()` wraps `quantlite.data.fetch()`
- `detect_regimes()` wraps `quantlite.regimes.hmm.fit_regime_model()`
- `construct_portfolio()` wraps `quantlite.regime_integration.portfolio.regime_aware_weights()`
- `backtest()` is a lightweight fixed-weight backtester
- `tearsheet()` wraps `quantlite.regime_integration.reporting.regime_tearsheet()`

For advanced use cases (custom allocation functions, circuit breakers, slippage models), use the lower-level modules directly. See [Architecture](architecture.md) for the full module map.
