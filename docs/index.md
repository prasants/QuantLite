# QuantLite

**A fat-tail-native quantitative finance toolkit.**

QuantLite provides honest, realistic tools for quantitative finance: extreme value theory, fat-tailed distributions, regime detection, portfolio optimisation, and backtesting that doesn't lie to you.

## The Dream API

Five lines from raw data to a full tearsheet:

```python
import quantlite as ql

data = ql.fetch(["AAPL", "MSFT", "GLD", "TLT"], period="5y")
regimes = ql.detect_regimes(data)
portfolio = ql.construct_portfolio(data, regimes)
result = ql.backtest(portfolio)
ql.tearsheet(result)
```

## Why QuantLite?

- **Fat tails are the default.** Every distribution, every simulation, every risk metric assumes markets bite.
- **Regimes matter.** Volatility clusters. Correlations spike in crises. QuantLite models this natively.
- **Honest backtesting.** Overfitting detection, strategy forensics, and resampled validation built in.
- **Real-time capable.** Stream prices, detect regimes online, and fire alerts as conditions change.

## Quick Links

- [Getting Started](getting-started.md) — install and run your first analysis
- [Quickstart](quickstart.md) — 10-minute tour of the key modules
- [API Reference](api.md) — every public function documented
