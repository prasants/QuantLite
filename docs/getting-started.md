# Getting Started

## Installation

Install QuantLite from PyPI:

```bash
pip install quantlite
```

For the full feature set (Yahoo Finance, crypto feeds, interactive charts, regime detection):

```bash
pip install "quantlite[all]"
```

Or pick what you need:

```bash
pip install "quantlite[yahoo]"      # Yahoo Finance data
pip install "quantlite[crypto]"     # Crypto exchange data
pip install "quantlite[stream]"     # Live price streaming
pip install "quantlite[plotly]"     # Interactive visualisation
pip install "quantlite[report]"     # HTML/PDF reports
```

### Requirements

- Python 3.9 or later
- NumPy, Pandas, SciPy, and Matplotlib are installed automatically

## Your First Analysis: The Dream API

QuantLite's pipeline API gets you from raw data to a full analysis in five lines:

```python
import quantlite as ql

# Fetch 5 years of daily data for a diversified portfolio
data = ql.fetch(["AAPL", "MSFT", "GLD", "TLT"], period="5y")

# Detect market regimes (calm, volatile, crisis)
regimes = ql.detect_regimes(data)

# Build an optimised portfolio that respects regime dynamics
portfolio = ql.construct_portfolio(data, regimes)

# Backtest with realistic transaction costs and slippage
result = ql.backtest(portfolio)

# Generate a comprehensive tearsheet
ql.tearsheet(result)
```

This produces a multi-page report covering:

- Cumulative returns and drawdown analysis
- Regime-conditional performance breakdown
- Risk metrics (VaR, CVaR, maximum drawdown) computed with fat-tailed assumptions
- Portfolio composition over time

## What to Explore Next

| If you want to...                          | Read this                                      |
|--------------------------------------------|------------------------------------------------|
| Understand risk metrics                    | [Risk Metrics](risk.md)                        |
| Detect market regimes                      | [Regime Detection](regimes.md)                 |
| Build and optimise portfolios              | [Portfolio Optimisation](portfolio.md)          |
| Stress-test with scenario analysis         | [Scenario Analysis](scenarios.md)              |
| Check if your backtest is lying to you     | [Overfitting Detection](overfitting.md)        |
| Stream live prices and detect regimes      | [Price Streaming](streaming.md)                |
| Measure systemic and contagion risk        | [Contagion Metrics](contagion.md)              |
| Run fat-tailed Monte Carlo simulations     | [EVT Simulation](simulation_evt.md)            |
| See every public function                  | [API Reference](api.md)                        |

## Project Structure

```
quantlite/
├── core/           # Correlation, covariance, clustering
├── risk/           # VaR, CVaR, drawdown, tail metrics
├── distributions/  # GPD, stable, fat-tailed fits
├── regimes/        # HMM, changepoint, online detection
├── portfolio/      # Optimisation, rebalancing, signals
├── backtesting/    # Engine, configs, risk limits
├── simulation/     # EVT, copula, regime Monte Carlo
├── crypto/         # Stablecoin, exchange, on-chain risk
├── factors/        # Classical, custom, tail risk factors
├── alerts/         # Rule-based and threshold alerts
├── data/           # Connectors, streaming
└── viz/            # Matplotlib and Plotly charts
```
