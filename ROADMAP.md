# QuantLite Roadmap

**Positioning:** QuantLite operationalises the ideas of Taleb, Peters, and Lopez de Prado. Fat tails are the default. Ergodicity is respected. Backtests are honest. If your current tools assume Gaussian, they are lying to you.

**Design law:** Every visualisation, matplotlib or Plotly, follows Stephen Few's principles without exception. Maximum data-ink ratio, direct labels, muted palette, no chartjunk. This is not a feature; it is a constraint on everything we build.

**Target users:**

- **Quant researchers** doing their own strategy development. They get: data connectors, fat-tail-native analytics, honest backtesting, ergodicity tools. The stuff they would otherwise cobble together from six different libraries, none of which talk to each other.
- **Institutional risk teams** looking for a drop-in toolkit. They get: regime-aware VaR/CVaR, contagion metrics (CoVaR, MES), stress testing, factor decomposition, auto-generated tearsheets. The stuff they currently pay Bloomberg or Axioma for, or build in-house and never maintain.

The common thread is the philosophy: if your tools assume Gaussian, they are lying to you. The researcher cares because they want correct answers. The risk team cares because their regulator cares. The API is designed so a researcher can use individual functions in a notebook, and a risk team can use the higher-level pipeline as a drop-in workflow. Same library, different depth.

---

## v0.2 (current)

Foundation. Fat-tail-native risk, EVT, copulas, regime detection, Stephen Few visualisations.

- Risk metrics: VaR (historical, parametric, Cornish-Fisher), CVaR, Sortino, Calmar, Omega, tail ratio
- Extreme Value Theory: GPD, GEV, Hill estimator, POT
- Fat-tailed distributions
- Copulas: Gaussian, Student-t, Clayton, Gumbel, Frank
- Correlation: rolling, EWMA, stress
- Clustering: Hierarchical Risk Parity
- Regime detection: HMM, Bayesian changepoint, conditional risk metrics
- Portfolio optimisation: mean-variance, mean-CVaR, risk parity, HRP, max Sharpe, Black-Litterman, Kelly
- Backtesting: multi-asset engine with circuit breakers, slippage, regime-filtered signals
- Visualisation: Stephen Few theme, risk/dependency/regime/portfolio charts
- Data generation: GBM, correlated GBM, Merton jump diffusion, Ornstein-Uhlenbeck
- Instruments: Black-Scholes, bonds, exotic options

---

## v0.3: Infrastructure

The plumbing that makes everything else usable. Fetch data from anywhere, generate reports, interact with charts.

### Phase 4: Data Connectors (`quantlite.data`)

Unified `fetch()` interface across all sources. One line to get clean OHLCV data.

```python
from quantlite.data import fetch
data = fetch(["AAPL", "BTC-USD", "GLD"], period="5y")
```

- **yfinance:** equities, ETFs, FX, futures, indices (free, no key)
- **CCXT:** 100+ crypto exchanges, OHLCV + order book depth (free for public data)
- **FRED:** macro data, rates, inflation, employment (free, key required)
- **CSV/Parquet:** local file loader with auto-detection and schema validation
- **Plugin architecture:** register custom data sources without touching core code
- Common `DataResult` return type (pandas DataFrame, OHLCV + metadata)
- Local disk caching with configurable TTL
- Optional install groups: `pip install quantlite[crypto]`, `quantlite[all]`

### Phase 5: Interactive Visualisation (Plotly Backend)

Plotly backend for every existing viz function, enforcing identical Stephen Few design language.

- Strict Plotly theme: strip all default chrome (toolbar, animations, hover effects)
- Same muted palette, same gridline rules, same direct labels as matplotlib
- `backend="plotly"` flag on every viz function (default remains matplotlib)
- Jupyter-native: interactive charts render inline without extra setup
- Consistent API: switching backends changes rendering, never the function signature

### Phase 6: Tearsheet Engine (`quantlite.report`)

One function call to a comprehensive, beautiful, multi-page report.

```python
from quantlite import report
report.tearsheet(backtest, save="portfolio_report.html")
```

- HTML output with interactive Plotly charts (Few-compliant)
- PDF output with static matplotlib charts (Few-compliant)
- Sections: executive summary, risk metrics, drawdown analysis, factor exposure, regime analysis, stress test results, monthly returns, rolling statistics
- Customisable: choose sections, add commentary, brand with logo

---

## v0.4: The Taleb Stack

Operationalise the ideas from Incerto and ergodicity economics. These exist in books. They do not exist as code. We build them first.

### Phase 7: Ergodicity Economics (`quantlite.ergodicity`)

The biggest gap in all of quantitative finance software.

```python
from quantlite.ergodicity import time_average, ensemble_average, ergodicity_gap

time_avg = time_average(returns)     # what you actually experience
ens_avg = ensemble_average(returns)  # what the textbook says
gap = ergodicity_gap(returns)        # if large, the strategy is dangerous
```

- Time average vs ensemble average computation
- Ergodicity gap measurement and visualisation
- Kelly criterion integration (optimal growth rate, not expected return)
- Geometric mean dominance testing
- Leverage effect on ergodicity (why 2x leverage can have negative time-average growth)

### Phase 8: Antifragility Framework (`quantlite.antifragile`)

Quantify antifragility. Not as a metaphor, as a number.

- **Antifragility score:** does the entity gain more from positive shocks than it loses from negative ones?
- **Convexity scoring:** payoff curvature measurement across the full distribution
- **Fourth quadrant detection:** "you are in a domain where your model's errors are consequential"
- **Non-naive barbell:** optimal allocation between hyperconservative and hyperaggressive
- **Lindy effect estimation:** survival probability based on current age
- **Skin in the game metrics:** payoff asymmetry, principal-agent alignment scoring

### Phase 9: Composable Scenario Engine (`quantlite.scenarios`)

Build stress tests like sentences.

```python
scenario = Scenario("China crisis") \
    .shock("CNY", -0.15) \
    .shock("BTC", -0.40) \
    .correlations(spike_to=0.85) \
    .duration(days=30)
result = portfolio.stress_test(weights, scenario)
```

- Named stress library: pre-built scenarios ("2008 GFC", "2020 COVID", "2022 Luna/FTX", "USDT depeg", "rates +200bps")
- Composable shocks: chain multiple effects
- Correlation regime overrides during stress
- Fragility heatmap: which positions are fragile, robust, or antifragile to each scenario
- Shock propagation: how does a shock in one asset cascade through the portfolio?

---

## v0.5: Honest Backtesting (Lopez de Prado)

Most backtesting libraries help you fool yourself. We do the opposite. Every metric comes with a confidence interval and an honesty check.

### Phase 10: Strategy Forensics (`quantlite.forensics`)

"Is this strategy actually good, or did you get lucky?"

- **Deflated Sharpe Ratio:** adjusts for number of trials, skewness, kurtosis
- **Probabilistic Sharpe Ratio:** probability that true Sharpe exceeds a benchmark
- **Haircut Sharpe Ratio:** adjusted for non-normality of returns
- **Minimum Track Record Length:** "this manager needs 4.3 more years before their alpha is statistically significant"
- **Signal decay analysis:** half-life of alpha, decay curve visualisation

### Phase 11: Overfitting Detection (`quantlite.overfit`)

The antidote to data mining.

- **Trial Tracker:** built-in logging of every backtest trial, not just the winner. CSCV is only honest if it knows about ALL trials. If you run a backtest outside the tracker, it warns you that your overfitting estimate is a lower bound.
  ```python
  with TrialTracker("momentum_strategy") as tracker:
      for lookback in [20, 40, 60, 120, 252]:
          for threshold in [0.5, 1.0, 1.5, 2.0]:
              result = run_backtest(data, strategy(lookback, threshold))
              tracker.log(params={...}, result=result)
      tracker.overfitting_probability()  # knows about ALL 20 trials
  ```
- **Probability of Backtest Overfitting (PBO):** CSCV method from Lopez de Prado, integrated with Trial Tracker
- **Multiple testing correction:** Bonferroni, Benjamini-Hochberg-Yekutieli
- **Minimum Backtest Length:** required data to trust a given Sharpe at a given confidence
- **Walk-forward validation:** built in, not optional. Rolling and expanding window.
- **Combinatorially purged cross-validation:** prevent leakage in time series

### Phase 12: Resampled Backtesting (`quantlite.resample`)

Confidence intervals on everything.

- Block bootstrap for time series (preserves autocorrelation)
- Stationary bootstrap (random block lengths)
- Confidence intervals on Sharpe, max drawdown, Calmar, and every other metric
- Distribution of outcomes, not point estimates

---

## v0.6: Systemic Risk and Contagion

Post-2008 tools that regulators actually use. No lightweight Python library provides them.

### Phase 13: Contagion Metrics (`quantlite.contagion`)

How does distress propagate?

- **CoVaR:** VaR of asset B conditional on asset A being in distress
- **Delta CoVaR:** marginal contribution to systemic risk
- **Marginal Expected Shortfall (MES):** each asset's contribution to system-wide tail risk
- **Granger causality chains:** does a BTC crash cause ETH to crash, or vice versa? Map the causal structure.

### Phase 14: Network Risk (`quantlite.network`)

The financial system as a graph.

- Counterparty graph construction from correlation or exposure data
- Eigenvector centrality: which nodes are systemically important?
- Cascade simulation: remove a node, measure the damage
- Community detection: identify clusters of interconnected risk
- Visualisation: network graphs with Few-compliant styling

### Phase 15: Concentration and Diversification (`quantlite.diversification`)

Beyond "how many assets do I hold."

- **Effective Number of Bets:** eigenvalue-based true diversification count
- **Entropy diversification:** information-theoretic measure
- **Tail diversification:** are you diversified when it matters (in crashes), or only in calm?
- **Marginal tail risk contribution:** which position hurts most in the left tail?
- **Herfindahl concentration index** with portfolio-aware adjustments

---

## v0.7: Crypto-Native Risk

Purpose-built risk tools for digital assets, sitting on top of the fat-tail core.

### Phase 16: Stablecoin Risk (`quantlite.crypto.stablecoin`)

- Depeg probability modelling (threshold breach, recovery time estimation)
- Peg deviation tracking and alerting
- Historical depeg database (UST, USDC March 2023, etc.)
- Reserve composition risk scoring

### Phase 17: Exchange and Counterparty Risk (`quantlite.crypto.exchange`)

- Exchange concentration scoring
- Hot wallet vs cold wallet risk assessment
- Proof-of-reserves analysis helpers
- Liquidity risk: estimated unwind time given order book depth
- Slippage estimation for large orders

### Phase 18: On-Chain Integration (`quantlite.crypto.onchain`)

- Wallet exposure tracking via public APIs (Etherscan, block explorers)
- TVL tracking via DefiLlama
- DeFi contagion modelling (protocol dependency graphs)
- Smart contract risk scoring (age, audit status, TVL stability)

---

## v0.8: Factor Models

Decompose returns into systematic and idiosyncratic components.

### Phase 19: Classical Factors (`quantlite.factors`)

- Fama-French 3-factor and 5-factor models with auto-downloaded factor data
- Carhart 4-factor (adds momentum)
- Factor attribution: "your alpha is 2.3% after adjusting for market, size, value, and momentum"

### Phase 20: Custom Factors

- Define custom factors from any time series
- Test factor significance with proper statistical tests
- Factor portfolio construction

### Phase 21: Fat-Tail Factor Risk

- Factor contribution to tail risk (CVaR decomposition by factor), not just variance
- Regime-conditional factor exposures: "in crisis, your value tilt becomes a liability"
- Factor crowding detection

---

## v0.9: Fat-Tail Monte Carlo

Replace Gaussian assumptions in simulation with the library's own fat-tail and copula machinery.

### Phase 22: EVT-Based Tail Simulation

- Use fitted GPD for extreme scenario generation instead of normal distribution
- Tail-aware confidence intervals on portfolio outcomes

### Phase 23: Copula-Dependent Paths

- Correlated multi-asset simulation preserving tail dependence
- Student-t and Clayton copulas for realistic joint crash modelling

### Phase 24: Regime-Switching Simulation

- Parameters shift based on HMM state transitions
- Scenario-conditional simulation: "simulate 10,000 paths but condition on starting in a crisis regime"
- Regime-aware probability cones for portfolio projections

---

## v1.0: Regime-Aware Everything + Production Ready

Connect regime detection through the entire library, end to end. Ship it.

### Phase 25: Regime-Aware Integration

- Regime-conditional risk metrics: "your 95% VaR is -2.1% in calm, -6.8% in crisis"
- Regime-aware portfolio construction: automatic defensive tilt when HMM signals crisis
- Regime-filtered backtesting: per-regime performance attribution
- Regime-conditional reporting: tearsheets that break down everything by detected regime

### Phase 26: Production Polish

- 90%+ test coverage
- mypy strict compliance
- Full documentation site (mkdocs or Sphinx) with search
- CI: auto-publish on tag, nightly test matrix, coverage badge
- Contribution guide and architecture docs

### Phase 27: The Dream API

```python
from quantlite import fetch, portfolio, risk, report

data = fetch(["AAPL", "BTC-USD", "GLD", "TLT"], period="5y")
regimes = data.detect_regimes()
weights = portfolio.hrp(data, regime_aware=True)
backtest = weights.backtest(data, rebalance="monthly")
report.tearsheet(backtest, save="my_portfolio.html")
```

Ten lines. Fetch, analyse, allocate, test, report. Fat-tail native throughout. Every chart looks like Stephen Few designed it. Every backtest metric comes with an honesty check. Every risk number respects ergodicity.

---

## The Narrative

| Version | Theme | What Nobody Else Has |
|---------|-------|---------------------|
| v0.2 | Foundation | Fat-tail-native risk, EVT, copulas, regimes |
| v0.3 | Infrastructure | Data connectors, interactive viz, tearsheets |
| v0.4 | Taleb | Ergodicity economics, antifragility as code |
| v0.5 | Lopez de Prado | Honest backtesting, overfitting detection |
| v0.6 | Systemic risk | CoVaR, contagion, network risk |
| v0.7 | Crypto-native | Stablecoin risk, exchange risk, on-chain |
| v0.8 | Factors | Classical + custom + fat-tail factor risk |
| v0.9 | Simulation | EVT/copula Monte Carlo, regime-switching |
| v1.0 | Everything connected | Regime-aware end to end, production ready |

Each version is independently useful. Each adds a layer that no other lightweight library has. By v1.0, QuantLite is the only library that operationalises Taleb, Peters, and Lopez de Prado in one place.

---

## What QuantLite Will Never Be

- A GUI or desktop application
- A brokerage integration or live trading system
- A real-time streaming platform
- An ML/AI framework (users can build on top of our features)
- A Bloomberg terminal replacement

We are a library. We do one thing: make quantitative finance honest about risk. Everything else is someone else's job.
