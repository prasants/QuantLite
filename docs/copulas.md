# Copulas and Dependency Analysis

## Copulas (`quantlite.dependency.copulas`)

Copulas separate the marginal behaviour of individual assets from their dependence structure. This is critical in risk management because linear correlation captures only one aspect of dependence and completely misses tail dependence, the tendency for assets to crash together.

### Five Copula Families Compared

All five copulas fitted to the same bivariate return data. Notice how Clayton captures lower-tail clustering (joint crashes), Gumbel captures upper-tail clustering, and the Student-t captures both:

![Copula Contours](images/copula_contours.png)

### Tail Dependence by Family

A direct comparison of lower and upper tail dependence coefficients across copula families. Only Clayton, Student-t, and Gumbel exhibit non-zero tail dependence:

![Tail Dependence](images/tail_dependence_comparison.png)

### Simulated Dependency Structures

Scatter plots from three copulas show how different families produce different joint extreme behaviour. Red points highlight joint crashes, green points highlight joint rallies:

![Copula Scatter](images/copula_scatter.png)

### Stress vs Calm Correlation

Correlation matrices estimated separately during calm and stress periods. Notice how correlations spike during crises, reducing diversification when it is most needed:

![Stress vs Calm](images/stress_vs_calm_correlation.png)

### EWMA vs Rolling Correlation

EWMA correlation reacts faster to regime changes than simple rolling windows, making it more useful for real-time risk monitoring:

![EWMA Correlation](images/ewma_correlation.png)

### Available Copula Families

| Copula | Parameters | Lower Tail Dep | Upper Tail Dep | Best For |
|--------|-----------|----------------|----------------|----------|
| Gaussian | rho | 0 | 0 | Baseline, no tail dependence |
| Student-t | rho, nu | Symmetric | Symmetric | Joint extremes, crisis modelling |
| Clayton | theta | Yes (2^{-1/theta}) | 0 | Joint crashes |
| Gumbel | theta | 0 | Yes (2 - 2^{1/theta}) | Joint booms |
| Frank | theta | 0 | 0 | Symmetric interior dependence |

### Fitting and Model Selection

```python
from quantlite.dependency.copulas import (
    GaussianCopula, StudentTCopula, ClaytonCopula,
    GumbelCopula, FrankCopula, select_best_copula,
)
import numpy as np

# Simulate bivariate returns with tail dependence
rng = np.random.default_rng(42)
# Use a Student-t distribution to create tail dependence
from scipy.stats import multivariate_t
data = multivariate_t.rvs(
    loc=[0, 0], shape=[[1, 0.6], [0.6, 1]], df=4, size=2000, random_state=42,
) * 0.015

# Fit each copula
for CopulaClass in [GaussianCopula, StudentTCopula, ClaytonCopula, GumbelCopula, FrankCopula]:
    cop = CopulaClass()
    cop.fit(data)
    ll = cop.log_likelihood(data)
    td = cop.tail_dependence()
    print(f"{cop!r}")
    print(f"  Log-lik: {ll:.1f}, AIC: {cop.aic(data):.1f}")
    print(f"  Tail dep: lower={td['lower']:.3f}, upper={td['upper']:.3f}")
    print()

# Automatic selection
best = select_best_copula(data)
print(f"Best copula: {best.name} (AIC={best.aic:.1f}, BIC={best.bic:.1f})")
```

### Simulation from Fitted Copulas

```python
cop = StudentTCopula()
cop.fit(data)

# Simulate 5000 observations from the fitted copula
simulated = cop.simulate(5000, rng_seed=42)
print(f"Shape: {simulated.shape}")  # (5000, 2) with uniform marginals

# Transform back to original scale using inverse CDF of your marginals
```

### Why Tail Dependence Matters

The Gaussian copula has zero tail dependence for any correlation < 1. This means that even with rho = 0.9, the Gaussian copula predicts that joint extreme events are vanishingly rare. The 2008 financial crisis demonstrated that this assumption is catastrophically wrong.

The Student-t copula, by contrast, has symmetric tail dependence that increases with lower degrees of freedom. A Student-t copula with nu = 4 and rho = 0.6 predicts meaningful probability of simultaneous crashes, which aligns with empirical observation.

```python
cop_gauss = GaussianCopula()
cop_gauss.fit(data)
cop_t = StudentTCopula()
cop_t.fit(data)

print(f"Gaussian tail dependence: {cop_gauss.lower_tail_dependence():.3f}")
print(f"Student-t tail dependence: {cop_t.lower_tail_dependence():.3f}")
```

## Correlation Analysis (`quantlite.dependency.correlation`)

### Rolling and EWMA Correlation

```python
from quantlite.dependency.correlation import (
    rolling_correlation, exponential_weighted_correlation,
)
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
n = 756
equities = pd.Series(rng.normal(0.0003, 0.012, n))
bonds = pd.Series(rng.normal(0.0001, 0.005, n))

# Rolling Pearson correlation (60-day window)
roll_corr = rolling_correlation(equities, bonds, window=60)

# EWMA correlation (half-life of 30 days, more responsive)
ewma_corr = exponential_weighted_correlation(equities, bonds, halflife=30)
```

### Stress Correlation

The well-documented phenomenon of correlation increasing during market drawdowns:

```python
from quantlite.dependency.correlation import stress_correlation, correlation_breakdown_test

returns_df = pd.DataFrame({
    "Equities": rng.normal(0.0003, 0.012, n),
    "Bonds": rng.normal(0.0001, 0.005, n),
    "Commodities": rng.normal(0.0002, 0.015, n),
})

# Correlation during stress periods (bottom 10th percentile)
stress_corr = stress_correlation(returns_df, threshold_percentile=10)
print("Stress correlation matrix:")
print(stress_corr)

# Statistical test for correlation breakdown
test = correlation_breakdown_test(returns_df, threshold_percentile=25)
print(f"Calm correlation:   {test['calm_corr']:.3f}")
print(f"Stress correlation: {test['stress_corr']:.3f}")
print(f"Test statistic:     {test['test_statistic']:.2f}")
print(f"p-value:            {test['p_value']:.4f}")
```

### Rank Correlation

More robust than Pearson correlation for fat-tailed data:

```python
from quantlite.dependency.correlation import rank_correlation

spearman_corr, p_val = rank_correlation(equities, bonds, method="spearman")
kendall_corr, p_val = rank_correlation(equities, bonds, method="kendall")
print(f"Spearman: {spearman_corr:.3f}")
print(f"Kendall:  {kendall_corr:.3f}")
```

## Hierarchical Risk Parity (`quantlite.dependency.clustering`)

HRP (Lopez de Prado, 2016) builds portfolio weights using hierarchical clustering on the correlation matrix, avoiding the instability problems of covariance matrix inversion.

### The Algorithm

1. Compute the correlation distance matrix: d(i,j) = sqrt(0.5 * (1 - rho(i,j)))
2. Hierarchically cluster assets by distance
3. Quasi-diagonalise the covariance matrix (reorder by cluster leaves)
4. Recursively bisect, allocating weight inversely proportional to cluster variance

```python
from quantlite.dependency.clustering import (
    correlation_distance, hierarchical_cluster,
    quasi_diagonalise, hrp_weights,
)
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
returns_df = pd.DataFrame({
    "US_Large": rng.normal(0.0004, 0.012, 504),
    "US_Small": rng.normal(0.0003, 0.016, 504),
    "EU_Equity": rng.normal(0.0002, 0.014, 504),
    "EM_Equity": rng.normal(0.0003, 0.018, 504),
    "Govt_Bonds": rng.normal(0.0001, 0.004, 504),
    "Corp_Bonds": rng.normal(0.00015, 0.006, 504),
    "Gold": rng.normal(0.0001, 0.010, 504),
    "REITs": rng.normal(0.0002, 0.013, 504),
})

# Full HRP pipeline
weights = hrp_weights(returns_df)
print("HRP Weights:")
for asset, w in weights.items():
    print(f"  {asset}: {w:.2%}")

# Step-by-step for analysis
corr = returns_df.corr()
dist = correlation_distance(corr)
link = hierarchical_cluster(corr, method="single")
reordered = quasi_diagonalise(link, corr)
```

### Why HRP Over Markowitz?

Markowitz mean-variance optimisation requires inverting the covariance matrix, which is notoriously unstable for large portfolios. Small estimation errors in the covariance matrix lead to extreme, concentrated portfolios. HRP avoids inversion entirely, producing diversified weights that are more stable out of sample.
