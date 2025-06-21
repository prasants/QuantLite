# Interactive Visualisation with Plotly

QuantLite includes a full Plotly backend that mirrors every matplotlib chart
with interactive equivalents. All charts follow Stephen Few's design
principles: maximum data-ink ratio, muted palette, no chartjunk.

## Installation

```bash
pip install quantlite[plotly]
```

This installs `plotly>=5.0` and `kaleido>=0.2` (for static image export).

## Quick Start

There are two ways to use the Plotly backend:

### Option 1: Import directly from the Plotly backend

```python
from quantlite.viz.plotly_backend.risk import plot_var_comparison

fig = plot_var_comparison(returns)
fig.show()
```

### Option 2: Use the `backend` parameter on existing functions

```python
from quantlite.viz.risk import plot_drawdown

fig = plot_drawdown(returns, backend="plotly")
fig.show()
```

When `backend="matplotlib"` (the default), the original matplotlib figure is
returned. When `backend="plotly"`, a `plotly.graph_objects.Figure` is returned
instead.

## Theme

The Plotly backend applies the same Stephen Few theme as the matplotlib
backend:

- **Colours**: primary `#4E79A7`, secondary `#F28E2B`, negative `#E15759`,
  positive `#59A14F`, neutral `#76B7B2`
- **Background**: white (`#FFFFFF`)
- **Gridlines**: horizontal only, light grey (`#E8E8E8`)
- **No top/right axes** (no mirror)
- **No Plotly chrome**: toolbar, animations, and bouncy transitions are stripped

To apply the theme globally:

```python
from quantlite.viz.plotly_backend import apply_few_theme_plotly
apply_few_theme_plotly()
```

To customise the template:

```python
from quantlite.viz.plotly_backend import FEW_TEMPLATE
import plotly.io as pio

# Modify and register
custom = FEW_TEMPLATE
custom.layout.font.size = 14
pio.templates["few_custom"] = custom
pio.templates.default = "few_custom"
```

## Chart Reference

### Risk Charts

#### `plot_var_comparison(returns, alpha=0.05, methods=None)`
Grouped bar chart comparing VaR and CVaR across estimation methods
(historical, parametric, Cornish-Fisher). Each bar is directly labelled
with the loss value. VaR bars in primary blue, CVaR in negative red.

#### `plot_return_distribution(returns, gpd_fit=None, bins=80)`
Histogram of returns with a dashed normal overlay in grey. If a GPD fit
is provided, the tail fit is overlaid in primary blue. VaR and CVaR
reference lines are annotated directly on the chart.

#### `plot_drawdown(returns)`
Filled area chart showing the drawdown (underwater) profile. The maximum
drawdown is annotated with an arrow pointing to the trough, showing both
the magnitude and duration.

#### `plot_qq(returns, dist="norm")`
QQ plot with sample quantiles on the y-axis and theoretical quantiles on
the x-axis. Points in primary blue, reference line dashed in red. Fat
tails appear as deviations from the line at the extremes.

#### `plot_gpd_fit(returns, gpd_fit)`
Log-scale survival function plot comparing the empirical tail against the
fitted GPD curve. Empirical points in grey, GPD fit line in primary blue.

#### `plot_return_level(gpd_fit, max_period=10000)`
Return level curve on a log-scaled x-axis with 95% confidence bands in
translucent blue. Key return levels (1-in-100, 1-in-1000, 1-in-5000) are
marked with red dots and direct labels.

#### `plot_hill(returns, max_k=None)`
Hill plot showing the tail index estimate as a function of the number of
order statistics. A stable plateau indicates a reliable estimate.

#### `plot_risk_bullet(metrics)`
Bullet graphs for risk metrics (Sortino, Calmar, Omega, etc.). Each metric
shows a value bar, a target marker, and background ranges for poor,
satisfactory, and good performance.

### Dependency Charts

#### `plot_copula_contour(copula, data)`
Bivariate copula contour plot in uniform margin space. The fitted copula
contours are in primary blue, with a Gaussian copula reference in grey.
Lower and upper tail dependence coefficients are annotated directly.

#### `plot_correlation_matrix(corr_matrix)`
Interactive heatmap with a blue-white-red diverging colourscale. Each cell
displays its value. Hover reveals the exact correlation. Axes show asset
names.

#### `plot_correlation_dynamics(x, y, window=60)`
Rolling correlation line chart with the overall correlation shown as a
dotted reference line. Useful for spotting regime changes in co-movement.

#### `plot_stress_correlation(calm_corr, stress_corr)`
Side-by-side heatmaps on the same colour scale, showing how correlations
shift between calm and stress periods.

### Regime Charts

#### `plot_regime_timeline(returns, regimes, changepoints=None)`
Cumulative return line chart with coloured background bands indicating the
active regime. Changepoints shown as dotted vertical lines.

#### `plot_transition_matrix(model)`
Annotated heatmap of transition probabilities. Colour intensity is
proportional to probability (white to primary blue).

#### `plot_regime_distributions(returns, regimes, bins=40)`
Small multiples (one per regime) showing return distributions. Each subplot
is annotated with mean, volatility, and sample count.

#### `plot_changepoints(returns, changepoints)`
Cumulative return line with detected changepoints marked as dashed red
vertical lines, each labelled with its index.

### Portfolio Charts

#### `plot_efficient_frontier(returns_df, n_portfolios=2000)`
Random portfolio scatter coloured by Sharpe ratio (grey to blue gradient).
Hover reveals portfolio weights. Min variance (green circle) and max Sharpe
(orange diamond) points are directly labelled.

#### `plot_weight_comparison(weights_dict)`
Grouped bar chart comparing portfolio weights across methods. Each bar is
labelled with the weight percentage.

#### `plot_monthly_returns(monthly_table)`
Diverging heatmap (red-white-blue) of monthly returns. Each cell shows the
return percentage. Hover provides year, month, and exact value.

#### `plot_hrp_dendrogram(returns_df, method="ward")`
Hierarchical clustering dendrogram showing the asset distance structure
used by HRP. All links rendered in primary blue.

## Jupyter Integration

Plotly figures render natively in Jupyter notebooks. Simply call `fig.show()`
or let the notebook auto-display the figure as the last expression in a cell.

For static export (e.g. in reports):

```python
fig.write_image("chart.png", scale=2)  # requires kaleido
fig.write_html("chart.html")           # standalone HTML
```

## Comparison: Matplotlib vs Plotly

| Feature | Matplotlib | Plotly |
|---------|-----------|--------|
| Output | Static image | Interactive HTML |
| Hover info | No | Yes |
| Zoom/pan | No | Yes (if enabled) |
| Export | PNG, PDF, SVG | PNG, HTML, PDF |
| Jupyter | Inline image | Native widget |
| Size | Lightweight | Heavier (~15 MB) |

Both backends apply the same Stephen Few theme and colour palette. Choose
matplotlib for publications and reports; choose Plotly for exploratory
analysis and dashboards.
