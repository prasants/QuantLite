"""Plotly backend for QuantLite visualisation.

Interactive equivalents of all matplotlib charts, following Stephen Few's
design principles: maximum data-ink ratio, muted palette, no chartjunk.
"""

from .dependency import (
    plot_copula_contour,
    plot_correlation_dynamics,
    plot_correlation_matrix,
    plot_stress_correlation,
)
from .portfolio import (
    plot_efficient_frontier,
    plot_hrp_dendrogram,
    plot_monthly_returns,
    plot_weight_comparison,
)
from .regimes import (
    plot_changepoints,
    plot_regime_distributions,
    plot_regime_timeline,
    plot_transition_matrix,
)
from .risk import (
    plot_drawdown,
    plot_gpd_fit,
    plot_hill,
    plot_qq,
    plot_return_distribution,
    plot_return_level,
    plot_risk_bullet,
    plot_var_comparison,
)
from .theme import (
    FEW_COLORWAY,
    FEW_PALETTE,
    FEW_TEMPLATE,
    apply_few_theme_plotly,
    few_figure,
)

__all__ = [
    # Theme
    "FEW_PALETTE",
    "FEW_COLORWAY",
    "FEW_TEMPLATE",
    "apply_few_theme_plotly",
    "few_figure",
    # Risk
    "plot_var_comparison",
    "plot_return_distribution",
    "plot_drawdown",
    "plot_qq",
    "plot_gpd_fit",
    "plot_return_level",
    "plot_hill",
    "plot_risk_bullet",
    # Dependency
    "plot_copula_contour",
    "plot_correlation_matrix",
    "plot_correlation_dynamics",
    "plot_stress_correlation",
    # Regimes
    "plot_regime_timeline",
    "plot_transition_matrix",
    "plot_regime_distributions",
    "plot_changepoints",
    # Portfolio
    "plot_efficient_frontier",
    "plot_weight_comparison",
    "plot_monthly_returns",
    "plot_hrp_dendrogram",
]
