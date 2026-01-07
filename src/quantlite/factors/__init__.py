"""Factor models: classical, custom, and tail risk factor analysis.

Provides tools for multi-factor attribution, custom factor construction,
factor significance testing, and tail-risk-aware factor decomposition.
"""

from quantlite.factors.classical import (
    carhart_four,
    factor_attribution,
    factor_summary,
    fama_french_five,
    fama_french_three,
)
from quantlite.factors.custom import (
    CustomFactor,
    factor_correlation_matrix,
    factor_decay,
    factor_portfolio,
    test_factor_significance,
)
from quantlite.factors.tail_risk import (
    factor_crowding_score,
    factor_cvar_decomposition,
    regime_factor_exposure,
    tail_factor_beta,
)

__all__ = [
    "fama_french_three",
    "fama_french_five",
    "carhart_four",
    "factor_attribution",
    "factor_summary",
    "CustomFactor",
    "test_factor_significance",
    "factor_correlation_matrix",
    "factor_portfolio",
    "factor_decay",
    "factor_cvar_decomposition",
    "regime_factor_exposure",
    "factor_crowding_score",
    "tail_factor_beta",
]
