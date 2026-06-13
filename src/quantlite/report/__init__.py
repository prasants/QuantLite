"""Tearsheet engine for generating professional portfolio reports.

Provides HTML and PDF report generation from backtest results,
with interactive Plotly charts (or matplotlib fallback) and
Stephen Few-inspired visual design.
"""

from __future__ import annotations

from .sections import (
    drawdown_section,
    executive_summary_section,
    factor_section,
    monthly_returns_section,
    regime_section,
    risk_section,
    rolling_section,
    stress_section,
)
from .tearsheet import tearsheet

__all__ = [
    "tearsheet",
    "executive_summary_section",
    "risk_section",
    "drawdown_section",
    "monthly_returns_section",
    "rolling_section",
    "regime_section",
    "factor_section",
    "stress_section",
]
