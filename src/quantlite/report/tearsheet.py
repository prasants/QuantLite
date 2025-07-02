"""Main tearsheet generator that assembles report sections.

Orchestrates section generation and rendering into HTML or PDF output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .html_renderer import render_html
from .pdf_renderer import render_pdf
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

__all__ = ["tearsheet"]

_SECTION_MAP: dict[str, str] = {
    "summary": "summary",
    "risk": "risk",
    "drawdown": "drawdown",
    "monthly": "monthly",
    "rolling": "rolling",
    "regimes": "regimes",
    "factors": "factors",
    "stress": "stress",
}

_DEFAULT_SECTIONS = [
    "summary", "risk", "drawdown", "monthly",
    "rolling", "regimes", "factors", "stress",
]


def tearsheet(
    backtest_result: Any,
    save: str = "tearsheet.html",
    sections: list[str] | None = None,
    title: str = "Portfolio Tearsheet",
    commentary: str | None = None,
) -> str:
    """Generate a tearsheet report from a backtest result.

    Assembles selected report sections into a complete HTML or PDF
    document. Output format is determined by the file extension in
    ``save``.

    Args:
        backtest_result: A BacktestResult object with ``portfolio_value``
            and optionally ``regime_labels`` attributes.
        save: Output file path. Use ``.html`` for HTML or ``.pdf`` for PDF.
        sections: List of section names to include. Defaults to all sections.
        title: Report title displayed at the top.
        commentary: Optional commentary paragraph.

    Returns:
        The HTML content as a string (even when saving as PDF).

    Raises:
        ValueError: If an unknown section name is provided.
    """
    portfolio_value: pd.Series = backtest_result.portfolio_value
    regime_labels: np.ndarray | None = getattr(
        backtest_result, "regime_labels", None
    )

    chosen = sections if sections is not None else _DEFAULT_SECTIONS

    unknown = set(chosen) - set(_SECTION_MAP)
    if unknown:
        raise ValueError(f"Unknown sections: {unknown}")

    section_html: list[str] = []
    for name in chosen:
        if name == "summary":
            section_html.append(executive_summary_section(portfolio_value))
        elif name == "risk":
            section_html.append(risk_section(portfolio_value))
        elif name == "drawdown":
            section_html.append(drawdown_section(portfolio_value))
        elif name == "monthly":
            section_html.append(monthly_returns_section(portfolio_value))
        elif name == "rolling":
            section_html.append(rolling_section(portfolio_value))
        elif name == "regimes":
            section_html.append(
                regime_section(portfolio_value, regime_labels=regime_labels)
            )
        elif name == "factors":
            section_html.append(factor_section())
        elif name == "stress":
            section_html.append(stress_section())

    html_content = render_html(
        section_html, title=title, commentary=commentary
    )

    if save.lower().endswith(".pdf"):
        render_pdf(html_content, save)
    else:
        with open(save, "w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content
