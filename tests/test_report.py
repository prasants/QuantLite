"""Tests for the tearsheet engine (quantlite.report)."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from quantlite.report import (
    drawdown_section,
    executive_summary_section,
    factor_section,
    monthly_returns_section,
    regime_section,
    risk_section,
    rolling_section,
    stress_section,
    tearsheet,
)


@dataclass
class _MockBacktestResult:
    """Minimal backtest result for testing."""

    portfolio_value: pd.Series
    regime_labels: np.ndarray | None = None


def _make_portfolio(n: int = 504, seed: int = 42) -> pd.Series:
    """Create a synthetic portfolio value series with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.01, n)
    values = 10000 * np.cumprod(1 + returns)
    dates = pd.bdate_range("2023-01-03", periods=n)
    return pd.Series(values, index=dates, name="portfolio")


def _make_result(
    n: int = 504,
    with_regimes: bool = False,
) -> _MockBacktestResult:
    """Create a mock backtest result."""
    pv = _make_portfolio(n)
    labels = None
    if with_regimes:
        rng = np.random.default_rng(99)
        labels = rng.choice([0, 1, 2], size=n - 1)
    return _MockBacktestResult(portfolio_value=pv, regime_labels=labels)


class TestTearsheetHTML:
    """Tests for HTML tearsheet generation."""

    def test_generates_html_file(self) -> None:
        """Tearsheet should create an HTML file."""
        result = _make_result()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            tearsheet(result, save=path)
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            assert "<!DOCTYPE html>" in content
            assert "Executive Summary" in content
        finally:
            os.unlink(path)

    def test_custom_sections(self) -> None:
        """Tearsheet should only include requested sections."""
        result = _make_result()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            html = tearsheet(
                result,
                save=path,
                sections=["summary", "drawdown"],
            )
            assert "Executive Summary" in html
            assert "Drawdown" in html
            assert "Rolling" not in html
        finally:
            os.unlink(path)

    def test_custom_title_and_commentary(self) -> None:
        """Title and commentary should appear in output."""
        result = _make_result()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            html = tearsheet(
                result,
                save=path,
                title="Test Strategy",
                commentary="This is a test.",
                sections=["summary"],
            )
            assert "Test Strategy" in html
            assert "This is a test." in html
        finally:
            os.unlink(path)

    def test_unknown_section_raises(self) -> None:
        """Unknown section names should raise ValueError."""
        result = _make_result()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unknown sections"):
                tearsheet(result, save=path, sections=["nonexistent"])
        finally:
            os.unlink(path)

    def test_with_regime_labels(self) -> None:
        """Regime section should show per-regime metrics when labels provided."""
        result = _make_result(with_regimes=True)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            html = tearsheet(result, save=path, sections=["regimes"])
            assert "Regime 0" in html
        finally:
            os.unlink(path)

    def test_without_regime_labels(self) -> None:
        """Regime section should show a message when no labels provided."""
        result = _make_result(with_regimes=False)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            html = tearsheet(result, save=path, sections=["regimes"])
            assert "No regime labels provided" in html
        finally:
            os.unlink(path)


class TestTearsheetPDF:
    """Tests for PDF tearsheet generation."""

    def test_generates_pdf_file(self) -> None:
        """Tearsheet should create a PDF file (fallback if no weasyprint)."""
        result = _make_result()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            tearsheet(result, save=path, sections=["summary"])
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


class TestIndividualSections:
    """Tests for individual section functions."""

    def test_executive_summary_returns_html(self) -> None:
        """Executive summary should return non-empty HTML."""
        pv = _make_portfolio()
        html = executive_summary_section(pv)
        assert len(html) > 0
        assert "Total Return" in html

    def test_risk_section_returns_html(self) -> None:
        """Risk section should return non-empty HTML."""
        pv = _make_portfolio()
        html = risk_section(pv)
        assert len(html) > 0
        assert "VaR" in html

    def test_drawdown_section_returns_html(self) -> None:
        """Drawdown section should return non-empty HTML."""
        pv = _make_portfolio()
        html = drawdown_section(pv)
        assert len(html) > 0
        assert "Drawdown" in html

    def test_monthly_returns_section_returns_html(self) -> None:
        """Monthly returns section should return non-empty HTML."""
        pv = _make_portfolio()
        html = monthly_returns_section(pv)
        assert len(html) > 0
        assert "Monthly" in html

    def test_rolling_section_returns_html(self) -> None:
        """Rolling section should return non-empty HTML."""
        pv = _make_portfolio()
        html = rolling_section(pv)
        assert len(html) > 0
        assert "Rolling" in html

    def test_regime_section_with_labels(self) -> None:
        """Regime section with labels should show regime metrics."""
        pv = _make_portfolio()
        labels = np.random.default_rng(0).choice([0, 1], size=len(pv) - 1)
        html = regime_section(pv, regime_labels=labels)
        assert "Regime 0" in html

    def test_regime_section_without_labels(self) -> None:
        """Regime section without labels should show placeholder."""
        pv = _make_portfolio()
        html = regime_section(pv, regime_labels=None)
        assert "No regime labels" in html

    def test_factor_section_placeholder(self) -> None:
        """Factor section should show v0.8 placeholder."""
        html = factor_section()
        assert "v0.8" in html

    def test_stress_section_placeholder(self) -> None:
        """Stress section should show v0.6 placeholder."""
        html = stress_section()
        assert "v0.6" in html


class TestEdgeCases:
    """Tests with minimal or edge-case data."""

    def test_minimal_data(self) -> None:
        """Tearsheet should handle very short time series."""
        pv = pd.Series([100.0, 101.0, 99.0])
        result = _MockBacktestResult(portfolio_value=pv)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            html = tearsheet(result, save=path, sections=["summary"])
            assert "Executive Summary" in html
        finally:
            os.unlink(path)

    def test_single_value(self) -> None:
        """Sections should handle single-value series gracefully."""
        pv = pd.Series([100.0])
        html = executive_summary_section(pv)
        assert "Insufficient data" in html

    def test_no_datetime_index_monthly(self) -> None:
        """Monthly returns should handle non-datetime index."""
        pv = pd.Series(np.linspace(100, 110, 100))
        html = monthly_returns_section(pv)
        assert "DatetimeIndex required" in html
