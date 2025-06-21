"""Tests for the Plotly visualisation backend.

Validates that every chart function returns a plotly.graph_objects.Figure,
the Few theme is correctly applied, and backend delegation works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

go = pytest.importorskip("plotly.graph_objects", reason="plotly not installed")

from quantlite.viz.plotly_backend import (  # noqa: E402
    FEW_PALETTE,
    FEW_TEMPLATE,
    apply_few_theme_plotly,
    few_figure,
)
from quantlite.viz.plotly_backend import dependency as plotly_dep  # noqa: E402
from quantlite.viz.plotly_backend import portfolio as plotly_port  # noqa: E402
from quantlite.viz.plotly_backend import regimes as plotly_reg  # noqa: E402
from quantlite.viz.plotly_backend import risk as plotly_risk  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def returns() -> np.ndarray:
    """Synthetic return series with fat tails."""
    rng = np.random.default_rng(42)
    return rng.standard_t(df=4, size=500) * 0.01


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    """Multi-asset return DataFrame."""
    rng = np.random.default_rng(42)
    data = rng.standard_t(df=5, size=(500, 4)) * 0.01
    return pd.DataFrame(data, columns=["A", "B", "C", "D"])


@pytest.fixture()
def corr_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    return returns_df.corr()


@pytest.fixture()
def regimes_arr() -> np.ndarray:
    """Synthetic regime labels."""
    rng = np.random.default_rng(42)
    return rng.choice([0, 1, 2], size=500, p=[0.3, 0.5, 0.2])


# ---------------------------------------------------------------------------
# Theme tests
# ---------------------------------------------------------------------------

class TestTheme:
    """Theme registration and properties."""

    def test_few_template_is_template(self) -> None:
        assert isinstance(FEW_TEMPLATE, go.layout.Template)

    def test_apply_few_theme(self) -> None:
        apply_few_theme_plotly()
        import plotly.io as pio
        assert pio.templates.default == "few"

    def test_few_figure_returns_figure(self) -> None:
        fig = few_figure(title="Test")
        assert isinstance(fig, go.Figure)

    def test_few_figure_has_correct_bg(self) -> None:
        fig = few_figure()
        # bg colours live in the template
        tmpl = fig.layout.template.layout
        assert tmpl.plot_bgcolor == FEW_PALETTE["bg"]
        assert tmpl.paper_bgcolor == FEW_PALETTE["bg"]

    def test_few_figure_no_toolbar(self) -> None:
        fig = few_figure()
        assert fig.layout.dragmode is False

    def test_colorway(self) -> None:
        assert FEW_TEMPLATE.layout.colorway[0] == FEW_PALETTE["primary"]


# ---------------------------------------------------------------------------
# Risk chart tests
# ---------------------------------------------------------------------------

class TestRiskCharts:
    """Risk visualisation Plotly charts."""

    def test_var_comparison(self, returns: np.ndarray) -> None:
        fig = plotly_risk.plot_var_comparison(returns)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # VaR + CVaR bars

    def test_return_distribution(self, returns: np.ndarray) -> None:
        fig = plotly_risk.plot_return_distribution(returns)
        assert isinstance(fig, go.Figure)

    def test_drawdown(self, returns: np.ndarray) -> None:
        fig = plotly_risk.plot_drawdown(returns)
        assert isinstance(fig, go.Figure)

    def test_qq(self, returns: np.ndarray) -> None:
        fig = plotly_risk.plot_qq(returns)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # points + reference line

    def test_hill(self, returns: np.ndarray) -> None:
        fig = plotly_risk.plot_hill(returns)
        assert isinstance(fig, go.Figure)

    def test_risk_bullet(self) -> None:
        metrics = {
            "Sortino": {"value": 1.5, "target": 2.0, "ranges": [1.0, 2.0, 3.0]},
            "Calmar": {"value": 0.8, "target": 1.0, "ranges": [0.5, 1.0, 1.5]},
        }
        fig = plotly_risk.plot_risk_bullet(metrics)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Dependency chart tests
# ---------------------------------------------------------------------------

class TestDependencyCharts:
    """Dependency visualisation Plotly charts."""

    def test_correlation_matrix(self, corr_matrix: pd.DataFrame) -> None:
        fig = plotly_dep.plot_correlation_matrix(corr_matrix)
        assert isinstance(fig, go.Figure)

    def test_correlation_dynamics(self, returns_df: pd.DataFrame) -> None:
        fig = plotly_dep.plot_correlation_dynamics(
            returns_df["A"], returns_df["B"], window=30,
        )
        assert isinstance(fig, go.Figure)

    def test_stress_correlation(self, corr_matrix: pd.DataFrame) -> None:
        # Use same matrix for both calm and stress (just testing it runs)
        fig = plotly_dep.plot_stress_correlation(corr_matrix, corr_matrix)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Regime chart tests
# ---------------------------------------------------------------------------

class TestRegimeCharts:
    """Regime visualisation Plotly charts."""

    def test_regime_timeline(
        self, returns: np.ndarray, regimes_arr: np.ndarray,
    ) -> None:
        fig = plotly_reg.plot_regime_timeline(returns, regimes_arr)
        assert isinstance(fig, go.Figure)

    def test_regime_timeline_with_changepoints(
        self, returns: np.ndarray, regimes_arr: np.ndarray,
    ) -> None:
        fig = plotly_reg.plot_regime_timeline(
            returns, regimes_arr, changepoints=[50, 200, 350],
        )
        assert isinstance(fig, go.Figure)

    def test_transition_matrix(self) -> None:
        class MockModel:
            transition_matrix = np.array([[0.9, 0.1], [0.3, 0.7]])
            n_regimes = 2

        fig = plotly_reg.plot_transition_matrix(MockModel())
        assert isinstance(fig, go.Figure)

    def test_regime_distributions(
        self, returns: np.ndarray, regimes_arr: np.ndarray,
    ) -> None:
        fig = plotly_reg.plot_regime_distributions(returns, regimes_arr)
        assert isinstance(fig, go.Figure)

    def test_changepoints(self, returns: np.ndarray) -> None:
        fig = plotly_reg.plot_changepoints(returns, [100, 250, 400])
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Portfolio chart tests
# ---------------------------------------------------------------------------

class TestPortfolioCharts:
    """Portfolio visualisation Plotly charts."""

    def test_weight_comparison(self) -> None:
        weights = {
            "Equal Weight": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
            "Risk Parity": {"A": 0.3, "B": 0.2, "C": 0.3, "D": 0.2},
        }
        fig = plotly_port.plot_weight_comparison(weights)
        assert isinstance(fig, go.Figure)

    def test_monthly_returns(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0.01, 0.03, (5, 12))
        table = pd.DataFrame(
            data,
            index=[2021, 2022, 2023, 2024, 2025],
            columns=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        )
        fig = plotly_port.plot_monthly_returns(table)
        assert isinstance(fig, go.Figure)

    def test_monthly_returns_empty(self) -> None:
        fig = plotly_port.plot_monthly_returns(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_hrp_dendrogram(self, returns_df: pd.DataFrame) -> None:
        fig = plotly_port.plot_hrp_dendrogram(returns_df)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Backend delegation tests
# ---------------------------------------------------------------------------

class TestBackendDelegation:
    """Test backend='plotly' on existing matplotlib functions."""

    def test_drawdown_delegation(self, returns: np.ndarray) -> None:
        from quantlite.viz.risk import plot_drawdown
        fig = plot_drawdown(returns, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_correlation_matrix_delegation(
        self, corr_matrix: pd.DataFrame,
    ) -> None:
        from quantlite.viz.dependency import plot_correlation_matrix
        fig = plot_correlation_matrix(corr_matrix, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_correlation_dynamics_delegation(
        self, returns_df: pd.DataFrame,
    ) -> None:
        from quantlite.viz.dependency import plot_correlation_dynamics
        fig = plot_correlation_dynamics(
            returns_df["A"], returns_df["B"], backend="plotly",
        )
        assert isinstance(fig, go.Figure)

    def test_regime_timeline_delegation(
        self, returns: np.ndarray, regimes_arr: np.ndarray,
    ) -> None:
        from quantlite.viz.regimes import plot_regime_timeline
        fig = plot_regime_timeline(returns, regimes_arr, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_transition_matrix_delegation(self) -> None:
        from quantlite.viz.regimes import plot_transition_matrix

        class MockModel:
            transition_matrix = np.array([[0.9, 0.1], [0.3, 0.7]])
            n_regimes = 2

        fig = plot_transition_matrix(MockModel(), backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_regime_distributions_delegation(
        self, returns: np.ndarray, regimes_arr: np.ndarray,
    ) -> None:
        from quantlite.viz.regimes import plot_regime_distributions
        fig = plot_regime_distributions(returns, regimes_arr, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_tail_distribution_delegation(self, returns: np.ndarray) -> None:
        from quantlite.viz.risk import plot_tail_distribution
        fig = plot_tail_distribution(returns, backend="plotly")
        assert isinstance(fig, go.Figure)
