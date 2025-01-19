"""Smoke tests for quantlite.viz (theme and risk charts)."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from quantlite.risk.evt import fit_gpd
from quantlite.viz.risk import (
    plot_drawdown,
    plot_return_levels,
    plot_risk_dashboard,
    plot_tail_distribution,
)
from quantlite.viz.theme import (
    FEW_PALETTE,
    apply_few_theme,
    bullet_graph,
    direct_label,
    few_figure,
    sparkline,
)


@pytest.fixture()
def sample_returns():
    rng = np.random.default_rng(42)
    return rng.standard_t(df=4, size=1000) * 0.015


class TestTheme:
    def test_apply_few_theme(self):
        apply_few_theme()
        assert matplotlib.rcParams["axes.spines.top"] is False
        assert matplotlib.rcParams["axes.spines.right"] is False

    def test_few_figure(self):
        fig, ax = few_figure()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_few_figure_subplots(self):
        fig, axes = few_figure(2, 2)
        assert axes.shape == (2, 2)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_palette_has_required_keys(self):
        required = {"primary", "secondary", "negative", "positive", "neutral", "grey_dark", "grey_mid", "grey_light", "bg"}
        assert required.issubset(FEW_PALETTE.keys())

    def test_sparkline(self):
        fig, ax = few_figure()
        sparkline(ax, [1, 2, 3, 2, 4])
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_bullet_graph(self):
        fig, ax = few_figure()
        bullet_graph(ax, value=75, target=90, ranges=[50, 75, 100], label="Score")
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_direct_label(self):
        fig, ax = few_figure()
        ax.plot([1, 2, 3], [1, 2, 3])
        direct_label(ax, 3, 3, "end")
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestRiskCharts:
    def test_plot_tail_distribution(self, sample_returns):
        fig, ax = plot_tail_distribution(sample_returns)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_tail_distribution_with_gpd(self, sample_returns):
        gpd = fit_gpd(sample_returns)
        fig, ax = plot_tail_distribution(sample_returns, gpd_fit=gpd)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_return_levels(self, sample_returns):
        gpd = fit_gpd(sample_returns)
        fig, ax = plot_return_levels(gpd)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_drawdown(self, sample_returns):
        fig, ax = plot_drawdown(sample_returns)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_risk_dashboard(self, sample_returns):
        fig, axes = plot_risk_dashboard(sample_returns)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
