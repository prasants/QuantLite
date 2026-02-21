"""Tests for the v1.3 Allocation Engine modules."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(n_assets: int = 4, n_periods: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic returns with different tail properties."""
    rng = np.random.RandomState(seed)
    # Asset 0: heavy tails (t-distribution)
    from scipy import stats

    cols = {}
    names = ["Equity", "Bonds", "Gold", "Crypto"][:n_assets]
    for i, name in enumerate(names):
        if name == "Crypto":
            cols[name] = stats.t.rvs(df=3, loc=0.0005, scale=0.03, size=n_periods, random_state=rng)
        elif name == "Equity":
            cols[name] = stats.t.rvs(df=5, loc=0.0003, scale=0.015, size=n_periods, random_state=rng)
        else:
            cols[name] = rng.normal(0.0002, 0.008, size=n_periods)
    return pd.DataFrame(cols)


def _make_regime_labels(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    # Simple: regime 0 for first half, regime 1 for second half, with some noise
    labels = np.zeros(n, dtype=int)
    labels[n // 2:] = 1
    # Sprinkle some regime 2 in the last quarter
    labels[3 * n // 4:] = rng.choice([1, 2], size=n - 3 * n // 4)
    return labels


# ---------------------------------------------------------------------------
# Tail Risk Parity
# ---------------------------------------------------------------------------

class TestTailRiskParity:
    def test_cvar_parity_equalises_contributions(self):
        from quantlite.portfolio.tail_risk_parity import cvar_parity_weights

        returns_df = _make_returns()
        result = cvar_parity_weights(returns_df, alpha=0.05)

        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert result.method == "cvar_parity"
        assert result.risk_measure == "CVaR"

        # Risk contributions should be roughly equal
        contribs = list(result.risk_contributions.values())
        total = sum(abs(c) for c in contribs)
        if total > 0:
            fracs = [abs(c) / total for c in contribs]
            target = 1.0 / len(fracs)
            for f in fracs:
                assert abs(f - target) < 0.15, f"CVaR contribution {f:.3f} far from target {target:.3f}"

    def test_es_parity(self):
        from quantlite.portfolio.tail_risk_parity import es_parity_weights

        returns_df = _make_returns()
        result = es_parity_weights(returns_df)
        assert result.method == "es_parity"
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_vol_parity(self):
        from quantlite.portfolio.tail_risk_parity import vol_parity_weights

        returns_df = _make_returns()
        result = vol_parity_weights(returns_df)
        assert result.method == "vol_parity"
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_compare_methods(self):
        from quantlite.portfolio.tail_risk_parity import compare_parity_methods

        returns_df = _make_returns()
        results = compare_parity_methods(returns_df)
        assert set(results.keys()) == {"vol_parity", "cvar_parity", "es_parity"}

    def test_regime_conditional(self):
        from quantlite.portfolio.tail_risk_parity import regime_conditional_tail_parity

        returns_df = _make_returns()
        labels = _make_regime_labels(len(returns_df))
        results = regime_conditional_tail_parity(returns_df, labels)
        assert len(results) >= 2
        for regime, result in results.items():
            assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_custom_risk_budget(self):
        from quantlite.portfolio.tail_risk_parity import cvar_parity_weights

        returns_df = _make_returns()
        budget = {name: 0.25 for name in returns_df.columns}
        result = cvar_parity_weights(returns_df, risk_budget=budget)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Regime Black-Litterman
# ---------------------------------------------------------------------------

class TestRegimeBL:
    def test_standard_bl_posterior(self):
        from quantlite.portfolio.regime_bl import black_litterman_posterior

        returns_df = _make_returns()
        caps = {name: 1000.0 for name in returns_df.columns}
        views = {"Equity": 0.10, "Bonds": 0.03}
        confs = {"Equity": 0.7, "Bonds": 0.5}

        mu, cov = black_litterman_posterior(returns_df, caps, views, confs)
        assert len(mu) == 4
        assert cov.shape == (4, 4)

    def test_regime_conditional_bl(self):
        from quantlite.portfolio.regime_bl import regime_conditional_bl

        returns_df = _make_returns()
        labels = _make_regime_labels(len(returns_df))
        caps = {name: 1000.0 for name in returns_df.columns}
        views = {"Equity": 0.10, "Bonds": 0.03}
        confs = {"Equity": 0.7, "Bonds": 0.5}

        result = regime_conditional_bl(
            returns_df, labels, caps, views, confs,
            regime_view_adjustments={0: {"Equity": 1.5}, 1: {"Equity": 0.5}},
            regime_confidence_scaling={0: 1.2, 1: 0.8},
        )

        assert len(result.regime_weights) >= 2
        assert abs(sum(result.blended_weights.values()) - 1.0) < 1e-6
        assert sum(result.regime_probabilities.values()) == pytest.approx(1.0, abs=1e-6)

    def test_blend_weights(self):
        from quantlite.portfolio.regime_bl import blend_regime_weights

        rw = {0: {"A": 0.6, "B": 0.4}, 1: {"A": 0.3, "B": 0.7}}
        probs = {0: 0.5, 1: 0.5}
        blended = blend_regime_weights(rw, probs)
        assert abs(blended["A"] - 0.45) < 1e-6
        assert abs(blended["B"] - 0.55) < 1e-6


# ---------------------------------------------------------------------------
# Dynamic Kelly
# ---------------------------------------------------------------------------

class TestDynamicKelly:
    def test_uses_scipy_optimize(self):
        """Kelly MUST use scipy.optimize.minimize_scalar, NOT grid search."""
        from quantlite.portfolio import dynamic_kelly

        source = inspect.getsource(dynamic_kelly)
        assert "minimize_scalar" in source, "Must use scipy.optimize.minimize_scalar"
        assert "linspace" not in source or "grid" not in source.lower(), \
            "Should not use grid search"

    def test_optimal_kelly(self):
        from quantlite.portfolio.dynamic_kelly import optimal_kelly_fraction

        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.02, 1000)
        f = optimal_kelly_fraction(returns)
        assert isinstance(f, float)
        assert 0 <= f <= 5.0

    def test_fractional_kelly(self):
        from quantlite.portfolio.dynamic_kelly import fractional_kelly

        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.02, 500)

        full = fractional_kelly(returns, fraction_of_kelly=1.0)
        half = fractional_kelly(returns, fraction_of_kelly=0.5)

        assert full.method == "full_kelly"
        assert "50%" in half.method
        assert half.fraction == pytest.approx(full.fraction / 2, rel=0.01)
        assert len(full.equity_curve) == len(returns) + 1

    def test_rolling_kelly(self):
        from quantlite.portfolio.dynamic_kelly import rolling_kelly

        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.02, 300)
        fracs, equity = rolling_kelly(returns, window=100)
        assert len(fracs) == 300
        assert len(equity) == 300

    def test_drawdown_control(self):
        from quantlite.portfolio.dynamic_kelly import kelly_with_drawdown_control

        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.02, 500)
        result = kelly_with_drawdown_control(
            returns, max_drawdown_threshold=-0.10, drawdown_reduction=0.25
        )
        assert result.method == "kelly_drawdown_control"
        assert result.max_drawdown <= 0


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class TestEnsemble:
    def test_ensemble_weights_sum_to_one(self):
        from quantlite.portfolio.ensemble import ensemble_allocate

        returns_df = _make_returns()
        result = ensemble_allocate(returns_df)
        total = sum(result.blended_weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_ensemble_with_custom_strategies(self):
        from quantlite.portfolio.ensemble import ensemble_allocate

        returns_df = _make_returns()
        strats = {
            "Strat A": {"Equity": 0.5, "Bonds": 0.3, "Gold": 0.1, "Crypto": 0.1},
            "Strat B": {"Equity": 0.2, "Bonds": 0.5, "Gold": 0.2, "Crypto": 0.1},
        }
        result = ensemble_allocate(returns_df, strategies=strats)
        total = sum(result.blended_weights.values())
        assert abs(total - 1.0) < 1e-6
        assert result.agreement_matrix.shape == (2, 2)

    def test_inverse_error_weights(self):
        from quantlite.portfolio.ensemble import inverse_error_weights

        errors = {"A": 0.05, "B": 0.10, "C": 0.20}
        weights = inverse_error_weights(errors)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert weights["A"] > weights["B"] > weights["C"]

    def test_consensus_portfolio(self):
        from quantlite.portfolio.ensemble import consensus_portfolio

        strats = {
            "A": {"X": 0.4, "Y": 0.3, "Z": 0.3},
            "B": {"X": 0.5, "Y": 0.1, "Z": 0.4},
        }
        consensus = consensus_portfolio(strats, threshold=0.05)
        # X min=0.4, Y min=0.1, Z min=0.3 → all above threshold
        assert len(consensus) >= 2
        assert abs(sum(consensus.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------

class TestWalkForward:
    def test_basic_walkforward(self):
        from quantlite.portfolio.walkforward import walk_forward

        returns_df = _make_returns(n_periods=1000)

        def simple_optimiser(df: pd.DataFrame) -> dict:
            n = df.shape[1]
            return {col: 1.0 / n for col in df.columns}

        result = walk_forward(
            returns_df, simple_optimiser,
            is_window=252, oos_window=63,
            window_type="sliding",
        )

        assert len(result.folds) > 0
        assert result.window_type == "sliding"

    def test_expanding_window(self):
        from quantlite.portfolio.walkforward import walk_forward

        returns_df = _make_returns(n_periods=800)

        def simple_optimiser(df: pd.DataFrame) -> dict:
            n = df.shape[1]
            return {col: 1.0 / n for col in df.columns}

        result = walk_forward(
            returns_df, simple_optimiser,
            is_window=200, oos_window=50,
            window_type="expanding",
        )

        # First fold IS should start at 0 (expanding)
        assert result.folds[0].is_start == 0
        assert result.window_type == "expanding"

    def test_fold_structure(self):
        from quantlite.portfolio.walkforward import walk_forward

        returns_df = _make_returns(n_periods=600)

        def opt(df):
            return {col: 1.0 / df.shape[1] for col in df.columns}

        result = walk_forward(returns_df, opt, is_window=200, oos_window=100)

        for fold in result.folds:
            assert fold.oos_start == fold.is_end
            assert fold.oos_end - fold.oos_start == 100
            assert len(fold.oos_returns) == 100

    def test_scoring_functions(self):
        from quantlite.portfolio.walkforward import (
            calmar_score,
            max_drawdown_score,
            sharpe_score,
            sortino_score,
        )

        rng = np.random.RandomState(42)
        rets = rng.normal(0.001, 0.02, 252)

        s = sharpe_score(rets)
        assert isinstance(s, float)

        so = sortino_score(rets)
        assert isinstance(so, float)

        c = calmar_score(rets)
        assert isinstance(c, float)

        md = max_drawdown_score(rets)
        assert md <= 0

    def test_invalid_window_type(self):
        from quantlite.portfolio.walkforward import walk_forward

        returns_df = _make_returns(n_periods=500)
        with pytest.raises(ValueError, match="window_type"):
            walk_forward(returns_df, lambda df: {}, is_window=200, oos_window=50,
                         window_type="invalid")

    def test_equity_curve(self):
        from quantlite.portfolio.walkforward import walk_forward

        returns_df = _make_returns(n_periods=800)

        def opt(df):
            return {col: 1.0 / df.shape[1] for col in df.columns}

        result = walk_forward(returns_df, opt, is_window=200, oos_window=100)
        assert result.equity_curve[0] == 1.0
        assert len(result.equity_curve) > 1
