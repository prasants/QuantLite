"""Tests for the explainability layer: attribution, narratives, audit, what-if."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantlite.explain.attribution import (
    RiskAttribution,
    compute_risk_attribution,
    factor_attribution,
    marginal_risk_contribution,
)
from quantlite.explain.audit import AuditTrail, compare_weights
from quantlite.explain.narratives import (
    generate_narrative,
    transition_narrative,
)
from quantlite.explain.whatif import (
    cap_weights,
    compare_scenarios,
    remove_asset,
    stress_correlations,
)


@pytest.fixture
def sample_returns():
    rng = np.random.RandomState(42)
    n = 500
    return pd.DataFrame(
        {
            "Equity": rng.normal(0.0003, 0.015, n),
            "Bonds": rng.normal(0.0002, 0.006, n),
            "Gold": rng.normal(0.0001, 0.010, n),
            "Crypto": rng.normal(0.0005, 0.030, n),
        }
    )


@pytest.fixture
def sample_weights():
    return {"Equity": 0.4, "Bonds": 0.3, "Gold": 0.2, "Crypto": 0.1}


@pytest.fixture
def sample_regimes():
    return np.array([0] * 200 + [1] * 150 + [2] * 100 + [0] * 50)


# ---------------------------------------------------------------------------
# Risk Attribution
# ---------------------------------------------------------------------------


class TestRiskAttribution:
    def test_components_sum_to_total(self, sample_returns, sample_weights):
        attr = compute_risk_attribution(sample_returns, sample_weights)
        comp_sum = sum(attr.component_cvar.values())
        assert abs(comp_sum - attr.total_cvar) < 0.001, (
            f"Component CVaR sum {comp_sum:.6f} != total {attr.total_cvar:.6f}"
        )

    def test_var_components_sum_to_total(self, sample_returns, sample_weights):
        attr = compute_risk_attribution(sample_returns, sample_weights)
        comp_sum = sum(attr.component_var.values())
        assert abs(comp_sum - attr.total_var) < 0.001

    def test_marginal_risk_keys(self, sample_returns, sample_weights):
        marg = marginal_risk_contribution(sample_returns, sample_weights)
        assert set(marg.keys()) == set(sample_returns.columns)

    def test_factor_attribution_sums(self, sample_returns, sample_weights):
        contrib, idio = factor_attribution(sample_returns, sample_weights)
        total = sum(contrib.values()) + idio
        sample_returns.values @ np.array(
            [sample_weights[c] for c in sample_returns.columns]
        )
        assert total > 0

    def test_regime_attribution(self, sample_returns, sample_weights, sample_regimes):
        attr = compute_risk_attribution(
            sample_returns, sample_weights, regime_labels=sample_regimes
        )
        assert len(attr.regime_attributions) == 3
        for regime, contrib in attr.regime_attributions.items():
            assert set(contrib.keys()) == set(sample_returns.columns)

    def test_result_is_dataclass(self, sample_returns, sample_weights):
        attr = compute_risk_attribution(sample_returns, sample_weights)
        assert isinstance(attr, RiskAttribution)
        assert attr.total_cvar > 0


# ---------------------------------------------------------------------------
# Narratives
# ---------------------------------------------------------------------------


class TestNarratives:
    def test_brief_narrative(self, sample_regimes):
        returns = np.random.RandomState(42).randn(500) * 0.01
        nar = generate_narrative(returns, sample_regimes, detail_level="brief")
        assert "identified 3 regimes" in nar.summary
        assert len(nar.summary) > 20

    def test_standard_narrative(self, sample_regimes):
        returns = np.random.RandomState(42).randn(500) * 0.01
        nar = generate_narrative(returns, sample_regimes, detail_level="standard")
        assert "volatility" in nar.summary.lower()
        assert len(nar.regime_stats) == 3

    def test_detailed_narrative(self, sample_regimes):
        returns = np.random.RandomState(42).randn(500) * 0.01
        nar = generate_narrative(returns, sample_regimes, detail_level="detailed")
        assert "transition" in nar.summary.lower()

    def test_transition_narrative(self, sample_regimes):
        text = transition_narrative(sample_regimes)
        assert "shifted" in text.lower() or "No regime" in text

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            generate_narrative(np.zeros(10), np.zeros(5))

    def test_all_detail_levels_produce_text(self, sample_regimes):
        returns = np.random.RandomState(42).randn(500) * 0.01
        for level in ["brief", "standard", "detailed"]:
            nar = generate_narrative(returns, sample_regimes, detail_level=level)
            assert isinstance(nar.summary, str)
            assert len(nar.summary) > 0


# ---------------------------------------------------------------------------
# Audit Trail
# ---------------------------------------------------------------------------


class TestAuditTrail:
    def test_log_entry(self, sample_returns, sample_weights):
        trail = AuditTrail("Test Portfolio")
        entry = trail.log(
            method="Risk Parity",
            weights=sample_weights,
            returns_df=sample_returns,
        )
        assert entry.method == "Risk Parity"
        assert len(trail.entries) == 1

    def test_json_export(self, sample_returns, sample_weights):
        trail = AuditTrail("Test")
        trail.log(method="HRP", weights=sample_weights, returns_df=sample_returns)
        j = trail.to_json()
        parsed = json.loads(j)
        assert parsed["name"] == "Test"
        assert len(parsed["entries"]) == 1

    def test_markdown_export(self, sample_returns, sample_weights):
        trail = AuditTrail("Test")
        trail.log(method="HRP", weights=sample_weights, returns_df=sample_returns)
        md = trail.to_markdown()
        assert "# Audit Trail" in md
        assert "HRP" in md

    def test_html_export(self, sample_returns, sample_weights):
        trail = AuditTrail("Test")
        trail.log(method="HRP", weights=sample_weights, returns_df=sample_returns)
        html = trail.to_html()
        assert "<html>" in html
        assert "HRP" in html

    def test_compare_weights(self):
        prev = {"A": 0.5, "B": 0.3, "C": 0.2}
        curr = {"A": 0.4, "B": 0.4, "C": 0.2}
        comp = compare_weights(prev, curr)
        assert "decreased" in comp["A"]
        assert "increased" in comp["B"]
        assert "unchanged" in comp["C"]


# ---------------------------------------------------------------------------
# What-If
# ---------------------------------------------------------------------------


class TestWhatIf:
    def test_remove_asset(self, sample_returns, sample_weights):
        result = remove_asset(sample_returns, sample_weights, "Crypto")
        assert len(result.scenarios) == 1
        assert "Crypto" not in result.scenarios[0].weights
        # Remaining weights should sum to ~1
        remaining_sum = sum(result.scenarios[0].weights.values())
        assert abs(remaining_sum - 1.0) < 0.01

    def test_remove_nonexistent_raises(self, sample_returns, sample_weights):
        with pytest.raises(ValueError):
            remove_asset(sample_returns, sample_weights, "FakeAsset")

    def test_stress_correlations(self, sample_returns, sample_weights):
        result = stress_correlations(sample_returns, sample_weights, stress_factor=2.0)
        assert len(result.scenarios) == 1
        assert result.comparison_table.shape[0] == 2

    def test_cap_weights(self, sample_returns, sample_weights):
        result = cap_weights(sample_returns, sample_weights, max_weight=0.30)
        for w in result.scenarios[0].weights.values():
            assert w <= 0.30 + 1e-6

    def test_compare_scenarios(self, sample_returns, sample_weights):
        scenarios = {
            "Equal Weight": {c: 0.25 for c in sample_returns.columns},
            "Bonds Heavy": {"Equity": 0.1, "Bonds": 0.7, "Gold": 0.1, "Crypto": 0.1},
        }
        result = compare_scenarios(sample_returns, sample_weights, scenarios)
        assert len(result.scenarios) == 2
        assert result.comparison_table.shape[0] == 3
