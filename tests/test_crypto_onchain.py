"""Tests for quantlite.crypto.onchain module."""

import pytest

from quantlite.crypto.onchain import (
    defi_dependency_graph,
    smart_contract_risk_score,
    tvl_tracker,
    wallet_exposure,
)


class TestWalletExposure:
    def test_returns_mock_data(self):
        result = wallet_exposure("0xabc123")
        assert result["address"] == "0xabc123"
        assert result["chain"] == "ethereum"
        assert len(result["tokens"]) > 0
        assert result["total_value_usd"] > 0

    def test_custom_chain(self):
        result = wallet_exposure("0xabc123", chain="polygon")
        assert result["chain"] == "polygon"

    def test_has_concentration(self):
        result = wallet_exposure("0xabc123")
        assert 0 < result["concentration"] < 1

    def test_has_note(self):
        result = wallet_exposure("0xabc123")
        assert "mock" in result["note"].lower() or "demonstration" in result["note"].lower()


class TestTvlTracker:
    def test_snapshot_input(self):
        tvls = {"Aave": 10_000_000_000, "Compound": 3_000_000_000, "MakerDAO": 8_000_000_000}
        result = tvl_tracker(tvls)
        assert result["total_tvl"] == pytest.approx(21_000_000_000)
        assert result["dominant_protocol"] == "Aave"
        assert sum(result["shares"].values()) == pytest.approx(1.0)

    def test_time_series_input(self):
        tvls = {
            "Aave": [10e9, 11e9, 12e9, 11.5e9],
            "Compound": [5e9, 4.5e9, 4e9, 3.5e9],
        }
        result = tvl_tracker(tvls)
        assert "Aave" in result["trends"]
        assert result["trends"]["Aave"]["direction"] == "growing"
        assert result["trends"]["Compound"]["direction"] == "shrinking"

    def test_concentration_flags(self):
        tvls = {"Dominant": 9_000_000, "Tiny": 1_000_000}
        result = tvl_tracker(tvls)
        assert any("concentration" in f.lower() for f in result["risk_flags"])

    def test_empty_tvls(self):
        result = tvl_tracker({})
        assert result["risk_rating"] == "critical"

    def test_declining_tvl_flag(self):
        tvls = {"Protocol": [100e6, 80e6, 50e6, 40e6]}
        result = tvl_tracker(tvls)
        assert any("declined" in f.lower() or "bank run" in f.lower() for f in result["risk_flags"])

    def test_risk_rating_low(self):
        tvls = {f"Proto{i}": 1_000_000 for i in range(10)}
        result = tvl_tracker(tvls)
        assert result["risk_rating"] in ("low", "medium")


class TestDefiDependencyGraph:
    def test_simple_graph(self):
        protocols = [
            {"name": "Aave", "dependencies": ["Chainlink"]},
            {"name": "Compound", "dependencies": ["Chainlink"]},
            {"name": "Chainlink", "dependencies": []},
        ]
        result = defi_dependency_graph(protocols)
        assert "Chainlink" in result["root_protocols"]
        assert "Chainlink" in result["critical_protocols"]
        assert len(result["edges"]) == 2

    def test_dependency_chain(self):
        protocols = [
            {"name": "Yearn", "dependencies": ["Aave"]},
            {"name": "Aave", "dependencies": ["Chainlink"]},
            {"name": "Chainlink", "dependencies": []},
        ]
        result = defi_dependency_graph(protocols)
        assert result["risk_layers"][0] == ["Chainlink"]
        assert "Yearn" in result["risk_layers"][2]

    def test_leaf_protocols(self):
        protocols = [
            {"name": "App", "dependencies": ["Lib"]},
            {"name": "Lib", "dependencies": []},
        ]
        result = defi_dependency_graph(protocols)
        assert "App" in result["leaf_protocols"]

    def test_no_dependencies(self):
        protocols = [
            {"name": "A", "dependencies": []},
            {"name": "B", "dependencies": []},
        ]
        result = defi_dependency_graph(protocols)
        assert len(result["edges"]) == 0
        assert len(result["root_protocols"]) == 2

    def test_adjacency_structure(self):
        protocols = [
            {"name": "A", "dependencies": ["B", "C"]},
            {"name": "B", "dependencies": []},
            {"name": "C", "dependencies": []},
        ]
        result = defi_dependency_graph(protocols)
        assert set(result["adjacency"]["A"]) == {"B", "C"}
        assert "A" in result["reverse_adjacency"]["B"]


class TestSmartContractRiskScore:
    def test_high_quality_contract(self):
        result = smart_contract_risk_score(
            age_days=800, audited=True, tvl=2_000_000_000, tvl_stability=0.95
        )
        assert result["risk_rating"] == "low"
        assert result["score"] > 0.75

    def test_new_unaudited_contract(self):
        result = smart_contract_risk_score(
            age_days=30, audited=False, tvl=500_000, tvl_stability=0.3
        )
        assert result["risk_rating"] in ("high", "critical")
        assert result["score"] < 0.30

    def test_recommendations_for_risky(self):
        result = smart_contract_risk_score(
            age_days=60, audited=False, tvl=1_000_000, tvl_stability=0.4
        )
        assert len(result["recommendations"]) >= 2

    def test_score_bounded(self):
        result = smart_contract_risk_score(
            age_days=1000, audited=True, tvl=10_000_000_000, tvl_stability=1.0
        )
        assert 0.0 <= result["score"] <= 1.0

    def test_factors_present(self):
        result = smart_contract_risk_score(
            age_days=365, audited=True, tvl=100_000_000
        )
        assert "age" in result["factors"]
        assert "audit" in result["factors"]
        assert "tvl" in result["factors"]
        assert "stability" in result["factors"]

    def test_audit_impact(self):
        audited = smart_contract_risk_score(365, True, 100e6, 0.9)
        unaudited = smart_contract_risk_score(365, False, 100e6, 0.9)
        assert audited["score"] > unaudited["score"]
