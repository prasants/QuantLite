"""Tests for the benchmarking suite."""

from __future__ import annotations

import json

import numpy as np

from quantlite.benchmark.compare import (
    ComparisonResult,
    _simulate_emerging_market,
    _simulate_multi_asset,
    _simulate_sp500,
    run_comparison,
)
from quantlite.benchmark.runner import BenchmarkReport, run_benchmarks
from quantlite.benchmark.speed import SpeedResult, run_speed_benchmarks
from quantlite.benchmark.tail_events import (
    CRISES,
    CrisisResult,
    _simulate_crisis,
    run_tail_event_analysis,
)


class TestComparison:
    """Tests for the head-to-head comparison framework."""

    def test_run_comparison_returns_results(self):
        results = run_comparison(datasets=["sp500"], seed=42)
        assert len(results) == 1
        assert isinstance(results[0], ComparisonResult)
        assert results[0].dataset == "sp500"

    def test_comparison_has_var_violations(self):
        results = run_comparison(datasets=["sp500"], seed=42)
        r = results[0]
        assert len(r.var_violations) > 0
        for rate in r.var_violations.values():
            assert 0.0 <= rate <= 1.0

    def test_multi_asset_has_sharpe_ratios(self):
        results = run_comparison(datasets=["multi_asset"], seed=42)
        r = results[0]
        assert len(r.sharpe_ratios) > 0

    def test_all_datasets_run(self):
        results = run_comparison(seed=42)
        assert len(results) == 3

    def test_simulated_data_shapes(self):
        sp = _simulate_sp500(n=100)
        assert sp.shape == (100,)
        ma = _simulate_multi_asset(n=100)
        assert ma.shape == (100, 4)
        em = _simulate_emerging_market(n=100)
        assert em.shape == (100,)


class TestTailEvents:
    """Tests for tail event backtesting."""

    def test_run_all_crises(self):
        results = run_tail_event_analysis(seed=42)
        assert len(results) == len(CRISES)
        for r in results:
            assert isinstance(r, CrisisResult)

    def test_single_crisis(self):
        results = run_tail_event_analysis(crises=["gfc_2008"], seed=42)
        assert len(results) == 1
        assert results[0].crisis_key == "gfc_2008"

    def test_violation_rates_bounded(self):
        results = run_tail_event_analysis(seed=42)
        for r in results:
            assert 0.0 <= r.gaussian_violation_rate <= 1.0
            assert 0.0 <= r.evt_violation_rate <= 1.0

    def test_crisis_data_plausible(self):
        for spec in CRISES:
            data = _simulate_crisis(spec, seed=42)
            assert len(data) == spec.duration_days
            assert np.min(data) < 0  # Should have losses
            # Worst loss should be close to spec
            # Worst loss should be at least as bad as the spec (more negative)
            assert np.min(data) <= spec.worst_daily_loss * 0.8

    def test_actual_worst_loss_populated(self):
        results = run_tail_event_analysis(seed=42)
        for r in results:
            assert r.actual_worst_loss < 0


class TestSpeed:
    """Tests for speed benchmarks."""

    def test_run_produces_results(self):
        results = run_speed_benchmarks(sizes=[100, 1000], seed=42)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, SpeedResult)

    def test_timing_data_positive(self):
        results = run_speed_benchmarks(sizes=[100], seed=42)
        for r in results:
            assert r.quantlite_time > 0
            assert r.baseline_time > 0
            assert r.speedup > 0

    def test_all_operations_benchmarked(self):
        results = run_speed_benchmarks(sizes=[1000], seed=42)
        ops = set(r.operation for r in results)
        assert len(ops) >= 3  # At least var, cvar, evt


class TestRunner:
    """Tests for the benchmark runner."""

    def test_run_all_benchmarks(self):
        report = run_benchmarks(sizes=[100, 1000], seed=42)
        assert isinstance(report, BenchmarkReport)
        assert report.elapsed_seconds > 0
        assert len(report.comparison_results) > 0
        assert len(report.crisis_results) > 0
        assert len(report.speed_results) > 0

    def test_json_serialisation(self):
        report = run_benchmarks(
            include=["comparison"],
            seed=42,
        )
        text = report.to_json()
        data = json.loads(text)
        assert "comparison" in data
        assert "elapsed_seconds" in data

    def test_summary_table(self):
        report = run_benchmarks(sizes=[100], seed=42)
        assert len(report.summary) > 0
        for row in report.summary:
            assert "category" in row
            assert "method" in row
            assert "value" in row

    def test_selective_benchmarks(self):
        report = run_benchmarks(include=["speed"], sizes=[100], seed=42)
        assert len(report.speed_results) > 0
        assert len(report.comparison_results) == 0
        assert len(report.crisis_results) == 0
