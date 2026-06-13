"""Benchmark runner: unified entry point for all benchmarks.

Provides ``run_benchmarks()`` which executes comparison, tail event,
and speed benchmarks and produces a consolidated report.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .compare import ComparisonResult, run_comparison
from .speed import SpeedResult, run_speed_benchmarks
from .tail_events import CrisisResult, run_tail_event_analysis


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


@dataclass
class BenchmarkReport:
    """Consolidated benchmark report.

    Attributes:
        comparison_results: Head-to-head comparison results.
        crisis_results: Tail event analysis results.
        speed_results: Speed benchmark results.
        summary: Summary table as list of dicts.
        elapsed_seconds: Total time to run all benchmarks.
    """
    comparison_results: list[ComparisonResult] = field(default_factory=list)
    crisis_results: list[CrisisResult] = field(default_factory=list)
    speed_results: list[SpeedResult] = field(default_factory=list)
    summary: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_json(self, path: str | None = None) -> str:
        """Serialise the report to JSON.

        Args:
            path: If provided, write to this file path.

        Returns:
            JSON string.
        """
        data = {
            "elapsed_seconds": self.elapsed_seconds,
            "comparison": [
                {
                    "dataset": r.dataset,
                    "methods": r.methods,
                    "var_violations": r.var_violations,
                    "sharpe_ratios": r.sharpe_ratios,
                    "computation_times": r.computation_times,
                }
                for r in self.comparison_results
            ],
            "crisis": [
                {
                    "crisis_name": r.crisis_name,
                    "crisis_key": r.crisis_key,
                    "year": r.year,
                    "actual_worst_loss": r.actual_worst_loss,
                    "gaussian_var": r.gaussian_var,
                    "evt_var": r.evt_var,
                    "gaussian_violation_rate": r.gaussian_violation_rate,
                    "evt_violation_rate": r.evt_violation_rate,
                    "gaussian_pass": r.gaussian_pass,
                    "evt_pass": r.evt_pass,
                }
                for r in self.crisis_results
            ],
            "speed": [
                {
                    "operation": r.operation,
                    "data_size": r.data_size,
                    "quantlite_time": r.quantlite_time,
                    "baseline_time": r.baseline_time,
                    "speedup": r.speedup,
                }
                for r in self.speed_results
            ],
            "summary": self.summary,
        }

        text = json.dumps(data, indent=2, cls=_NumpyEncoder)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        return text


def _build_summary(report: BenchmarkReport) -> list[dict[str, Any]]:
    """Build a summary table from benchmark results.

    Args:
        report: The partially-filled report.

    Returns:
        List of summary row dicts.
    """
    rows = []

    # VaR accuracy summary
    for cr in report.comparison_results:
        for method, rate in cr.var_violations.items():
            rows.append({
                "category": "VaR Accuracy",
                "dataset": cr.dataset,
                "method": method,
                "metric": "violation_rate",
                "value": rate,
                "baseline": 0.05,
                "improvement_pct": (0.05 - abs(rate - 0.05)) / 0.05 * 100
                if rate != 0.05 else 0.0,
            })

    # Crisis performance summary
    for cr in report.crisis_results:
        rows.append({
            "category": "Crisis Performance",
            "dataset": cr.crisis_key,
            "method": "Gaussian VaR",
            "metric": "violation_rate",
            "value": cr.gaussian_violation_rate,
            "baseline": cr.evt_violation_rate,
            "improvement_pct": (
                (cr.gaussian_violation_rate - cr.evt_violation_rate)
                / max(cr.gaussian_violation_rate, 1e-10) * 100
            ),
        })

    # Speed summary (at 10K)
    for sr in report.speed_results:
        if sr.data_size == 10_000:
            rows.append({
                "category": "Speed",
                "dataset": "10K observations",
                "method": sr.operation,
                "metric": "speedup",
                "value": sr.speedup,
                "baseline": 1.0,
                "improvement_pct": (sr.speedup - 1.0) * 100,
            })

    return rows


def run_benchmarks(
    include: Sequence[str] | None = None,
    sizes: Sequence[int] | None = None,
    output_path: str | None = None,
    seed: int = 42,
) -> BenchmarkReport:
    """Run all benchmarks and produce a consolidated report.

    Args:
        include: Which benchmark suites to run. Options:
            ``"comparison"``, ``"tail_events"``, ``"speed"``.
            Defaults to all.
        sizes: Data sizes for speed benchmarks.
        output_path: If provided, save JSON report to this path.
        seed: Random seed for reproducibility.

    Returns:
        A ``BenchmarkReport`` with all results.
    """
    if include is None:
        include = ["comparison", "tail_events", "speed"]

    report = BenchmarkReport()
    t0 = time.perf_counter()

    if "comparison" in include:
        report.comparison_results = run_comparison(seed=seed)

    if "tail_events" in include:
        report.crisis_results = run_tail_event_analysis(seed=seed)

    if "speed" in include:
        report.speed_results = run_speed_benchmarks(sizes=sizes, seed=seed)

    report.elapsed_seconds = time.perf_counter() - t0
    report.summary = _build_summary(report)

    if output_path is not None:
        report.to_json(output_path)

    return report
