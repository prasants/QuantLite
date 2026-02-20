"""Benchmarking suite for QuantLite.

Provides head-to-head comparisons against baseline methods,
tail event backtesting, speed benchmarks, and a unified runner.
"""

from .compare import ComparisonResult, run_comparison
from .runner import BenchmarkReport, run_benchmarks
from .speed import SpeedResult, run_speed_benchmarks
from .tail_events import CrisisResult, run_tail_event_analysis

__all__ = [
    "run_comparison",
    "ComparisonResult",
    "run_tail_event_analysis",
    "CrisisResult",
    "run_speed_benchmarks",
    "SpeedResult",
    "run_benchmarks",
    "BenchmarkReport",
]
