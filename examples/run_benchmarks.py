#!/usr/bin/env python3
"""Generate all benchmark charts and save to docs/images/.

Usage:
    python examples/run_benchmarks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")

from quantlite.benchmark.runner import run_benchmarks
from quantlite.viz.benchmark import (
    plot_benchmark_summary,
    plot_crisis_timeline,
    plot_crisis_var_comparison,
    plot_method_comparison,
    plot_risk_estimate_scatter,
    plot_scaling,
    plot_speed_comparison,
    plot_var_accuracy,
    plot_var_violations,
)

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUTPUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run all benchmarks and generate charts."""
    print("Running benchmarks...")
    report = run_benchmarks(output_path=str(OUTPUT / "benchmark_results.json"))
    print(f"Benchmarks complete in {report.elapsed_seconds:.1f}s")

    # Head-to-head charts
    print("Generating head-to-head charts...")
    fig = plot_var_accuracy(report.comparison_results)
    fig.savefig(str(OUTPUT / "var_accuracy.png"), bbox_inches="tight", dpi=150)

    fig = plot_method_comparison(report.comparison_results)
    fig.savefig(str(OUTPUT / "method_comparison.png"), bbox_inches="tight", dpi=150)

    fig = plot_risk_estimate_scatter(report.comparison_results)
    fig.savefig(str(OUTPUT / "risk_estimate_scatter.png"), bbox_inches="tight", dpi=150)

    # Tail event charts
    print("Generating tail event charts...")
    fig = plot_crisis_var_comparison(report.crisis_results)
    fig.savefig(str(OUTPUT / "crisis_var_comparison.png"), bbox_inches="tight", dpi=150)

    # Use the first crisis with enough data for the violations chart
    for cr in report.crisis_results:
        fig = plot_var_violations(cr)
        fig.savefig(str(OUTPUT / f"var_violations_{cr.crisis_key}.png"),
                    bbox_inches="tight", dpi=150)

    fig = plot_crisis_timeline(report.crisis_results)
    fig.savefig(str(OUTPUT / "crisis_timeline.png"), bbox_inches="tight", dpi=150)

    # Speed charts
    print("Generating speed charts...")
    fig = plot_scaling(report.speed_results)
    fig.savefig(str(OUTPUT / "scaling.png"), bbox_inches="tight", dpi=150)

    fig = plot_speed_comparison(report.speed_results)
    fig.savefig(str(OUTPUT / "speed_comparison.png"), bbox_inches="tight", dpi=150)

    # Summary dashboard
    print("Generating summary dashboard...")
    fig = plot_benchmark_summary(
        report.comparison_results,
        report.crisis_results,
        report.speed_results,
    )
    fig.savefig(str(OUTPUT / "benchmark_summary.png"), bbox_inches="tight", dpi=150)

    print(f"All charts saved to {OUTPUT}")
    print("JSON report saved to {}".format(OUTPUT / "benchmark_results.json"))


if __name__ == "__main__":
    main()
