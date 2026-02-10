"""Regime-aware integration demo.

Generates charts showing regime-conditional risk, defensive weight
tilting, and regime-filtered backtest results.

Charts saved to docs/images/.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure the source tree is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlite.regime_integration.risk import (
    regime_conditional_var,
    regime_conditional_cvar,
    regime_risk_summary,
    regime_transition_risk,
)
from quantlite.regime_integration.portfolio import (
    regime_aware_weights,
    regime_filtered_backtest,
)
from quantlite.regime_integration.reporting import (
    regime_comparison_table,
    regime_performance_attribution,
    regime_tearsheet,
)

# Palette (Stephen Few)
PRIMARY = "#4E79A7"
SECONDARY = "#F28E2B"
NEGATIVE = "#E15759"
POSITIVE = "#59A14F"
NEUTRAL = "#76B7B2"
REGIME_COLOURS = {0: NEGATIVE, 1: SECONDARY, 2: PRIMARY}
REGIME_NAMES = {0: "Crisis", 1: "Bear", 2: "Bull"}

IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(IMG_DIR, exist_ok=True)


def generate_synthetic_data(seed=42):
    """Create synthetic multi-asset returns with regime structure."""
    rng = np.random.RandomState(seed)
    n_bull, n_bear, n_crisis = 500, 200, 100
    n = n_bull + n_bear + n_crisis
    dates = pd.date_range("2018-01-02", periods=n, freq="B")

    # Regime labels
    regimes = np.array([2] * n_bull + [1] * n_bear + [0] * n_crisis)

    # Asset returns with regime-dependent characteristics
    df = pd.DataFrame(index=dates)
    for regime_idx, start, count in [(2, 0, n_bull), (1, n_bull, n_bear), (0, n_bull + n_bear, n_crisis)]:
        sl = slice(start, start + count)
        if regime_idx == 2:  # Bull
            df.loc[dates[sl], "Equity"] = rng.normal(0.0008, 0.012, count)
            df.loc[dates[sl], "Gold"] = rng.normal(0.0002, 0.007, count)
            df.loc[dates[sl], "Bonds"] = rng.normal(0.0001, 0.004, count)
        elif regime_idx == 1:  # Bear
            df.loc[dates[sl], "Equity"] = rng.normal(-0.001, 0.02, count)
            df.loc[dates[sl], "Gold"] = rng.normal(0.0005, 0.009, count)
            df.loc[dates[sl], "Bonds"] = rng.normal(0.0003, 0.005, count)
        else:  # Crisis
            df.loc[dates[sl], "Equity"] = rng.normal(-0.004, 0.035, count)
            df.loc[dates[sl], "Gold"] = rng.normal(0.001, 0.012, count)
            df.loc[dates[sl], "Bonds"] = rng.normal(0.0005, 0.008, count)

    return df, regimes, dates


def chart_regime_risk_summary(returns, regimes):
    """Chart 1: Regime risk metrics comparison."""
    summary = regime_risk_summary(returns, regimes)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    regime_labels = ["Crisis", "Bear", "Bull"]
    regime_keys = ["0", "1", "2"]

    # Volatility
    vols = [summary[k]["volatility"] for k in regime_keys]
    bars = axes[0].bar(regime_labels, vols,
                       color=[NEGATIVE, SECONDARY, PRIMARY], edgecolor="white", linewidth=0.5)
    axes[0].set_ylabel("Annualised volatility")
    axes[0].set_title("Volatility by regime", fontsize=11, fontweight="bold")
    for bar, v in zip(bars, vols):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                     f"{v:.1%}", ha="center", va="bottom", fontsize=9)

    # VaR
    vars_ = [summary[k]["var"] for k in regime_keys]
    bars = axes[1].bar(regime_labels, vars_,
                       color=[NEGATIVE, SECONDARY, PRIMARY], edgecolor="white", linewidth=0.5)
    axes[1].set_ylabel("VaR (5%)")
    axes[1].set_title("Value at Risk by regime", fontsize=11, fontweight="bold")
    for bar, v in zip(bars, vars_):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v - 0.002,
                     f"{v:.4f}", ha="center", va="top", fontsize=9)

    # Kurtosis
    kurts = [summary[k]["kurtosis"] for k in regime_keys]
    bars = axes[2].bar(regime_labels, kurts,
                       color=[NEGATIVE, SECONDARY, PRIMARY], edgecolor="white", linewidth=0.5)
    axes[2].set_ylabel("Excess kurtosis")
    axes[2].set_title("Tail heaviness by regime", fontsize=11, fontweight="bold")
    for bar, v in zip(bars, kurts):
        axes[2].text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "regime_risk_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def chart_regime_filtered_backtest(df, regimes):
    """Chart 2: Equity curves for regime-filtered vs static allocation."""
    assets = list(df.columns)

    # Define regime-specific weights
    weights_by_regime = {
        0: {"Equity": 0.1, "Gold": 0.5, "Bonds": 0.4},   # Crisis: defensive
        1: {"Equity": 0.3, "Gold": 0.35, "Bonds": 0.35},  # Bear: balanced
        2: {"Equity": 0.6, "Gold": 0.2, "Bonds": 0.2},    # Bull: aggressive
    }
    static_weights = {"Equity": 1.0 / 3, "Gold": 1.0 / 3, "Bonds": 1.0 / 3}

    # Regime-filtered backtest
    result = regime_filtered_backtest(df, weights_by_regime, regimes, rebalance="monthly")

    # Static backtest
    port_ret_static = (df.values * np.array([static_weights[a] for a in assets])).sum(axis=1)
    static_equity = 10000.0 * np.cumprod(1.0 + port_ret_static)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, result["equity_curve"].values, color=PRIMARY,
            linewidth=1.5, label="Regime-aware allocation")
    ax.plot(df.index, static_equity, color=NEUTRAL,
            linewidth=1.5, linestyle="--", label="Static equal-weight")

    # Shade regimes
    prev_regime = regimes[0]
    start_idx = 0
    for i in range(1, len(regimes)):
        if regimes[i] != prev_regime or i == len(regimes) - 1:
            ax.axvspan(df.index[start_idx], df.index[i - 1],
                       alpha=0.08, color=REGIME_COLOURS[prev_regime])
            start_idx = i
            prev_regime = regimes[i]

    ax.set_ylabel("Portfolio value")
    ax.set_title("Regime-filtered vs static allocation", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "regime_filtered_backtest.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def chart_regime_attribution(returns, regimes):
    """Chart 3: Performance attribution by regime."""
    attr = regime_performance_attribution(returns, regimes)

    labels = []
    values = []
    colours = []
    for key in sorted(attr.keys()):
        labels.append(REGIME_NAMES.get(int(key), f"Regime {key}"))
        values.append(attr[key]["contribution_pct"])
        colours.append(REGIME_COLOURS.get(int(key), NEUTRAL))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(labels, values, color=colours, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        x = bar.get_width()
        ax.text(x + 1 if x >= 0 else x - 1, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", ha="left" if x >= 0 else "right", va="center", fontsize=10)

    ax.set_xlabel("Contribution to total return (%)")
    ax.set_title("Performance attribution by regime", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.axvline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "regime_performance_attribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    df, regimes, dates = generate_synthetic_data()

    # Mean return across assets for single-series analysis
    mean_returns = df.mean(axis=1).values

    print("Regime risk summary:")
    summary = regime_risk_summary(mean_returns, regimes)
    for regime, metrics in sorted(summary.items()):
        print(f"  {regime}: vol={metrics['volatility']:.2%}, "
              f"VaR={metrics['var']:.4f}, count={metrics['count']:.0f}")

    print("\nComparison table:")
    print(regime_comparison_table(mean_returns, regimes))

    chart_regime_risk_summary(mean_returns, regimes)
    chart_regime_filtered_backtest(df, regimes)
    chart_regime_attribution(mean_returns, regimes)

    print("\nAll charts saved to docs/images/")
