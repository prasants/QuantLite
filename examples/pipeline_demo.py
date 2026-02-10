"""Pipeline (Dream API) demo.

Demonstrates the full QuantLite workflow using synthetic data:
fetch-equivalent returns, regime detection, portfolio construction,
backtesting, and tearsheet generation.

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

from quantlite.pipeline import (
    backtest,
    construct_portfolio,
    detect_regimes,
    tearsheet,
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


def generate_synthetic_returns(seed=42):
    """Create synthetic multi-asset returns mimicking real market data."""
    rng = np.random.RandomState(seed)
    n = 1000
    dates = pd.date_range("2016-01-04", periods=n, freq="B")

    # Three distinct periods
    bull1 = 400
    bear = 200
    bull2 = 300
    crisis = 100

    df = pd.DataFrame(index=dates)

    # Bull period 1
    df.iloc[:bull1, :] = 0  # placeholder
    df["Stock A"] = 0.0
    df["Stock B"] = 0.0
    df["GLD"] = 0.0
    df["TLT"] = 0.0

    idx = 0
    for count, mu_eq, vol_eq, mu_gld, mu_tlt in [
        (bull1, 0.0006, 0.013, 0.0001, 0.00005),
        (bear, -0.0008, 0.022, 0.0004, 0.0003),
        (bull2, 0.0005, 0.014, 0.00015, 0.0001),
        (crisis, -0.003, 0.04, 0.0008, 0.0005),
    ]:
        sl = slice(idx, idx + count)
        df.iloc[sl, df.columns.get_loc("Stock A")] = rng.normal(mu_eq, vol_eq, count)
        df.iloc[sl, df.columns.get_loc("Stock B")] = rng.normal(mu_eq * 0.8, vol_eq * 1.2, count)
        df.iloc[sl, df.columns.get_loc("GLD")] = rng.normal(mu_gld, 0.008, count)
        df.iloc[sl, df.columns.get_loc("TLT")] = rng.normal(mu_tlt, 0.005, count)
        idx += count

    return df


def chart_pipeline_equity(data, result, regimes):
    """Chart 1: Equity curve with regime shading."""
    fig, ax = plt.subplots(figsize=(10, 5))

    eq = result["equity_curve"]
    ax.plot(eq.index, eq.values, color=PRIMARY, linewidth=1.5, label="Portfolio")

    # Shade regimes
    prev = regimes[0]
    start = 0
    for i in range(1, len(regimes)):
        if regimes[i] != prev or i == len(regimes) - 1:
            colour = REGIME_COLOURS.get(int(prev), NEUTRAL)
            ax.axvspan(data.index[start], data.index[min(i, len(data) - 1)],
                       alpha=0.1, color=colour)
            start = i
            prev = regimes[i]

    # Annotate final value
    final = eq.values[-1]
    ax.annotate(f"${final:,.0f}",
                xy=(eq.index[-1], final),
                xytext=(-60, 15), textcoords="offset points",
                fontsize=10, color=PRIMARY, fontweight="bold")

    ax.set_ylabel("Portfolio value ($)")
    ax.set_title("Pipeline backtest: regime-aware portfolio",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "pipeline_equity_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def chart_pipeline_weights(weights):
    """Chart 2: Portfolio weights pie chart."""
    fig, ax = plt.subplots(figsize=(6, 6))

    labels = list(weights.keys())
    values = list(weights.values())
    colours = [PRIMARY, SECONDARY, POSITIVE, NEUTRAL][:len(labels)]

    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%",
        colors=colours, startangle=90,
        textprops={"fontsize": 10},
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")

    ax.set_title("Regime-aware portfolio weights",
                 fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "pipeline_weights.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def chart_pipeline_drawdown(result):
    """Chart 3: Drawdown chart."""
    eq = result["equity_curve"].values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1.0)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(result["equity_curve"].index, dd, 0,
                    color=NEGATIVE, alpha=0.4)
    ax.plot(result["equity_curve"].index, dd, color=NEGATIVE, linewidth=0.8)

    max_dd_idx = np.argmin(dd)
    ax.annotate(f"Max: {dd[max_dd_idx]:.1%}",
                xy=(result["equity_curve"].index[max_dd_idx], dd[max_dd_idx]),
                xytext=(30, -15), textcoords="offset points",
                fontsize=9, color=NEGATIVE, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": NEGATIVE, "lw": 0.8})

    ax.set_ylabel("Drawdown")
    ax.set_title("Portfolio drawdown", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "pipeline_drawdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    # Generate synthetic data (in production, use ql.fetch())
    data = generate_synthetic_returns()

    # Step 1: Detect regimes
    regimes = detect_regimes(data, n_regimes=3, rng_seed=42)
    unique, counts = np.unique(regimes, return_counts=True)
    print("Regime distribution:")
    for u, c in zip(unique, counts):
        print(f"  Regime {u} ({REGIME_NAMES.get(u, '?')}): {c} days ({c / len(regimes):.1%})")

    # Step 2: Construct portfolio
    weights = construct_portfolio(
        data, method="hrp", regime_aware=True, regimes=regimes,
    )
    print(f"\nPortfolio weights: {weights}")

    # Step 3: Backtest
    result = backtest(data, weights, rebalance="monthly")
    print(f"\nBacktest results:")
    print(f"  Total return: {result['total_return']:.2%}")
    print(f"  Annualised return: {result['annualised_return']:.2%}")
    print(f"  Annualised vol: {result['annualised_volatility']:.2%}")
    print(f"  Sharpe ratio: {result['sharpe_ratio']:.2f}")
    print(f"  Max drawdown: {result['max_drawdown']:.2%}")

    # Step 4: Tearsheet
    ts = tearsheet(result, regimes=regimes)
    if "comparison_table" in ts:
        print(f"\n{ts['comparison_table']}")

    # Generate charts
    chart_pipeline_equity(data, result, regimes)
    chart_pipeline_weights(weights)
    chart_pipeline_drawdown(result)

    print("\nAll charts saved to docs/images/")
