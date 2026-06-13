#!/usr/bin/env python3
"""Generate all explainability chart PNGs for documentation."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")

from quantlite.explain.attribution import compute_risk_attribution
from quantlite.explain.narratives import generate_narrative
from quantlite.explain.whatif import (
    cap_weights,
    compare_scenarios,
    remove_asset,
    stress_correlations,
)
from quantlite.viz.explain import (
    plot_correlation_stress,
    plot_factor_attribution,
    plot_marginal_risk,
    plot_regime_summary,
    plot_regime_transition_matrix,
    plot_risk_waterfall,
    plot_weight_changes,
    plot_whatif_comparison,
    plot_whatif_tornado,
)

IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(IMG_DIR, exist_ok=True)


def _make_returns(seed=42):
    rng = np.random.RandomState(seed)
    n = 750
    return pd.DataFrame({
        "Equity": stats.t.rvs(df=5, loc=0.0003, scale=0.015, size=n, random_state=rng),
        "Bonds": rng.normal(0.0002, 0.006, n),
        "Gold": rng.normal(0.0001, 0.010, n),
        "Crypto": stats.t.rvs(df=3, loc=0.0005, scale=0.03, size=n, random_state=rng),
    })


def _make_regimes(n, seed=42):
    rng = np.random.RandomState(seed)
    labels = np.zeros(n, dtype=int)
    labels[n // 3: 2 * n // 3] = 1
    labels[2 * n // 3:] = rng.choice([1, 2], size=n - 2 * n // 3)
    return labels


def main():
    returns_df = _make_returns()
    weights = {"Equity": 0.4, "Bonds": 0.3, "Gold": 0.2, "Crypto": 0.1}
    regime_labels = _make_regimes(len(returns_df))

    # --- Risk Attribution ---
    print("Generating risk attribution charts...")
    attr = compute_risk_attribution(
        returns_df, weights, regime_labels=regime_labels
    )

    # Use standalone (undiversified) CVaRs for the waterfall so the
    # diversification benefit is visible as a negative bar.
    standalone_cvar = {}
    for name in returns_df.columns:
        w = weights.get(name, 0.0)
        asset_ret = returns_df[name].values * w
        threshold = np.percentile(asset_ret, 5)
        tail = asset_ret[asset_ret <= threshold]
        standalone_cvar[name] = -float(np.mean(tail))

    plot_risk_waterfall(
        standalone_cvar, attr.total_cvar,
        save_path=os.path.join(IMG_DIR, "risk_waterfall.png"),
    )
    print("  -> risk_waterfall.png")

    plot_marginal_risk(
        attr.marginal_cvar,
        save_path=os.path.join(IMG_DIR, "marginal_risk.png"),
    )
    print("  -> marginal_risk.png")

    plot_factor_attribution(
        attr.factor_contributions, attr.idiosyncratic_risk,
        save_path=os.path.join(IMG_DIR, "factor_attribution.png"),
    )
    print("  -> factor_attribution.png")

    # --- Regime Narratives ---
    print("Generating regime narrative charts...")
    port_ret = returns_df.values @ np.array([weights[c] for c in returns_df.columns])
    narrative = generate_narrative(port_ret, regime_labels, detail_level="detailed")

    plot_regime_summary(
        port_ret, regime_labels, regime_stats=narrative.regime_stats,
        title="Regime Summary Dashboard",
        save_path=os.path.join(IMG_DIR, "regime_summary.png"),
    )
    print("  -> regime_summary.png")

    # Rename regime labels for readability
    regime_names = {"0": "Calm", "1": "Transitional", "2": "Crisis"}
    tm = narrative.transition_matrix.rename(index=regime_names, columns=regime_names)

    plot_regime_transition_matrix(
        tm,
        save_path=os.path.join(IMG_DIR, "regime_transitions.png"),
    )
    print("  -> regime_transitions.png")

    plot_regime_transition_matrix(
        tm,
        save_path=os.path.join(IMG_DIR, "regime_transition_heatmap.png"),
    )
    print("  -> regime_transition_heatmap.png")

    # --- What-If Analysis ---
    print("Generating what-if charts...")
    removal = remove_asset(returns_df, weights, "Crypto")
    stress = stress_correlations(returns_df, weights, stress_factor=2.0)
    capped = cap_weights(returns_df, weights, max_weight=0.25)

    # Comparison chart
    combined = compare_scenarios(
        returns_df, weights,
        {
            "Remove Crypto": removal.scenarios[0].weights,
            "Stressed Corr": weights,  # same weights, different interpretation
            "Capped 25%": capped.scenarios[0].weights,
        },
    )
    plot_whatif_comparison(
        combined.comparison_table,
        save_path=os.path.join(IMG_DIR, "whatif_comparison.png"),
    )
    print("  -> whatif_comparison.png")

    # Tornado chart
    base_m = {
        "cvar_95": removal.base.cvar_95,
        "sharpe": removal.base.sharpe,
        "max_dd": removal.base.max_drawdown,
    }
    scenario_m = {
        "Remove Crypto": {
            "cvar_95": removal.scenarios[0].cvar_95,
            "sharpe": removal.scenarios[0].sharpe,
            "max_dd": removal.scenarios[0].max_drawdown,
        },
        "Stressed Corr (x2)": {
            "cvar_95": stress.scenarios[0].cvar_95,
            "sharpe": stress.scenarios[0].sharpe,
            "max_dd": stress.scenarios[0].max_drawdown,
        },
        "Capped at 25%": {
            "cvar_95": capped.scenarios[0].cvar_95,
            "sharpe": capped.scenarios[0].sharpe,
            "max_dd": capped.scenarios[0].max_drawdown,
        },
    }
    plot_whatif_tornado(
        base_m, scenario_m, metric="cvar_95",
        title="CVaR Impact by Scenario",
        save_path=os.path.join(IMG_DIR, "whatif_tornado.png"),
    )
    print("  -> whatif_tornado.png")

    # Correlation stress
    normal_corr = returns_df.corr()
    stressed_corr = normal_corr.copy()
    for i in range(len(stressed_corr)):
        for j in range(len(stressed_corr)):
            if i != j:
                stressed_corr.iloc[i, j] = np.clip(stressed_corr.iloc[i, j] * 2, -0.99, 0.99)

    plot_correlation_stress(
        normal_corr, stressed_corr,
        save_path=os.path.join(IMG_DIR, "correlation_stress.png"),
    )
    print("  -> correlation_stress.png")

    # --- Audit Trail ---
    print("Generating audit trail chart...")
    prev_weights = {"Equity": 0.35, "Bonds": 0.35, "Gold": 0.20, "Crypto": 0.10}
    plot_weight_changes(
        prev_weights, weights,
        save_path=os.path.join(IMG_DIR, "weight_changes.png"),
    )
    print("  -> weight_changes.png")

    print("\nAll explainability charts generated successfully!")


if __name__ == "__main__":
    main()
