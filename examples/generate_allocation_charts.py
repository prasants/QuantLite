#!/usr/bin/env python3
"""Generate all allocation engine chart PNGs for documentation."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlite.portfolio.tail_risk_parity import (
    cvar_parity_weights, vol_parity_weights, es_parity_weights, compare_parity_methods,
)
from quantlite.portfolio.regime_bl import regime_conditional_bl, black_litterman_posterior
from quantlite.portfolio.dynamic_kelly import (
    fractional_kelly, rolling_kelly, kelly_with_drawdown_control,
)
from quantlite.portfolio.ensemble import ensemble_allocate, consensus_portfolio
from quantlite.portfolio.walkforward import walk_forward, sharpe_score
from quantlite.portfolio.optimisation import risk_parity_weights
from quantlite.viz.allocation import (
    plot_risk_contribution_comparison,
    plot_tail_parity_weights,
    plot_tail_risk_budget,
    plot_regime_bl_weights,
    plot_view_confidence,
    plot_bl_frontier,
    plot_kelly_drawdown_control,
    plot_kelly_fraction_evolution,
    plot_kelly_risk_reward,
    plot_ensemble_agreement,
    plot_ensemble_weights,
    plot_walkforward_folds,
    plot_walkforward_equity,
)

import matplotlib
matplotlib.use("Agg")

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
    regime_labels = _make_regimes(len(returns_df))

    print("Generating tail risk parity charts...")
    # Tail Risk Parity
    vol = vol_parity_weights(returns_df)
    cvar = cvar_parity_weights(returns_df)
    es = es_parity_weights(returns_df)

    plot_risk_contribution_comparison(
        vol.risk_contributions, cvar.risk_contributions,
        save_path=os.path.join(IMG_DIR, "risk_contribution_comparison.png"),
    )
    plot_tail_parity_weights(
        {"Vol Parity": vol.weights, "CVaR Parity": cvar.weights, "ES Parity": es.weights},
        save_path=os.path.join(IMG_DIR, "tail_parity_weights.png"),
    )
    plot_tail_risk_budget(
        cvar.risk_contributions,
        save_path=os.path.join(IMG_DIR, "tail_risk_budget.png"),
    )

    print("Generating regime BL charts...")
    # Regime BL
    caps = {name: 1000.0 for name in returns_df.columns}
    views = {"Equity": 0.10, "Bonds": 0.03, "Gold": 0.05}
    confs = {"Equity": 0.7, "Bonds": 0.5, "Gold": 0.4}

    bl_result = regime_conditional_bl(
        returns_df, regime_labels, caps, views, confs,
        regime_view_adjustments={0: {"Equity": 1.3}, 1: {"Equity": 0.6}, 2: {"Equity": 0.3}},
        regime_confidence_scaling={0: 1.2, 1: 0.8, 2: 0.5},
    )
    regime_names = {0: "Bull", 1: "Bear", 2: "Crisis"}

    plot_regime_bl_weights(
        bl_result.regime_weights, regime_names=regime_names,
        save_path=os.path.join(IMG_DIR, "regime_bl_weights.png"),
    )

    # Confidence over time (synthetic)
    conf_over_time = np.clip(
        0.5 + 0.3 * np.sin(np.linspace(0, 6 * np.pi, len(returns_df))) +
        np.random.RandomState(42).normal(0, 0.05, len(returns_df)),
        0, 1,
    )
    plot_view_confidence(
        conf_over_time, regime_labels, regime_names=regime_names,
        save_path=os.path.join(IMG_DIR, "view_confidence.png"),
    )

    # BL frontier
    regime_portfolios = {}
    for r, name in regime_names.items():
        if r in bl_result.regime_returns:
            mu = bl_result.regime_returns[r]
            cov = bl_result.regime_covariances[r]
            w = np.array([bl_result.regime_weights[r].get(a, 0.0) for a in returns_df.columns])
            ret = float(w @ mu.values)
            risk = float(np.sqrt(w @ cov.values @ w))
            regime_portfolios[name] = (risk, ret)

    plot_bl_frontier(
        returns_df, regime_portfolios,
        save_path=os.path.join(IMG_DIR, "bl_frontier.png"),
    )

    print("Generating Kelly charts...")
    # Kelly
    equity_returns = returns_df["Equity"].values

    full = fractional_kelly(equity_returns, fraction_of_kelly=1.0)
    half = fractional_kelly(equity_returns, fraction_of_kelly=0.5)
    controlled = kelly_with_drawdown_control(
        equity_returns, fraction_of_kelly=0.5,
        max_drawdown_threshold=-0.10, drawdown_reduction=0.25,
    )

    plot_kelly_drawdown_control(
        {"Full Kelly": full.equity_curve, "Half Kelly": half.equity_curve,
         "DD Control": controlled.equity_curve},
        save_path=os.path.join(IMG_DIR, "kelly_drawdown_control.png"),
    )

    kelly_fracs, _ = rolling_kelly(equity_returns, window=126)
    plot_kelly_fraction_evolution(
        kelly_fracs, regime_labels=regime_labels, regime_names=regime_names,
        save_path=os.path.join(IMG_DIR, "kelly_fraction_evolution.png"),
    )

    # Sharpe estimates
    def _sharpe(eq):
        rets = np.diff(eq) / eq[:-1]
        if len(rets) < 2:
            return 0.0
        return float(np.mean(rets) / max(np.std(rets, ddof=1), 1e-12) * np.sqrt(252))

    plot_kelly_risk_reward(
        {
            "Full Kelly": (full.fraction, _sharpe(full.equity_curve)),
            "Half Kelly": (half.fraction, _sharpe(half.equity_curve)),
            "DD Control": (controlled.fraction, _sharpe(controlled.equity_curve)),
        },
        save_path=os.path.join(IMG_DIR, "kelly_risk_reward.png"),
    )

    print("Generating ensemble charts...")
    # Ensemble — use 4+ strategies for an informative agreement matrix
    from quantlite.portfolio.optimisation import hrp_weights, black_litterman

    hrp = hrp_weights(returns_df)
    rp = risk_parity_weights(returns_df)

    # Black-Litterman
    caps = {c: 1e9 for c in returns_df.columns}
    views = {returns_df.columns[0]: 0.08, returns_df.columns[-1]: 0.12}
    confs = {returns_df.columns[0]: 0.7, returns_df.columns[-1]: 0.5}
    bl_posterior, _ = black_litterman(returns_df, caps, views, confs)
    bl_w_raw = bl_posterior.clip(lower=0)
    bl_total = bl_w_raw.sum()
    bl_weights = {c: float(bl_w_raw[c] / bl_total) if bl_total > 0 else 0.25
                  for c in returns_df.columns}

    # Kelly (fractional)
    kelly_res = fractional_kelly(returns_df.mean(axis=1).values, fraction_of_kelly=0.5)
    # Distribute Kelly across assets by Sharpe
    sharpes = returns_df.mean() / returns_df.std()
    sharpes_pos = sharpes.clip(lower=0)
    s_total = sharpes_pos.sum()
    kelly_weights = {c: float(sharpes_pos[c] / s_total) if s_total > 0 else 0.25
                     for c in returns_df.columns}

    strategies_4 = {
        "HRP": hrp.weights,
        "Risk Parity": rp.weights,
        "Black-Litterman": bl_weights,
        "Kelly": kelly_weights,
    }
    ens = ensemble_allocate(returns_df, strategies=strategies_4)
    plot_ensemble_agreement(
        ens.agreement_matrix,
        save_path=os.path.join(IMG_DIR, "ensemble_agreement.png"),
    )
    plot_ensemble_weights(
        ens.blended_weights, ens.strategy_allocations, ens.strategy_weights,
        save_path=os.path.join(IMG_DIR, "ensemble_weights.png"),
    )

    print("Generating walk-forward charts...")
    # Walk-forward
    def simple_opt(df):
        n = df.shape[1]
        return {col: 1.0 / n for col in df.columns}

    wf = walk_forward(returns_df, simple_opt, is_window=252, oos_window=63)
    plot_walkforward_folds(
        wf.folds, total_periods=len(returns_df),
        save_path=os.path.join(IMG_DIR, "walkforward_folds.png"),
    )

    # Naive equity for comparison
    w_naive = np.ones(returns_df.shape[1]) / returns_df.shape[1]
    naive_rets = returns_df.values @ w_naive
    naive_eq = np.cumprod(1.0 + naive_rets)
    naive_eq = np.insert(naive_eq, 0, 1.0)
    # Trim to match WF length
    wf_len = len(wf.equity_curve)
    naive_trimmed = naive_eq[-wf_len:] / naive_eq[-wf_len] if len(naive_eq) >= wf_len else naive_eq

    plot_walkforward_equity(
        wf.equity_curve, naive_trimmed,
        save_path=os.path.join(IMG_DIR, "walkforward_equity.png"),
    )

    print(f"All charts saved to {IMG_DIR}")


if __name__ == "__main__":
    main()
