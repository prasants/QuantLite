"""Allocation engine visualisations: tail risk, regime BL, Kelly, ensemble, walk-forward.

All charts follow Stephen Few principles: muted palette, maximum
data-ink ratio, direct labels, horizontal gridlines only, no chartjunk.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .theme import FEW_PALETTE, apply_few_theme

__all__ = [
    "plot_risk_contribution_comparison",
    "plot_tail_parity_weights",
    "plot_tail_risk_budget",
    "plot_regime_bl_weights",
    "plot_view_confidence",
    "plot_bl_frontier",
    "plot_kelly_drawdown_control",
    "plot_kelly_fraction_evolution",
    "plot_kelly_risk_reward",
    "plot_ensemble_agreement",
    "plot_ensemble_weights",
    "plot_walkforward_folds",
    "plot_walkforward_equity",
]

_COLOURS = [
    FEW_PALETTE["primary"],
    FEW_PALETTE["secondary"],
    FEW_PALETTE["negative"],
    FEW_PALETTE["positive"],
    FEW_PALETTE["neutral"],
    FEW_PALETTE["grey_mid"],
]


def _save_or_show(fig: Figure, save_path: str | None) -> Figure:
    """Save to file or display."""
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Tail Risk Parity
# ---------------------------------------------------------------------------

def plot_risk_contribution_comparison(
    vol_contributions: dict[str, float],
    cvar_contributions: dict[str, float],
    title: str = "Risk Contributions: Volatility Parity vs CVaR Parity",
    save_path: str | None = None,
) -> Figure:
    """Side-by-side grouped bars of vol parity vs CVaR parity risk contributions.

    Args:
        vol_contributions: Per-asset volatility risk contributions.
        cvar_contributions: Per-asset CVaR risk contributions.
        title: Chart title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=(12, 6))

    assets = list(vol_contributions.keys())
    n = len(assets)
    x = np.arange(n)
    width = 0.35

    # Normalise to percentages
    vol_total = sum(abs(v) for v in vol_contributions.values())
    cvar_total = sum(abs(v) for v in cvar_contributions.values())

    vol_pcts = [abs(vol_contributions[a]) / vol_total * 100 if vol_total > 0 else 0 for a in assets]
    cvar_pcts = [abs(cvar_contributions[a]) / cvar_total * 100 if cvar_total > 0 else 0 for a in assets]

    bars1 = ax.bar(x - width / 2, vol_pcts, width, color=FEW_PALETTE["primary"], label="Vol Parity")
    bars2 = ax.bar(x + width / 2, cvar_pcts, width, color=FEW_PALETTE["secondary"], label="CVaR Parity")

    # Direct labels
    for bar, val in zip(bars1, vol_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
                color=FEW_PALETTE["primary"])
    for bar, val in zip(bars2, cvar_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
                color=FEW_PALETTE["secondary"])

    # Equal line
    target = 100.0 / n
    ax.axhline(y=target, color=FEW_PALETTE["grey_mid"], linestyle="--", linewidth=1,
               alpha=0.7)
    ax.text(n - 0.5, target + 0.5, f"Equal = {target:.1f}%", fontsize=9,
            color=FEW_PALETTE["grey_mid"], ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(assets)
    ax.set_ylabel("Risk Contribution (%)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    return _save_or_show(fig, save_path)


def plot_tail_parity_weights(
    weight_sets: dict[str, dict[str, float]],
    title: str = "Portfolio Weights: Parity Method Comparison",
    save_path: str | None = None,
) -> Figure:
    """Grouped bar chart comparing weights across parity methods.

    Args:
        weight_sets: Dict mapping method name to weight dict.
        title: Chart title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = list(weight_sets.keys())
    assets = list(list(weight_sets.values())[0].keys())
    n_methods = len(methods)
    n_assets = len(assets)
    x = np.arange(n_assets)
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        vals = [weight_sets[method].get(a, 0.0) * 100 for a in assets]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, color=_COLOURS[i % len(_COLOURS)],
                       label=method)
        for bar, val in zip(bars, vals):
            if val > 2:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
                        color=_COLOURS[i % len(_COLOURS)])

    ax.set_xticks(x)
    ax.set_xticklabels(assets)
    ax.set_ylabel("Weight (%)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    return _save_or_show(fig, save_path)


def plot_tail_risk_budget(
    cvar_contributions: dict[str, float],
    title: str = "CVaR Risk Budget Allocation",
    save_path: str | None = None,
) -> Figure:
    """Horizontal stacked bar showing CVaR budget allocation.

    Args:
        cvar_contributions: Per-asset CVaR contributions (absolute values used).
        title: Chart title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=(12, 3))

    assets = list(cvar_contributions.keys())
    vals = [abs(v) for v in cvar_contributions.values()]
    total = sum(vals)
    pcts = [v / total * 100 if total > 0 else 0 for v in vals]

    left = 0.0
    for i, (asset, pct) in enumerate(zip(assets, pcts)):
        colour = _COLOURS[i % len(_COLOURS)]
        ax.barh(0, pct, left=left, height=0.5, color=colour)
        if pct > 3:
            ax.text(left + pct / 2, 0, f"{asset}\n{pct:.1f}%",
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold")
        left += pct

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("CVaR Budget (%)")
    ax.set_title(title)

    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Regime Black-Litterman
# ---------------------------------------------------------------------------

def plot_regime_bl_weights(
    regime_weights: dict[int, dict[str, float]],
    regime_names: dict[int, str] | None = None,
    title: str = "Black-Litterman Weights by Regime",
    save_path: str | None = None,
) -> Figure:
    """Grouped bars showing how BL weights shift across regimes.

    Args:
        regime_weights: Per-regime weight dictionaries.
        regime_names: Optional human-readable regime names.
        title: Chart title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    if regime_names is None:
        regime_names = {r: f"Regime {r}" for r in regime_weights}

    methods = {regime_names[r]: w for r, w in regime_weights.items()}
    return plot_tail_parity_weights(methods, title=title, save_path=save_path)


def plot_view_confidence(
    confidence_over_time: np.ndarray,
    regime_labels: np.ndarray,
    regime_names: dict[int, str] | None = None,
    title: str = "View Confidence and Regime State",
    save_path: str | None = None,
) -> Figure:
    """Line chart of regime confidence over time with regime shading.

    Args:
        confidence_over_time: Array of confidence values (0 to 1).
        regime_labels: Array of integer regime labels.
        regime_names: Optional human-readable regime names.
        title: Chart title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=(12, 6))

    t = np.arange(len(confidence_over_time))
    ax.plot(t, confidence_over_time, color=FEW_PALETTE["primary"], linewidth=1.5)

    # Regime shading
    unique_regimes = sorted(set(regime_labels.tolist()))
    regime_colours = [FEW_PALETTE["positive"], FEW_PALETTE["secondary"],
                      FEW_PALETTE["negative"], FEW_PALETTE["neutral"]]
    if regime_names is None:
        regime_names = {r: f"Regime {r}" for r in unique_regimes}

    for i, regime in enumerate(unique_regimes):
        mask = regime_labels == regime
        colour = regime_colours[i % len(regime_colours)]
        starts = np.where(np.diff(np.concatenate(([0], mask.astype(int), [0]))) == 1)[0]
        ends = np.where(np.diff(np.concatenate(([0], mask.astype(int), [0]))) == -1)[0]
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, alpha=0.1, color=colour)
        # One label in legend
        ax.fill_between([], [], [], alpha=0.2, color=colour,
                        label=regime_names[regime])

    ax.set_xlabel("Time")
    ax.set_ylabel("View Confidence")
    ax.set_title(title)
    ax.legend(loc="upper right")

    return _save_or_show(fig, save_path)


def plot_bl_frontier(
    returns_df: pd.DataFrame,
    regime_portfolios: dict[str, tuple[float, float]],
    n_points: int = 100,
    title: str = "Efficient Frontier with Regime BL Portfolios",
    save_path: str | None = None,
) -> Figure:
    """Efficient frontier with regime-conditional BL portfolios marked.

    Args:
        returns_df: DataFrame of asset returns.
        regime_portfolios: Dict mapping label to (risk, return) tuples.
        n_points: Number of frontier points.
        title: Chart title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Compute simple frontier via random portfolios
    mu = returns_df.mean().values * 252
    cov = returns_df.cov().values * 252
    n_assets = len(mu)

    risks, rets = [], []
    rng = np.random.RandomState(42)
    for _ in range(5000):
        w = rng.dirichlet(np.ones(n_assets))
        ret = float(w @ mu)
        risk = float(np.sqrt(w @ cov @ w))
        risks.append(risk)
        rets.append(ret)

    ax.scatter(risks, rets, s=1, color=FEW_PALETTE["grey_light"], alpha=0.5, rasterized=True)

    # Mark regime portfolios
    for i, (label, (risk, ret)) in enumerate(regime_portfolios.items()):
        colour = _COLOURS[i % len(_COLOURS)]
        ax.scatter(risk, ret, s=120, color=colour, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(label, (risk, ret), textcoords="offset points",
                    xytext=(8, 8), fontsize=10, color=colour, fontweight="bold")

    ax.set_xlabel("Annualised Risk (Volatility)")
    ax.set_ylabel("Annualised Return")
    ax.set_title(title)

    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Dynamic Kelly
# ---------------------------------------------------------------------------

def plot_kelly_drawdown_control(
    equity_curves: dict[str, np.ndarray],
    drawdown_threshold: float | None = None,
    title: str = "Kelly Criterion: Equity Curves with Drawdown Control",
    save_path: str | None = None,
) -> Figure:
    """Equity curves comparing Kelly variants with drawdown circuit breaker.

    Args:
        equity_curves: Dict mapping strategy name to equity array.
        drawdown_threshold: If provided, draw a reference line.
        title: Chart title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect end values for staggered label placement
    end_values = []
    for i, (name, equity) in enumerate(equity_curves.items()):
        colour = _COLOURS[i % len(_COLOURS)]
        ax.plot(equity, color=colour, linewidth=1.5)
        end_values.append((equity[-1], name, colour, len(equity) - 1))

    # Stagger labels so they don't overlap
    end_values.sort(key=lambda x: x[0])
    min_gap = (max(v[0] for v in end_values) - min(v[0] for v in end_values)) * 0.05
    if min_gap < 0.01:
        min_gap = 0.05
    placed: list[float] = []
    for val, name, colour, x_pos in end_values:
        y = val
        for yp in placed:
            if abs(y - yp) < min_gap:
                y = yp + min_gap
        placed.append(y)
        ax.text(x_pos, y, f"  {name}",
                color=colour, fontsize=10, va="center")

    ax.set_xlabel("Period")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title(title)

    return _save_or_show(fig, save_path)


def plot_kelly_fraction_evolution(
    kelly_fractions: np.ndarray,
    regime_labels: np.ndarray | None = None,
    regime_names: dict[int, str] | None = None,
    title: str = "Rolling Kelly Fraction Over Time",
    save_path: str | None = None,
) -> Figure:
    """Rolling Kelly fraction with optional regime shading.

    Args:
        kelly_fractions: Array of Kelly fractions over time.
        regime_labels: Optional array of regime labels for shading.
        regime_names: Optional human-readable regime names.
        title: Chart title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=(12, 6))

    t = np.arange(len(kelly_fractions))
    ax.plot(t, kelly_fractions, color=FEW_PALETTE["primary"], linewidth=1.5)

    if regime_labels is not None:
        unique_regimes = sorted(set(regime_labels.tolist()))
        regime_colours = [FEW_PALETTE["positive"], FEW_PALETTE["secondary"],
                          FEW_PALETTE["negative"]]
        if regime_names is None:
            regime_names = {r: f"Regime {r}" for r in unique_regimes}
        for i, regime in enumerate(unique_regimes):
            mask = regime_labels == regime
            colour = regime_colours[i % len(regime_colours)]
            starts = np.where(np.diff(np.concatenate(([0], mask.astype(int), [0]))) == 1)[0]
            ends = np.where(np.diff(np.concatenate(([0], mask.astype(int), [0]))) == -1)[0]
            for s, e in zip(starts, ends):
                ax.axvspan(s, e, alpha=0.1, color=colour)
            ax.fill_between([], [], [], alpha=0.2, color=colour,
                            label=regime_names[regime])
        ax.legend(loc="upper right")

    ax.set_xlabel("Period")
    ax.set_ylabel("Kelly Fraction")
    ax.set_title(title)

    return _save_or_show(fig, save_path)


def plot_kelly_risk_reward(
    strategies: dict[str, tuple[float, float]],
    title: str = "Kelly Fraction vs Realised Sharpe",
    save_path: str | None = None,
) -> Figure:
    """Scatter of strategies showing Kelly fraction vs realised Sharpe.

    Args:
        strategies: Dict mapping name to (kelly_fraction, sharpe) tuples.
        title: Chart title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (name, (kelly_f, sharpe)) in enumerate(strategies.items()):
        colour = _COLOURS[i % len(_COLOURS)]
        ax.scatter(kelly_f, sharpe, s=150, color=colour, zorder=5,
                   edgecolors="white", linewidth=1.5)
        ax.annotate(name, (kelly_f, sharpe), textcoords="offset points",
                    xytext=(8, 8), fontsize=10, color=colour)

    ax.set_xlabel("Kelly Fraction")
    ax.set_ylabel("Realised Sharpe Ratio")
    ax.set_title(title)

    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def plot_ensemble_agreement(
    agreement_matrix: pd.DataFrame,
    title: str = "Strategy Agreement Matrix",
    save_path: str | None = None,
) -> Figure:
    """Heatmap showing pairwise agreement between strategies.

    Args:
        agreement_matrix: DataFrame of pairwise correlations.
        title: Chart title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=(10, 8))

    n = len(agreement_matrix)
    # Custom diverging colourmap using Few palette (blue-white-orange)
    from matplotlib.colors import LinearSegmentedColormap
    _few_div = LinearSegmentedColormap.from_list(
        "few_div",
        [FEW_PALETTE["primary"], "#FFFFFF", FEW_PALETTE["secondary"]],
    )
    im = ax.imshow(agreement_matrix.values, cmap=_few_div, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agreement_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(agreement_matrix.index)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = agreement_matrix.values[i, j]
            colour = "white" if abs(val) > 0.5 else FEW_PALETTE["grey_dark"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=colour)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")

    return _save_or_show(fig, save_path)


def plot_ensemble_weights(
    blended_weights: dict[str, float],
    strategy_allocations: dict[str, dict[str, float]],
    strategy_weights: dict[str, float],
    title: str = "Ensemble Blended Weights with Strategy Contributions",
    save_path: str | None = None,
) -> Figure:
    """Stacked bar showing blended weights with per-strategy contributions.

    Args:
        blended_weights: Final blended weights.
        strategy_allocations: Per-strategy weight dicts.
        strategy_weights: Weight given to each strategy in the blend.
        title: Chart title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=(12, 6))

    assets = list(blended_weights.keys())
    strategies = list(strategy_allocations.keys())
    n_assets = len(assets)
    x = np.arange(n_assets)

    bottom = np.zeros(n_assets)
    for i, strat in enumerate(strategies):
        sw = strategy_weights.get(strat, 0.0)
        vals = [strategy_allocations[strat].get(a, 0.0) * sw * 100 for a in assets]
        colour = _COLOURS[i % len(_COLOURS)]
        ax.bar(x, vals, bottom=bottom, color=colour, label=strat, width=0.6)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(assets)
    ax.set_ylabel("Blended Weight (%)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------

def plot_walkforward_folds(
    folds: list,
    total_periods: int,
    title: str = "Walk-Forward Folds: In-Sample and Out-of-Sample Windows",
    save_path: str | None = None,
) -> Figure:
    """Timeline showing IS/OOS windows with per-fold Sharpe.

    Args:
        folds: List of ``WalkForwardFold`` objects.
        total_periods: Total number of periods in the dataset.
        title: Chart title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=(14, max(4, len(folds) * 0.6 + 1)))

    for fold in folds:
        y = fold.fold_index
        # In-sample bar
        ax.barh(y, fold.is_end - fold.is_start, left=fold.is_start,
                height=0.4, color=FEW_PALETTE["primary"], alpha=0.6)
        # Out-of-sample bar
        ax.barh(y, fold.oos_end - fold.oos_start, left=fold.oos_start,
                height=0.4, color=FEW_PALETTE["secondary"], alpha=0.8)
        # Score label
        ax.text(fold.oos_end + 2, y, f"Sharpe: {fold.oos_score:.2f}",
                va="center", fontsize=9, color=FEW_PALETTE["grey_dark"])

    ax.set_xlabel("Period")
    ax.set_ylabel("Fold")
    ax.set_yticks(range(len(folds)))
    ax.set_title(title)

    # Legend
    ax.barh([], [], color=FEW_PALETTE["primary"], alpha=0.6, label="In-Sample")
    ax.barh([], [], color=FEW_PALETTE["secondary"], alpha=0.8, label="Out-of-Sample")
    ax.legend(loc="lower right")

    return _save_or_show(fig, save_path)


def plot_walkforward_equity(
    wf_equity: np.ndarray,
    naive_equity: np.ndarray | None = None,
    title: str = "Walk-Forward vs Naive Backtest Equity Curve",
    save_path: str | None = None,
) -> Figure:
    """Cumulative equity curve from walk-forward vs naive backtest.

    Args:
        wf_equity: Walk-forward out-of-sample equity curve.
        naive_equity: Optional naive (in-sample) equity curve.
        title: Chart title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(wf_equity, color=FEW_PALETTE["primary"], linewidth=1.5)
    ax.text(len(wf_equity) - 1, wf_equity[-1], "  Walk-Forward",
            color=FEW_PALETTE["primary"], fontsize=10, va="center")

    if naive_equity is not None:
        ax.plot(naive_equity, color=FEW_PALETTE["grey_mid"], linewidth=1.2, linestyle="--")
        ax.text(len(naive_equity) - 1, naive_equity[-1], "  Naive",
                color=FEW_PALETTE["grey_mid"], fontsize=10, va="center")

    ax.set_xlabel("Period")
    ax.set_ylabel("Portfolio Value")
    ax.set_title(title)

    return _save_or_show(fig, save_path)
