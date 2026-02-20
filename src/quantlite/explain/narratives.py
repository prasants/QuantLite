"""Regime narratives: auto-generate human-readable explanations of regime detection.

Produces structured text at configurable detail levels (brief, standard,
detailed) describing identified regimes, their characteristics, and
transition dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "RegimeStats",
    "RegimeNarrative",
    "generate_narrative",
    "transition_narrative",
]


@dataclass(frozen=True)
class RegimeStats:
    """Statistics for a single regime.

    Attributes:
        label: Regime identifier.
        name: Human-readable regime name.
        count: Number of observations in this regime.
        proportion: Fraction of total observations.
        mean_return: Mean return during this regime.
        volatility: Annualised volatility during this regime.
        mean_duration: Average consecutive duration (in periods).
        sharpe: Annualised Sharpe ratio during this regime.
    """

    label: int
    name: str
    count: int
    proportion: float
    mean_return: float
    volatility: float
    mean_duration: float
    sharpe: float


@dataclass(frozen=True)
class RegimeNarrative:
    """Container for regime narrative output.

    Attributes:
        summary: The generated narrative text.
        regime_stats: Per-regime statistics.
        transition_matrix: Regime transition probability matrix.
        current_regime: Label of the most recent regime.
        detail_level: Detail level used to generate the narrative.
    """

    summary: str
    regime_stats: list[RegimeStats]
    transition_matrix: pd.DataFrame
    current_regime: int
    detail_level: str


def _classify_regime(
    mean_ret: float, vol: float, all_means: list[float], all_vols: list[float]
) -> str:
    """Assign a descriptive name based on return/volatility characteristics."""
    n = len(all_means)
    mean_rank = sorted(all_means).index(mean_ret)
    vol_rank = sorted(all_vols).index(vol)

    if n <= 2:
        if vol_rank == 0:
            return "low-volatility" if mean_rank >= vol_rank else "calm"
        return "high-volatility" if mean_rank <= vol_rank else "turbulent"

    # Three or more regimes
    if vol_rank == 0 and mean_rank >= n // 2:
        return "low-volatility bull market"
    if vol_rank == n - 1:
        return "crisis regime"
    if vol_rank == 0:
        return "calm period"
    return "transitional period"


def _compute_durations(labels: np.ndarray) -> dict[int, list[int]]:
    """Compute consecutive run lengths per regime."""
    durations = {}  # type: Dict[int, List[int]]
    if len(labels) == 0:
        return durations

    current = labels[0]
    run = 1
    for i in range(1, len(labels)):
        if labels[i] == current:
            run += 1
        else:
            durations.setdefault(int(current), []).append(run)
            current = labels[i]
            run = 1
    durations.setdefault(int(current), []).append(run)
    return durations


def _compute_transition_matrix(labels: np.ndarray) -> pd.DataFrame:
    """Compute transition probability matrix from regime labels."""
    unique = sorted(set(labels))
    n = len(unique)
    label_to_idx = {lbl: i for i, lbl in enumerate(unique)}
    counts = np.zeros((n, n))

    for i in range(len(labels) - 1):
        fr = label_to_idx[labels[i]]
        to = label_to_idx[labels[i + 1]]
        counts[fr, to] += 1

    # Normalise rows
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs = counts / row_sums

    idx = [str(u) for u in unique]
    return pd.DataFrame(probs, index=idx, columns=idx)


def generate_narrative(
    returns: pd.Series | np.ndarray,
    regime_labels: np.ndarray,
    detail_level: str = "standard",
    dates: pd.DatetimeIndex | None = None,
    freq: int = 252,
) -> RegimeNarrative:
    """Generate a human-readable narrative explaining regime detection results.

    Args:
        returns: Array or Series of portfolio/asset returns.
        regime_labels: Array of integer regime labels (same length as returns).
        detail_level: One of ``"brief"``, ``"standard"``, or ``"detailed"``.
        dates: Optional datetime index for date-aware narratives.
        freq: Annualisation frequency (default 252 for daily).

    Returns:
        ``RegimeNarrative`` with summary text and supporting statistics.

    Raises:
        ValueError: If returns and regime_labels have different lengths.

    Example:
        >>> import numpy as np
        >>> returns = np.random.randn(500) * 0.01
        >>> labels = np.array([0]*200 + [1]*150 + [2]*100 + [0]*50)
        >>> narrative = generate_narrative(returns, labels)
        >>> print(narrative.summary)
    """
    returns_arr = returns.values if isinstance(returns, pd.Series) else np.asarray(returns)

    labels = np.asarray(regime_labels)
    if len(returns_arr) != len(labels):
        raise ValueError(
            f"returns ({len(returns_arr)}) and regime_labels ({len(labels)}) "
            f"must have the same length."
        )

    unique_regimes = sorted(set(labels))
    n_total = len(labels)
    durations = _compute_durations(labels)
    trans_matrix = _compute_transition_matrix(labels)

    # Compute per-regime stats
    all_means = []
    all_vols = []
    raw_stats = []

    for regime in unique_regimes:
        mask = labels == regime
        r = returns_arr[mask]
        mean_r = float(np.mean(r))
        vol = float(np.std(r)) * np.sqrt(freq)
        all_means.append(mean_r)
        all_vols.append(vol)
        raw_stats.append((regime, mask.sum(), mean_r, vol, r))

    regime_stats_list = []
    for regime, count, mean_r, vol, r in raw_stats:
        name = _classify_regime(mean_r, vol, all_means, all_vols)
        mean_dur = float(np.mean(durations.get(int(regime), [1])))
        ann_ret = mean_r * freq
        sharpe = ann_ret / vol if vol > 0 else 0.0
        regime_stats_list.append(
            RegimeStats(
                label=int(regime),
                name=name,
                count=count,
                proportion=count / n_total,
                mean_return=mean_r,
                volatility=vol,
                mean_duration=mean_dur,
                sharpe=sharpe,
            )
        )

    current_regime = int(labels[-1])

    # Build narrative
    summary = _build_narrative(
        regime_stats_list, trans_matrix, current_regime, detail_level, dates, labels
    )

    return RegimeNarrative(
        summary=summary,
        regime_stats=regime_stats_list,
        transition_matrix=trans_matrix,
        current_regime=current_regime,
        detail_level=detail_level,
    )


def _build_narrative(
    stats: list[RegimeStats],
    trans: pd.DataFrame,
    current: int,
    level: str,
    dates: pd.DatetimeIndex | None,
    labels: np.ndarray,
) -> str:
    """Build narrative text at the specified detail level."""
    n_regimes = len(stats)
    current_stat = next(s for s in stats if s.label == current)

    # Brief
    regime_desc = ", ".join(
        f"a {s.name} ({s.proportion:.0%} of observations)" for s in stats
    )
    brief = (
        f"The model identified {n_regimes} regimes: {regime_desc}. "
        f"The current regime is {current_stat.name}."
    )

    if level == "brief":
        return brief

    # Standard: add stats
    stats_lines = []
    for s in stats:
        stats_lines.append(
            f"Regime {s.label} ({s.name}): mean daily return {s.mean_return:.4%}, "
            f"annualised volatility {s.volatility:.1%}, "
            f"average duration {s.mean_duration:.0f} periods, "
            f"Sharpe ratio {s.sharpe:.2f}."
        )

    standard = brief + "\n\n" + "\n".join(stats_lines)

    if level == "standard":
        return standard

    # Detailed: add transition info and date context
    trans_lines = ["\nTransition probabilities:"]
    for i, s in enumerate(stats):
        row = trans.iloc[i]
        transitions = ", ".join(
            f"{row.index[j]}: {row.iloc[j]:.1%}"
            for j in range(len(row))
            if row.iloc[j] > 0.005
        )
        trans_lines.append(f"  From regime {s.label}: {transitions}")

    detailed = standard + "\n" + "\n".join(trans_lines)

    # Add date-based transition narrative if dates available
    if dates is not None and len(dates) == len(labels):
        shift_lines = ["\nRegime transitions:"]
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                from_s = next(s for s in stats if s.label == labels[i - 1])
                to_s = next(s for s in stats if s.label == labels[i])
                shift_lines.append(
                    f"  {dates[i].strftime('%Y-%m-%d')}: "
                    f"{from_s.name} -> {to_s.name}"
                )
                if len(shift_lines) > 12:
                    shift_lines.append("  ... and more transitions")
                    break
        detailed += "\n" + "\n".join(shift_lines)

    return detailed


def transition_narrative(
    regime_labels: np.ndarray,
    dates: pd.DatetimeIndex | None = None,
    regime_names: dict[int, str] | None = None,
) -> str:
    """Generate a narrative focused on regime transitions.

    Args:
        regime_labels: Array of regime labels.
        dates: Optional datetime index.
        regime_names: Optional mapping of regime label to name.

    Returns:
        Human-readable string describing regime transitions.
    """
    labels = np.asarray(regime_labels)
    durations = _compute_durations(labels)

    if regime_names is None:
        regime_names = {int(lbl): f"Regime {lbl}" for lbl in sorted(set(labels))}

    transitions = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            from_name = regime_names.get(int(labels[i - 1]), f"Regime {labels[i - 1]}")
            to_name = regime_names.get(int(labels[i]), f"Regime {labels[i]}")
            date_str = (
                dates[i].strftime("%Y-%m-%d") if dates is not None else f"period {i}"
            )
            # Historical duration of the destination regime
            dest_durs = durations.get(int(labels[i]), [])
            avg_dur = np.mean(dest_durs) if dest_durs else 0

            transitions.append(
                f"Regime shifted from {from_name} to {to_name} on {date_str}. "
                f"Historical {to_name} periods lasted an average of {avg_dur:.0f} days."
            )

    if not transitions:
        return "No regime transitions detected in the sample period."

    return "\n".join(transitions)
