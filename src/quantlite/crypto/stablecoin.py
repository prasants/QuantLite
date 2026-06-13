"""Stablecoin risk analysis: depeg probability, deviation tracking, and reserve scoring.

Tools for assessing the stability and risk profile of stablecoins,
including historical depeg analysis, recovery time estimation, and
reserve composition quality scoring.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "HISTORICAL_DEPEGS",
    "depeg_probability",
    "depeg_recovery_time",
    "peg_deviation_tracker",
    "reserve_risk_score",
]

# Notable historical depeg events with dates, magnitude, and recovery time
HISTORICAL_DEPEGS = {
    "UST": {
        "name": "TerraUSD (UST)",
        "date": "2022-05-09",
        "trough_price": 0.044,
        "magnitude": 0.956,
        "recovery_days": None,  # Never recovered
        "cause": "Algorithmic design failure and death spiral",
        "notes": "Complete collapse; $40B+ in value destroyed",
    },
    "USDC_2023": {
        "name": "USD Coin (USDC)",
        "date": "2023-03-10",
        "trough_price": 0.878,
        "magnitude": 0.122,
        "recovery_days": 3,
        "cause": "Silicon Valley Bank exposure ($3.3B reserves)",
        "notes": "Recovered after Fed backstop announcement",
    },
    "DAI_2020": {
        "name": "Dai (DAI)",
        "date": "2020-03-12",
        "trough_price": 1.11,
        "magnitude": 0.11,
        "recovery_days": 5,
        "cause": "Black Thursday ETH crash; liquidation cascade",
        "notes": "Depegged upward due to flight to safety within DeFi",
    },
    "USDT_2022": {
        "name": "Tether (USDT)",
        "date": "2022-05-12",
        "trough_price": 0.9485,
        "magnitude": 0.0515,
        "recovery_days": 2,
        "cause": "Contagion from UST collapse; market panic",
        "notes": "Brief depeg driven by fear, not reserve issues",
    },
}

# Reserve quality weights by asset type (higher = safer)
_RESERVE_WEIGHTS = {
    "cash": 1.0,
    "treasuries": 0.95,
    "money_market": 0.85,
    "bonds": 0.75,
    "commercial_paper": 0.60,
    "secured_loans": 0.50,
    "crypto": 0.25,
    "other": 0.15,
}


def depeg_probability(price_series, threshold=0.005):
    """Estimate the probability of a peg breach using historical deviations.

    Computes the fraction of observations where the absolute deviation
    from 1.0 exceeds the given threshold, plus a parametric estimate
    using fitted normal distribution on deviations.

    Parameters
    ----------
    price_series : array-like
        Historical price observations for the stablecoin.
    threshold : float, optional
        Deviation threshold defining a depeg event (default 0.005).

    Returns
    -------
    dict
        Dictionary with keys:
        - empirical_prob: fraction of observations exceeding threshold
        - parametric_prob: probability from fitted normal distribution
        - n_breaches: count of threshold breaches
        - n_observations: total observations
        - mean_deviation: mean absolute deviation from peg
        - max_deviation: maximum absolute deviation from peg
        - risk_rating: qualitative risk rating
    """
    prices = np.asarray(price_series, dtype=float)
    deviations = np.abs(prices - 1.0)

    n_obs = len(deviations)
    n_breaches = int(np.sum(deviations > threshold))
    empirical_prob = n_breaches / n_obs if n_obs > 0 else 0.0

    mean_dev = float(np.mean(deviations))
    std_dev = float(np.std(deviations))
    max_dev = float(np.max(deviations))

    # Parametric estimate using half-normal approximation
    if std_dev > 0:
        from scipy import stats

        # Use the distribution of absolute deviations
        z_score = (threshold - mean_dev) / std_dev
        parametric_prob = float(1.0 - stats.norm.cdf(z_score))
    else:
        parametric_prob = 0.0

    # Risk rating
    if empirical_prob > 0.10 or max_dev > 0.05:
        risk_rating = "critical"
    elif empirical_prob > 0.05 or max_dev > 0.02:
        risk_rating = "high"
    elif empirical_prob > 0.01 or max_dev > 0.01:
        risk_rating = "medium"
    else:
        risk_rating = "low"

    return {
        "empirical_prob": empirical_prob,
        "parametric_prob": parametric_prob,
        "n_breaches": n_breaches,
        "n_observations": n_obs,
        "mean_deviation": mean_dev,
        "max_deviation": max_dev,
        "risk_rating": risk_rating,
    }


def peg_deviation_tracker(price_series, peg=1.0):
    """Compute rolling deviation statistics from the peg.

    Analyses the price series for deviations from the target peg,
    identifying excursions and computing summary statistics.

    Parameters
    ----------
    price_series : array-like
        Historical price observations.
    peg : float, optional
        Target peg value (default 1.0).

    Returns
    -------
    dict
        Dictionary with keys:
        - deviations: array of signed deviations from peg
        - abs_deviations: array of absolute deviations
        - mean_deviation: mean absolute deviation
        - max_deviation: maximum absolute deviation
        - std_deviation: standard deviation of deviations
        - excursions: list of dicts describing each excursion period
        - longest_excursion: duration of longest excursion (in periods)
        - pct_time_off_peg: percentage of time deviation > 0.001
    """
    prices = np.asarray(price_series, dtype=float)
    deviations = prices - peg
    abs_devs = np.abs(deviations)

    # Identify excursions (periods where abs deviation > 0.001)
    off_peg_threshold = 0.001
    off_peg = abs_devs > off_peg_threshold

    excursions = []
    in_excursion = False
    start_idx = 0

    for i in range(len(off_peg)):
        if off_peg[i] and not in_excursion:
            in_excursion = True
            start_idx = i
        elif not off_peg[i] and in_excursion:
            in_excursion = False
            excursion_devs = abs_devs[start_idx:i]
            excursions.append({
                "start": int(start_idx),
                "end": int(i - 1),
                "duration": int(i - start_idx),
                "max_deviation": float(np.max(excursion_devs)),
                "mean_deviation": float(np.mean(excursion_devs)),
                "direction": "above" if deviations[start_idx] > 0 else "below",
            })

    # Handle excursion that extends to end of series
    if in_excursion:
        excursion_devs = abs_devs[start_idx:]
        excursions.append({
            "start": int(start_idx),
            "end": int(len(prices) - 1),
            "duration": int(len(prices) - start_idx),
            "max_deviation": float(np.max(excursion_devs)),
            "mean_deviation": float(np.mean(excursion_devs)),
            "direction": "above" if deviations[start_idx] > 0 else "below",
        })

    longest = max((e["duration"] for e in excursions), default=0)
    pct_off = float(np.sum(off_peg)) / len(prices) * 100 if len(prices) > 0 else 0.0

    return {
        "deviations": deviations,
        "abs_deviations": abs_devs,
        "mean_deviation": float(np.mean(abs_devs)),
        "max_deviation": float(np.max(abs_devs)),
        "std_deviation": float(np.std(deviations)),
        "excursions": excursions,
        "longest_excursion": longest,
        "pct_time_off_peg": pct_off,
    }


def depeg_recovery_time(price_series, threshold=0.005, peg=1.0):
    """Estimate time to recover from depeg events.

    Identifies depeg events (where absolute deviation from peg exceeds
    threshold) and measures the number of periods until the price
    returns within threshold of the peg.

    Parameters
    ----------
    price_series : array-like
        Historical price observations.
    threshold : float, optional
        Deviation threshold defining a depeg event (default 0.005).
    peg : float, optional
        Target peg value (default 1.0).

    Returns
    -------
    dict
        Dictionary with keys:
        - events: list of dicts with start, end, duration, max_deviation
        - mean_recovery_time: average recovery time across events
        - max_recovery_time: longest recovery time
        - unrecovered: number of events that never recovered
        - total_events: total depeg events detected
    """
    prices = np.asarray(price_series, dtype=float)
    abs_devs = np.abs(prices - peg)
    depegged = abs_devs > threshold

    events = []
    in_depeg = False
    start_idx = 0

    for i in range(len(depegged)):
        if depegged[i] and not in_depeg:
            in_depeg = True
            start_idx = i
        elif not depegged[i] and in_depeg:
            in_depeg = False
            events.append({
                "start": int(start_idx),
                "end": int(i),
                "duration": int(i - start_idx),
                "max_deviation": float(np.max(abs_devs[start_idx:i])),
                "recovered": True,
            })

    # Handle ongoing depeg at end of series
    if in_depeg:
        events.append({
            "start": int(start_idx),
            "end": None,
            "duration": int(len(prices) - start_idx),
            "max_deviation": float(np.max(abs_devs[start_idx:])),
            "recovered": False,
        })

    recovered_events = [e for e in events if e["recovered"]]
    unrecovered = sum(1 for e in events if not e["recovered"])

    if recovered_events:
        durations = [e["duration"] for e in recovered_events]
        mean_recovery = float(np.mean(durations))
        max_recovery = int(np.max(durations))
    else:
        mean_recovery = 0.0
        max_recovery = 0

    return {
        "events": events,
        "mean_recovery_time": mean_recovery,
        "max_recovery_time": max_recovery,
        "unrecovered": unrecovered,
        "total_events": len(events),
    }


def reserve_risk_score(reserve_composition):
    """Score reserve quality based on asset type composition.

    Higher scores indicate safer, more liquid reserves. The score
    is a weighted average of asset quality weights, where each
    asset type has a predefined safety weight.

    Parameters
    ----------
    reserve_composition : dict
        Mapping of asset type to percentage allocation.
        Keys should be from: cash, treasuries, money_market, bonds,
        commercial_paper, secured_loans, crypto, other.
        Values should sum to approximately 100.

    Returns
    -------
    dict
        Dictionary with keys:
        - score: float in [0, 1], higher is safer
        - rating: qualitative rating (excellent, good, fair, poor, critical)
        - breakdown: dict of asset type contributions to score
        - total_allocation: sum of input allocations
        - warnings: list of risk warnings
    """
    total_alloc = sum(reserve_composition.values())
    breakdown = {}
    weighted_sum = 0.0

    for asset_type, pct in reserve_composition.items():
        weight = _RESERVE_WEIGHTS.get(asset_type.lower(), 0.15)
        contribution = weight * (pct / 100.0) if total_alloc > 0 else 0.0
        weighted_sum += contribution
        breakdown[asset_type] = {
            "allocation_pct": pct,
            "quality_weight": weight,
            "contribution": contribution,
        }

    # Normalise if allocations do not sum to 100
    if total_alloc > 0 and abs(total_alloc - 100) > 1:
        score = weighted_sum * (100.0 / total_alloc)
    else:
        score = weighted_sum

    score = float(np.clip(score, 0.0, 1.0))

    # Warnings
    warnings = []
    crypto_pct = reserve_composition.get("crypto", 0)
    other_pct = reserve_composition.get("other", 0)
    cash_pct = reserve_composition.get("cash", 0) + reserve_composition.get("treasuries", 0)

    if crypto_pct > 20:
        warnings.append("Crypto allocation exceeds 20%: high volatility risk")
    if other_pct > 30:
        warnings.append("Opaque 'other' allocation exceeds 30%: transparency risk")
    if cash_pct < 20:
        warnings.append("Cash and treasuries below 20%: liquidity risk")
    if total_alloc < 95:
        warnings.append(f"Allocations sum to {total_alloc:.1f}%: incomplete disclosure")

    # Rating
    if score >= 0.85:
        rating = "excellent"
    elif score >= 0.70:
        rating = "good"
    elif score >= 0.50:
        rating = "fair"
    elif score >= 0.30:
        rating = "poor"
    else:
        rating = "critical"

    return {
        "score": score,
        "rating": rating,
        "breakdown": breakdown,
        "total_allocation": total_alloc,
        "warnings": warnings,
    }
