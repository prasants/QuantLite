"""Exchange risk analysis: concentration, wallet risk, reserves, and liquidity.

Tools for assessing counterparty risk across crypto exchanges,
including balance concentration, hot/cold wallet ratios, proof of
reserves verification, and order book liquidity analysis.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "concentration_score",
    "liquidity_risk",
    "proof_of_reserves_check",
    "slippage_estimate",
    "wallet_risk_assessment",
]


def concentration_score(balances_by_exchange):
    """Compute HHI-based concentration score across exchanges.

    The Herfindahl-Hirschman Index (HHI) measures how concentrated
    holdings are across exchanges. An HHI near 1.0 means all funds
    sit on a single exchange (maximum counterparty risk).

    Parameters
    ----------
    balances_by_exchange : dict
        Mapping of exchange name to balance value.

    Returns
    -------
    dict
        Dictionary with keys:
        - hhi: float in [0, 1], Herfindahl-Hirschman Index
        - normalised_hhi: HHI normalised to [0, 1] accounting for number of exchanges
        - risk_rating: qualitative rating
        - shares: dict of exchange to percentage share
        - dominant_exchange: name of exchange with largest share
        - n_exchanges: number of exchanges with non-zero balance
    """
    total = sum(balances_by_exchange.values())
    if total <= 0:
        return {
            "hhi": 1.0,
            "normalised_hhi": 1.0,
            "risk_rating": "critical",
            "shares": {},
            "dominant_exchange": None,
            "n_exchanges": 0,
        }

    shares = {}
    for name, balance in balances_by_exchange.items():
        shares[name] = balance / total

    hhi = sum(s ** 2 for s in shares.values())
    n = len(shares)
    normalised_hhi = (hhi - 1.0 / n) / (1.0 - 1.0 / n) if n > 1 else 1.0

    normalised_hhi = float(np.clip(normalised_hhi, 0.0, 1.0))

    dominant = max(shares, key=shares.get)

    if normalised_hhi > 0.75:
        risk_rating = "critical"
    elif normalised_hhi > 0.50:
        risk_rating = "high"
    elif normalised_hhi > 0.25:
        risk_rating = "medium"
    else:
        risk_rating = "low"

    return {
        "hhi": float(hhi),
        "normalised_hhi": normalised_hhi,
        "risk_rating": risk_rating,
        "shares": {k: float(v) for k, v in shares.items()},
        "dominant_exchange": dominant,
        "n_exchanges": n,
    }


def wallet_risk_assessment(hot_pct, cold_pct, total_value):
    """Score hot/cold wallet ratio risk.

    Evaluates the risk of a wallet allocation strategy based on
    the proportion of assets in hot wallets (online, higher risk)
    versus cold wallets (offline, lower risk).

    Parameters
    ----------
    hot_pct : float
        Percentage of total value in hot wallets (0-100).
    cold_pct : float
        Percentage of total value in cold wallets (0-100).
    total_value : float
        Total value across all wallets.

    Returns
    -------
    dict
        Dictionary with keys:
        - risk_score: float in [0, 1], higher = riskier
        - risk_rating: qualitative rating
        - hot_value: absolute value in hot wallets
        - cold_value: absolute value in cold wallets
        - recommendations: list of risk mitigation suggestions
    """
    hot_frac = hot_pct / 100.0
    cold_frac = cold_pct / 100.0

    hot_value = total_value * hot_frac
    cold_value = total_value * cold_frac

    # Risk score: heavily penalise high hot wallet ratios
    # Industry best practice: < 5% hot
    if hot_pct <= 2:
        risk_score = 0.05
    elif hot_pct <= 5:
        risk_score = 0.15
    elif hot_pct <= 10:
        risk_score = 0.30
    elif hot_pct <= 20:
        risk_score = 0.50
    elif hot_pct <= 50:
        risk_score = 0.75
    else:
        risk_score = 0.95

    # Scale by total value (larger holdings are higher risk)
    if total_value > 1_000_000_000:
        risk_score = min(1.0, risk_score * 1.15)
    elif total_value > 100_000_000:
        risk_score = min(1.0, risk_score * 1.05)

    recommendations = []
    if hot_pct > 10:
        recommendations.append(
            f"Reduce hot wallet exposure from {hot_pct:.1f}% to below 10%"
        )
    if hot_pct > 5:
        recommendations.append("Implement multi-signature controls on hot wallets")
    if hot_value > 50_000_000:
        recommendations.append(
            f"Hot wallet value ({hot_value / 1e6:.0f}M) exceeds prudent limits; "
            "consider tiered withdrawal architecture"
        )
    if cold_pct < 80:
        recommendations.append("Increase cold storage allocation to at least 80%")
    if total_value > 500_000_000 and hot_pct > 3:
        recommendations.append(
            "For holdings above $500M, target hot wallet ratio below 3%"
        )

    if risk_score >= 0.75:
        risk_rating = "critical"
    elif risk_score >= 0.50:
        risk_rating = "high"
    elif risk_score >= 0.25:
        risk_rating = "medium"
    else:
        risk_rating = "low"

    return {
        "risk_score": float(risk_score),
        "risk_rating": risk_rating,
        "hot_value": float(hot_value),
        "cold_value": float(cold_value),
        "recommendations": recommendations,
    }


def proof_of_reserves_check(claimed_reserves, on_chain_verified):
    """Compare claimed reserves against on-chain verified amounts.

    Assesses the credibility of an exchange's proof-of-reserves
    disclosure by comparing claimed totals with independently
    verifiable on-chain data.

    Parameters
    ----------
    claimed_reserves : dict
        Mapping of asset name to claimed reserve amount.
    on_chain_verified : dict
        Mapping of asset name to on-chain verified amount.

    Returns
    -------
    dict
        Dictionary with keys:
        - overall_ratio: total verified / total claimed
        - per_asset: dict of asset to verification details
        - fully_verified: bool, True if all assets >= 100% verified
        - risk_rating: qualitative rating
        - warnings: list of specific concerns
    """
    per_asset = {}
    total_claimed = 0.0
    total_verified = 0.0
    warnings = []

    all_assets = set(list(claimed_reserves.keys()) + list(on_chain_verified.keys()))

    for asset in sorted(all_assets):
        claimed = claimed_reserves.get(asset, 0.0)
        verified = on_chain_verified.get(asset, 0.0)

        total_claimed += claimed
        total_verified += verified

        ratio = verified / claimed if claimed > 0 else 0.0

        per_asset[asset] = {
            "claimed": float(claimed),
            "verified": float(verified),
            "ratio": float(ratio),
            "shortfall": float(max(0, claimed - verified)),
        }

        if claimed > 0 and verified == 0:
            warnings.append(f"{asset}: claimed {claimed:.2f} but nothing verified on-chain")
        elif ratio < 0.95:
            warnings.append(
                f"{asset}: only {ratio:.1%} verified (shortfall: {claimed - verified:.2f})"
            )

    if asset in on_chain_verified and asset not in claimed_reserves:
        warnings.append(f"{asset}: found on-chain but not in claimed reserves")

    overall_ratio = total_verified / total_claimed if total_claimed > 0 else 0.0
    fully_verified = all(
        per_asset[a]["ratio"] >= 1.0 for a in per_asset if per_asset[a]["claimed"] > 0
    )

    if overall_ratio >= 1.0 and fully_verified:
        risk_rating = "low"
    elif overall_ratio >= 0.95:
        risk_rating = "medium"
    elif overall_ratio >= 0.80:
        risk_rating = "high"
    else:
        risk_rating = "critical"

    return {
        "overall_ratio": float(overall_ratio),
        "per_asset": per_asset,
        "fully_verified": fully_verified,
        "risk_rating": risk_rating,
        "warnings": warnings,
    }


def liquidity_risk(order_book_depth, position_size):
    """Estimate unwind time and risk given order book depth.

    Calculates how many periods it would take to unwind a position
    given the available order book depth, and assesses the associated
    liquidity risk.

    Parameters
    ----------
    order_book_depth : float
        Average available depth per period (e.g. daily volume at
        acceptable slippage).
    position_size : float
        Total position size to unwind.

    Returns
    -------
    dict
        Dictionary with keys:
        - periods_to_unwind: estimated periods to fully unwind
        - position_to_depth_ratio: position size relative to depth
        - risk_rating: qualitative rating
        - daily_participation_rate: suggested daily rate (max 10% of depth)
        - recommended_unwind_periods: periods at conservative participation
    """
    if order_book_depth <= 0:
        return {
            "periods_to_unwind": float("inf"),
            "position_to_depth_ratio": float("inf"),
            "risk_rating": "critical",
            "daily_participation_rate": 0.0,
            "recommended_unwind_periods": float("inf"),
        }

    ratio = position_size / order_book_depth
    periods = ratio  # 100% participation

    # Conservative: participate at most 10% of depth per period
    conservative_rate = order_book_depth * 0.10
    recommended_periods = position_size / conservative_rate if conservative_rate > 0 else float("inf")

    if ratio <= 0.5:
        risk_rating = "low"
    elif ratio <= 2.0:
        risk_rating = "medium"
    elif ratio <= 10.0:
        risk_rating = "high"
    else:
        risk_rating = "critical"

    return {
        "periods_to_unwind": float(periods),
        "position_to_depth_ratio": float(ratio),
        "risk_rating": risk_rating,
        "daily_participation_rate": float(conservative_rate),
        "recommended_unwind_periods": float(recommended_periods),
    }


def slippage_estimate(order_book, trade_size):
    """Estimate price impact for a given trade size.

    Walks through an order book (list of price/quantity levels)
    to estimate the volume-weighted average price and resulting
    slippage for a given trade size.

    Parameters
    ----------
    order_book : list of tuples
        List of (price, quantity) tuples, sorted by price
        (ascending for asks, descending for bids).
    trade_size : float
        Total quantity to trade.

    Returns
    -------
    dict
        Dictionary with keys:
        - vwap: volume-weighted average execution price
        - best_price: best available price (first level)
        - worst_price: worst price touched
        - slippage_pct: percentage slippage from best price
        - levels_consumed: number of order book levels consumed
        - unfilled: quantity that could not be filled
        - risk_rating: qualitative rating
    """
    if not order_book or trade_size <= 0:
        return {
            "vwap": 0.0,
            "best_price": 0.0,
            "worst_price": 0.0,
            "slippage_pct": 0.0,
            "levels_consumed": 0,
            "unfilled": trade_size if trade_size > 0 else 0.0,
            "risk_rating": "critical" if trade_size > 0 else "low",
        }

    remaining = trade_size
    total_cost = 0.0
    total_filled = 0.0
    best_price = order_book[0][0]
    worst_price = best_price
    levels_consumed = 0

    for price, qty in order_book:
        if remaining <= 0:
            break
        fill = min(remaining, qty)
        total_cost += fill * price
        total_filled += fill
        remaining -= fill
        worst_price = price
        levels_consumed += 1

    vwap = total_cost / total_filled if total_filled > 0 else 0.0
    slippage_pct = abs(vwap - best_price) / best_price * 100 if best_price > 0 else 0.0

    if slippage_pct <= 0.1:
        risk_rating = "low"
    elif slippage_pct <= 0.5:
        risk_rating = "medium"
    elif slippage_pct <= 2.0:
        risk_rating = "high"
    else:
        risk_rating = "critical"

    return {
        "vwap": float(vwap),
        "best_price": float(best_price),
        "worst_price": float(worst_price),
        "slippage_pct": float(slippage_pct),
        "levels_consumed": levels_consumed,
        "unfilled": float(remaining),
        "risk_rating": risk_rating,
    }
