"""Regime-conditional Black-Litterman model.

Extends the standard Black-Litterman framework by making investor
views and their confidence levels functions of the detected market
regime. In a bull regime, bullish views are amplified; in a crisis,
defensive views dominate. The output is a set of optimal weights per
regime and a blended portfolio across regimes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "RegimeBLResult",
    "black_litterman_posterior",
    "regime_conditional_bl",
    "blend_regime_weights",
]


@dataclass(frozen=True)
class RegimeBLResult:
    """Result container for regime-conditional Black-Litterman.

    Attributes:
        regime_weights: Per-regime optimal weight dicts.
        blended_weights: Probability-weighted blend across regimes.
        regime_returns: Per-regime posterior expected returns.
        regime_covariances: Per-regime posterior covariance matrices.
        regime_probabilities: Probability of each regime.
    """

    regime_weights: dict[int, dict[str, float]]
    blended_weights: dict[str, float]
    regime_returns: dict[int, pd.Series]
    regime_covariances: dict[int, pd.DataFrame]
    regime_probabilities: dict[int, float]


def black_litterman_posterior(
    returns_df: pd.DataFrame,
    market_caps: dict[str, float],
    views: dict[str, float],
    view_confidences: dict[str, float],
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    freq: int = 252,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute Black-Litterman posterior returns and covariance.

    Standard BL implementation: combines market equilibrium with
    subjective views weighted by confidence.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        market_caps: Market capitalisation per asset.
        views: Absolute view on expected return per asset (annualised).
        view_confidences: Confidence in each view (0 to 1).
        tau: Scaling parameter for uncertainty in equilibrium.
        risk_aversion: Market risk aversion coefficient (lambda).
        freq: Trading periods per year.

    Returns:
        Tuple of (posterior_returns as Series, posterior_cov as DataFrame).
    """
    names = list(returns_df.columns)
    n = len(names)
    cov = returns_df.cov().values * freq  # annualised

    # Market-implied equilibrium returns
    caps = np.array([market_caps[name] for name in names])
    w_mkt = caps / caps.sum()
    pi = risk_aversion * cov @ w_mkt

    # Build P (pick matrix) and Q (view vector)
    view_assets = [a for a in names if a in views]
    k = len(view_assets)
    if k == 0:
        return pd.Series(pi, index=names), pd.DataFrame(cov, index=names, columns=names)

    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega_diag = np.zeros(k)

    for i, asset in enumerate(view_assets):
        j = names.index(asset)
        P[i, j] = 1.0
        Q[i] = views[asset]
        conf = max(min(view_confidences.get(asset, 0.5), 0.999), 0.001)
        omega_diag[i] = (1.0 / conf - 1.0) * (P[i] @ (tau * cov) @ P[i])

    Omega = np.diag(omega_diag)

    tau_cov = tau * cov
    tau_cov_inv = np.linalg.inv(tau_cov)
    Pt_Omega_inv = P.T @ np.linalg.inv(Omega)

    posterior_cov = np.linalg.inv(tau_cov_inv + Pt_Omega_inv @ P)
    posterior_mu = posterior_cov @ (tau_cov_inv @ pi + Pt_Omega_inv @ Q)

    return (
        pd.Series(posterior_mu, index=names),
        pd.DataFrame(posterior_cov + cov, index=names, columns=names),
    )


def _optimal_weights_from_posterior(
    posterior_mu: pd.Series,
    posterior_cov: pd.DataFrame,
    risk_aversion: float = 2.5,
    long_only: bool = True,
) -> dict[str, float]:
    """Derive optimal weights from BL posterior.

    Args:
        posterior_mu: Posterior expected returns.
        posterior_cov: Posterior covariance matrix.
        risk_aversion: Risk aversion coefficient.
        long_only: If True, clip negative weights and renormalise.

    Returns:
        Dictionary of asset weights.
    """
    cov_inv = np.linalg.inv(posterior_cov.values)
    w = (1.0 / risk_aversion) * cov_inv @ posterior_mu.values

    if long_only:
        w = np.maximum(w, 0.0)

    w_sum = w.sum()
    w = w / w_sum if abs(w_sum) > 1e-12 else np.ones(len(w)) / len(w)

    return dict(zip(posterior_mu.index.tolist(), w.tolist()))


def regime_conditional_bl(
    returns_df: pd.DataFrame,
    regime_labels: np.ndarray,
    market_caps: dict[str, float],
    base_views: dict[str, float],
    base_confidences: dict[str, float],
    regime_view_adjustments: dict[int, dict[str, float]] | None = None,
    regime_confidence_scaling: dict[int, float] | None = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    freq: int = 252,
    long_only: bool = True,
) -> RegimeBLResult:
    """Compute regime-conditional Black-Litterman allocations.

    Adjusts views and confidence levels based on the detected regime.
    In each regime, the model:
    1. Filters returns to that regime's observations.
    2. Adjusts views by regime-specific multipliers.
    3. Scales confidence by regime certainty.
    4. Computes the BL posterior and optimal weights.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        regime_labels: Array of integer regime labels.
        market_caps: Market capitalisation per asset.
        base_views: Base absolute views on expected returns (annualised).
        base_confidences: Base confidence in each view (0 to 1).
        regime_view_adjustments: Per-regime multipliers for views.
            E.g. ``{0: {"Equity": 1.2}, 1: {"Equity": 0.5}}`` amplifies
            the equity view in regime 0, dampens it in regime 1.
        regime_confidence_scaling: Per-regime confidence scale factor.
            E.g. ``{0: 1.5, 1: 0.5}`` increases confidence in regime 0.
        tau: BL uncertainty scaling parameter.
        risk_aversion: Market risk aversion coefficient.
        freq: Trading periods per year.
        long_only: If True, constrain weights >= 0.

    Returns:
        ``RegimeBLResult`` with per-regime and blended weights.
    """
    labels = np.asarray(regime_labels)
    unique_regimes = sorted(set(labels.tolist()))
    total_obs = len(labels)

    regime_weights_map: dict[int, dict[str, float]] = {}
    regime_returns_map: dict[int, pd.Series] = {}
    regime_cov_map: dict[int, pd.DataFrame] = {}
    regime_probs: dict[int, float] = {}

    for regime in unique_regimes:
        mask = labels == regime
        regime_probs[regime] = float(mask.sum()) / total_obs

        regime_returns = returns_df.iloc[mask]
        if regime_returns.shape[0] < 5:
            # Fallback to equal weights
            names = list(returns_df.columns)
            n = len(names)
            regime_weights_map[regime] = dict(zip(names, [1.0 / n] * n))
            regime_returns_map[regime] = pd.Series(0.0, index=names)
            regime_cov_map[regime] = returns_df.cov() * freq
            continue

        # Adjust views for this regime
        adjusted_views = dict(base_views)
        if regime_view_adjustments and regime in regime_view_adjustments:
            for asset, mult in regime_view_adjustments[regime].items():
                if asset in adjusted_views:
                    adjusted_views[asset] = adjusted_views[asset] * mult

        # Scale confidences for this regime
        adjusted_conf = dict(base_confidences)
        if regime_confidence_scaling and regime in regime_confidence_scaling:
            scale = regime_confidence_scaling[regime]
            adjusted_conf = {
                k: min(max(v * scale, 0.001), 0.999)
                for k, v in adjusted_conf.items()
            }

        post_mu, post_cov = black_litterman_posterior(
            regime_returns,
            market_caps,
            adjusted_views,
            adjusted_conf,
            tau=tau,
            risk_aversion=risk_aversion,
            freq=freq,
        )

        regime_returns_map[regime] = post_mu
        regime_cov_map[regime] = post_cov
        regime_weights_map[regime] = _optimal_weights_from_posterior(
            post_mu, post_cov, risk_aversion=risk_aversion, long_only=long_only
        )

    # Blend across regimes
    blended = blend_regime_weights(regime_weights_map, regime_probs)

    return RegimeBLResult(
        regime_weights=regime_weights_map,
        blended_weights=blended,
        regime_returns=regime_returns_map,
        regime_covariances=regime_cov_map,
        regime_probabilities=regime_probs,
    )


def blend_regime_weights(
    regime_weights: dict[int, dict[str, float]],
    regime_probabilities: dict[int, float],
) -> dict[str, float]:
    """Blend portfolio weights across regimes by probability.

    Args:
        regime_weights: Per-regime weight dictionaries.
        regime_probabilities: Probability of each regime.

    Returns:
        Blended weight dictionary.
    """
    all_assets: list[str] = []
    for weights in regime_weights.values():
        for asset in weights:
            if asset not in all_assets:
                all_assets.append(asset)

    blended: dict[str, float] = {a: 0.0 for a in all_assets}
    for regime, weights in regime_weights.items():
        prob = regime_probabilities.get(regime, 0.0)
        for asset, w in weights.items():
            blended[asset] += w * prob

    # Renormalise
    total = sum(blended.values())
    if abs(total) > 1e-12:
        blended = {k: v / total for k, v in blended.items()}

    return blended
