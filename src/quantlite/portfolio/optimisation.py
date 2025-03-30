"""Portfolio optimisation: Markowitz, CVaR, risk parity, HRP, Black-Litterman, Kelly.

All optimisation functions return a ``PortfolioWeights`` dataclass for
consistent downstream consumption.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..dependency.clustering import hrp_weights as _hrp_weights_raw
from ..risk.metrics import cvar as _cvar

__all__ = [
    "PortfolioWeights",
    "mean_variance_weights",
    "min_variance_weights",
    "mean_cvar_weights",
    "risk_parity_weights",
    "hrp_weights",
    "black_litterman",
    "kelly_criterion",
    "half_kelly",
    "max_sharpe_weights",
]


@dataclass(frozen=True)
class PortfolioWeights:
    """Result container for portfolio weight optimisation.

    Attributes:
        weights: Mapping of asset name to weight.
        method: Optimisation method used.
        expected_return: Annualised expected return, if computed.
        expected_risk: Annualised expected risk (volatility or CVaR).
        sharpe: Sharpe ratio, if computed.
    """

    weights: dict[str, float]
    method: str
    expected_return: float | None = None
    expected_risk: float | None = None
    sharpe: float | None = None

    def __repr__(self) -> str:
        n = len(self.weights)
        return (
            f"PortfolioWeights(method={self.method!r}, assets={n}, "
            f"ret={self.expected_return}, risk={self.expected_risk})"
        )


def _annualise(mean_ret: float, std_ret: float, freq: int = 252) -> tuple[float, float]:
    """Annualise mean and standard deviation of periodic returns."""
    ann_ret = float((1 + mean_ret) ** freq - 1)
    ann_vol = float(std_ret * np.sqrt(freq))
    return ann_ret, ann_vol


def mean_variance_weights(
    returns_df: pd.DataFrame,
    target_return: float | None = None,
    risk_free_rate: float = 0.0,
    long_only: bool = True,
    freq: int = 252,
) -> PortfolioWeights:
    """Classic Markowitz mean-variance optimisation.

    Minimises portfolio variance subject to a target return constraint.
    If ``target_return`` is None, minimises variance without a return target
    (equivalent to ``min_variance_weights``).

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        target_return: Target annualised return. If None, no return constraint.
        risk_free_rate: Annualised risk-free rate for Sharpe computation.
        long_only: If True, constrain weights >= 0.
        freq: Trading periods per year.

    Returns:
        ``PortfolioWeights`` with optimised allocations.
    """
    n = returns_df.shape[1]
    mu = returns_df.mean().values
    cov = returns_df.cov().values
    names = list(returns_df.columns)

    x0 = np.ones(n) / n

    def portfolio_var(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    constraints: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if target_return is not None:
        # Convert annualised target to periodic
        periodic_target = (1 + target_return) ** (1 / freq) - 1
        constraints.append(
            {"type": "eq", "fun": lambda w: w @ mu - periodic_target}
        )

    bounds = [(0.0, 1.0)] * n if long_only else [(None, None)] * n

    result = minimize(
        portfolio_var,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    w = result.x
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))
    ann_ret, ann_vol = _annualise(port_ret, port_vol, freq)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else float("nan")

    return PortfolioWeights(
        weights=dict(zip(names, w.tolist())),
        method="mean_variance",
        expected_return=ann_ret,
        expected_risk=ann_vol,
        sharpe=sharpe,
    )


def min_variance_weights(
    returns_df: pd.DataFrame,
    long_only: bool = True,
    freq: int = 252,
) -> PortfolioWeights:
    """Compute the minimum variance portfolio.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        long_only: If True, constrain weights >= 0.
        freq: Trading periods per year.

    Returns:
        ``PortfolioWeights`` for the minimum variance portfolio.
    """
    return mean_variance_weights(
        returns_df, target_return=None, long_only=long_only, freq=freq
    )


def mean_cvar_weights(
    returns_df: pd.DataFrame,
    alpha: float = 0.05,
    target_return: float | None = None,
    long_only: bool = True,
    freq: int = 252,
    risk_free_rate: float = 0.0,
) -> PortfolioWeights:
    """Optimise portfolio weights using CVaR as the risk measure.

    Minimises Conditional Value at Risk (Expected Shortfall) at the
    given significance level, optionally subject to a return constraint.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        alpha: Significance level for CVaR (e.g. 0.05 for 95%).
        target_return: Target annualised return. If None, no constraint.
        long_only: If True, constrain weights >= 0.
        freq: Trading periods per year.
        risk_free_rate: Annualised risk-free rate for Sharpe computation.

    Returns:
        ``PortfolioWeights`` with CVaR-optimised allocations.
    """
    n = returns_df.shape[1]
    mu = returns_df.mean().values
    returns_arr = returns_df.values
    names = list(returns_df.columns)

    x0 = np.ones(n) / n

    def portfolio_cvar(w: np.ndarray) -> float:
        port_rets = returns_arr @ w
        return -_cvar(port_rets, alpha=alpha)  # negate: cvar returns negative, we minimise loss

    constraints: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if target_return is not None:
        periodic_target = (1 + target_return) ** (1 / freq) - 1
        constraints.append(
            {"type": "eq", "fun": lambda w: w @ mu - periodic_target}
        )

    bounds = [(0.0, 1.0)] * n if long_only else [(None, None)] * n

    result = minimize(
        portfolio_cvar,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    w = result.x
    port_rets = returns_arr @ w
    port_ret = float(w @ mu)
    port_vol = float(np.std(port_rets, ddof=1))
    ann_ret, ann_vol = _annualise(port_ret, port_vol, freq)
    cvar_val = float(_cvar(port_rets, alpha=alpha))
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else float("nan")

    return PortfolioWeights(
        weights=dict(zip(names, w.tolist())),
        method="mean_cvar",
        expected_return=ann_ret,
        expected_risk=cvar_val,
        sharpe=sharpe,
    )


def risk_parity_weights(
    returns_df: pd.DataFrame,
    freq: int = 252,
    risk_free_rate: float = 0.0,
) -> PortfolioWeights:
    """Compute risk parity (equal risk contribution) weights.

    Each asset contributes equally to total portfolio variance.
    Uses numerical optimisation to find the weight vector where
    marginal risk contributions are equalised.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        freq: Trading periods per year.
        risk_free_rate: Annualised risk-free rate for Sharpe computation.

    Returns:
        ``PortfolioWeights`` with risk parity allocations.
    """
    n = returns_df.shape[1]
    cov = returns_df.cov().values
    mu = returns_df.mean().values
    names = list(returns_df.columns)

    x0 = np.ones(n) / n

    def risk_parity_obj(w: np.ndarray) -> float:
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-12:
            return 0.0
        mrc = (cov @ w) / port_vol  # marginal risk contributions
        rc = w * mrc  # risk contributions
        target_rc = port_vol / n
        return float(np.sum((rc - target_rc) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    result = minimize(
        risk_parity_obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-15, "maxiter": 1000},
    )

    w = result.x
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))
    ann_ret, ann_vol = _annualise(port_ret, port_vol, freq)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else float("nan")

    return PortfolioWeights(
        weights=dict(zip(names, w.tolist())),
        method="risk_parity",
        expected_return=ann_ret,
        expected_risk=ann_vol,
        sharpe=sharpe,
    )


def hrp_weights(
    returns_df: pd.DataFrame,
    freq: int = 252,
    risk_free_rate: float = 0.0,
) -> PortfolioWeights:
    """Hierarchical Risk Parity weights (convenience wrapper).

    Delegates to ``dependency.clustering.hrp_weights`` and wraps the
    result in a ``PortfolioWeights`` dataclass.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        freq: Trading periods per year.
        risk_free_rate: Annualised risk-free rate for Sharpe computation.

    Returns:
        ``PortfolioWeights`` with HRP allocations.
    """
    raw: OrderedDict[str, float] = _hrp_weights_raw(returns_df)
    w = np.array(list(raw.values()))
    mu = returns_df.mean().values
    cov = returns_df.cov().values

    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))
    ann_ret, ann_vol = _annualise(port_ret, port_vol, freq)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else float("nan")

    return PortfolioWeights(
        weights=dict(raw),
        method="hrp",
        expected_return=ann_ret,
        expected_risk=ann_vol,
        sharpe=sharpe,
    )


def black_litterman(
    returns_df: pd.DataFrame,
    market_caps: dict[str, float],
    views: dict[str, float],
    view_confidences: dict[str, float],
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    freq: int = 252,
) -> tuple[pd.Series, pd.DataFrame]:
    """Black-Litterman model: combine equilibrium with investor views.

    Computes posterior expected returns and covariance matrix by blending
    market-implied equilibrium returns with subjective views weighted
    by confidence.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        market_caps: Market capitalisation per asset.
        views: Absolute view on expected return per asset (annualised).
        view_confidences: Confidence in each view (0 to 1, where 1 = certain).
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
    pi = risk_aversion * cov @ w_mkt  # equilibrium excess returns

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
        conf = view_confidences.get(asset, 0.5)
        conf = max(min(conf, 0.999), 0.001)
        omega_diag[i] = (1.0 / conf - 1.0) * (P[i] @ (tau * cov) @ P[i])

    Omega = np.diag(omega_diag)

    # Posterior
    tau_cov = tau * cov
    tau_cov_inv = np.linalg.inv(tau_cov)
    Pt_Omega_inv = P.T @ np.linalg.inv(Omega)

    posterior_cov = np.linalg.inv(tau_cov_inv + Pt_Omega_inv @ P)
    posterior_mu = posterior_cov @ (tau_cov_inv @ pi + Pt_Omega_inv @ Q)

    return (
        pd.Series(posterior_mu, index=names),
        pd.DataFrame(posterior_cov + cov, index=names, columns=names),
    )


def kelly_criterion(
    returns: np.ndarray | pd.Series | list[float],
    win_prob: float | None = None,
    win_loss_ratio: float | None = None,
) -> float:
    """Compute the optimal Kelly fraction for position sizing.

    If ``win_prob`` and ``win_loss_ratio`` are provided, uses the
    classic Kelly formula: f* = p - q/b where p = win probability,
    q = 1-p, b = win/loss ratio.

    If only ``returns`` is provided, estimates from the return series:
    f* = mean / variance (continuous Kelly).

    Args:
        returns: Historical returns (used if win_prob not provided).
        win_prob: Probability of a winning trade.
        win_loss_ratio: Ratio of average win to average loss.

    Returns:
        Optimal Kelly fraction (can be > 1 for levered bets).
    """
    if win_prob is not None and win_loss_ratio is not None:
        q = 1.0 - win_prob
        return float(win_prob - q / win_loss_ratio)

    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    mu = float(np.mean(arr))
    var = float(np.var(arr, ddof=1))
    if var < 1e-15:
        return 0.0
    return mu / var


def half_kelly(
    returns: np.ndarray | pd.Series | list[float],
    win_prob: float | None = None,
    win_loss_ratio: float | None = None,
) -> float:
    """Half-Kelly: the practical, reduced-variance Kelly fraction.

    Returns half the full Kelly criterion, which empirically produces
    ~75% of the growth rate with significantly lower drawdowns.

    Args:
        returns: Historical returns.
        win_prob: Probability of a winning trade.
        win_loss_ratio: Ratio of average win to average loss.

    Returns:
        Half-Kelly fraction.
    """
    return kelly_criterion(returns, win_prob, win_loss_ratio) / 2.0


def max_sharpe_weights(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    long_only: bool = True,
    freq: int = 252,
) -> PortfolioWeights:
    """Find the tangency (maximum Sharpe ratio) portfolio.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        risk_free_rate: Annualised risk-free rate.
        long_only: If True, constrain weights >= 0.
        freq: Trading periods per year.

    Returns:
        ``PortfolioWeights`` for the maximum Sharpe ratio portfolio.
    """
    n = returns_df.shape[1]
    mu = returns_df.mean().values
    cov = returns_df.cov().values
    names = list(returns_df.columns)
    rf_periodic = (1 + risk_free_rate) ** (1 / freq) - 1

    x0 = np.ones(n) / n

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(w @ cov @ w))
        if port_vol < 1e-12:
            return 0.0
        return -(port_ret - rf_periodic) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n if long_only else [(None, None)] * n

    result = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    w = result.x
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))
    ann_ret, ann_vol = _annualise(port_ret, port_vol, freq)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else float("nan")

    return PortfolioWeights(
        weights=dict(zip(names, w.tolist())),
        method="max_sharpe",
        expected_return=ann_ret,
        expected_risk=ann_vol,
        sharpe=sharpe,
    )
