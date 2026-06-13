"""Hidden Markov Model regime detection.

Provides Gaussian HMM-based regime identification for financial
returns. Detects distinct market regimes (e.g. calm, volatile, crisis)
by fitting a Hidden Markov Model to the return series.

Requires ``hmmlearn`` as an optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "RegimeModel",
    "fit_regime_model",
    "predict_regimes",
    "regime_probabilities",
    "select_n_regimes",
]


def _check_hmmlearn() -> None:
    """Raise a helpful ImportError if hmmlearn is not installed."""
    try:
        import hmmlearn  # noqa: F401
    except ImportError:
        raise ImportError(
            "hmmlearn is required for HMM regime detection. "
            "Install it with: pip install hmmlearn"
        ) from None


@dataclass(frozen=True)
class RegimeModel:
    """Fitted Hidden Markov Model for regime detection.

    Attributes:
        means: Mean return per regime, shape ``(n_regimes,)``.
        variances: Variance per regime, shape ``(n_regimes,)``.
        transition_matrix: Row-stochastic transition matrix,
            shape ``(n_regimes, n_regimes)``.
        stationary_distribution: Long-run regime probabilities.
        regime_labels: Viterbi-decoded regime sequence.
        log_likelihood: Model log-likelihood.
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.
        n_regimes: Number of regimes.
        hmm: The underlying fitted hmmlearn model.
    """

    means: np.ndarray
    variances: np.ndarray
    transition_matrix: np.ndarray
    stationary_distribution: np.ndarray
    regime_labels: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    n_regimes: int = field(default=2)
    hmm: object = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"RegimeModel(n_regimes={self.n_regimes}, "
            f"AIC={self.aic:.2f}, BIC={self.bic:.2f})"
        )


def _compute_stationary(trans_mat: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of a transition matrix.

    Solves pi @ T = pi with sum(pi) = 1 via eigendecomposition.

    Args:
        trans_mat: Row-stochastic transition matrix.

    Returns:
        Stationary distribution vector.
    """
    evals, evecs = np.linalg.eig(trans_mat.T)
    # Find eigenvector for eigenvalue closest to 1
    idx = np.argmin(np.abs(evals - 1.0))
    pi = np.real(evecs[:, idx])
    pi = pi / pi.sum()
    return np.abs(pi)


def fit_regime_model(
    returns: np.ndarray | pd.Series,
    n_regimes: int = 2,
    rng_seed: int | None = None,
) -> RegimeModel:
    """Fit a Gaussian HMM to a return series.

    Args:
        returns: Simple periodic returns.
        n_regimes: Number of hidden states.
        rng_seed: Seed for reproducibility.

    Returns:
        A ``RegimeModel`` dataclass with fitted parameters.
    """
    _check_hmmlearn()
    from hmmlearn.hmm import GaussianHMM

    arr = np.asarray(returns, dtype=float).reshape(-1, 1)
    n_obs = arr.shape[0]

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="diag",
        n_iter=200,
        random_state=rng_seed,
        tol=1e-6,
    )
    model.fit(arr)

    means = model.means_.flatten()
    variances = model.covars_.flatten()
    trans_mat = model.transmat_
    stationary = _compute_stationary(trans_mat)
    labels = model.predict(arr)
    ll = float(model.score(arr))

    # Sort regimes by mean (lowest mean = regime 0, typically "crisis")
    order = np.argsort(means)
    means = means[order]
    variances = variances[order]
    trans_mat = trans_mat[np.ix_(order, order)]
    stationary = stationary[order]
    label_map = {old: new for new, old in enumerate(order)}
    labels = np.array([label_map[lbl] for lbl in labels])

    # AIC / BIC
    n_params = n_regimes**2 + 2 * n_regimes - 1
    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(n_obs) - 2 * ll

    return RegimeModel(
        means=means,
        variances=variances,
        transition_matrix=trans_mat,
        stationary_distribution=stationary,
        regime_labels=labels,
        log_likelihood=ll,
        aic=aic,
        bic=bic,
        n_regimes=n_regimes,
        hmm=model,
    )


def predict_regimes(
    model: RegimeModel,
    returns: np.ndarray | pd.Series,
) -> np.ndarray:
    """Predict the most likely regime sequence via Viterbi decoding.

    Args:
        model: A fitted ``RegimeModel``.
        returns: Return series to decode.

    Returns:
        Array of regime labels.
    """
    _check_hmmlearn()
    arr = np.asarray(returns, dtype=float).reshape(-1, 1)
    return model.hmm.predict(arr)


def regime_probabilities(
    model: RegimeModel,
    returns: np.ndarray | pd.Series,
) -> np.ndarray:
    """Compute per-observation regime probabilities via forward-backward.

    Args:
        model: A fitted ``RegimeModel``.
        returns: Return series.

    Returns:
        Array of shape ``(n_obs, n_regimes)`` with posterior
        regime probabilities.
    """
    _check_hmmlearn()
    arr = np.asarray(returns, dtype=float).reshape(-1, 1)
    return model.hmm.predict_proba(arr)


def select_n_regimes(
    returns: np.ndarray | pd.Series,
    max_regimes: int = 4,
    rng_seed: int | None = None,
) -> RegimeModel:
    """Select the optimal number of regimes by BIC.

    Fits models with 2 through ``max_regimes`` states and returns
    the one with the lowest BIC.

    Args:
        returns: Simple periodic returns.
        max_regimes: Maximum number of regimes to test.
        rng_seed: Seed for reproducibility.

    Returns:
        The best ``RegimeModel`` by BIC.
    """
    best: RegimeModel | None = None

    for n in range(2, max_regimes + 1):
        try:
            model = fit_regime_model(returns, n_regimes=n, rng_seed=rng_seed)
            if best is None or model.bic < best.bic:
                best = model
        except Exception:
            continue

    if best is None:
        raise RuntimeError("Could not fit any HMM model")
    return best
