"""Fat-tailed return generators: Student-t, Levy stable, regime-switching, and Kou's model.

These processes generate return series with realistic tail behaviour,
unlike Gaussian models that systematically underestimate extreme events.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "student_t_process",
    "levy_stable_process",
    "regime_switching_gbm",
    "kou_double_exponential_jump",
    "RegimeParams",
]


@dataclass(frozen=True)
class RegimeParams:
    """Parameters for a single regime in a Markov-switching model.

    Args:
        mu: Annualised drift.
        sigma: Annualised volatility.
    """

    mu: float
    sigma: float

    def __repr__(self) -> str:
        return f"RegimeParams(mu={self.mu:.4f}, sigma={self.sigma:.4f})"


def student_t_process(
    nu: float = 4.0,
    mu: float = 0.0,
    sigma: float = 0.01,
    n_steps: int = 252,
    rng_seed: int | None = None,
) -> np.ndarray:
    """Generate returns from a scaled Student-t distribution.

    Student-t returns exhibit power-law tails with tail index
    approximately equal to ``nu``. Lower ``nu`` means fatter tails;
    ``nu > 30`` approximates Gaussian behaviour.

    Args:
        nu: Degrees of freedom (controls tail heaviness).
        mu: Location (mean return per step).
        sigma: Scale (volatility per step).
        n_steps: Number of return observations.
        rng_seed: Seed for reproducibility.

    Returns:
        Array of ``n_steps`` returns.

    Raises:
        ValueError: If ``nu <= 2`` (variance undefined).
    """
    if nu <= 2:
        raise ValueError("nu must be > 2 for finite variance")

    rng = np.random.default_rng(rng_seed)
    # Scale so that the variance equals sigma^2
    scale = sigma * np.sqrt((nu - 2) / nu)
    raw = rng.standard_t(df=nu, size=n_steps)
    return mu + scale * raw


def levy_stable_process(
    alpha: float = 1.7,
    beta: float = 0.0,
    mu: float = 0.0,
    sigma: float = 0.01,
    n_steps: int = 252,
    rng_seed: int | None = None,
) -> np.ndarray:
    """Generate returns from a Levy alpha-stable distribution.

    Stable distributions generalise the Gaussian and allow for
    infinite variance (when ``alpha < 2``) and skewness.

    Args:
        alpha: Stability parameter (0 < alpha <= 2). Lower = fatter tails.
        beta: Skewness parameter (-1 <= beta <= 1).
        mu: Location parameter.
        sigma: Scale parameter.
        n_steps: Number of return observations.
        rng_seed: Seed for reproducibility.

    Returns:
        Array of ``n_steps`` returns.

    Raises:
        ValueError: On invalid alpha or beta.
    """
    if not (0 < alpha <= 2):
        raise ValueError("alpha must be in (0, 2]")
    if not (-1 <= beta <= 1):
        raise ValueError("beta must be in [-1, 1]")

    from scipy.stats import levy_stable

    rng = np.random.default_rng(rng_seed)
    # scipy levy_stable uses Zolotarev's (M) parameterisation
    samples = levy_stable.rvs(
        alpha, beta, loc=mu, scale=sigma, size=n_steps, random_state=rng
    )
    return np.asarray(samples)


def regime_switching_gbm(
    params_by_regime: list[RegimeParams],
    transition_matrix: np.ndarray,
    n_steps: int = 252,
    dt: float = 1 / 252,
    S0: float = 100.0,
    rng_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a Markov regime-switching GBM.

    Each regime has its own drift and volatility. Regime transitions
    follow a discrete Markov chain with the given transition matrix.

    Args:
        params_by_regime: List of ``RegimeParams``, one per regime.
        transition_matrix: Row-stochastic matrix of shape
            ``(n_regimes, n_regimes)``.
        n_steps: Number of time steps.
        dt: Time increment in years.
        S0: Initial price.
        rng_seed: Seed for reproducibility.

    Returns:
        Tuple of (prices array of length ``n_steps + 1``,
        regimes array of length ``n_steps``).

    Raises:
        ValueError: On dimension mismatches.
    """
    n_regimes = len(params_by_regime)
    if transition_matrix.shape != (n_regimes, n_regimes):
        raise ValueError("transition_matrix shape must match number of regimes")

    rng = np.random.default_rng(rng_seed)

    # Pre-draw all random variates
    dW = rng.normal(0, np.sqrt(dt), size=n_steps)
    regime_draws = rng.random(size=n_steps)

    prices = np.empty(n_steps + 1)
    prices[0] = S0
    regimes = np.empty(n_steps, dtype=int)

    # Start in regime 0
    current_regime = 0

    for t in range(n_steps):
        regimes[t] = current_regime
        p = params_by_regime[current_regime]
        drift = (p.mu - 0.5 * p.sigma**2) * dt
        diffusion = p.sigma * dW[t]
        prices[t + 1] = prices[t] * np.exp(drift + diffusion)

        # Transition
        cum_probs = np.cumsum(transition_matrix[current_regime])
        current_regime = int(np.searchsorted(cum_probs, regime_draws[t]))
        current_regime = min(current_regime, n_regimes - 1)

    return prices, regimes


def kou_double_exponential_jump(
    S0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    lam: float = 1.0,
    p: float = 0.4,
    eta1: float = 10.0,
    eta2: float = 5.0,
    dt: float = 1 / 252,
    n_steps: int = 252,
    rng_seed: int | None = None,
) -> np.ndarray:
    """Simulate Kou's double-exponential jump-diffusion model.

    Unlike Merton's model (Gaussian jumps), Kou uses asymmetric
    double-exponential jump sizes, producing a better fit to the
    leptokurtic, asymmetric returns observed in practice.

    Args:
        S0: Initial price.
        mu: Annualised drift.
        sigma: Annualised diffusion volatility.
        lam: Jump intensity (expected jumps per year).
        p: Probability of an upward jump.
        eta1: Rate parameter for upward jumps (mean = 1/eta1).
            Must be > 1 for finite expectation of exp(J).
        eta2: Rate parameter for downward jumps (mean = 1/eta2).
        dt: Time increment in years.
        n_steps: Number of time steps.
        rng_seed: Seed for reproducibility.

    Returns:
        Price path of length ``n_steps + 1``.

    Raises:
        ValueError: If ``eta1 <= 1`` (required for finite moments).
    """
    if eta1 <= 1:
        raise ValueError("eta1 must be > 1 for finite E[exp(J)]")

    rng = np.random.default_rng(rng_seed)

    dW = rng.normal(0, np.sqrt(dt), size=n_steps)
    N_jumps = rng.poisson(lam * dt, size=n_steps)

    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * dW

    for t in range(n_steps):
        if N_jumps[t] > 0:
            for _ in range(N_jumps[t]):
                if rng.random() < p:
                    # Upward jump: exponential with rate eta1
                    jump = rng.exponential(1.0 / eta1)
                else:
                    # Downward jump: negative exponential with rate eta2
                    jump = -rng.exponential(1.0 / eta2)
                log_increments[t] += jump

    prices = np.empty(n_steps + 1)
    prices[0] = S0
    prices[1:] = S0 * np.exp(np.cumsum(log_increments))

    return prices
