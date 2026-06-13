"""Copula fitting, simulation, and tail dependence analysis.

Provides Gaussian, Student-t, Clayton, Gumbel, and Frank copulas with
a common interface for fitting, simulation, log-likelihood, and tail
dependence measurement. All copulas work in the bivariate case.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize, special, stats
from scipy.stats import kendalltau, rankdata, spearmanr

__all__ = [
    "GaussianCopula",
    "StudentTCopula",
    "ClaytonCopula",
    "GumbelCopula",
    "FrankCopula",
    "CopulaFitResult",
    "select_best_copula",
]


def _to_uniform(data: np.ndarray) -> np.ndarray:
    """Transform data columns to pseudo-uniform margins via rank transform.

    Args:
        data: Array of shape ``(n, 2)`` with raw observations.

    Returns:
        Array of shape ``(n, 2)`` with values in ``(0, 1)``.
    """
    n = data.shape[0]
    u = np.column_stack([
        rankdata(data[:, j], method="ordinal") / (n + 1)
        for j in range(data.shape[1])
    ])
    return u


@dataclass(frozen=True)
class CopulaFitResult:
    """Result of copula model selection."""

    name: str
    copula: Any
    log_likelihood: float
    aic: float
    bic: float
    n_params: int

    def __repr__(self) -> str:
        return (
            f"CopulaFitResult(name={self.name!r}, "
            f"AIC={self.aic:.2f}, BIC={self.bic:.2f})"
        )


class BaseCopula(ABC):
    """Abstract base class for bivariate copulas."""

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Fit copula parameters from bivariate data.

        Args:
            data: Array of shape ``(n, 2)`` with raw observations.
        """

    @abstractmethod
    def simulate(self, n: int, rng_seed: int | None = None) -> np.ndarray:
        """Simulate ``n`` samples from the fitted copula.

        Args:
            n: Number of samples.
            rng_seed: Seed for reproducibility.

        Returns:
            Array of shape ``(n, 2)`` with uniform marginals.
        """

    @abstractmethod
    def log_likelihood(self, data: np.ndarray) -> float:
        """Compute log-likelihood on uniform-transformed data.

        Args:
            data: Array of shape ``(n, 2)`` with raw observations.

        Returns:
            Total log-likelihood.
        """

    @abstractmethod
    def n_params(self) -> int:
        """Number of fitted parameters."""

    def tail_dependence(self) -> dict[str, float]:
        """Return lower and upper tail dependence coefficients.

        Returns:
            Dict with keys ``"lower"`` and ``"upper"``.
        """
        return {
            "lower": self.lower_tail_dependence(),
            "upper": self.upper_tail_dependence(),
        }

    @abstractmethod
    def lower_tail_dependence(self) -> float:
        """Lower tail dependence coefficient."""

    @abstractmethod
    def upper_tail_dependence(self) -> float:
        """Upper tail dependence coefficient."""

    def aic(self, data: np.ndarray) -> float:
        """Akaike Information Criterion."""
        return 2 * self.n_params() - 2 * self.log_likelihood(data)

    def bic(self, data: np.ndarray) -> float:
        """Bayesian Information Criterion."""
        n = data.shape[0]
        return self.n_params() * np.log(n) - 2 * self.log_likelihood(data)


class GaussianCopula(BaseCopula):
    """Bivariate Gaussian copula.

    Parametrised by the Pearson correlation of the Gaussian latent
    variables. Has zero tail dependence for all correlations < 1.
    """

    def __init__(self) -> None:
        self.rho: float = 0.0

    def fit(self, data: np.ndarray) -> None:
        """Fit via Spearman rank correlation (consistent estimator).

        Args:
            data: Array of shape ``(n, 2)``.
        """
        u = _to_uniform(data)
        rho_s, _ = spearmanr(u[:, 0], u[:, 1])
        # Convert Spearman to Pearson for Gaussian copula
        self.rho = float(2 * np.sin(np.pi * rho_s / 6))

    def simulate(self, n: int, rng_seed: int | None = None) -> np.ndarray:
        """Simulate from bivariate Gaussian copula.

        Args:
            n: Number of samples.
            rng_seed: Random seed.

        Returns:
            Array of shape ``(n, 2)`` with uniform marginals.
        """
        rng = np.random.default_rng(rng_seed)
        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        z = rng.multivariate_normal([0, 0], cov, size=n)
        return stats.norm.cdf(z)

    def log_likelihood(self, data: np.ndarray) -> float:
        """Log-likelihood of the Gaussian copula density.

        Args:
            data: Array of shape ``(n, 2)``.

        Returns:
            Total log-likelihood.
        """
        u = _to_uniform(data)
        u = np.clip(u, 1e-10, 1 - 1e-10)
        z = stats.norm.ppf(u)
        rho = self.rho
        if abs(rho) > 0.9999:
            rho = np.sign(rho) * 0.9999

        det = 1 - rho**2
        n_obs = z.shape[0]
        ll = -0.5 * n_obs * np.log(det) - (1 / (2 * det)) * np.sum(
            rho**2 * (z[:, 0]**2 + z[:, 1]**2) - 2 * rho * z[:, 0] * z[:, 1]
        )
        return float(ll)

    def n_params(self) -> int:
        return 1

    def lower_tail_dependence(self) -> float:
        """Gaussian copula has zero tail dependence (unless rho = 1)."""
        return 0.0

    def upper_tail_dependence(self) -> float:
        """Gaussian copula has zero tail dependence (unless rho = 1)."""
        return 0.0

    def __repr__(self) -> str:
        return f"GaussianCopula(rho={self.rho:.4f})"


class StudentTCopula(BaseCopula):
    """Bivariate Student-t copula.

    Has symmetric tail dependence that increases with lower degrees
    of freedom and higher correlation. This is the key advantage
    over the Gaussian copula for modelling joint extremes.
    """

    def __init__(self) -> None:
        self.rho: float = 0.0
        self.nu: float = 10.0

    def fit(self, data: np.ndarray) -> None:
        """Fit via rank correlation (rho) and MLE for degrees of freedom.

        Args:
            data: Array of shape ``(n, 2)``.
        """
        u = _to_uniform(data)
        rho_s, _ = spearmanr(u[:, 0], u[:, 1])
        self.rho = float(2 * np.sin(np.pi * rho_s / 6))

        # MLE for nu, conditional on rho
        u_clipped = np.clip(u, 1e-10, 1 - 1e-10)

        def neg_ll(log_nu: float) -> float:
            nu = np.exp(log_nu)
            if nu < 2.01 or nu > 200:
                return 1e10
            z = stats.t.ppf(u_clipped, df=nu)
            rho = self.rho
            if abs(rho) > 0.9999:
                rho = np.sign(rho) * 0.9999
            det = 1 - rho**2

            # t-copula density
            n_obs = z.shape[0]
            quad = (z[:, 0]**2 + z[:, 1]**2 - 2 * rho * z[:, 0] * z[:, 1]) / det
            ll = n_obs * (
                special.gammaln((nu + 2) / 2)
                - special.gammaln(nu / 2)
                - np.log(np.pi * nu)
                - 0.5 * np.log(det)
            )
            ll += -(nu + 2) / 2 * np.sum(np.log(1 + quad / nu))
            # Subtract marginal t densities
            ll -= np.sum(stats.t.logpdf(z[:, 0], df=nu))
            ll -= np.sum(stats.t.logpdf(z[:, 1], df=nu))
            return -ll

        result = optimize.minimize_scalar(neg_ll, bounds=(np.log(2.1), np.log(100)), method="bounded")
        self.nu = float(np.exp(result.x))

    def simulate(self, n: int, rng_seed: int | None = None) -> np.ndarray:
        """Simulate from bivariate Student-t copula.

        Args:
            n: Number of samples.
            rng_seed: Random seed.

        Returns:
            Array of shape ``(n, 2)`` with uniform marginals.
        """
        rng = np.random.default_rng(rng_seed)
        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        z = rng.multivariate_normal([0, 0], cov, size=n)
        chi2 = rng.chisquare(df=self.nu, size=n)
        t_samples = z / np.sqrt(chi2[:, np.newaxis] / self.nu)
        return stats.t.cdf(t_samples, df=self.nu)

    def log_likelihood(self, data: np.ndarray) -> float:
        """Log-likelihood of the Student-t copula density.

        Args:
            data: Array of shape ``(n, 2)``.

        Returns:
            Total log-likelihood.
        """
        u = _to_uniform(data)
        u = np.clip(u, 1e-10, 1 - 1e-10)
        z = stats.t.ppf(u, df=self.nu)
        rho = self.rho
        if abs(rho) > 0.9999:
            rho = np.sign(rho) * 0.9999
        det = 1 - rho**2
        nu = self.nu

        n_obs = z.shape[0]
        quad = (z[:, 0]**2 + z[:, 1]**2 - 2 * rho * z[:, 0] * z[:, 1]) / det
        ll = n_obs * (
            special.gammaln((nu + 2) / 2)
            - special.gammaln(nu / 2)
            - np.log(np.pi * nu)
            - 0.5 * np.log(det)
        )
        ll += -(nu + 2) / 2 * np.sum(np.log(1 + quad / nu))
        ll -= np.sum(stats.t.logpdf(z[:, 0], df=nu))
        ll -= np.sum(stats.t.logpdf(z[:, 1], df=nu))
        return float(ll)

    def n_params(self) -> int:
        return 2

    def lower_tail_dependence(self) -> float:
        """Analytical lower tail dependence for Student-t copula.

        lambda_L = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
        """
        nu, rho = self.nu, self.rho
        if abs(1 + rho) < 1e-10:
            return 0.0
        arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        return float(2 * stats.t.cdf(arg, df=nu + 1))

    def upper_tail_dependence(self) -> float:
        """Student-t copula has symmetric tail dependence."""
        return self.lower_tail_dependence()

    def __repr__(self) -> str:
        return f"StudentTCopula(rho={self.rho:.4f}, nu={self.nu:.2f})"


class ClaytonCopula(BaseCopula):
    """Clayton copula: lower tail dependence, no upper tail dependence.

    Particularly useful for modelling joint crashes in financial markets.
    """

    def __init__(self) -> None:
        self.theta: float = 1.0

    def fit(self, data: np.ndarray) -> None:
        """Fit via Kendall's tau inversion: theta = 2*tau / (1 - tau).

        Args:
            data: Array of shape ``(n, 2)``.
        """
        u = _to_uniform(data)
        tau, _ = kendalltau(u[:, 0], u[:, 1])
        tau = float(tau)
        if tau <= 0:
            self.theta = 0.01  # near-independence
        else:
            self.theta = max(2 * tau / (1 - tau), 0.01)

    def simulate(self, n: int, rng_seed: int | None = None) -> np.ndarray:
        """Simulate from Clayton copula using conditional method.

        Args:
            n: Number of samples.
            rng_seed: Random seed.

        Returns:
            Array of shape ``(n, 2)`` with uniform marginals.
        """
        rng = np.random.default_rng(rng_seed)
        u1 = rng.uniform(size=n)
        w = rng.uniform(size=n)
        # Conditional inverse: u2 = (u1^{-theta} * (w^{-theta/(1+theta)} - 1) + 1)^{-1/theta}
        theta = self.theta
        u2 = (u1**(-theta) * (w**(-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
        return np.column_stack([u1, u2])

    def log_likelihood(self, data: np.ndarray) -> float:
        """Log-likelihood of the Clayton copula density.

        Args:
            data: Array of shape ``(n, 2)``.

        Returns:
            Total log-likelihood.
        """
        u = _to_uniform(data)
        u = np.clip(u, 1e-10, 1 - 1e-10)
        theta = self.theta
        n_obs = u.shape[0]

        ll = n_obs * np.log(1 + theta)
        ll += -(1 + theta) * np.sum(np.log(u[:, 0]) + np.log(u[:, 1]))
        ll += -(2 + 1 / theta) * np.sum(
            np.log(u[:, 0]**(-theta) + u[:, 1]**(-theta) - 1)
        )
        return float(ll)

    def n_params(self) -> int:
        return 1

    def lower_tail_dependence(self) -> float:
        """lambda_L = 2^{-1/theta} for Clayton copula."""
        return float(2 ** (-1 / self.theta))

    def upper_tail_dependence(self) -> float:
        """Clayton copula has zero upper tail dependence."""
        return 0.0

    def __repr__(self) -> str:
        return f"ClaytonCopula(theta={self.theta:.4f})"


class GumbelCopula(BaseCopula):
    """Gumbel copula: upper tail dependence, no lower tail dependence.

    Models joint booms or simultaneous upside extremes.
    """

    def __init__(self) -> None:
        self.theta: float = 1.5

    def fit(self, data: np.ndarray) -> None:
        """Fit via Kendall's tau inversion: theta = 1 / (1 - tau).

        Args:
            data: Array of shape ``(n, 2)``.
        """
        u = _to_uniform(data)
        tau, _ = kendalltau(u[:, 0], u[:, 1])
        tau = float(tau)
        if tau <= 0:
            self.theta = 1.01  # near-independence
        else:
            self.theta = max(1 / (1 - tau), 1.01)

    def simulate(self, n: int, rng_seed: int | None = None) -> np.ndarray:
        """Simulate from Gumbel copula via Marshall-Olkin method.

        Uses a stable distribution to generate the frailty variable.

        Args:
            n: Number of samples.
            rng_seed: Random seed.

        Returns:
            Array of shape ``(n, 2)`` with uniform marginals.
        """
        rng = np.random.default_rng(rng_seed)
        theta = self.theta
        alpha = 1.0 / theta

        # Generate stable(alpha) variate via Chambers-Mallows-Stuck
        v = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
        w = rng.exponential(1.0, size=n)

        if abs(alpha - 1.0) < 1e-10:
            s = np.ones(n)
        else:
            num = np.sin(alpha * (v + np.pi / 2))
            den = (np.cos(v)) ** (1 / alpha)
            frac = np.cos(v - alpha * (v + np.pi / 2)) / w
            s = num / den * frac ** ((1 - alpha) / alpha)

        # Generate exponentials and transform
        e1 = rng.exponential(1.0, size=n)
        e2 = rng.exponential(1.0, size=n)

        u1 = np.exp(-(e1 / s) ** alpha)
        u2 = np.exp(-(e2 / s) ** alpha)
        return np.column_stack([u1, u2])

    def log_likelihood(self, data: np.ndarray) -> float:
        """Log-likelihood of the Gumbel copula density.

        Args:
            data: Array of shape ``(n, 2)``.

        Returns:
            Total log-likelihood.
        """
        u = _to_uniform(data)
        u = np.clip(u, 1e-10, 1 - 1e-10)
        theta = self.theta

        t1 = (-np.log(u[:, 0])) ** theta
        t2 = (-np.log(u[:, 1])) ** theta
        s = t1 + t2
        C = np.exp(-s ** (1 / theta))

        # Bivariate density via second mixed partial
        log_c = (
            np.log(C)
            + np.log(s ** (1 / theta) + theta - 1)
            + (1 / theta - 1) * np.log(s)
            + (theta - 1) * (np.log(-np.log(u[:, 0])) + np.log(-np.log(u[:, 1])))
            - np.log(u[:, 0]) - np.log(u[:, 1])
            - np.log(-np.log(u[:, 0])) - np.log(-np.log(u[:, 1]))
        )
        return float(np.sum(log_c))

    def n_params(self) -> int:
        return 1

    def lower_tail_dependence(self) -> float:
        """Gumbel copula has zero lower tail dependence."""
        return 0.0

    def upper_tail_dependence(self) -> float:
        """lambda_U = 2 - 2^{1/theta} for Gumbel copula."""
        return float(2 - 2 ** (1 / self.theta))

    def __repr__(self) -> str:
        return f"GumbelCopula(theta={self.theta:.4f})"


class FrankCopula(BaseCopula):
    """Frank copula: symmetric, zero tail dependence.

    Useful as a null model or when dependence is symmetric and
    concentrated in the interior rather than the tails.
    """

    def __init__(self) -> None:
        self.theta: float = 1.0

    def fit(self, data: np.ndarray) -> None:
        """Fit via Kendall's tau inversion (numerical).

        Args:
            data: Array of shape ``(n, 2)``.
        """
        u = _to_uniform(data)
        tau, _ = kendalltau(u[:, 0], u[:, 1])
        tau = float(tau)

        # tau = 1 - 4/theta * (1 - D_1(theta))
        # where D_1 is the first Debye function
        # Solve numerically
        def _debye1(t: float) -> float:
            if abs(t) < 1e-10:
                return 1.0
            from scipy.integrate import quad
            result, _ = quad(lambda x: x / (np.exp(x) - 1), 0, t)
            return result / t

        def tau_from_theta(t: float) -> float:
            if abs(t) < 1e-10:
                return 0.0
            return 1 - 4 / t * (1 - _debye1(t))

        def objective(t: float) -> float:
            return tau_from_theta(t) - tau

        # Search for theta
        if abs(tau) < 0.01:
            self.theta = 0.01
        else:
            try:
                sign = 1 if tau > 0 else -1
                result = optimize.brentq(objective, sign * 0.01, sign * 50)
                self.theta = float(result)
            except ValueError:
                self.theta = 0.01

    def simulate(self, n: int, rng_seed: int | None = None) -> np.ndarray:
        """Simulate from Frank copula using conditional method.

        Args:
            n: Number of samples.
            rng_seed: Random seed.

        Returns:
            Array of shape ``(n, 2)`` with uniform marginals.
        """
        rng = np.random.default_rng(rng_seed)
        u1 = rng.uniform(size=n)
        w = rng.uniform(size=n)
        theta = self.theta

        if abs(theta) < 1e-10:
            return np.column_stack([u1, rng.uniform(size=n)])

        # Conditional inverse
        u2 = -np.log(
            1 + w * (np.exp(-theta) - 1) / (
                w + (1 - w) * np.exp(-theta * u1)
            )
        ) / theta
        u2 = np.clip(u2, 0, 1)
        return np.column_stack([u1, u2])

    def log_likelihood(self, data: np.ndarray) -> float:
        """Log-likelihood of the Frank copula density.

        Args:
            data: Array of shape ``(n, 2)``.

        Returns:
            Total log-likelihood.
        """
        u = _to_uniform(data)
        u = np.clip(u, 1e-10, 1 - 1e-10)
        theta = self.theta

        if abs(theta) < 1e-10:
            return 0.0

        et = np.exp(-theta)
        etu1 = np.exp(-theta * u[:, 0])
        etu2 = np.exp(-theta * u[:, 1])
        etu12 = np.exp(-theta * (u[:, 0] + u[:, 1]))

        num = -theta * (et - 1) * etu12
        den = ((et - 1) + (etu1 - 1) * (etu2 - 1)) ** 2

        log_c = np.log(np.abs(num)) - np.log(np.abs(den))
        return float(np.sum(log_c))

    def n_params(self) -> int:
        return 1

    def lower_tail_dependence(self) -> float:
        """Frank copula has zero tail dependence."""
        return 0.0

    def upper_tail_dependence(self) -> float:
        """Frank copula has zero tail dependence."""
        return 0.0

    def __repr__(self) -> str:
        return f"FrankCopula(theta={self.theta:.4f})"


def select_best_copula(data: np.ndarray) -> CopulaFitResult:
    """Fit all copulas and return the best by AIC.

    Args:
        data: Array of shape ``(n, 2)`` with raw observations.

    Returns:
        A ``CopulaFitResult`` for the best-fitting copula.
    """
    copulas: list[tuple[str, BaseCopula]] = [
        ("Gaussian", GaussianCopula()),
        ("Student-t", StudentTCopula()),
        ("Clayton", ClaytonCopula()),
        ("Gumbel", GumbelCopula()),
        ("Frank", FrankCopula()),
    ]

    results: list[CopulaFitResult] = []
    n = data.shape[0]

    for name, cop in copulas:
        try:
            cop.fit(data)
            ll = cop.log_likelihood(data)
            k = cop.n_params()
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll
            results.append(CopulaFitResult(
                name=name, copula=cop, log_likelihood=ll,
                aic=aic, bic=bic, n_params=k,
            ))
        except Exception:
            continue

    if not results:
        raise RuntimeError("No copula could be fitted to the data")

    return min(results, key=lambda r: r.aic)
