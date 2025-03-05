"""Tests for copula fitting, simulation, and tail dependence."""

from __future__ import annotations

import numpy as np
import pytest

from quantlite.dependency.copulas import (
    ClaytonCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    StudentTCopula,
    select_best_copula,
)


@pytest.fixture()
def correlated_data() -> np.ndarray:
    """Generate bivariate normal data with known correlation."""
    rng = np.random.default_rng(42)
    cov = [[1.0, 0.6], [0.6, 1.0]]
    return rng.multivariate_normal([0, 0], cov, size=500)


class TestGaussianCopula:
    def test_fit_recovers_correlation(self, correlated_data: np.ndarray) -> None:
        cop = GaussianCopula()
        cop.fit(correlated_data)
        assert 0.4 < cop.rho < 0.8

    def test_simulate_shape(self, correlated_data: np.ndarray) -> None:
        cop = GaussianCopula()
        cop.fit(correlated_data)
        samples = cop.simulate(200, rng_seed=1)
        assert samples.shape == (200, 2)
        assert np.all((samples >= 0) & (samples <= 1))

    def test_tail_dependence_zero(self) -> None:
        cop = GaussianCopula()
        cop.rho = 0.5
        assert cop.lower_tail_dependence() == 0.0
        assert cop.upper_tail_dependence() == 0.0

    def test_log_likelihood_finite(self, correlated_data: np.ndarray) -> None:
        cop = GaussianCopula()
        cop.fit(correlated_data)
        ll = cop.log_likelihood(correlated_data)
        assert np.isfinite(ll)


class TestStudentTCopula:
    def test_fit_recovers_parameters(self, correlated_data: np.ndarray) -> None:
        cop = StudentTCopula()
        cop.fit(correlated_data)
        assert 0.3 < cop.rho < 0.9
        assert cop.nu > 2

    def test_simulate_shape(self, correlated_data: np.ndarray) -> None:
        cop = StudentTCopula()
        cop.fit(correlated_data)
        samples = cop.simulate(200, rng_seed=1)
        assert samples.shape == (200, 2)

    def test_tail_dependence_positive(self) -> None:
        cop = StudentTCopula()
        cop.rho = 0.5
        cop.nu = 4.0
        td_lower = cop.lower_tail_dependence()
        td_upper = cop.upper_tail_dependence()
        assert td_lower > 0
        assert td_lower == td_upper  # symmetric


class TestClaytonCopula:
    def test_fit(self, correlated_data: np.ndarray) -> None:
        cop = ClaytonCopula()
        cop.fit(correlated_data)
        assert cop.theta > 0

    def test_simulate_shape(self, correlated_data: np.ndarray) -> None:
        cop = ClaytonCopula()
        cop.fit(correlated_data)
        samples = cop.simulate(200, rng_seed=1)
        assert samples.shape == (200, 2)

    def test_lower_tail_dependence(self) -> None:
        cop = ClaytonCopula()
        cop.theta = 2.0
        td = cop.lower_tail_dependence()
        expected = 2 ** (-1 / 2.0)
        assert abs(td - expected) < 1e-10

    def test_upper_tail_dependence_zero(self) -> None:
        cop = ClaytonCopula()
        cop.theta = 2.0
        assert cop.upper_tail_dependence() == 0.0


class TestGumbelCopula:
    def test_fit(self, correlated_data: np.ndarray) -> None:
        cop = GumbelCopula()
        cop.fit(correlated_data)
        assert cop.theta >= 1.0

    def test_upper_tail_dependence(self) -> None:
        cop = GumbelCopula()
        cop.theta = 2.0
        expected = 2 - 2 ** 0.5
        assert abs(cop.upper_tail_dependence() - expected) < 1e-10

    def test_lower_tail_dependence_zero(self) -> None:
        cop = GumbelCopula()
        cop.theta = 2.0
        assert cop.lower_tail_dependence() == 0.0


class TestFrankCopula:
    def test_fit(self, correlated_data: np.ndarray) -> None:
        cop = FrankCopula()
        cop.fit(correlated_data)
        assert cop.theta > 0  # positive dependence data

    def test_tail_dependence_zero(self) -> None:
        cop = FrankCopula()
        assert cop.lower_tail_dependence() == 0.0
        assert cop.upper_tail_dependence() == 0.0


class TestSelectBestCopula:
    def test_returns_result(self, correlated_data: np.ndarray) -> None:
        result = select_best_copula(correlated_data)
        assert result.name in {"Gaussian", "Student-t", "Clayton", "Gumbel", "Frank"}
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)
