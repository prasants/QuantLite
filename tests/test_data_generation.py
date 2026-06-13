"""Tests for quantlite.data_generation."""

import numpy as np

from quantlite.data_generation import (
    correlated_gbm,
    geometric_brownian_motion,
    merton_jump_diffusion,
    ornstein_uhlenbeck,
)


def test_gbm_basic():
    path = geometric_brownian_motion(steps=5)
    assert len(path) == 6


def test_gbm_reproducibility():
    a = geometric_brownian_motion(S0=50, mu=0.1, sigma=0.3, steps=5, rng_seed=42)
    b = geometric_brownian_motion(S0=50, mu=0.1, sigma=0.3, steps=5, rng_seed=42)
    np.testing.assert_allclose(a, b)


def test_ou_basic():
    path = ornstein_uhlenbeck(steps=5)
    assert len(path) == 6


def test_correlated_gbm():
    S0 = [100, 50]
    mu = [0.05, 0.02]
    cov = np.array([[0.04, 0.01], [0.01, 0.03]])
    df = correlated_gbm(S0, mu, cov, steps=5, rng_seed=42)
    assert df.shape == (6, 2)
    df2 = correlated_gbm(S0, mu, cov, steps=5, rng_seed=42)
    assert df.equals(df2)


def test_merton_jump_diffusion():
    path = merton_jump_diffusion(steps=5, rng_seed=123)
    assert len(path) == 6
    assert not np.allclose(path, path[0])
