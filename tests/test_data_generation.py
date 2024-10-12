import numpy as np
import pytest
import pandas as pd
from quantlite.data_generation import (
    geometric_brownian_motion,
    ornstein_uhlenbeck,
    correlated_gbm,
    merton_jump_diffusion
)

def test_geometric_brownian_motion_basic():
    """
    Ensures basic functionality of GBM with default parameters.
    """
    path = geometric_brownian_motion(steps=5)
    assert len(path) == 6, "Should have 5 steps + initial = 6 data points"
    # Check it doesn't return a trivial array
    assert not np.allclose(path, path[0]), "Should not be flat"

def test_geometric_brownian_motion_custom():
    """
    Test a custom S0, mu, sigma, and rng_seed for reproducibility.
    """
    path = geometric_brownian_motion(S0=50, mu=0.1, sigma=0.3, steps=5, rng_seed=42)
    # Repeated call with same seed should give same results
    path2 = geometric_brownian_motion(S0=50, mu=0.1, sigma=0.3, steps=5, rng_seed=42)
    assert np.allclose(path, path2), "Same seed must yield identical paths"

def test_ornstein_uhlenbeck_basic():
    """
    Ensures OU process returns correct length and changes over time.
    """
    path = ornstein_uhlenbeck(steps=5)
    assert len(path) == 6, "OU should also have 5 steps + initial"

def test_correlated_gbm():
    """
    Basic test for multi-asset correlated GBM. 
    We check shape and also some correlation properties.
    """
    S0_list = [100, 50]
    mu_list = [0.05, 0.02]
    # Simple 2x2 covariance matrix with mild correlation
    cov_matrix = np.array([[0.04, 0.01],
                           [0.01, 0.03]])
    df = correlated_gbm(S0_list, mu_list, cov_matrix, steps=5, rng_seed=42, return_as="dataframe")
    assert df.shape == (6, 2), "Should have 6 rows (5 steps + initial) and 2 assets"
    # Re-run with same seed => same results
    df2 = correlated_gbm(S0_list, mu_list, cov_matrix, steps=5, rng_seed=42, return_as="dataframe")
    assert df.equals(df2), "Same seed must yield identical DataFrame"

def test_merton_jump_diffusion():
    """
    Test the Merton Jump Diffusion for correct length and non-trivial jumps.
    """
    path = merton_jump_diffusion(S0=100, mu=0.05, sigma=0.2, lamb=0.5,
                                 jump_mean=0.0, jump_std=0.1,
                                 steps=5, rng_seed=123)
    assert len(path) == 6, "Should have steps+1 data points"
    # Ensure it's not flat
    assert not np.allclose(path, path[0]), "Path should vary"
