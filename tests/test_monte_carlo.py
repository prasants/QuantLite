"""Tests for quantlite.monte_carlo."""

import pandas as pd

from quantlite.monte_carlo import run_monte_carlo_sims


def test_mc_sims_basic():
    prices = pd.Series([100, 101, 102], index=[0, 1, 2])
    results = run_monte_carlo_sims(prices, lambda i, s: 1, n_sims=3, rng_seed=42)
    assert len(results) == 3
    for r in results:
        assert "final_value" in r


def test_mc_sims_replace_mode():
    prices = pd.Series([100, 101, 102], index=[0, 1, 2])
    r1 = run_monte_carlo_sims(prices, lambda i, s: 1, n_sims=1, rng_seed=123)
    r2 = run_monte_carlo_sims(prices, lambda i, s: 1, n_sims=1, rng_seed=123, mode="replace")
    assert r1[0]["final_value"] != r2[0]["final_value"]
