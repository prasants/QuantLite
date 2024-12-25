import pytest
import pandas as pd
from quantlite.monte_carlo import run_monte_carlo_sims, multi_asset_correlated_sim

def test_run_monte_carlo_sims_basic():
    """
    Tests single-asset approach (default 'perturb' mode).
    Checks we get multiple sim results, each with a final_value.
    """
    price_data = pd.Series([100, 101, 102], index=[0,1,2])
    def always_buy(idx, series):
        return 1

    results = run_monte_carlo_sims(price_data, always_buy, n_sims=3, rng_seed=42)
    assert len(results) == 3, "Should generate 3 simulation results"

    for res in results:
        assert "final_value" in res
        assert "portfolio_value" in res

def test_run_monte_carlo_sims_replace_mode():
    """
    Tests 'replace' mode for run_monte_carlo_sims.
    Ensures we get a different final_value from default mode.
    """
    price_data = pd.Series([100, 101, 102], index=[0,1,2])
    def always_buy(idx, series):
        return 1

    res_default = run_monte_carlo_sims(price_data, always_buy, n_sims=1, rng_seed=123)
    res_replace = run_monte_carlo_sims(price_data, always_buy, n_sims=1, rng_seed=123, mode="replace")

    default_final = res_default[0]["final_value"]
    replace_final = res_replace[0]["final_value"]
    assert default_final != replace_final, "Different modes should yield different final values"

def test_multi_asset_correlated_sim():
    """
    Tests generating multi-asset correlated paths 
    to see we get DataFrame of correct shape and repeated seeds match.
    """
    S0_list = [100, 50]
    mu_list = [0.05, 0.02]
    # Minimal 2x2 cov matrix
    import numpy as np
    cov_matrix = np.array([[0.04, 0.01],
                           [0.01, 0.03]])

    df1 = multi_asset_correlated_sim(S0_list, mu_list, cov_matrix, steps=5, rng_seed=42)
    df2 = multi_asset_correlated_sim(S0_list, mu_list, cov_matrix, steps=5, rng_seed=42)

    assert df1.shape == (6, 2), "6 rows (steps+1) and 2 columns"
    assert df1.equals(df2), "Same seed => identical output"

