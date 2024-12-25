import numpy as np
import pandas as pd
from .backtesting import run_backtest

def run_monte_carlo_sims(
    price_data,
    signal_function,
    n_sims=100,
    noise_scale=0.01,
    mode="perturb",
    rng_seed=None
):
    """
    (Unchanged from previous) Runs multiple random simulations for a single-asset case.
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    if not isinstance(price_data, pd.Series):
        raise ValueError("price_data must be a pandas Series")

    results = []
    base_index = price_data.index
    base_values = price_data.values

    for _ in range(n_sims):
        if mode == "perturb":
            noise = np.random.normal(loc=0.0, scale=noise_scale, size=len(base_values))
            sim_vals = base_values * (1 + noise)
        elif mode == "gbm":
            sim_vals = _simulate_gbm_series(base_values[0], len(base_values))
        elif mode == "replace":
            daily_returns = np.random.normal(loc=0.0, scale=noise_scale, size=len(base_values) - 1)
            sim_vals = [base_values[0]]
            for dr in daily_returns:
                sim_vals.append(sim_vals[-1] * (1 + dr))
        else:
            raise ValueError("Unknown mode. Choose from ['perturb', 'gbm', 'replace'].")

        sim_prices = pd.Series(sim_vals, index=base_index)
        result = run_backtest(sim_prices, signal_function)
        results.append(result)

    return results

def multi_asset_correlated_sim(
    S0_list,
    mu_list,
    cov_matrix,
    steps=252,
    dt=1/252,
    rng_seed=None
):
    """
    Generates correlated multi-asset price paths using a GBM approach.

    Parameters
    ----------
    S0_list : list of floats
        Initial prices for each asset (e.g., [100.0, 50.0]).
    mu_list : list of floats
        Annual drift for each asset (e.g., [0.05, 0.02] for 5% and 2%).
    cov_matrix : 2D numpy array
        Covariance matrix for the assets' returns. Should be size NxN where N is len(S0_list).
    steps : int
        Number of time steps to simulate.
    dt : float
        Time step in years (1/252 for daily).
    rng_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        DataFrame of shape (steps+1, len(S0_list)), each column is an asset price path.
        Index is a simple range for time steps (0 to steps).
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    n_assets = len(S0_list)
    if len(mu_list) != n_assets:
        raise ValueError("mu_list must have the same length as S0_list")
    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError("cov_matrix dimension mismatch with S0_list")

    # Cholesky decomposition for correlation
    L = np.linalg.cholesky(cov_matrix)

    # Initialize the result array
    prices = np.zeros((steps + 1, n_assets))
    prices[0, :] = S0_list

    for t in range(1, steps + 1):
        # Draw standard normal
        z = np.random.normal(size=n_assets)
        # Correlate them
        correlated_z = L @ z

        for i in range(n_assets):
            # GBM step
            dW = correlated_z[i] * np.sqrt(dt)
            drift = (mu_list[i] - 0.5 * cov_matrix[i, i]) * dt
            prices[t, i] = prices[t - 1, i] * np.exp(drift + dW)

    columns = [f"Asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, columns=columns)

def _simulate_gbm_series(S0, steps, mu=0.05, sigma=0.2, dt=1/252):
    """
    A helper for single-asset GBM (unchanged from older version).
    """
    vals = [S0]
    for _ in range(steps - 1):
        dW = np.random.normal(0, np.sqrt(dt))
        new_val = vals[-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)
        vals.append(new_val)
    return np.array(vals)
