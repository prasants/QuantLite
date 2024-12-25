import numpy as np
import pandas as pd

def geometric_brownian_motion(
    S0=100.0,
    mu=0.05,
    sigma=0.2,
    dt=1/252,
    steps=252,
    rng_seed=None,
    return_as="array"
):
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if sigma < 0:
        raise ValueError("sigma cannot be negative")
    if rng_seed is not None:
        np.random.seed(rng_seed)

    prices = np.zeros(steps + 1)
    prices[0] = S0
    for i in range(1, steps + 1):
        dW = np.random.normal(0, np.sqrt(dt))
        prices[i] = prices[i-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)

    if return_as == "series":
        return pd.Series(prices, index=range(steps+1), name="GBM")
    return prices

def correlated_gbm(
    S0_list,
    mu_list,
    cov_matrix,
    steps=252,
    dt=1/252,
    rng_seed=None,
    return_as="dataframe"
):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    n_assets = len(S0_list)
    if len(mu_list) != n_assets:
        raise ValueError("mu_list length must match S0_list")
    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError("cov_matrix must be NxN where N=len(S0_list)")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    L = np.linalg.cholesky(cov_matrix)
    prices = np.zeros((steps+1, n_assets))
    prices[0, :] = S0_list

    for t in range(1, steps + 1):
        z = np.random.normal(size=n_assets)
        corr_z = L @ z

        for i in range(n_assets):
            sigma_i = np.sqrt(cov_matrix[i,i])
            drift = (mu_list[i] - 0.5*sigma_i**2) * dt
            dW = corr_z[i] * np.sqrt(dt)
            prices[t, i] = prices[t-1, i] * np.exp(drift + sigma_i*dW)

    if return_as == "dataframe":
        cols = [f"Asset_{i}" for i in range(n_assets)]
        return pd.DataFrame(prices, columns=cols)
    return prices

def ornstein_uhlenbeck(
    x0=0.0,
    theta=0.07,
    mu=0.0,
    sigma=0.1,
    dt=1/252,
    steps=252,
    rng_seed=None,
    return_as="array"
):
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if sigma < 0:
        raise ValueError("sigma cannot be negative")

    if rng_seed is not None:
        np.random.seed(rng_seed)

    x_vals = np.zeros(steps + 1)
    x_vals[0] = x0

    for t in range(1, steps+1):
        dx = theta*(mu - x_vals[t-1])*dt + sigma*np.sqrt(dt)*np.random.normal()
        x_vals[t] = x_vals[t-1] + dx

    if return_as == "series":
        return pd.Series(x_vals, index=range(steps+1), name="OU")
    return x_vals

def merton_jump_diffusion(
    S0=100.0,
    mu=0.05,
    sigma=0.2,
    lamb=0.5,
    jump_mean=0.0,
    jump_std=0.1,
    dt=1/252,
    steps=252,
    rng_seed=None,
    return_as="array"
):
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if rng_seed is not None:
        np.random.seed(rng_seed)

    prices = np.zeros(steps + 1)
    prices[0] = S0
    for t in range(1, steps+1):
        N_t = np.random.poisson(lamb * dt)
        jump_factor = 1.0
        if N_t > 0:
            total_log_jump = np.sum(np.random.normal(jump_mean, jump_std, N_t))
            jump_factor = np.exp(total_log_jump)

        dW = np.random.normal(0, np.sqrt(dt))
        drift = (mu - 0.5*sigma**2) * dt
        diffusion = sigma * dW
        prices[t] = prices[t-1] * np.exp(drift + diffusion) * jump_factor

    if return_as == "series":
        return pd.Series(prices, index=range(steps+1), name="MJD")
    return prices
