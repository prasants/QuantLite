import math
import numpy as np
from scipy.stats import norm

def barrier_option_knock_out(
    S0, 
    K, 
    H, 
    T, 
    r, 
    sigma, 
    option_type="call", 
    barrier_type="down-and-out",
    steps=1000,
    sims=10000,
    rng_seed=None
):
    """
    Monte Carlo pricing for a knock-out barrier option.
    If the underlying price touches the barrier H, the option immediately becomes worthless (knocks out).

    Parameters
    ----------
    S0 : float
        Initial underlying price.
    K : float
        Strike.
    H : float
        Barrier level.
    T : float
        Time to maturity (in years).
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility.
    option_type : str
        "call" or "put".
    barrier_type : str
        For this example, "down-and-out" only. (You could extend to up-and-out, etc.)
    steps : int
        Number of time steps in each simulation path.
    sims : int
        Number of simulated paths.
    rng_seed : int, optional
        Reproducibility.

    Returns
    -------
    float
        Estimated option value.
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    dt = T / steps
    disc_factor = math.exp(-r * T)
    payoffs = []

    for _ in range(sims):
        price = S0
        knocked_out = False
        for __ in range(steps):
            dW = np.random.normal(0, math.sqrt(dt))
            price *= math.exp((r - 0.5 * sigma**2)*dt + sigma*dW)

            # Barrier check
            if barrier_type == "down-and-out":
                if price <= H:
                    knocked_out = True
                    break
            else:
                # Extend for up-and-out, etc.
                pass

        if knocked_out:
            payoffs.append(0.0)
        else:
            if option_type == "call":
                payoffs.append(max(price - K, 0.0))
            else:  # put
                payoffs.append(max(K - price, 0.0))

    return disc_factor * np.mean(payoffs)

def asian_option_arithmetic(
    S0,
    K,
    T,
    r,
    sigma,
    option_type="call",
    steps=1000,
    sims=10000,
    rng_seed=None
):
    """
    Monte Carlo pricing of an arithmetic average price Asian option.

    The payoff depends on the average of prices over the path.
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    dt = T / steps
    disc_factor = math.exp(-r * T)
    payoffs = []

    for _ in range(sims):
        price = S0
        path_prices = [price]
        for __ in range(steps):
            dW = np.random.normal(0, math.sqrt(dt))
            price *= math.exp((r - 0.5 * sigma**2)*dt + sigma*dW)
            path_prices.append(price)
        average_price = np.mean(path_prices)
        if option_type == "call":
            payoffs.append(max(average_price - K, 0.0))
        else:
            payoffs.append(max(K - average_price, 0.0))

    return disc_factor * np.mean(payoffs)
