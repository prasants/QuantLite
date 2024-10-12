import math
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Returns a dictionary of Greeks for a European call or put using Black–Scholes.

    Parameters
    ----------
    option_type : str
        "call" or "put"

    Returns
    -------
    dict
        Keys: "Delta", "Gamma", "Vega", "Theta", "Rho"
    """
    if T <= 0:
        # At expiry, greeks for digital payoff are ambiguous
        return {"Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0, "Rho": 0}
    
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = 1.0 / math.sqrt(2 * math.pi) * math.exp(-0.5 * d1**2)

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (- (S * pdf_d1 * sigma) / (2 * math.sqrt(T))
                 - r * K * math.exp(-r*T) * norm.cdf(d2))
        rho = K * T * math.exp(-r*T) * norm.cdf(d2)
    else:  # put
        delta = norm.cdf(d1) - 1
        theta = (- (S * pdf_d1 * sigma) / (2 * math.sqrt(T))
                 + r * K * math.exp(-r*T) * norm.cdf(-d2))
        rho = -K * T * math.exp(-r*T) * norm.cdf(-d2)

    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * math.sqrt(T) * pdf_d1

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho
    }
