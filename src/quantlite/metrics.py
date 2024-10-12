import numpy as np
import pandas as pd

def annualised_return(returns, freq=252):
    if isinstance(returns, pd.Series):
        returns = returns.values
    arr = np.array(returns)
    if len(arr) == 0:
        return 0.0
    cumulative = (1 + arr).prod()
    return cumulative ** (freq / len(arr)) - 1

def annualised_volatility(returns, freq=252):
    if isinstance(returns, pd.Series):
        returns = returns.values
    arr = np.array(returns)
    if len(arr) == 0:
        return 0.0
    return np.std(arr, ddof=1) * np.sqrt(freq)

def sharpe_ratio(returns, risk_free_rate=0.0, freq=252):
    ar = annualised_return(returns, freq=freq)
    av = annualised_volatility(returns, freq=freq)
    if av == 0:
        return float("nan")
    return (ar - risk_free_rate) / av

def max_drawdown(returns):
    if isinstance(returns, pd.Series):
        returns = returns.values
    arr = np.array(returns)
    if len(arr) == 0:
        return 0.0
    cum = (1 + arr).cumprod()
    roll_max = np.maximum.accumulate(cum)
    drawdown = (cum - roll_max) / roll_max
    return drawdown.min()

