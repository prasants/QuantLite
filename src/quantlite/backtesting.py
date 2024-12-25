import pandas as pd

def run_backtest(
    price_data,
    signal_function,
    initial_capital=10_000.0,
    fee=0.0,
    partial_capital=False,
    capital_fraction=1.0,
    allow_short=True,
    per_share_cost=0.0
):
    """
    Enhanced backtest with partial capital allocation and short-selling toggles.

    Parameters
    ----------
    price_data : pd.Series
        Price series of the asset.
    signal_function : callable
        (idx, price_data) -> +1 (long), -1 (short), or 0 (flat).
    initial_capital : float
        Starting capital.
    fee : float
        A flat transaction fee each time you change position.
    partial_capital : bool
        If True, we only allocate a fraction (capital_fraction) of current capital 
        to the position, leaving the rest in cash.
    capital_fraction : float
        Fraction of available capital allocated to the position. (e.g., 0.5 means half in the trade)
    allow_short : bool
        If False, signals that return -1 will be treated as 0 (no shorting allowed).
    per_share_cost : float
        Commission per share traded. e.g., 0.01 means 1 cent commission per share.

    Returns
    -------
    dict
        - "portfolio_value": pd.Series of portfolio value
        - "positions": pd.Series of position size (in shares)
        - "trades": list of (date, action, shares, price, cost)
        - "final_value": float
    """
    if not isinstance(price_data, pd.Series):
        raise ValueError("price_data must be a pd.Series")

    price_data = price_data.sort_index()
    dates = price_data.index
    capital = initial_capital
    current_shares = 0
    portfolio_values = []
    positions = []
    trades = []

    for i, date in enumerate(dates):
        price = price_data.iloc[i]
        raw_signal = signal_function(i, price_data)
        # If shorting is not allowed, clamp raw_signal to 0 or +1
        if not allow_short and raw_signal < 0:
            raw_signal = 0
        # The new target position is raw_signal: 1 = long, -1 = short, 0 = flat
        target_position = raw_signal

        # Determine how many shares we want
        # partial_capital => only a fraction of capital goes into shares
        desired_capital_for_position = capital * capital_fraction if partial_capital else capital
        if desired_capital_for_position < 0:
            desired_capital_for_position = 0

        if target_position == 0:
            desired_shares = 0
        else:
            # If target_position == 1, we buy as many shares as desired_capital_for_position can afford
            # If target_position == -1, we 'short' that many shares
            # floor shares to an integer
            desired_shares = int(desired_capital_for_position // price) * target_position

        if desired_shares != current_shares:
            # We have a trade
            delta_shares = desired_shares - current_shares
            trade_cost = fee  # flat fee
            # Add per-share cost
            trade_cost += abs(delta_shares) * per_share_cost

            # If delta_shares > 0 => buying; if < 0 => selling/covering
            value_of_shares = delta_shares * price * -1  # negative for capital if buying
            capital += value_of_shares - trade_cost  # buy => capital decreases

            trades.append((date, 
                           "buy" if delta_shares > 0 else "sell", 
                           delta_shares,
                           price,
                           trade_cost))

            current_shares = desired_shares

        # Compute portfolio value => capital + shares * price
        port_val = capital + current_shares * price
        portfolio_values.append(port_val)
        positions.append(current_shares)

    portfolio_series = pd.Series(portfolio_values, index=dates)
    positions_series = pd.Series(positions, index=dates)

    return {
        "portfolio_value": portfolio_series,
        "positions": positions_series,
        "trades": trades,
        "final_value": portfolio_series.iloc[-1]
    }
