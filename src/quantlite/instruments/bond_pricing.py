def bond_price(face_value, coupon_rate, market_rate, maturity, payments_per_year=1):
    """
    Calculates the price of a coupon bond.

    payments_per_year : int
        e.g., 2 for semi-annual, 4 for quarterly.
    """
    coupon = face_value * coupon_rate / payments_per_year
    periods = maturity * payments_per_year
    rate_per_period = market_rate / payments_per_year

    total = 0.0
    for t in range(1, periods + 1):
        total += coupon / ((1 + rate_per_period)**t)
    total += face_value / ((1 + rate_per_period)**periods)
    return total

def bond_duration(face_value, coupon_rate, market_rate, maturity, payments_per_year=1):
    """
    Macaulay Duration for a standard coupon bond.
    """
    coupon = face_value * coupon_rate / payments_per_year
    periods = maturity * payments_per_year
    rate_per_period = market_rate / payments_per_year

    total_price = bond_price(face_value, coupon_rate, market_rate, maturity, payments_per_year)
    weighted_sum = 0.0

    for t in range(1, periods + 1):
        # time in years from now
        time = t / payments_per_year
        pv = coupon / ((1 + rate_per_period)**t)
        weighted_sum += time * pv
    # Add principal redemption
    weighted_sum += maturity * face_value / ((1 + rate_per_period)**periods)

    return weighted_sum / total_price

def bond_yield_to_maturity(face_value, coupon_rate, current_price, maturity, payments_per_year=1, guess=0.05, tol=1e-7):
    def price_at_rate(r):
        return bond_price(face_value, coupon_rate, r, maturity, payments_per_year)

    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2
        diff = price_at_rate(mid) - current_price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2
