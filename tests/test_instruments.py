import pytest
import math
from quantlite.instruments.bond_pricing import (
    bond_price,
    bond_yield_to_maturity,
    bond_duration
)
from quantlite.instruments.option_pricing import (
    black_scholes_call,
    black_scholes_put,
    black_scholes_greeks
)
from quantlite.instruments.exotic_options import (
    barrier_option_knock_out,
    asian_option_arithmetic
)

def test_bond_price_basic():
    """
    Check bond_price for par bond (coupon_rate == market_rate).
    Should be near face_value if maturity is moderate.
    """
    price = bond_price(face_value=1000, coupon_rate=0.05, market_rate=0.05, maturity=5)
    assert abs(price - 1000) < 2, "Par bond should be ~ face_value"

def test_bond_duration():
    """
    Check bond_duration for a simple annual coupon bond. 
    Duration should be > 0 and less than maturity.
    """
    dur = bond_duration(face_value=1000, coupon_rate=0.05, market_rate=0.05, maturity=5)
    assert 0 < dur < 5, "Macaulay duration is typically less than full maturity"

def test_bond_yield_to_maturity():
    """
    If bond_price is close to face_value, YTM ~ coupon_rate.
    """
    ytm = bond_yield_to_maturity(face_value=1000, coupon_rate=0.05,
                                 current_price=1000, maturity=5)
    assert abs(ytm - 0.05) < 0.01, "YTM should be ~ 5% if price=face_value"

def test_black_scholes_call_put():
    """
    Basic check on standard call/put outputs.
    """
    call_val = black_scholes_call(S=100, K=95, T=1, r=0.01, sigma=0.2)
    put_val = black_scholes_put(S=100, K=95, T=1, r=0.01, sigma=0.2)
    assert call_val > 0, "Vanilla call should be > 0 if S>K"
    assert put_val >= 0, "Vanilla put >= 0 always"

def test_black_scholes_greeks():
    """
    Ensure greeks dictionary has all required keys and are valid floats.
    """
    greeks = black_scholes_greeks(S=100, K=95, T=1, r=0.01, sigma=0.2, option_type="call")
    for key in ["Delta", "Gamma", "Vega", "Theta", "Rho"]:
        assert key in greeks
        assert isinstance(greeks[key], float), f"{key} must be a float"

def test_barrier_option_knock_out():
    """
    Test down-and-out barrier call with simple parameters, ensuring it's not zero or negative.
    """
    val = barrier_option_knock_out(
        S0=100, K=90, H=80, T=1, r=0.01, sigma=0.2,
        option_type="call", barrier_type="down-and-out",
        steps=50, sims=1000, rng_seed=42
    )
    assert val >= 0, "Option value can't be negative"
    # Typically some positive value
    assert val < 100, "Should be realistically less than S0"

def test_asian_option_arithmetic():
    """
    Checks arithmetic average Asian call is non-negative.
    """
    val = asian_option_arithmetic(
        S0=100, K=90, T=1.0, r=0.01, sigma=0.2,
        option_type="call", steps=50, sims=1000, rng_seed=42
    )
    assert val >= 0, "Asian option value can't be negative"
    assert val < 100, "Should be realistically less than S0"
