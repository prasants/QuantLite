"""Tests for quantlite.instruments (options, bonds, exotics)."""

import pytest

from quantlite.instruments.bond_pricing import bond_duration, bond_price, bond_yield_to_maturity
from quantlite.instruments.exotic_options import asian_option_arithmetic, barrier_option_knock_out
from quantlite.instruments.option_pricing import (
    black_scholes_call,
    black_scholes_greeks,
    black_scholes_put,
)


def test_bond_price_par():
    price = bond_price(1000, 0.05, 0.05, 5)
    assert abs(price - 1000) < 2


def test_bond_duration():
    dur = bond_duration(1000, 0.05, 0.05, 5)
    assert 0 < dur < 5


def test_bond_ytm():
    ytm = bond_yield_to_maturity(1000, 0.05, 1000, 5)
    assert abs(ytm - 0.05) < 0.01


def test_bs_call_put():
    call = black_scholes_call(100, 95, 1, 0.01, 0.2)
    put = black_scholes_put(100, 95, 1, 0.01, 0.2)
    assert call > 0
    assert put >= 0


def test_bs_greeks():
    g = black_scholes_greeks(100, 95, 1, 0.01, 0.2)
    assert hasattr(g, "delta")
    assert hasattr(g, "gamma")
    assert isinstance(g.vega, float)


def test_barrier_knock_out():
    val = barrier_option_knock_out(
        S0=100, K=90, H=80, T=1, r=0.01, sigma=0.2,
        steps=50, sims=1000, rng_seed=42,
    )
    assert 0 <= val < 100


def test_asian_arithmetic():
    val = asian_option_arithmetic(
        S0=100, K=90, T=1, r=0.01, sigma=0.2,
        steps=50, sims=1000, rng_seed=42,
    )
    assert 0 <= val < 100
