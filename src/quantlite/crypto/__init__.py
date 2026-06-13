"""Crypto-native risk analysis tools.

Provides stablecoin depeg analysis, exchange risk assessment,
and on-chain risk metrics for crypto-native portfolios.
"""

from .exchange import (
    concentration_score,
    liquidity_risk,
    proof_of_reserves_check,
    slippage_estimate,
    wallet_risk_assessment,
)
from .onchain import (
    defi_dependency_graph,
    smart_contract_risk_score,
    tvl_tracker,
    wallet_exposure,
)
from .stablecoin import (
    HISTORICAL_DEPEGS,
    depeg_probability,
    depeg_recovery_time,
    peg_deviation_tracker,
    reserve_risk_score,
)

__all__ = [
    "HISTORICAL_DEPEGS",
    "concentration_score",
    "defi_dependency_graph",
    "depeg_probability",
    "depeg_recovery_time",
    "liquidity_risk",
    "peg_deviation_tracker",
    "proof_of_reserves_check",
    "reserve_risk_score",
    "slippage_estimate",
    "smart_contract_risk_score",
    "tvl_tracker",
    "wallet_exposure",
    "wallet_risk_assessment",
]
