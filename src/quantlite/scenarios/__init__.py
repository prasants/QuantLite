"""Composable scenario engine for stress testing and fragility analysis.

Build custom crisis scenarios or use pre-built ones, then stress-test
portfolios to understand fragility before it finds you.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "Scenario",
    "SCENARIO_LIBRARY",
    "stress_test",
    "fragility_heatmap",
    "shock_propagation",
]


class Scenario:
    """A composable crisis scenario with a fluent API.

    Example
    -------
    >>> scenario = Scenario("China crisis") \\
    ...     .shock("CNY", -0.15) \\
    ...     .shock("BTC", -0.40) \\
    ...     .correlations(spike_to=0.85) \\
    ...     .duration(days=30)
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.shocks: dict[str, float] = {}
        self.correlation_spike: float | None = None
        self.duration_days: int | None = None
        self.metadata: dict[str, Any] = {}

    def shock(self, asset: str, magnitude: float) -> Scenario:
        """Add a shock to an asset.

        Parameters
        ----------
        asset : str
            Asset identifier.
        magnitude : float
            Shock magnitude (e.g. -0.40 for a 40% drop).
        """
        self.shocks[asset] = magnitude
        return self

    def correlations(self, spike_to: float) -> Scenario:
        """Set the correlation spike level during the scenario.

        Parameters
        ----------
        spike_to : float
            Target correlation level (typically 0.7 to 0.95).
        """
        self.correlation_spike = spike_to
        return self

    def duration(self, days: int) -> Scenario:
        """Set the scenario duration.

        Parameters
        ----------
        days : int
            Number of trading days the scenario persists.
        """
        self.duration_days = days
        return self

    def __repr__(self) -> str:
        shocks_str = ", ".join(f"{k}: {v:+.1%}" for k, v in self.shocks.items())
        return f"Scenario('{self.name}', shocks=[{shocks_str}])"


def _build_library() -> dict[str, Scenario]:
    """Construct the pre-built scenario library."""
    scenarios = {}

    gfc = Scenario("2008 GFC")
    gfc.shock("SPX", -0.55).shock("HY_CREDIT", -0.30).shock("COMMODITIES", -0.40)
    gfc.correlations(spike_to=0.90).duration(days=250)
    scenarios["2008 GFC"] = gfc

    covid = Scenario("2020 COVID")
    covid.shock("SPX", -0.34).shock("OIL", -0.65).shock("BTC", -0.50)
    covid.correlations(spike_to=0.85).duration(days=30)
    scenarios["2020 COVID"] = covid

    luna = Scenario("2022 Luna/FTX")
    luna.shock("BTC", -0.65).shock("ETH", -0.70).shock("SOL", -0.95)
    luna.correlations(spike_to=0.92).duration(days=60)
    scenarios["2022 Luna/FTX"] = luna

    usdt = Scenario("USDT depeg")
    usdt.shock("USDT", -0.10).shock("BTC", -0.25).shock("ETH", -0.30)
    usdt.correlations(spike_to=0.80).duration(days=14)
    scenarios["USDT depeg"] = usdt

    rates = Scenario("rates +200bps")
    rates.shock("BONDS_10Y", -0.15).shock("SPX", -0.20).shock("GROWTH", -0.30)
    rates.correlations(spike_to=0.70).duration(days=90)
    scenarios["rates +200bps"] = rates

    return scenarios


SCENARIO_LIBRARY: dict[str, Scenario] = _build_library()


def stress_test(
    weights: dict[str, float],
    scenario: Scenario,
    returns: dict[str, ArrayLike] | None = None,
) -> dict[str, Any]:
    """Apply a scenario to a portfolio and return impact metrics.

    Parameters
    ----------
    weights : dict
        Mapping of asset names to portfolio weights.
    scenario : Scenario
        The crisis scenario to apply.
    returns : dict of array-like, optional
        Historical returns per asset. Used for volatility scaling
        if provided.

    Returns
    -------
    dict
        Keys: 'scenario_name', 'portfolio_impact' (weighted shock),
        'asset_impacts' (per-asset), 'worst_asset', 'best_asset',
        'survival' (bool: whether portfolio survives with > 0 value).
    """
    asset_impacts: dict[str, float] = {}
    for asset, weight in weights.items():
        shock_mag = scenario.shocks.get(asset, 0.0)

        # If we have historical returns, scale by relative volatility
        if returns is not None and asset in returns:
            hist = np.asarray(returns[asset], dtype=float).ravel()
            vol = np.std(hist)
            if vol > 0:
                # Amplify shock for high-vol assets
                vol_scale = min(vol / 0.01, 3.0)  # cap at 3x
                shock_mag = shock_mag * max(1.0, vol_scale * 0.5)

        asset_impacts[asset] = weight * shock_mag

    portfolio_impact = sum(asset_impacts.values())

    # Find worst and best assets
    if asset_impacts:
        worst_asset = min(asset_impacts, key=asset_impacts.get)  # type: ignore[arg-type]
        best_asset = max(asset_impacts, key=asset_impacts.get)  # type: ignore[arg-type]
    else:
        worst_asset = None
        best_asset = None

    survival = (1.0 + portfolio_impact) > 0

    return {
        "scenario_name": scenario.name,
        "portfolio_impact": portfolio_impact,
        "asset_impacts": asset_impacts,
        "worst_asset": worst_asset,
        "best_asset": best_asset,
        "survival": survival,
    }


def fragility_heatmap(
    weights: dict[str, float],
    scenarios: list[Scenario],
    returns: dict[str, ArrayLike] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute a fragility heatmap across scenarios.

    For each scenario, shows the impact on each position, classifying
    them as fragile, robust, or antifragile.

    Parameters
    ----------
    weights : dict
        Portfolio weights per asset.
    scenarios : list of Scenario
        Scenarios to test against.
    returns : dict of array-like, optional
        Historical returns per asset.

    Returns
    -------
    dict
        Nested dict: scenario_name -> asset -> impact value.
    """
    heatmap: dict[str, dict[str, float]] = {}
    for scenario in scenarios:
        result = stress_test(weights, scenario, returns)
        heatmap[scenario.name] = result["asset_impacts"]
    return heatmap


def shock_propagation(
    returns: dict[str, ArrayLike],
    initial_shock: dict[str, float],
    correlation_matrix: np.ndarray | None = None,
) -> dict[str, float]:
    """Model how an initial shock cascades through correlated assets.

    Parameters
    ----------
    returns : dict of array-like
        Historical returns per asset. Used to estimate correlations
        if no explicit matrix is provided.
    initial_shock : dict
        Asset -> shock magnitude for the initial event.
    correlation_matrix : ndarray, optional
        Pre-computed correlation matrix. If None, estimated from returns.

    Returns
    -------
    dict
        Asset -> propagated shock magnitude after cascade.
    """
    assets = list(returns.keys())
    n = len(assets)

    # Build return matrix
    min_len = min(np.asarray(returns[a], dtype=float).ravel().size for a in assets)
    ret_matrix = np.column_stack(
        [np.asarray(returns[a], dtype=float).ravel()[:min_len] for a in assets]
    )

    if correlation_matrix is None:
        if n < 2:
            # Single asset: just return the initial shock
            return {a: initial_shock.get(a, 0.0) for a in assets}
        correlation_matrix = np.corrcoef(ret_matrix, rowvar=False)

    # Build initial shock vector
    shock_vec = np.array([initial_shock.get(a, 0.0) for a in assets])

    # Propagate: shocked assets influence others via correlation
    # Each non-shocked asset absorbs correlated impact
    propagated = shock_vec.copy()
    for i, asset in enumerate(assets):
        if asset not in initial_shock:
            # Weighted average of correlations with shocked assets
            corr_impact = 0.0
            for j, other in enumerate(assets):
                if other in initial_shock:
                    corr_impact += correlation_matrix[i, j] * shock_vec[j]
            propagated[i] = corr_impact

    return {asset: float(propagated[i]) for i, asset in enumerate(assets)}
