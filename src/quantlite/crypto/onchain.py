"""On-chain risk analysis: wallet exposure, TVL tracking, and smart contract scoring.

Tools for assessing DeFi and on-chain risks including protocol
dependency mapping, TVL concentration analysis, and composite
smart contract risk scoring.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "defi_dependency_graph",
    "smart_contract_risk_score",
    "tvl_tracker",
    "wallet_exposure",
]


def wallet_exposure(address, chain="ethereum"):
    """Return a mock wallet exposure summary for a given address.

    This is a skeleton implementation that returns synthetic data.
    In production, this would integrate with on-chain data providers
    (e.g. Etherscan, Alchemy, Moralis).

    Parameters
    ----------
    address : str
        Wallet address to analyse.
    chain : str, optional
        Blockchain network (default "ethereum").

    Returns
    -------
    dict
        Dictionary with keys:
        - address: the queried address
        - chain: the blockchain network
        - tokens: list of token holdings with name, balance, value_usd
        - total_value_usd: total portfolio value
        - concentration: HHI of token values
        - risk_flags: list of identified risks
        - note: disclaimer about mock data
    """
    # Mock data for demonstration and testing
    mock_tokens = [
        {"name": "ETH", "balance": 150.0, "value_usd": 450_000.0},
        {"name": "USDC", "balance": 500_000.0, "value_usd": 500_000.0},
        {"name": "USDT", "balance": 200_000.0, "value_usd": 200_000.0},
        {"name": "WBTC", "balance": 5.0, "value_usd": 250_000.0},
        {"name": "AAVE", "balance": 1000.0, "value_usd": 80_000.0},
    ]

    total = sum(t["value_usd"] for t in mock_tokens)
    shares = [(t["value_usd"] / total) ** 2 for t in mock_tokens]
    hhi = sum(shares)

    risk_flags = []
    stablecoin_pct = sum(
        t["value_usd"] for t in mock_tokens if t["name"] in ("USDC", "USDT", "DAI")
    ) / total * 100

    if hhi > 0.4:
        risk_flags.append("High concentration in few tokens")
    if stablecoin_pct > 60:
        risk_flags.append("Heavy stablecoin exposure: depeg risk")

    return {
        "address": address,
        "chain": chain,
        "tokens": mock_tokens,
        "total_value_usd": total,
        "concentration": float(hhi),
        "risk_flags": risk_flags,
        "note": "Mock data for demonstration. Integrate with on-chain provider for live data.",
    }


def tvl_tracker(protocol_tvls):
    """Compute TVL concentration, trend, and risk flags.

    Analyses total value locked across DeFi protocols to identify
    concentration risks and potential systemic vulnerabilities.

    Parameters
    ----------
    protocol_tvls : dict
        Mapping of protocol name to TVL value (current or time series).
        If values are floats, treated as current snapshot.
        If values are lists/arrays, treated as time series (most recent last).

    Returns
    -------
    dict
        Dictionary with keys:
        - total_tvl: aggregate TVL across all protocols
        - hhi: Herfindahl-Hirschman Index of TVL concentration
        - shares: dict of protocol to TVL share
        - dominant_protocol: protocol with largest share
        - trends: dict of protocol to trend info (if time series provided)
        - risk_flags: list of identified risks
        - risk_rating: qualitative rating
    """
    # Handle both snapshot and time series inputs
    current_tvls = {}
    trends = {}

    for name, value in protocol_tvls.items():
        if isinstance(value, (list, np.ndarray)):
            arr = np.asarray(value, dtype=float)
            current_tvls[name] = float(arr[-1])
            if len(arr) >= 2:
                pct_change = (arr[-1] - arr[0]) / arr[0] * 100 if arr[0] > 0 else 0.0
                recent_change = (arr[-1] - arr[-2]) / arr[-2] * 100 if arr[-2] > 0 else 0.0
                trends[name] = {
                    "total_change_pct": float(pct_change),
                    "recent_change_pct": float(recent_change),
                    "volatility": float(np.std(np.diff(arr) / arr[:-1]) * 100) if len(arr) > 2 else 0.0,
                    "direction": "growing" if pct_change > 5 else "shrinking" if pct_change < -5 else "stable",
                }
        else:
            current_tvls[name] = float(value)

    total_tvl = sum(current_tvls.values())

    if total_tvl <= 0:
        return {
            "total_tvl": 0.0,
            "hhi": 1.0,
            "shares": {},
            "dominant_protocol": None,
            "trends": trends,
            "risk_flags": ["No TVL detected"],
            "risk_rating": "critical",
        }

    shares = {name: tvl / total_tvl for name, tvl in current_tvls.items()}
    hhi = sum(s ** 2 for s in shares.values())

    dominant = max(shares, key=shares.get)
    risk_flags = []

    if shares[dominant] > 0.5:
        risk_flags.append(f"{dominant} holds {shares[dominant]:.0%} of TVL: extreme concentration")
    elif shares[dominant] > 0.3:
        risk_flags.append(f"{dominant} holds {shares[dominant]:.0%} of TVL: high concentration")

    if hhi > 0.4:
        risk_flags.append("TVL highly concentrated (HHI > 0.4)")

    # Check for rapid TVL declines
    for name, trend in trends.items():
        if trend["total_change_pct"] < -30:
            risk_flags.append(f"{name}: TVL declined {trend['total_change_pct']:.0f}% (potential bank run)")
        if trend["volatility"] > 20:
            risk_flags.append(f"{name}: high TVL volatility ({trend['volatility']:.1f}%)")

    if hhi > 0.4:
        risk_rating = "critical"
    elif hhi > 0.25 or len(risk_flags) > 2:
        risk_rating = "high"
    elif hhi > 0.15 or len(risk_flags) > 0:
        risk_rating = "medium"
    else:
        risk_rating = "low"

    return {
        "total_tvl": float(total_tvl),
        "hhi": float(hhi),
        "shares": {k: float(v) for k, v in shares.items()},
        "dominant_protocol": dominant,
        "trends": trends,
        "risk_flags": risk_flags,
        "risk_rating": risk_rating,
    }


def defi_dependency_graph(protocols):
    """Build a protocol dependency network.

    Maps which DeFi protocols depend on which others, creating
    a directed graph useful for contagion analysis.

    Parameters
    ----------
    protocols : list of dict
        Each dict must have:
        - name: protocol name
        - dependencies: list of protocol names this depends on
        Optional:
        - tvl: total value locked
        - category: protocol category (lending, dex, yield, etc.)

    Returns
    -------
    dict
        Dictionary with keys:
        - nodes: list of protocol names
        - edges: list of (dependent, dependency) tuples
        - adjacency: dict mapping each protocol to its dependencies
        - reverse_adjacency: dict mapping each protocol to its dependants
        - root_protocols: protocols with no dependencies (foundations)
        - leaf_protocols: protocols nothing depends on
        - critical_protocols: protocols with most dependants (systemic risk)
        - risk_layers: protocols grouped by dependency depth
    """
    nodes = []
    edges = []
    adjacency: dict[str, list[str]] = {}
    reverse_adj: dict[str, list[str]] = {}

    for proto in protocols:
        name = proto["name"]
        deps = proto.get("dependencies", [])
        nodes.append(name)
        adjacency[name] = list(deps)
        for dep in deps:
            edges.append((name, dep))
            if dep not in reverse_adj:
                reverse_adj[dep] = []
            reverse_adj[dep].append(name)

    # Ensure all dependency targets appear in nodes
    all_names = set(nodes)
    for proto in protocols:
        for dep in proto.get("dependencies", []):
            if dep not in all_names:
                nodes.append(dep)
                all_names.add(dep)
                adjacency.setdefault(dep, [])

    for name in nodes:
        reverse_adj.setdefault(name, [])
        adjacency.setdefault(name, [])

    root_protocols = [n for n in nodes if len(adjacency[n]) == 0]
    leaf_protocols = [n for n in nodes if len(reverse_adj[n]) == 0]

    # Critical protocols: sorted by number of (transitive) dependants
    dependant_counts = {n: len(reverse_adj[n]) for n in nodes}
    critical = sorted(
        [n for n in nodes if dependant_counts[n] > 0],
        key=lambda x: dependant_counts[x],
        reverse=True,
    )

    # Risk layers (BFS from roots)
    layers: dict[int, list[str]] = {}
    depth: dict[str, int] = {}
    for root in root_protocols:
        depth[root] = 0

    # Simple iterative depth assignment
    changed = True
    max_iter = len(nodes) + 1
    iteration = 0
    while changed and iteration < max_iter:
        changed = False
        iteration += 1
        for name in nodes:
            deps = adjacency[name]
            if not deps:
                if name not in depth:
                    depth[name] = 0
                    changed = True
            else:
                dep_depths = [depth[d] for d in deps if d in depth]
                if dep_depths:
                    new_depth = max(dep_depths) + 1
                    if name not in depth or depth[name] != new_depth:
                        depth[name] = new_depth
                        changed = True

    for name, d in depth.items():
        layers.setdefault(d, [])
        layers[d].append(name)

    return {
        "nodes": nodes,
        "edges": edges,
        "adjacency": {k: list(v) for k, v in adjacency.items()},
        "reverse_adjacency": {k: list(v) for k, v in reverse_adj.items()},
        "root_protocols": root_protocols,
        "leaf_protocols": leaf_protocols,
        "critical_protocols": critical,
        "risk_layers": {int(k): v for k, v in sorted(layers.items())},
    }


def smart_contract_risk_score(age_days, audited, tvl, tvl_stability=0.9):
    """Composite risk score for a smart contract or DeFi protocol.

    Combines multiple factors into a single risk score. Lower
    scores indicate higher risk.

    Parameters
    ----------
    age_days : int
        Age of the smart contract in days since deployment.
    audited : bool
        Whether the contract has been professionally audited.
    tvl : float
        Current total value locked in the protocol.
    tvl_stability : float, optional
        TVL stability metric in [0, 1], where 1.0 means perfectly
        stable TVL over time (default 0.9).

    Returns
    -------
    dict
        Dictionary with keys:
        - score: float in [0, 1], higher is safer
        - risk_rating: qualitative rating
        - factors: dict of individual factor scores and weights
        - recommendations: list of risk considerations
    """
    factors = {}

    # Age factor: older contracts have survived longer (Lindy effect)
    if age_days >= 730:  # 2+ years
        age_score = 1.0
    elif age_days >= 365:
        age_score = 0.75
    elif age_days >= 180:
        age_score = 0.50
    elif age_days >= 90:
        age_score = 0.30
    else:
        age_score = 0.10

    factors["age"] = {"score": age_score, "weight": 0.25, "value": age_days}

    # Audit factor
    audit_score = 0.80 if audited else 0.10
    factors["audit"] = {"score": audit_score, "weight": 0.30, "value": audited}

    # TVL factor: higher TVL means more battle-tested, but diminishing returns
    if tvl >= 1_000_000_000:
        tvl_score = 1.0
    elif tvl >= 100_000_000:
        tvl_score = 0.80
    elif tvl >= 10_000_000:
        tvl_score = 0.60
    elif tvl >= 1_000_000:
        tvl_score = 0.40
    else:
        tvl_score = 0.15

    factors["tvl"] = {"score": tvl_score, "weight": 0.20, "value": tvl}

    # Stability factor
    stability_score = float(np.clip(tvl_stability, 0.0, 1.0))
    factors["stability"] = {"score": stability_score, "weight": 0.25, "value": tvl_stability}

    # Weighted composite
    score = sum(f["score"] * f["weight"] for f in factors.values())
    score = float(np.clip(score, 0.0, 1.0))

    recommendations = []
    if not audited:
        recommendations.append("No audit detected: strongly recommend professional audit")
    if age_days < 180:
        recommendations.append(f"Contract is only {age_days} days old: limited track record")
    if tvl < 10_000_000:
        recommendations.append("Low TVL: limited real-world battle testing")
    if tvl_stability < 0.5:
        recommendations.append("TVL instability suggests potential confidence issues")

    if score >= 0.75:
        risk_rating = "low"
    elif score >= 0.50:
        risk_rating = "medium"
    elif score >= 0.30:
        risk_rating = "high"
    else:
        risk_rating = "critical"

    return {
        "score": score,
        "risk_rating": risk_rating,
        "factors": factors,
        "recommendations": recommendations,
    }
