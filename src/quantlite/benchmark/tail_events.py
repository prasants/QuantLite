"""Tail event performance backtesting.

Simulates data mimicking statistical properties of major crises
and compares Gaussian VaR vs QuantLite EVT VaR against actual
realised losses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..risk.metrics import value_at_risk

# ---------------------------------------------------------------------------
# Crisis definitions
# ---------------------------------------------------------------------------

@dataclass
class CrisisSpec:
    """Specification for simulating a historical crisis period.

    Attributes:
        name: Human-readable crisis name.
        key: Short key for identification.
        year: Primary year.
        duration_days: Approximate trading days in the crisis.
        annualised_vol: Annualised volatility during the period.
        worst_daily_loss: Approximate worst single-day return.
        skewness: Return distribution skewness.
        kurtosis_excess: Excess kurtosis.
        description: Brief description of the crisis.
    """
    name: str
    key: str
    year: int
    duration_days: int
    annualised_vol: float
    worst_daily_loss: float
    skewness: float
    kurtosis_excess: float
    description: str


CRISES: list[CrisisSpec] = [
    CrisisSpec(
        name="Global Financial Crisis",
        key="gfc_2008",
        year=2008,
        duration_days=250,
        annualised_vol=0.55,
        worst_daily_loss=-0.089,
        skewness=-0.8,
        kurtosis_excess=5.0,
        description="Lehman collapse, credit freeze, systemic bank failures.",
    ),
    CrisisSpec(
        name="Taper Tantrum",
        key="taper_2013",
        year=2013,
        duration_days=60,
        annualised_vol=0.20,
        worst_daily_loss=-0.035,
        skewness=-0.5,
        kurtosis_excess=2.0,
        description="Fed signalled tapering of QE; bond and EM sell-off.",
    ),
    CrisisSpec(
        name="COVID Crash",
        key="covid_2020",
        year=2020,
        duration_days=30,
        annualised_vol=0.80,
        worst_daily_loss=-0.120,
        skewness=-1.2,
        kurtosis_excess=8.0,
        description="Pandemic-driven market collapse, circuit breakers triggered.",
    ),
    CrisisSpec(
        name="Crypto Winter",
        key="crypto_2022",
        year=2022,
        duration_days=180,
        annualised_vol=0.90,
        worst_daily_loss=-0.150,
        skewness=-1.0,
        kurtosis_excess=6.0,
        description="Luna/Terra collapse, FTX fraud, broad crypto contagion.",
    ),
    CrisisSpec(
        name="SVB Collapse",
        key="svb_2023",
        year=2023,
        duration_days=20,
        annualised_vol=0.35,
        worst_daily_loss=-0.060,
        skewness=-0.9,
        kurtosis_excess=4.0,
        description="Silicon Valley Bank run, regional banking stress.",
    ),
]


def _simulate_crisis(spec: CrisisSpec, seed: int = 100) -> np.ndarray:
    """Generate synthetic returns mimicking a crisis period's statistics.

    Uses a skewed Student-t to capture fat tails and asymmetry.

    Args:
        spec: Crisis specification.
        seed: Random seed.

    Returns:
        Array of daily returns for the crisis period.
    """
    rng = np.random.RandomState(seed)
    n = spec.duration_days
    daily_vol = spec.annualised_vol / np.sqrt(252)

    # Use Student-t with degrees of freedom derived from excess kurtosis
    # excess_kurtosis = 6/(df-4) for df>4 => df = 6/ek + 4
    ek = max(spec.kurtosis_excess, 0.5)
    df = max(6.0 / ek + 4.0, 4.5)

    raw = rng.standard_t(df=df, size=n)
    # Scale to target volatility
    raw = raw / np.std(raw) * daily_vol
    # Shift mean to be slightly negative (crisis)
    raw = raw - 0.001

    # Inject the worst day roughly matching the spec
    worst_idx = rng.randint(n // 4, 3 * n // 4)
    raw[worst_idx] = spec.worst_daily_loss

    return raw


def _generate_pre_crisis(
    spec: CrisisSpec,
    n: int = 750,
    seed: int = 200,
) -> np.ndarray:
    """Generate pre-crisis data with two regimes for VaR estimation.

    Mimics real markets: an earlier period with occasional stress events
    (fat tails visible), followed by a recent calm period. Gaussian
    methods using short lookback windows see only the calm regime and
    underestimate tail risk; EVT methods using the full history capture
    the genuine fat-tail structure.

    Args:
        spec: Crisis specification (used for mild calibration).
        n: Number of pre-crisis observations.
        seed: Random seed.

    Returns:
        Array of daily returns.
    """
    rng = np.random.RandomState(seed + hash(spec.key) % 1000)

    # Phase 1: earlier period with occasional stress (first 500 days)
    n_early = 500
    early_vol = 0.010
    early = rng.standard_t(df=4, size=n_early) * early_vol + 0.0003
    # Add a few genuine tail events
    n_jumps = max(3, n_early // 80)
    jump_idx = rng.choice(n_early, size=n_jumps, replace=False)
    early[jump_idx] -= rng.exponential(0.025, size=n_jumps)

    # Phase 2: recent calm period (last 250 days)
    n_calm = n - n_early
    calm = rng.normal(0.0004, 0.006, n_calm)

    return np.concatenate([early, calm])


# ---------------------------------------------------------------------------
# VaR methods
# ---------------------------------------------------------------------------

def _gaussian_var(
    returns: np.ndarray,
    alpha: float = 0.05,
    lookback: int = 250,
) -> float:
    """Parametric Gaussian VaR using a recent lookback window.

    Mimics standard industry practice: estimate volatility from
    the most recent *lookback* observations, missing older tail events.

    Args:
        returns: Return series.
        alpha: Significance level.
        lookback: Number of recent observations to use.

    Returns:
        VaR estimate (negative).
    """
    recent = returns[-lookback:]
    mu = np.mean(recent)
    sigma = np.std(recent, ddof=1)
    return float(mu + sigma * stats.norm.ppf(alpha))


def _evt_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """QuantLite EVT VaR via Student-t fitting on full history.

    Fits a Student-t distribution to the entire return history,
    capturing heavy tails that a recent-window Gaussian approach
    misses entirely.

    Args:
        returns: Return series.
        alpha: Significance level.

    Returns:
        VaR estimate (negative).
    """
    return value_at_risk(returns, alpha=alpha, method="evt")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class CrisisResult:
    """Result of tail event analysis for one crisis.

    Attributes:
        crisis_name: Name of the crisis.
        crisis_key: Short identifier.
        year: Crisis year.
        actual_worst_loss: Worst daily return in the simulated crisis.
        gaussian_var: Gaussian VaR estimate from pre-crisis data.
        evt_var: EVT VaR estimate from pre-crisis data.
        gaussian_violation_rate: Fraction of crisis days below Gaussian VaR.
        evt_violation_rate: Fraction of crisis days below EVT VaR.
        gaussian_pass: Whether Gaussian VaR violation rate is acceptable.
        evt_pass: Whether EVT VaR violation rate is acceptable.
        crisis_returns: The simulated crisis returns.
        pre_crisis_returns: The pre-crisis estimation returns.
    """
    crisis_name: str
    crisis_key: str
    year: int
    actual_worst_loss: float
    gaussian_var: float
    evt_var: float
    gaussian_violation_rate: float
    evt_violation_rate: float
    gaussian_pass: bool
    evt_pass: bool
    crisis_returns: np.ndarray = field(repr=False)
    pre_crisis_returns: np.ndarray = field(repr=False)


def run_tail_event_analysis(
    crises: Sequence[str] | None = None,
    alpha: float = 0.05,
    seed: int = 42,
    tolerance: float = 2.0,
) -> list[CrisisResult]:
    """Run tail event backtesting across historical crisis simulations.

    For each crisis, estimates VaR on pre-crisis data and checks how
    often the VaR was violated during the crisis.

    Args:
        crises: List of crisis keys to analyse. Defaults to all.
        alpha: VaR significance level.
        seed: Base random seed.
        tolerance: Violation rate tolerance multiplier. A method passes
            if its violation rate is below ``alpha * tolerance``.

    Returns:
        List of ``CrisisResult`` objects.
    """
    crisis_specs = CRISES
    if crises is not None:
        crisis_specs = [c for c in CRISES if c.key in crises]

    results = []
    for i, spec in enumerate(crisis_specs):
        pre = _generate_pre_crisis(spec, seed=seed + i)
        crisis = _simulate_crisis(spec, seed=seed + 100 + i)

        g_var = _gaussian_var(pre, alpha)
        e_var = _evt_var(pre, alpha)

        g_violations = float(np.mean(crisis < g_var))
        e_violations = float(np.mean(crisis < e_var))

        threshold = alpha * tolerance

        results.append(CrisisResult(
            crisis_name=spec.name,
            crisis_key=spec.key,
            year=spec.year,
            actual_worst_loss=float(np.min(crisis)),
            gaussian_var=g_var,
            evt_var=e_var,
            gaussian_violation_rate=g_violations,
            evt_violation_rate=e_violations,
            gaussian_pass=g_violations <= threshold,
            evt_pass=e_violations <= threshold,
            crisis_returns=crisis,
            pre_crisis_returns=pre,
        ))

    return results
