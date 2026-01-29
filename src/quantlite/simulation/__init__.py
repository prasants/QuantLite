"""Fat-tail Monte Carlo simulation: EVT scenarios, copula MC, and regime switching.

Provides three families of simulation:

1. **EVT simulation** -- GPD-based tail sampling with empirical body.
2. **Copula Monte Carlo** -- multivariate simulation with fat-tailed marginals.
3. **Regime Monte Carlo** -- regime-switching paths, stress tests, and reverse stress tests.
"""

from .copula_mc import (
    gaussian_copula_mc,
    joint_tail_probability,
    stress_correlation_mc,
    t_copula_mc,
)
from .evt_simulation import (
    evt_tail_simulation,
    historical_bootstrap_evt,
    parametric_tail_simulation,
    scenario_fan,
)
from .regime_mc import (
    regime_switching_simulation,
    reverse_stress_test,
    simulation_summary,
    stress_test_scenario,
)

__all__ = [
    # EVT simulation
    "evt_tail_simulation",
    "parametric_tail_simulation",
    "historical_bootstrap_evt",
    "scenario_fan",
    # Copula MC
    "gaussian_copula_mc",
    "t_copula_mc",
    "stress_correlation_mc",
    "joint_tail_probability",
    # Regime MC
    "regime_switching_simulation",
    "stress_test_scenario",
    "reverse_stress_test",
    "simulation_summary",
]
