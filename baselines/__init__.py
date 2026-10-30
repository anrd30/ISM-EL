"""Non-learned baseline policies and the paired evaluation harness.

``baselines.policies`` holds the policies; ``baselines.evaluate`` runs them
(``python -m baselines.evaluate``). The harness is deliberately not imported
here so that running it with ``-m`` does not import it twice.

All results come from the simulator in ``sim/``, which is uncalibrated and
synthetic (``sim/LIMITATIONS.md``).
"""

from baselines.policies import (
    FixedRate,
    LatencyRule,
    OracleLookahead,
    Policy,
    RiskAwareRuleBased,
    SlowestRiskFeasible,
    WorkloadOnlyThrottle,
    baseline_policies,
    diagnostic_policies,
    fixed_rate_policies,
)

__all__ = [
    "FixedRate",
    "LatencyRule",
    "OracleLookahead",
    "Policy",
    "RiskAwareRuleBased",
    "SlowestRiskFeasible",
    "WorkloadOnlyThrottle",
    "baseline_policies",
    "diagnostic_policies",
    "fixed_rate_policies",
]
