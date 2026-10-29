"""Recovery simulator for the rebuild-bandwidth controller.

The simulator is the only source of rebuild start and end times, allocated
rebuild bandwidth, application latency during a rebuild, and outcomes under
alternative actions: no dataset supplies them. Everything it produces is
simulated and must be labelled so wherever it is reported.

Entry points: :class:`sim.recovery_env.RecoveryEnv` and its factory
:func:`sim.recovery_env.make_env`, configured by :class:`sim.config.SimConfig`.
:func:`sim.stress.stress_config` returns a labelled elevated-hazard
configuration whose only purpose is to make data loss frequent enough to
measure. Limitations are collected in ``sim/LIMITATIONS.md``.
"""

from sim.config import SimConfig
from sim.recovery_env import RecoveryEnv, make_env
from sim.stress import STRESS_HAZARD_MULTIPLIER, ScaledHazard, stress_config

__all__ = [
    "SimConfig",
    "RecoveryEnv",
    "make_env",
    "STRESS_HAZARD_MULTIPLIER",
    "ScaledHazard",
    "stress_config",
]
