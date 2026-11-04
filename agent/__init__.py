"""Constrained PPO agent for the rebuild-bandwidth controller.

``agent.ppo`` holds the learner (clipped PPO with two Lagrangian constraints),
``agent.rollout`` the data collection, ``agent.ablation`` the hazard-input mask,
``agent.policy_adapter`` the wrapper that runs a trained actor inside
``baselines.evaluate``, and ``agent.train`` the command line
(``python -m agent.train``).

All behaviour is defined against the simulator in ``sim/``, whose constants are
design decisions and whose latency model is uncalibrated
(``sim/LIMITATIONS.md``). This package contains an implementation, not a result.
"""
