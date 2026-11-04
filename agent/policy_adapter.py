"""Adapter that exposes a learned agent through the ``Policy`` protocol.

``baselines.policies.Policy`` is a ``name``, ``__call__(obs) -> int`` and
``reset()``. Wrapping an agent in :class:`LearnedPolicy` lets it run inside
``baselines.evaluate`` on the same scenario seeds as the fixed rates, the
workload-only throttle, the risk-aware rule and the oracle.

Modes. The default is deterministic (argmax of the logits): evaluation then
measures the policy's mode and is repeatable without a seed. ``deterministic=
False`` samples from the policy; the sampling generator is re-seeded on every
``reset()``, so repeated evaluations of the same scenarios give the same result.

Ablation. If the agent was trained with the hazard input masked, the same mask
must be applied here, because the harness passes the full observation. The
adapter applies it from ``ablate_hazard`` (restored from a checkpoint).

The adapter deliberately has no attribute called ``config``: the harness reads
``policy.config`` to compare latency models and would misinterpret a PPO
configuration object.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from agent.ablation import mask_observation
from agent.checkpoint import load_checkpoint
from agent.networks import Actor
from contracts.contracts import N_ACTIONS, OBS_DIM, ContractViolation

__all__ = ["LearnedPolicy"]


class LearnedPolicy:
    """A trained actor as a ``Policy``.

    Args:
        actor: The policy network.
        name: Label used in every evaluation output.
        deterministic: Argmax if true (default), sample otherwise.
        ablate_hazard: Zero observation fields 5 and 6 before the actor runs.
        seed: Seed of the sampling generator (stochastic mode only).
    """

    def __init__(
        self,
        actor: Actor,
        name: str = "ppo_lagrangian",
        deterministic: bool = True,
        ablate_hazard: bool = False,
        seed: int = 0,
    ) -> None:
        self.actor: Actor = actor.eval()
        self.name: str = name
        self.deterministic: bool = deterministic
        self.ablate_hazard: bool = ablate_hazard
        self._seed: int = int(seed)
        self._generator: torch.Generator = torch.Generator().manual_seed(self._seed)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | os.PathLike[str],
        name: str | None = None,
        deterministic: bool = True,
        seed: int = 0,
    ) -> "LearnedPolicy":
        """Build from a checkpoint written by ``agent.train``; the ablation flag is restored from it."""
        ckpt = load_checkpoint(path)
        actor = Actor()
        actor.load_state_dict(ckpt["agent"]["actor"])
        ablate = bool(ckpt["meta"]["ablate_hazard"])
        label = name if name is not None else ("ppo_lagrangian_no_hazard" if ablate else "ppo_lagrangian")
        return cls(actor, name=label, deterministic=deterministic, ablate_hazard=ablate, seed=seed)

    def reset(self) -> None:
        """Re-seed the sampling generator; the networks carry no per-episode state."""
        self._generator.manual_seed(self._seed)

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> int:
        """Return an action index in ``[0, N_ACTIONS)``.

        Raises:
            ContractViolation: if ``obs`` is not a vector of ``OBS_DIM`` entries.
        """
        arr = np.asarray(obs, dtype=np.float32)
        if arr.shape != (OBS_DIM,):
            raise ContractViolation(f"observation must have shape ({OBS_DIM},), got {arr.shape}")
        x = torch.as_tensor(mask_observation(arr, self.ablate_hazard)).unsqueeze(0)
        logits = self.actor(x)
        if self.deterministic:
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, 1, generator=self._generator).item())
        assert 0 <= action < N_ACTIONS
        return action
