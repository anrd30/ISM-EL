"""Residual policy network and its ``Policy`` adapter.

Networks. Actor and critic are separate ``FEATURE_DIM -> 64 -> 64 -> out``
tanh multilayer perceptrons. The actor has one output per entry of
``RESIDUAL_DELTAS``: action logits under ``method="ppo"``, and the predicted
advantage of each correction over no correction under
``method="counterfactual"`` (see :mod:`agent.train_residual`). Either way the
policy acts by the argmax. The head starts with zero weights and a bias of
``INIT_NO_CHANGE_LOGIT`` on the "no change" output, so before any training the
argmax is "take the rule's action" in every state (and a PPO policy samples it
with probability ``INIT_NO_CHANGE_PROB``). Training starts from the rule.

Adapter. :class:`ResidualPolicy` satisfies ``baselines.policies.Policy``
(``name``, ``reset``, ``__call__``) and exposes ``config``, the ``SimConfig``
of its wrapped rule, so ``baselines.evaluate`` checks that the rule's latency
model matches the environment's. It acts by the argmax of the logits.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch
from torch import nn

from agent.residual_features import (
    FEATURE_DIM,
    FEATURE_VERSION,
    N_RESIDUAL,
    NO_CHANGE_INDEX,
    ResidualFeatures,
    compose_action,
)
from baselines.policies import RiskAwareRuleBased
from sim.config import SimConfig

__all__ = [
    "HIDDEN_SIZE",
    "INIT_NO_CHANGE_LOGIT",
    "INIT_NO_CHANGE_PROB",
    "ResidualActor",
    "ResidualCritic",
    "ResidualPolicy",
]

HIDDEN_SIZE: int = 64
#: Initial "no change" logit; the other two start at 0.
INIT_NO_CHANGE_LOGIT: float = 1.5
INIT_NO_CHANGE_PROB: float = math.exp(INIT_NO_CHANGE_LOGIT) / (math.exp(INIT_NO_CHANGE_LOGIT) + N_RESIDUAL - 1)


def _mlp(out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = FEATURE_DIM
    for _ in range(2):
        linear = nn.Linear(width, HIDDEN_SIZE)
        nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.Tanh()]
        width = HIDDEN_SIZE
    head = nn.Linear(width, out_dim)
    layers.append(head)
    return nn.Sequential(*layers)


class ResidualActor(nn.Module):
    """Features ``(B, FEATURE_DIM)`` to correction logits ``(B, N_RESIDUAL)``."""

    def __init__(self) -> None:
        super().__init__()
        self.net: nn.Sequential = _mlp(N_RESIDUAL)
        head = self.net[-1]
        assert isinstance(head, nn.Linear)
        nn.init.zeros_(head.weight)
        with torch.no_grad():
            head.bias.zero_()
            head.bias[NO_CHANGE_INDEX] = INIT_NO_CHANGE_LOGIT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualCritic(nn.Module):
    """Features ``(B, FEATURE_DIM)`` to values ``(B,)``."""

    def __init__(self) -> None:
        super().__init__()
        self.net: nn.Sequential = _mlp(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResidualPolicy:
    """Trained residual actor on top of ``RiskAwareRuleBased``, as a ``Policy``.

    Args:
        actor: The correction network.
        config: Simulator configuration of the wrapped rule; must match the
            evaluation environment's latency model.
        name: Label used in every evaluation output.
    """

    deployable: bool = True

    def __init__(self, actor: ResidualActor, config: SimConfig, name: str = "residual_ppo") -> None:
        self.actor: ResidualActor = actor.eval()
        self.config: SimConfig = config
        self.name: str = name
        self.features: ResidualFeatures = ResidualFeatures(RiskAwareRuleBased(config))
        self.n_steps: int = 0
        self.n_changed: int = 0

    @classmethod
    def from_checkpoint(
        cls, path: str | os.PathLike[str], config: SimConfig, name: str | None = None
    ) -> "ResidualPolicy":
        """Build from a checkpoint written by ``agent.train_residual``.

        Raises:
            ValueError: if the checkpoint's feature version differs from this code's.
        """
        ckpt = load_residual_checkpoint(path)
        version = int(ckpt["meta"]["feature_version"])
        if version != FEATURE_VERSION:
            raise ValueError(f"checkpoint feature_version={version}, code FEATURE_VERSION={FEATURE_VERSION}")
        actor = ResidualActor()
        actor.load_state_dict(ckpt["actor"])
        label = name if name is not None else f"residual_ppo seed{int(ckpt['meta']['seed'])}"
        return cls(actor, config, name=label)

    def reset(self) -> None:
        """Clear the per-episode history and the wrapped rule's belief."""
        self.features.reset()

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> int:
        x, step = self.features.update(obs)
        logits = self.actor(torch.as_tensor(x).unsqueeze(0))
        residual = int(torch.argmax(logits, dim=-1).item())
        action = compose_action(step.action, step.floor, residual)
        self.n_steps += 1
        self.n_changed += int(action != step.action)
        return action


def load_residual_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a residual checkpoint (tensors and plain values only)."""
    return torch.load(path, map_location="cpu", weights_only=True)
