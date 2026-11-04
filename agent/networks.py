"""Actor and critic networks.

Two separate multilayer perceptrons, each ``12 -> 64 -> 64 -> out`` with tanh
activations. The actor emits ``N_ACTIONS`` unnormalised logits; the critic emits
one scalar value. They share no parameters.

Why this small and why separate. The observation is 12 floats in ``[0, 1]`` and
the action set has four elements, so a larger model has nothing to fit and is
harder to describe in a report. Separate networks keep the value loss from
shaping the policy's features and make the parameter count of each part a plain
product of layer sizes.

Initialisation. Hidden layers use orthogonal initialisation with gain
``sqrt(2)``; the policy head uses a small gain (0.01) so the initial policy is
close to uniform over the four actions and early exploration does not begin
from an arbitrary preference; the value head uses gain 1.0. Biases start at 0.
This is a convention for on-policy methods; it has not been tuned for this
problem.

Everything runs on CPU in ``float32``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from contracts.contracts import N_ACTIONS, OBS_DIM

__all__ = ["HIDDEN_SIZE", "POLICY_HEAD_GAIN", "Actor", "Critic", "mlp"]

HIDDEN_SIZE: int = 64
POLICY_HEAD_GAIN: float = 0.01
_HIDDEN_GAIN: float = math.sqrt(2.0)
_VALUE_HEAD_GAIN: float = 1.0


def _init_linear(layer: nn.Linear, gain: float) -> nn.Linear:
    """Orthogonal weights with the given gain and zero bias, in place."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def mlp(in_dim: int, out_dim: int, head_gain: float, hidden: int = HIDDEN_SIZE) -> nn.Sequential:
    """Two tanh hidden layers of width ``hidden`` followed by a linear head."""
    return nn.Sequential(
        _init_linear(nn.Linear(in_dim, hidden), _HIDDEN_GAIN),
        nn.Tanh(),
        _init_linear(nn.Linear(hidden, hidden), _HIDDEN_GAIN),
        nn.Tanh(),
        _init_linear(nn.Linear(hidden, out_dim), head_gain),
    )


class Actor(nn.Module):
    """Policy network: normalised observation ``(B, 12)`` to logits ``(B, 4)``."""

    def __init__(self) -> None:
        super().__init__()
        self.net: nn.Sequential = mlp(OBS_DIM, N_ACTIONS, POLICY_HEAD_GAIN)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return unnormalised action logits of shape ``(B, N_ACTIONS)``."""
        return self.net(obs)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """Categorical distribution over action indices."""
        return torch.distributions.Categorical(logits=self.forward(obs))


class Critic(nn.Module):
    """Value network: normalised observation ``(B, 12)`` to values ``(B,)``."""

    def __init__(self) -> None:
        super().__init__()
        self.net: nn.Sequential = mlp(OBS_DIM, 1, _VALUE_HEAD_GAIN)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return state values of shape ``(B,)``."""
        return self.net(obs).squeeze(-1)
