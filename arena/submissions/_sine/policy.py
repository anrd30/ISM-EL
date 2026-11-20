"""Nonsense reference: a sine of foreground bandwidth, with no reasoning behind it.

Committed deliberately. It scores close to the tuned rule, so it is the floor any
submission must clear before a result means anything. A model that loses to this has
not written a controller.
"""

import math


def policy(obs, state):
    return 1 if math.sin(obs[2]) > 0.3 else 2
