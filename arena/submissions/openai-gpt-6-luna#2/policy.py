import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    latency = float(obs[3])
    target = max(float(obs[4]), 1e-6)
    headroom = float(obs[11])
    ratio = latency / target

    # Infer the applied action from its normalized rate.
    rates = (20.0 / 120.0, 50.0 / 120.0, 80.0 / 120.0, 1.0)
    applied = float(obs[10])
    current = min(range(4), key=lambda i: abs(applied - rates[i]))

    # Keep a margin below the latency limit; back off more sharply after a violation.
    if ratio > 1.05:
        desired = current - 2
    elif ratio > 1.0:
        desired = current - 1
    elif ratio > 0.96:
        desired = current
    elif ratio > 0.88:
        desired = current + 1
    else:
        desired = 3
    desired = max(0, min(3, desired))

    # Avoid choosing a rebuild rate that consumes nearly all estimated bandwidth headroom.
    cap = max(
        (i for i, rate in enumerate(rates) if rate <= headroom + 0.08),
        default=0,
    )
    desired = min(desired, cap)

    state["last_action"] = int(desired)
    return int(desired)
