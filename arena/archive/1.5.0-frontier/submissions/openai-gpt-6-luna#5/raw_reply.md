import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    latency = float(obs[3])
    target = max(float(obs[4]), 1e-6)
    ratio = latency / target
    headroom = float(obs[11])

    # Choose a latency-based rate, leaving a larger margin when latency is
    # close to its limit.
    if ratio >= 1.0:
        action = 0
    elif ratio >= 0.94:
        action = 0
    elif ratio >= 0.86:
        action = 1
    elif ratio >= 0.74:
        action = 2
    else:
        action = 3

    # Limit the rate when little bandwidth headroom remains.
    if headroom < 0.12:
        action = min(action, 0)
    elif headroom < 0.25:
        action = min(action, 1)
    elif headroom < 0.42:
        action = min(action, 2)

    # A rising latency trend is a warning that the current margin may not last.
    previous_ratio = state.get("previous_latency_ratio")
    if previous_ratio is not None and ratio - float(previous_ratio) > 0.05:
        action = max(0, action - 1)
    state["previous_latency_ratio"] = ratio

    # Use a higher rate to finish a nearly complete rebuild, unless latency
    # or bandwidth constraints above already limit it.
    if float(obs[9]) < 0.03 and ratio < 0.90 and headroom >= 0.25:
        action = min(3, action + 1)

    return int(action)