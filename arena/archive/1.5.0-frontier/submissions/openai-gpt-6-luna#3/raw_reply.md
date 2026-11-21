import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    latency = float(obs[3])
    target = max(float(obs[4]), 1e-6)
    headroom = float(obs[11])

    previous = state.get("previous_latency")
    state["previous_latency"] = latency

    # Reduce the usable margin when latency has recently been rising.
    margin = (target - latency) / target
    if previous is not None:
        rise = max(0.0, (latency - float(previous)) / target)
        margin -= 0.5 * rise

    if margin <= 0.0:
        latency_cap = 0
    elif margin < 0.03:
        latency_cap = 0
    elif margin < 0.10:
        latency_cap = 1
    elif margin < 0.20:
        latency_cap = 2
    else:
        latency_cap = 3

    if headroom < 0.08:
        bandwidth_cap = 0
    elif headroom < 0.18:
        bandwidth_cap = 1
    elif headroom < 0.32:
        bandwidth_cap = 2
    else:
        bandwidth_cap = 3

    return int(min(latency_cap, bandwidth_cap))