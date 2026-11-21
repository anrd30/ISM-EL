import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    latency = float(obs[3])
    target = float(obs[4])
    ratio = latency / max(target, 1e-6)
    bandwidth_headroom = float(obs[11])

    if ratio < 0.82:
        action = 3
    elif ratio < 0.91:
        action = 2
    elif ratio < 0.97:
        action = 1
    else:
        action = 0

    if bandwidth_headroom < 0.08:
        action = min(action, 0)
    elif bandwidth_headroom < 0.18:
        action = min(action, 1)
    elif bandwidth_headroom < 0.32:
        action = min(action, 2)

    previous_ratio = state.get("previous_ratio")
    if previous_ratio is not None and ratio - previous_ratio > 0.025 and ratio > 0.75:
        action = max(0, action - 1)

    state["previous_ratio"] = ratio
    return int(action)
