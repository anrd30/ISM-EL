import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    latency = float(obs[3])
    target = float(obs[4])
    headroom = float(obs[11])
    current_alloc = float(obs[10])

    # Be conservative when latency is near or above its limit; otherwise favor
    # faster rebuild rates to reduce time spent degraded.
    margin = (target - latency) / max(abs(target), 1e-6)
    if margin >= 0.20:
        latency_cap = 3
    elif margin >= 0.12:
        latency_cap = 2
    elif margin >= 0.06:
        latency_cap = 1
    else:
        latency_cap = 0

    # Approximate each rate as a fraction of the maximum rate. Retain the
    # minimum action even when measured headroom is below that fraction.
    rate_fractions = (20.0 / 120.0, 50.0 / 120.0, 80.0 / 120.0, 1.0)
    bandwidth_cap = 0
    for i, fraction in enumerate(rate_fractions):
        if headroom >= fraction:
            bandwidth_cap = i

    # If the currently applied rate is already high and latency has a healthy
    # margin, avoid stepping down solely because of a small headroom fluctuation.
    if margin >= 0.20 and current_alloc >= rate_fractions[2] and headroom >= rate_fractions[1]:
        bandwidth_cap = max(bandwidth_cap, 1)

    return int(min(latency_cap, bandwidth_cap))
