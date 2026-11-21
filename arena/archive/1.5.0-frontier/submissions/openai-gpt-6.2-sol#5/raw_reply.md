import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    rates = np.array([1.0 / 6.0, 5.0 / 12.0, 2.0 / 3.0, 1.0])
    alloc = float(obs[10])
    current = int(alloc >= 0.23) + int(alloc >= 0.51) + int(alloc >= 0.83)
    rate_now = rates[current]

    read, write, bw = (float(obs[i]) for i in range(3))
    headroom = min(1.0, max(0.0, float(obs[11])))
    pressure = 0.5 * (1.0 - headroom) + 0.5 * min(1.0, max(0.0, bw))
    log_latency = np.log(max(1e-6, float(obs[3])))
    log_target = np.log(max(1e-6, float(obs[4])))
    progress = float(obs[8])

    features = np.array([
        read, write, bw, 1.0 - headroom,
        rate_now, rate_now * rate_now, pressure * rate_now,
    ])

    history = state.get("rebuild_model")
    if history is None:
        weights = np.array([0.15, 0.20, 0.40, 0.20, 0.15, 0.25, 0.40])
        covariance = 4.0 * np.eye(7)
        volatility = 0.04
        step = 0.0
    else:
        old_features, old_latency, old_progress, weights, covariance, volatility, step = history
        change = features - old_features

        # Fit latency changes, separating changes in load from changes in rate.
        if log_latency > -13.8 and old_latency > -13.8:
            observed_change = log_latency - old_latency
            load_change = observed_change - float(change[4:] @ weights[4:])
            volatility = 0.94 * volatility + 0.06 * min(abs(load_change), 0.25)
            if float(change @ change) > 1e-7:
                direction = covariance @ change
                denom = 1.0 + float(change @ direction)
                residual = observed_change - float(change @ weights)
                weights = weights + (np.clip(residual, -0.35, 0.35) / denom) * direction
                covariance = covariance - np.outer(direction, direction) / denom

        advance = progress - old_progress
        if advance > 1e-7:
            sample = advance / rate_now
            step = sample if step == 0.0 else 0.85 * step + 0.15 * sample

    state["rebuild_model"] = (
        features, log_latency, progress, weights, covariance, volatility, step
    )

    # Anchor predictions at the measured latency and keep the rate response monotone.
    linear = float(np.clip(weights[4] + pressure * weights[6], 0.03, 2.0))
    curvature = float(np.clip(weights[5], 0.0, 1.0))
    extra_delay = linear * rates + curvature * rates * rates
    current_delay = linear * rate_now + curvature * rate_now * rate_now
    predicted_gap = log_latency - log_target + extra_delay - current_delay

    # Leave a buffer around the latency limit, especially when load is changing.
    width = min(0.14, max(0.04, 1.4 * volatility))
    risk = np.clip((predicted_gap + width) / (2.0 * width), 0.0, 1.0)
    risk = risk * risk * (3.0 - 2.0 * risk)
    cost = 0.1 + 0.25 * np.expm1(extra_delay) + 10.0 * risk

    # A rate that finishes the rebuild gets no credit for unused bandwidth.
    useful_rate = rates
    if step > 0.0:
        remaining = max(0.0, 1.0 - progress)
        useful_rate = np.minimum(rates, max(1e-6, remaining / step))

    return int(np.argmin(cost / useful_rate))