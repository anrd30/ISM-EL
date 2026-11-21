```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    read, write, fg_bw, latency, target, hazard, uncertainty, redundancy, \
        progress, remaining, current, headroom = map(float, obs)

    # Decode the applied rate, which may differ from our previous request.
    applied = int(current > 0.28) + int(current > 0.53) + int(current > 0.83)
    rates = (1.0 / 6.0, 5.0 / 12.0, 2.0 / 3.0, 1.0)
    a = rates[applied]
    load = min(1.0, max(0.0, 0.7 * (1.0 - headroom) + 0.3 * fg_bw))
    z = load - 0.5

    # Fit log latency from the observed foreground load and applied rate.
    x = np.array((1.0, z, z * z, read - 0.5, write - 0.5,
                  a, a * z, a * a))
    if "coefficients" not in state:
        state["coefficients"] = np.array(
            (-2.0, 1.5, 1.1, 0.12, 0.24, 0.57, 1.05, 0.23)
        )
        state["covariance"] = np.diag(
            (3.0, 1.0, 1.0, 0.4, 0.4, 0.75, 0.9, 0.5)
        )
        state["error"] = 0.08

    beta = state["coefficients"]
    covariance = state["covariance"]
    offset = 0.008
    measured = max(0.0, latency) + offset
    error = float(np.log(measured) - beta @ x)
    direction = covariance @ x
    denominator = 1.0 + float(x @ direction)
    beta += (min(0.65, max(-0.65, error)) / denominator) * direction
    covariance -= np.outer(direction, direction) / denominator
    noise = 0.96 * state["error"] + 0.04 * min(0.30, abs(error))
    state["error"] = noise

    # Estimate the progress delivered by one interval at the fastest rate.
    # This lets the last interval use a lower rate if it can still finish.
    full_step = state.get("full_step", 0.0)
    if "progress" in state:
        increment = progress - state["progress"]
        if increment > 1e-7:
            sample = increment / a
            if full_step > 0.0:
                sample = min(1.5 * full_step, max(0.67 * full_step, sample))
                full_step = 0.85 * full_step + 0.15 * sample
            else:
                full_step = sample
            state["full_step"] = full_step
    state["progress"] = progress

    # Anchor each prediction to the measured latency. Use the model only
    # for the change in latency caused by changing the rebuild rate.
    linear = min(2.8, max(0.03, float(beta[5] + beta[6] * z)))
    curvature = min(1.0, max(0.0, float(beta[7])))
    rate_covariance = covariance[5:, 5:]
    latency_limit = max(0.015, target)
    urgency = min(1.0, max(0.0, hazard + 0.35 * uncertainty - redundancy))
    progress_value = 0.36 + 0.05 * urgency

    best_action = 0
    best_cost = float("inf")
    for action, rate in enumerate(rates):
        delta = rate - a
        delta_log = delta * (linear + curvature * (rate + a))
        predicted = max(0.0, measured * float(np.exp(delta_log)) - offset)

        change = np.array((delta, delta * z, delta * (rate + a)))
        variance = max(0.0, float(change @ rate_covariance @ change))
        # Widen the transition around the SLO when the counterfactual
        # is uncertain; the current rate requires no counterfactual.
        margin = min(0.18, 0.025 + 0.20 * noise
                     + 0.18 * variance ** 0.5)
        ratio = predicted / latency_limit
        risk = min(1.0, max(0.0, (ratio - 1.0 + margin) / (2.0 * margin)))
        risk = risk * risk * (3.0 - 2.0 * risk)

        useful_rate = rate
        if full_step > 0.0:
            useful_rate = min(rate, max(0.0, 1.0 - progress) / full_step)

        # Retain a preference for lower latency even when every choice
        # is predicted to violate the target.
        cost = (0.28 * ratio + 0.22 * predicted + 10.0 * risk
                - progress_value * useful_rate)
        if cost < best_cost:
            best_cost = cost
            best_action = action

    return best_action
```