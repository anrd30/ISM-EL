import math
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    read, write, fg, latency, target, hazard, uncertainty, redundancy, progress, remaining, alloc, headroom = map(float, obs)

    rates = (1.0 / 6.0, 5.0 / 12.0, 2.0 / 3.0, 1.0)
    effort = (0.0, 0.2125, 0.465, 1.0)
    current = 0 if alloc < 0.28 else 1 if alloc < 0.54 else 2 if alloc < 0.83 else 3
    headroom = min(1.0, max(0.0, headroom))
    busy = 1.0 - headroom
    congestion = busy * busy
    y = math.log(max(0.002, latency) + 0.02)
    limit = math.log(max(0.002, target) + 0.02)

    # Fit log latency to foreground load and the allocation actually applied.
    a = effort[current]
    x = np.array((1.0, read, write, fg, busy, congestion, a, a * congestion))
    if "fit" not in state:
        prior_slope = 0.32 + 1.05 * congestion
        weights = np.array((y - a * prior_slope, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.32, 1.05))
        cov = np.diag((1.0, 1.5, 1.5, 1.5, 1.5, 2.0, 0.22, 0.40))
        residual_scale = 0.06
        typical_busy = busy
    else:
        weights, cov, residual_scale, typical_busy = state["fit"]

    prediction = float(x @ weights)
    error = y - prediction
    direction = cov @ x
    leverage = max(0.0, float(x @ direction))
    weights = weights + direction * (min(0.65, max(-0.65, error)) / (1.0 + leverage))
    cov = (cov - np.outer(direction, direction) / (1.0 + leverage)) / 0.998
    residual_scale += 0.04 * (min(0.30, abs(error)) - residual_scale)
    typical_busy += 0.02 * (busy - typical_busy)
    state["fit"] = weights, cov, residual_scale, typical_busy

    slope = min(2.4, max(0.08, float(weights[6] + congestion * weights[7])))
    slope_var = max(0.0, float(cov[6, 6] + 2.0 * congestion * cov[6, 7]
                               + congestion * congestion * cov[7, 7]))
    anchor = y - 0.15 * min(0.25, max(-0.25, y - float(x @ weights)))
    noise = min(0.095, max(0.025, 0.65 * residual_scale))

    # Infer work per interval from observed progress, including filter overrides.
    previous = state.get("remaining", remaining)
    work_rate, samples = state.get("work_rate", (0.0, 0))
    completed = previous - remaining
    if completed > 0.0:
        sample = completed / rates[current]
        if samples:
            sample = min(1.6 * work_rate, max(0.6 * work_rate, sample))
            work_rate += 0.15 * (sample - work_rate)
        else:
            work_rate = sample
        samples += 1
    state["remaining"] = remaining
    state["work_rate"] = work_rate, samples

    urgency = min(1.0, max(0.0, hazard + 0.5 * uncertainty)) * (
        1.0 - min(1.0, max(0.0, redundancy))
    )
    work_value = 0.24 + 0.14 * typical_busy + 0.10 * urgency

    best, best_cost = 0, math.inf
    for action in range(4):
        delta = effort[action] - a
        margin = anchor + slope * delta - limit
        spread = noise + abs(delta) * min(0.14, 0.18 * math.sqrt(slope_var))
        violation = 0.5 * math.erfc(-margin / (math.sqrt(2.0) * spread))
        degradation = 0.5 * math.expm1(min(1.5, margin))

        useful_rate = rates[action]
        if samples >= 3 and work_rate > 0.0:
            useful_rate = min(useful_rate, max(0.0, remaining) / (0.9 * work_rate))

        cost = degradation + 10.0 * violation - work_value * useful_rate
        if cost < best_cost:
            best, best_cost = action, cost

    return best
