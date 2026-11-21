```python
import math

import numpy as np


def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    read = float(obs[0])
    write = float(obs[1])
    foreground = float(obs[2])
    latency = max(0.001, float(obs[3]))
    target = max(0.001, float(obs[4]))
    progress = float(obs[8])
    allocation = float(obs[10])
    pressure = 1.0 - min(1.0, max(0.0, float(obs[11])))

    rates = (1.0 / 6.0, 5.0 / 12.0, 2.0 / 3.0, 1.0)
    current = int(allocation >= 0.25) + int(allocation >= 0.54) + int(allocation >= 0.83)
    rate = rates[current]
    history = state.get("controller")

    if history is None and allocation <= 0.01:
        # The first observation may have been taken before any rebuild I/O.
        rate = 0.0

    log_latency = math.log(latency)
    gap = log_latency - math.log(target)
    features = np.array(
        (read, write, foreground, pressure, rate, rate * pressure, rate * rate),
        dtype=np.float64,
    )

    if history is None:
        weights = np.array((0.2, 0.4, 0.7, 0.6, 0.65, 0.9, 0.55))
        covariance = np.diag((2.0, 2.0, 3.0, 3.0, 8.0, 5.0, 5.0))
        switch_error = 0.07
        speed = 0.0
        speed_samples = 0
    else:
        (
            previous_features, previous_latency, previous_progress,
            weights, covariance, switch_error, speed, speed_samples,
        ) = history

        # Learn from applied rates, including any increases made by the filter.
        delta = features - previous_features
        error = log_latency - previous_latency - float(weights @ delta)
        if abs(float(delta[4])) > 0.1:
            load_change = max(abs(float(v)) for v in delta[:4])
            if load_change < 0.15:
                switch_error = 0.90 * switch_error + 0.10 * min(0.30, abs(error))

        if float(delta @ delta) > 1e-7:
            projected = covariance @ delta
            denominator = 0.999 + float(delta @ projected)
            weights = weights + projected * (min(0.7, max(-0.7, error)) / denominator)
            covariance = (covariance - np.outer(projected, projected) / denominator) / 0.999

        gained = progress - previous_progress
        if gained > 0.0 and 0.0 < progress < 1.0:
            sample = gained / rate
            if speed_samples == 0:
                speed = sample
            else:
                speed = 0.8 * speed + 0.2 * min(1.5 * speed, max(0.5 * speed, sample))
            speed_samples += 1

    state["controller"] = (
        features, log_latency, progress, weights, covariance,
        switch_error, speed, speed_samples,
    )

    # Anchor counterfactuals to the measured latency. The fit estimates only
    # how much the latency would change if rebuild traffic changed.
    safe = []
    for action, candidate in enumerate(rates):
        change = candidate - rate
        if change == 0.0:
            if gap <= 0.0:
                safe.append(action)
            continue

        slope = float(weights[4] + weights[5] * pressure + weights[6] * (candidate + rate))
        slope = min(4.5, max(0.25, slope))
        predicted_gap = gap + change * slope
        if change > 0.0:
            margin = 0.025 + 0.025 * pressure + (0.055 + 0.65 * switch_error) * math.sqrt(change)
        else:
            margin = 0.025 if gap > 0.0 else 0.0
        if predicted_gap <= -margin:
            safe.append(action)

    if not safe:
        return 0

    # Extra bandwidth buys no time if a lower safe rate finishes this tick.
    if speed_samples >= 3:
        remaining = max(0.0, 1.0 - progress)
        for action in safe:
            if remaining <= 0.95 * speed * rates[action]:
                return action

    return safe[-1]
```