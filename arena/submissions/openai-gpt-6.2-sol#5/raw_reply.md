```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    read, write, bw, latency, target, _, _, _, progress, _, alloc, headroom = (
        float(value) for value in obs
    )
    rates = np.array([1.0 / 6.0, 5.0 / 12.0, 2.0 / 3.0, 1.0])

    # Identify the applied action from the normalized allocation. The last
    # request is a lower bound, since the filter can only raise it.
    encodings = np.array([
        [1.0 / 6.0, 5.0 / 12.0, 2.0 / 3.0, 1.0],
        [0.0, 0.3, 0.6, 1.0],
        [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        [0.25, 0.5, 0.75, 1.0],
    ])
    previous_request = state.get("request", 0)
    errors = (encodings - alloc) ** 2
    errors[:, :previous_request] = np.inf
    applied = errors.argmin(axis=1)
    encoding_loss = (
        0.97 * state.get("encoding_loss", np.zeros(4))
        + errors[np.arange(4), applied]
    )
    state["encoding_loss"] = encoding_loss
    encoding = int(encoding_loss.argmin())
    current = int(applied[encoding])

    # Fit changes in log latency, separating changes in foreground traffic
    # from changes in rebuild allocation. Anchor each forecast to latency_now.
    pressure = min(1.0, max(0.0, 0.5 * (bw + 1.0 - headroom)))
    rebuild = (0.8 * rates + 0.2 * rates ** 2) * (0.65 + 0.85 * pressure)
    features = np.array([read, write, bw, rebuild[current]])
    log_latency = float(np.log(max(latency, 1e-5)))
    log_ratio = log_latency - float(np.log(max(target, 1e-5)))

    prior = np.array([0.4, 0.5, 0.8, 0.7])
    ridge = np.diag([0.45, 0.45, 0.35, 0.22])
    gram = state.get("gram", np.zeros((4, 4)))
    rhs = state.get("rhs", np.zeros(4))
    noise = state.get("noise", 0.055)
    speed = state.get("speed", 0.0)
    finish_speed = 0.0

    previous = state.get("previous")
    if previous is not None:
        old_features, old_latency, old_progress, old_encoding = previous
        if old_encoding == encoding:
            x = features - old_features
            if latency > 1e-5 and old_latency > -11.51 and x @ x > 1e-5:
                residual = (
                    log_latency - old_latency
                    - float(x @ state.get("coefficients", prior))
                )
                limit = max(0.12, 3.0 * noise)
                clipped = min(limit, max(-limit, residual))
                response = log_latency - old_latency - residual + clipped
                gram = 0.994 * gram + np.outer(x, x)
                rhs = 0.994 * rhs + x * (response - float(x @ prior))
                noise = 0.97 * noise + 0.03 * min(0.30, abs(residual))

            measured_speed = (progress - old_progress) / rates[current]
            if measured_speed > 0.0:
                if speed > 0.0:
                    finish_speed = 0.98 * min(speed, measured_speed)
                    speed = 0.85 * speed + 0.15 * measured_speed
                else:
                    speed = measured_speed
        else:
            speed = 0.0

    fit = np.linalg.solve(
        gram + ridge, np.column_stack((rhs, [0.0, 0.0, 0.0, 1.0]))
    )
    coefficients = prior + fit[:, 0]
    slope = min(3.0, max(0.04, float(coefficients[3])))
    uncertainty = max(0.025, noise) * float(np.sqrt(fit[3, 1]))

    action = 0
    for candidate in range(3, -1, -1):
        change = float(rebuild[candidate] - rebuild[current])
        predicted = log_ratio + slope * change
        margin = abs(change) * (0.04 + 1.5 * uncertainty)
        if candidate > current:
            margin += 0.015 + 0.20 * noise
        if predicted + margin <= 0.0:
            action = candidate
            break

    # If a slower safe rate will also finish this interval, use it.
    remaining = max(0.0, 1.0 - progress)
    if finish_speed > 0.0:
        for candidate in range(action):
            if finish_speed * rates[candidate] >= remaining:
                action = candidate
                break

    state["previous"] = (features, log_latency, progress, encoding)
    state["gram"] = gram
    state["rhs"] = rhs
    state["coefficients"] = coefficients
    state["noise"] = noise
    state["speed"] = speed
    state["request"] = action
    return action
```