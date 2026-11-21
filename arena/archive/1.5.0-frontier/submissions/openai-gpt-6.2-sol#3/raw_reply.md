```python
import math

import numpy as np


def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    rates = np.array((1.0 / 6.0, 5.0 / 12.0, 2.0 / 3.0, 1.0))
    allocation = float(obs[10])
    first = "coef" not in state

    # The allocation field is normalized. Identify its encoding from
    # the applied allocations, not from the actions that were requested.
    encodings = np.array((
        (1.0 / 6.0, 5.0 / 12.0, 2.0 / 3.0, 1.0),
        (0.0, 0.3, 0.6, 1.0),
        (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
    ))
    errors = state.get("encoding_errors")
    if errors is None:
        errors = np.zeros(3)
    distances = np.abs(encodings - allocation)
    if not first:
        # The filter may raise the action, but it cannot lower it.
        distances[:, :state["requested"]] += 0.25
    indices = np.argmin(distances, axis=1)
    if not first or allocation > 0.02:
        errors += distances[np.arange(3), indices]
    encoding = int(np.argmin(errors))
    applied = float(rates[indices[encoding]])
    if first and allocation < 0.02:
        applied = 0.0
    state["encoding_errors"] = errors

    read = float(obs[0])
    write = float(obs[1])
    foreground = float(obs[2])
    headroom = float(obs[11])
    busy = min(1.0, max(0.0, 0.5 * (foreground + 1.0 - headroom)))
    target = max(1e-6, float(obs[4]))
    ratio = max(1e-4, float(obs[3]) / target)
    measured = math.log(min(20.0, ratio))

    # Fit log latency online. Separate foreground terms from rebuild
    # terms so that counterfactuals hold the foreground load fixed.
    q = np.array((0.0, rates[0], rates[1], rates[2], rates[3], applied))
    features = np.empty((6, 13))
    features[:, :7] = (
        1.0, busy, busy * busy, busy ** 3,
        read - 0.5, write - 0.5, foreground - (1.0 - headroom),
    )
    features[:, 7] = q
    features[:, 8] = q * q
    features[:, 9] = q * busy
    features[:, 10] = q * busy * busy
    features[:, 11] = q * q * busy
    features[:, 12] = q * (write - read)

    if first:
        coef = np.array((
            measured - applied * (0.22 + 0.40 * busy + 0.10 * applied),
            0.25, 0.30, 0.20, 0.0, 0.0, 0.0,
            0.22, 0.10, 0.40, 0.0, 0.0, 0.0,
        ))
        prior_var = np.square(np.array((
            0.40, 0.55, 0.65, 0.55, 0.40, 0.40, 0.35,
            0.30, 0.21, 0.40, 0.40, 0.25, 0.25,
        )))
        cov = np.diag(prior_var)
        noise = 0.0049
        shadow = 0.58
    else:
        coef = state["coef"]
        cov = state["cov"]
        prior_var = state["prior_var"]
        noise = state["noise"]
        shadow = state["shadow"]

    x = features[-1]
    residual = measured - float(x @ coef)
    vx = cov @ x
    innovation = max(1e-8, noise + float(x @ vx))
    bound = 3.0 * math.sqrt(innovation)
    coef = coef + (vx / innovation) * max(-bound, min(bound, residual))
    cov = 0.9995 * (cov - np.outer(vx, vx) / innovation)
    cov.flat[::14] += 0.0005 * prior_var
    noise = 0.98 * noise + 0.02 * min(0.045, residual * residual)
    noise = max(0.0009, noise)
    state["coef"] = coef
    state["cov"] = cov
    state["prior_var"] = prior_var
    state["noise"] = noise

    # Anchor forecasts at the observed latency. Enforce a monotone
    # latency curve in the rebuild rate at this foreground load.
    fitted = features @ coef
    increments = np.diff(fitted[:5])
    increments = np.maximum(increments, 0.012 * np.diff(q[:5]))
    curve = np.concatenate(([0.0], np.cumsum(increments)))
    baseline = float(np.interp(applied, q[:5], curve))
    log_latency = measured + curve[1:] - baseline
    delta = features[1:5] - x
    variance = np.einsum("ij,ij->i", delta @ cov, delta)
    change = np.abs(rates - applied)
    sigma = np.sqrt(
        0.64 * np.maximum(0.0, variance)
        + 0.0001 + 0.16 * noise * change
    )

    current_risk = 0.5 * math.erfc(-measured / (math.sqrt(2.0) * 0.01))
    risk = np.empty(4)
    for i in range(4):
        p = 0.5 * math.erfc(
            -float(log_latency[i]) / (math.sqrt(2.0) * float(sigma[i]))
        )
        if rates[i] <= applied:
            p = min(p, current_risk)
        else:
            p = max(p, current_risk)
        risk[i] = p

    latency = np.exp(np.clip(log_latency, -9.0, 3.0))
    no_rebuild_latency = math.exp(max(-9.0, min(3.0, measured - baseline)))
    interval_cost = 0.1 + 0.8 * np.maximum(0.0, latency - no_rebuild_latency)

    # Use the recently available cost of doing work as its marginal
    # value. Update more quickly when cheap rebuild opportunities appear.
    safe = risk < 0.08
    if np.any(safe):
        unit_cost = float(np.min((interval_cost[safe] + risk[safe]) / rates[safe]))
        unit_cost = min(1.25, max(0.23, unit_cost))
        weight = 0.08 if unit_cost < shadow else 0.015
        shadow += weight * (unit_cost - shadow)
    state["shadow"] = shadow

    # Avoid rewarding bandwidth that would go unused on the final step.
    progress = float(obs[8])
    stride = state.get("stride", 0.0)
    stride_count = state.get("stride_count", 0)
    if not first and applied > 0.0:
        advance = progress - state["progress"]
        if advance > 1e-7 and progress < 0.999999:
            sample = advance / applied
            stride = sample if stride_count == 0 else 0.85 * stride + 0.15 * sample
            stride_count += 1
    state["progress"] = progress
    state["stride"] = stride
    state["stride_count"] = stride_count

    useful = rates
    if stride_count >= 4 and stride > 0.0:
        remaining = max(0.0, 1.0 - progress) / stride
        useful = np.minimum(rates, 1.03 * remaining)

    action = int(np.argmin(interval_cost + 10.0 * risk - shadow * useful))
    state["requested"] = action
    return action
```