```python
import math

import numpy as np


def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    rates = np.array((1.0 / 6.0, 5.0 / 12.0, 2.0 / 3.0, 1.0))
    read, write, bw = (float(np.clip(obs[i], 0.0, 1.0)) for i in range(3))
    latency = max(0.0, float(obs[3]))
    target = max(0.0, float(obs[4]))
    work = max(0.0, float(obs[9]))
    alloc = float(obs[10])
    headroom = float(np.clip(obs[11], 0.0, 1.0))
    busy = 0.7 * (1.0 - headroom) + 0.3 * bw
    y = math.log((latency + 0.025) / (target + 0.025))

    if "n" not in state:
        state["n"] = 0
        state["grid"] = 0
        state["grids"] = np.array((rates, (rates - rates[0]) / (1.0 - rates[0]),
                                   (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)))
        state["grid_loss"] = np.array((0.0, 0.012, 0.020))
        state["step_work"] = 0.0
        state["work_samples"] = 0

    n = state["n"]
    grid = state["grid"]
    changed_grid = False
    if n:
        requested = state["requested"]
        if n == 1 and alloc > 0.0:
            # Also allow allocation to be normalized by a capacity other
            # than the maximum rebuild rate.
            scales = alloc / rates[requested:]
            scales = scales[(scales >= 0.06) & (scales <= 1.5)]
            state["grids"] = np.vstack((state["grids"], scales[:, None] * rates))
            state["grid_loss"] = np.concatenate(
                (state["grid_loss"], np.full(len(scales), 0.030))
            )

        grids = state["grids"]
        distances = np.abs(grids - alloc) / np.maximum(grids[:, -1:], 0.06)
        nearest = np.argmin(distances, axis=1)
        loss = distances[np.arange(len(grids)), nearest]
        state["grid_loss"] *= 0.99
        state["grid_loss"] += np.minimum(loss, 0.25) + 0.035 * (nearest < requested)
        grid = int(np.argmin(state["grid_loss"]))
        applied = int(nearest[grid])
        changed_grid = grid != state["grid"]
        state["grid"] = grid

        delta = state["work"] - work
        if delta > 1e-8:
            sample = delta / rates[applied]
            old = state["step_work"]
            if changed_grid or old == 0.0:
                state["step_work"] = sample
                state["work_samples"] = 1
            else:
                sample = min(2.0 * old, max(0.5 * old, sample))
                state["step_work"] = 0.88 * old + 0.12 * sample
                state["work_samples"] += 1
    else:
        applied = int(np.argmin(np.abs(rates - alloc)))

    if not n or changed_grid:
        state["theta"] = np.array(
            (-0.8, 0.12, 0.15, 0.15, 0.25, 0.25, 0.0, 0.0, 0.22, 0.55, 0.24)
        )
        state["cov"] = np.diag((6., 3., 3., 3., 3., 2., 2., 2., 1.4, 2., 0.8))
        state["noise"] = 0.10

    a = rates[applied]
    theta = state["theta"]
    cov = state["cov"]
    if n:
        x = np.array((1., read, write, bw, busy, busy * busy,
                      read * busy, write * busy, a, a * busy, a * a))
        px = cov @ x
        leverage = float(x @ px)
        err = float(np.clip(y - x @ theta, -1.0, 1.0))
        gain = px / (0.996 + leverage)
        theta += gain * err
        state["cov"] = (cov - np.outer(gain, px)) / 0.996
        state["noise"] = 0.97 * state["noise"] + 0.03 * min(
            0.35, abs(err) / math.sqrt(1.0 + leverage)
        )

    # Hold foreground load fixed and anchor the rate-response curve at the
    # latency we actually observe, rather than at the regression intercept.
    slope = float(np.clip(theta[8] + busy * theta[9], 0.04, 3.0))
    curvature = float(np.clip(theta[10], 0.0, 2.0))
    no_rebuild = y - slope * a - curvature * a * a
    base_latency = math.exp(min(2.0, no_rebuild))
    rate_cov = state["cov"][8:, 8:]

    best = 0
    best_score = float("inf")
    violation = 0.0
    for i, rate in enumerate(rates):
        d = rate - a
        d2 = rate * rate - a * a
        prediction = y + slope * d + curvature * d2

        z = np.array((d, busy * d, d2))
        variance = max(0.0, float(z @ rate_cov @ z))
        noise = max(0.045, state["noise"])
        sigma = math.sqrt(0.012 ** 2 + noise ** 2 * variance + (0.035 * d) ** 2)
        risk = 0.5 * math.erfc(-prediction / (math.sqrt(2.0) * sigma))
        violation = max(violation, risk)

        degradation = max(0.0, math.exp(min(2.0, prediction)) - base_latency)
        useful_rate = rate
        if state["work_samples"] >= 5:
            # Don't credit bandwidth beyond what is needed to finish.
            needed = work / (0.90 * state["step_work"])
            useful_rate = min(rate, max(rates[0], needed))

        ordinary_cost = (0.10 + 0.60 * degradation) / useful_rate
        score = 0.55 * ordinary_cost + violation * (10.0 - 0.30 * useful_rate)
        if score < best_score:
            best_score = score
            best = i

    state["n"] = n + 1
    state["work"] = work
    state["requested"] = best
    return best
```