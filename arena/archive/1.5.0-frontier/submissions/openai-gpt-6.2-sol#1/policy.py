import math
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    headroom = min(1.0, max(0.0, float(obs[11])))
    allocation = min(1.0, max(0.0, float(obs[10])))
    progress = float(obs[8])
    latency = max(1e-5, float(obs[3]))
    target = max(1e-5, float(obs[4]))
    y = math.log(latency)

    # Learn the latency response from applied allocations, not requests:
    # the safety filter can make those two allocations different.
    saved = state.get("fit")
    minmax = allocation < 0.02 or (saved is not None and saved[5])
    scale = 1.0 / (headroom + 0.30)
    x = allocation * scale
    f = np.array([
        1.0, -math.log(headroom + 0.20), float(obs[0]),
        float(obs[1]), float(obs[2]), x, x * x,
    ])

    if saved is None:
        beta = np.array([0.0, 0.65, 0.10, 0.15, 0.0, 0.38, 0.04])
        beta[0] = y - float(f @ beta)
        cov = np.diag([4.0, 3.0, 2.0, 2.0, 2.0, 0.8, 0.12])
        noise = 0.0225
        speed = 0.0
    else:
        beta, cov, noise, prev_progress, speed, _ = saved
        err = y - float(f @ beta)
        v = cov @ f
        leverage = max(0.0, float(f @ v))
        denom = 0.998 + leverage
        beta = beta + (max(-0.75, min(0.75, err)) / denom) * v
        cov = (cov - np.outer(v, v) / denom) / 0.998
        noise = 0.97 * noise + 0.03 * min(0.36, err * err / (1.0 + leverage))

        # Estimate progress per interval at full rate for the final decision.
        moved = progress - prev_progress
        rate = (1.0 / 6.0 + (5.0 / 6.0) * allocation) if minmax else allocation
        if moved > 0.0 and rate > 0.05:
            sample = moved / rate
            speed = sample if speed == 0.0 else 0.90 * speed + 0.10 * sample

    state["fit"] = (beta, cov, noise, progress, speed, minmax)

    needed = (max(0.0, 1.0 - progress) / (0.95 * speed)
              if speed > 0.0 else 1.0)
    margin = math.log(latency / target)
    best_action = 0
    best_score = math.inf

    for action, rate in enumerate((1.0 / 6.0, 5.0 / 12.0, 2.0 / 3.0, 1.0)):
        candidate = (rate - 1.0 / 6.0) * 1.2 if minmax else rate
        z = candidate * scale
        dx = z - x
        dx2 = dx * (z + x)

        # Anchor each prediction at the measured latency. Only extrapolate
        # the change due to allocation; keep that response monotone.
        slope = float(beta[5] + beta[6] * (z + x))
        slope = max(0.04, min(1.8, slope))
        delta = dx * slope
        predicted_margin = margin + delta

        variance = (cov[5, 5] * dx * dx + 2.0 * cov[5, 6] * dx * dx2
                    + cov[6, 6] * dx2 * dx2)
        uncertainty = math.sqrt(
            0.024 ** 2 + 0.06 * noise
            + noise * min(4.0, max(0.0, float(variance)))
        )
        violation = 0.5 * math.erfc(
            -(predicted_margin + 0.012) / (math.sqrt(2.0) * uncertainty)
        )
        degradation = 0.22 * math.exp(max(-6.0, min(2.0, predicted_margin)))
        score = 10.0 * violation + degradation - 0.43 * min(rate, needed)
        if score < best_score:
            best_score = score
            best_action = action

    return best_action
