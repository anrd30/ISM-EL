import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    read, write, bw, latency, target, _, _, _, progress, _, alloc, headroom = map(float, obs)
    y = max(0.0, latency) / max(1e-6, target)

    # Infer which of two common allocation normalizations is in use.
    rates = (1.0 / 6.0, 5.0 / 12.0, 2.0 / 3.0, 1.0)
    zero_based = (0.0, 0.3, 0.6, 1.0)
    votes = state.get("allocation_scale", 0.0)
    if 0.02 < alloc < 0.98 or (alloc <= 0.02 and "progress" in state):
        e_ratio = min(abs(alloc - r) for r in rates)
        e_zero = min(abs(alloc - r) for r in zero_based)
        votes = max(-0.3, min(0.3, votes + e_ratio - e_zero))
    state["allocation_scale"] = votes
    levels = zero_based if votes > 0.015 else rates
    applied = min(range(4), key=lambda i: abs(alloc - levels[i]))

    pressure = max(0.0, min(1.0, 0.65 * (1.0 - headroom) + 0.35 * bw))

    # Fit latency relative to the target. The last three terms describe
    # how rebuild allocation affects latency at the observed load.
    x = rates[applied]
    features = np.array(
        [1.0, read, write, bw, pressure, pressure * pressure,
         x, x * x, x * pressure]
    )
    if "coef" not in state:
        state["coef"] = np.array(
            [0.16, 0.10, 0.14, 0.15, 0.15, 0.15, 0.32, 0.14, 0.48]
        )
        state["cov"] = np.diag([16.0, 5.0, 5.0, 5.0, 5.0, 4.0, 3.0, 2.0, 3.0])

    coef = state["coef"]
    cov = state["cov"]
    residual = y - float(features @ coef)
    direction = cov @ features
    denominator = 1.0 + float(features @ direction)
    coef += (max(-0.5, min(0.5, residual)) / denominator) * direction
    cov -= np.outer(direction, direction) / denominator
    noise = 0.96 * state.get("noise", 0.04) + 0.04 * min(0.25, abs(residual))
    state["noise"] = noise

    # Anchor counterfactual predictions to measured latency rather than
    # relying on the fitted foreground baseline.
    choice = 0
    for i in (3, 2, 1):
        r = rates[i]
        middle = 0.5 * (r + x)
        prior_slope = 0.32 + 0.28 * middle + 0.48 * pressure
        slope = float(coef[6] + 2.0 * middle * coef[7] + pressure * coef[8])
        slope = max(0.08, min(2.5, 0.90 * slope + 0.10 * prior_slope))
        delta = r - x
        predicted = y + delta * slope

        contrast = np.array([delta, delta * (r + x), delta * pressure])
        leverage = max(0.0, float(contrast @ cov[6:, 6:] @ contrast))
        margin = min(0.15, 0.022 + 0.55 * noise + 0.045 * leverage ** 0.5)
        if i > applied:
            margin += 0.01
        if predicted <= 1.0 - margin:
            choice = i
            break

    # If a slower rate can finish in the same interval, use that rate.
    previous_progress = state.get("progress", progress)
    step = progress - previous_progress
    state["progress"] = progress
    throughput = state.get("throughput", 0.0)
    if step > 0.0:
        sample = step / x
        throughput = sample if throughput == 0.0 else 0.8 * throughput + 0.2 * sample
        state["throughput"] = throughput
    if throughput > 0.0:
        remaining = max(0.0, 1.0 - progress)
        for i in range(choice):
            if remaining <= 0.9 * throughput * rates[i]:
                return i

    return choice
