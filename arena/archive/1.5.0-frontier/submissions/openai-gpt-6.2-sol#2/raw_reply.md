import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    read, write, load, latency, target = map(float, obs[:5])
    progress = float(obs[8])
    allocation = max(0.0, float(obs[10]))
    pressure = 1.0 - min(1.0, max(0.0, float(obs[11])))
    rates = (1.0 / 6.0, 5.0 / 12.0, 2.0 / 3.0, 1.0)
    level_rates = (0.0, 0.3, 0.6, 1.0)

    # Infer the observation's rate encoding from applied, not requested, rates.
    # A request for the highest rate cannot be raised by the safety filter.
    previous = state.get("request")
    scale = state.get("scale", 1.0)
    known_scale = state.get("known_scale", False)
    level_encoding = state.get("level_encoding", False)
    if previous == 3 and allocation > 0.0:
        scale = allocation
        known_scale = True
    elif previous is not None and not known_scale:
        if min(abs(allocation - x) for x in level_rates) < 0.008:
            if allocation < 0.95:
                level_encoding = True
        elif not level_encoding and allocation > 0.0:
            scale = min(scale, allocation / rates[previous])
            scale = max(scale, allocation)
    if known_scale and scale > 0.0:
        fraction = allocation / scale
        if min(abs(fraction - x) for x in level_rates[:3]) < 0.008:
            level_encoding = True
    state["scale"] = scale
    state["known_scale"] = known_scale
    state["level_encoding"] = level_encoding
    encoding = level_rates if level_encoding else rates
    applied = min(range(4), key=lambda i: abs(allocation - scale * encoding[i]))

    # Estimate interval progress at full speed for the final few decisions.
    gain = state.get("full_speed_gain", 0.0)
    if previous is not None:
        delta = progress - state["progress"]
        if delta > 0.0:
            sample = delta / rates[applied]
            gain = sample if gain == 0.0 else 0.85 * gain + 0.15 * sample
    state["progress"] = progress
    state["full_speed_gain"] = gain

    # Learn how foreground load and rebuild allocation affect log latency.
    # The allocation terms capture increasing marginal cost near saturation.
    a = allocation / max(scale, 0.05)
    x = np.array(
        [1.0, read, write, load, load * load, pressure,
         a, a * a, a * pressure],
        dtype=float,
    )
    y = float(np.log(max(latency, 1e-5)))
    beta = state.get("beta")
    if beta is None:
        beta = np.array([0.0, 0.15, 0.20, 0.30, 0.20, 0.35,
                         0.15, 0.20, 0.65])
        beta[0] = y - float(x @ beta)
        cov = np.diag([3.0, 3.0, 3.0, 3.0, 4.0, 3.0, 5.0, 4.0, 6.0])
        error = 0.04
    else:
        cov = state["cov"]
        residual = y - float(x @ beta)
        error = 0.97 * state["error"] + 0.03 * min(abs(residual), 0.35)
        cx = cov @ x
        denom = 0.9995 + float(x @ cx)
        beta = beta + cx * (min(0.6, max(-0.6, residual)) / denom)
        cov = (cov - np.outer(cx, cx) / denom) / 0.9995
    state["beta"] = beta
    state["cov"] = cov
    state["error"] = error

    # Anchor counterfactual predictions at the measured latency. Keep the
    # estimated rate response monotone even when observations are noisy.
    linear = float(beta[6] + beta[8] * pressure)
    quadratic = min(1.5, max(-0.5, float(beta[7])))
    linear = max(0.035 - min(0.0, 2.0 * quadratic * max(1.0, a)), linear)
    slope_now = linear + quadratic * a
    log_limit = float(np.log(max(target, 1e-5)))
    no_rebuild = float(np.exp(max(-20.0, y - a * slope_now)))
    remaining = max(0.0, 1.0 - progress)
    best = 0
    best_score = float("inf")
    for i, (rate, candidate) in enumerate(zip(rates, encoding)):
        change = candidate - a
        predicted = y + change * (linear + quadratic * (candidate + a))
        margin = 0.035 + min(0.065, 0.55 * error)
        margin += max(0.0, change) * (0.06 if known_scale else 0.10)

        if predicted + margin >= log_limit:
            # If none is comfortably below the limit, request the lowest rate.
            break

        degradation = max(0.0, float(np.exp(predicted)) - no_rebuild)
        effective_rate = rate
        if gain > 0.0 and remaining > 0.0:
            effective_rate = min(rate, remaining / gain)
        score = (0.1 + degradation) / max(1e-6, effective_rate)
        if score < best_score:
            best, best_score = i, score

    state["request"] = best
    return best