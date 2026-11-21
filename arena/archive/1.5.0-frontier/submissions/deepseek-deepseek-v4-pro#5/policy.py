import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1,2,3 for rebuild rate index."""
    # Unpack observation (see spec for indices)
    latency_now      = obs[3]
    latency_target   = obs[4]
    work_remaining  = obs[9]
    bw_available    = obs[11]   # headroom, normalized

    # If rebuild is essentially finished, do nothing.
    if work_remaining < 0.001:
        return 0

    # Immediate reaction to a latency violation: cut rate entirely.
    if latency_now > latency_target:
        # Reset integral to avoid wind-up after violation.
        state['integral'] = 0.0
        return 0

    # No violation: use a PI controller to keep latency just under target.
    error = latency_target - latency_now  # positive = we have slack

    # Initialise state on first call.
    if 'integral' not in state:
        # Start integral at a sensible value derived from headroom.
        state['integral'] = float(np.clip(bw_available * 3.0, 0.0, 3.0))
    if 'prev_error' not in state:
        state['prev_error'] = error

    integral = state['integral']

    # Controller gains – chosen to give quick response while avoiding oscillation.
    Kp = 2.0
    Ki = 0.3

    # Proportional term.
    P = Kp * error

    # Integral update with clamping to [0, 3].
    integral += Ki * error
    integral = float(np.clip(integral, 0.0, 3.0))

    # Raw action = integral + proportional, then clamp.
    raw = integral + P
    raw = float(np.clip(raw, 0.0, 3.0))

    # Convert to discrete action index.
    action = int(round(raw))
    action = max(0, min(3, action))

    # Store state for next step.
    state['integral'] = integral
    state['prev_error'] = error

    return action
