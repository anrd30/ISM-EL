import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    # Get the current observation
    fg_read_iops = obs[0]
    fg_write_iops = obs[1]
    fg_bw = obs[2]
    latency_now = obs[3]
    latency_target = obs[4]
    hazard = obs[5]
    hazard_uncertainty = obs[6]
    redundancy_left = obs[7]
    progress = obs[8]
    work_remaining = obs[9]
    alloc_current = obs[10]
    bw_available = obs[11]

    # State variables to remember
    state['fg_read_iops'] = fg_read_iops
    state['fg_write_iops'] = fg_write_iops
    state['fg_bw'] = fg_bw
    state['latency_now'] = latency_now
    state['latency_target'] = latency_target
    state['hazard'] = hazard
    state['hazard_uncertainty'] = hazard_uncertainty
    state['redundancy_left'] = redundancy_left
    state['progress'] = progress
    state['work_remaining'] = work_remaining
    state['alloc_current'] = alloc_current
    state['bw_available'] = bw_available

    # Determine the optimal action based on the current state
    if progress < 1.0:  # Not finished
        # Prioritize finishing quickly
        if progress < 0.5:
            return 0  # Slowest rate
        elif progress < 0.75:
            return 1  # Medium rate
        else:
            return 2  # Fastest rate
    else:  # Finished
        # Now focus on minimizing latency violations and degraded state
        if latency_now > latency_target:
            # If latency is too high, need to reduce rate
            if latency_now - latency_target > 0.5:
                return 3  # Lowest rate to avoid violation
            else:
                # Minimal reduction needed
                return 0
        else:
            # If latency is okay, prioritize reducing degraded state
            if state['progress'] < 0.5:
                return 0  # Slowest rate to reduce degraded state
            elif state['progress'] < 0.75:
                return 1  # Medium rate
            else:
                return 2  # Fastest rate

    # In case of tie, default to fastest (2)
    return 2

# Note: The function is written to be deterministic and under 10ms per call.
# It uses only `numpy` and the Python standard library.
