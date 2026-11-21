import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    current_rate = state['alloc_current']
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
    bw_available = obs[11]
    actions = [0, 1, 2, 3]

    if current_rate == 0:
        # Start with lowest risk rate
        return 0

    # Check if we can increase
    if current_rate < 3:
        # Prefer higher rates when close to finishing
        if progress + 1 >= 1:
            if current_rate < 2:
                return 1
            elif current_rate < 3:
                return 2
        else:
            # If we're not close to finishing, prefer lower risk
            return 0

    # If we can't increase, stay at current rate
    return current_rate
