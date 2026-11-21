import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    # Extract current state
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

    # Simple threshold-based decision
    if hazard > 0.5:
        return 0
    elif hazard >= 0.2 and hazard <= 0.5:
        return 1
    elif hazard >= 0.1 and hazard <= 0.2:
        return 2
    else:
        return 3
