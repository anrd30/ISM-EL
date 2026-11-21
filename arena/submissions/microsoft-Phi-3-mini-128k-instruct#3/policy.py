import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    
    # Extract relevant fields from the observation
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
    
    # Calculate the remaining headroom for the rebuild
    bw_headroom = bw_available - fg_bw
    
    # Define a safety margin to avoid latency violation
    safety_margin = 0.2 * latency_target
    
    # If latency is already exceeding the target, do not increase the allocation
    if latency_now > latency_target + safety_margin:
        return 0
    
    # If there is enough headroom and hazard is high, try to increase allocation
    if bw_headroom > 0 and hazard > 0.5:
        if alloc_current < 120:  # Assuming the highest rate is 120 MBps
            state['last_action'] = 3
            return 3
    
    # If there is not enough headroom, try to use less allocation
    if bw_headroom < 0:
        if alloc_current > 20:  # Assuming the lowest rate is 20 MBps
            state['last_action'] = 0
            return 0
    
    # If the rebuild is almost complete, try to finish it faster
    if work_remaining < 0.1 * (10 - progress):  # Assuming 10 is the completion mark
        if alloc_current < 120:  # Assuming the highest rate is 120 MBps
            state['last_action'] = 3
            return 3
    
    # If none of the above conditions are met, maintain current allocation
    state['last_action'] = alloc_current
    return alloc_current
