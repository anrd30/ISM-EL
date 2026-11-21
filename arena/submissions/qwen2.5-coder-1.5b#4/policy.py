import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Return 0, 1, 2, or 3 based on the observation and state.
    
    Parameters:
    obs (np.ndarray): A 12-element float32 vector with normalised fields:
        [0] fg_read_iops
        [1] fg_write_iops
        [2] fg_bw
        [3] latency_now
        [4] latency_target
        [5] hazard
        [6] hazard_uncertainty
        [7] redundancy_left
        [8] progress
        [9] work_remaining
        [10] alloc_current
        [11] bw_available
    
    state (dict): A dictionary to store episode-specific information.
    
    Returns:
    int: An integer in the range [0, 3].
    """
    # Extract relevant fields from obs
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
    
    # Calculate the total cost
    latency_degradation = 2.5 * (latency_now - latency_target)
    degraded_state = 1 if progress < 1 else 0
    latency_violation = 1 if latency_now > latency_target else 0
    
    total_cost = 1.0 * latency_degradation + 0.1 * degraded_state + 10.0 * latency_violation
    
    # Determine the action based on the current state and cost
    actions_mbps = [20, 50, 80, 120]
    
    if total_cost > 15.0:  # High cost, choose slowest action
        action = actions_mbps[3]
    elif total_cost > 10.0:  # Moderate cost, choose average action
        action = actions_mbps[2]
    elif total_cost > 5.0:  # Low cost, choose medium action
        action = actions_mbps[1]
    else:  # Very low cost, choose fastest action
        action = actions_mbps[0]
    
    return action
