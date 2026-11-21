import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Determine the rebuild rate index to minimize the cost function.
    obs: 12-element float32 vector representing the current observation.
    state: dict to store state information across intervals.
    Returns an integer in [0, 3] corresponding to the chosen action.
    """
    # Extract the relevant fields from the observation
    fg_read_iops, fg_write_iops, fg_bw, latency_now, latency_target, hazard, hazard_uncertainty, redundancy_left, progress, work_remaining, alloc_current, bw_available = obs
    
    # Calculate the available bandwidth for the rebuild
    bw_available -= fg_bw
    
    # Initialize state if not already present
    if 'work_done' not in state:
        state['work_done'] = 0
    
    # Calculate the rate of work done per interval
    work_done_rate = progress / (state['work_done'] + 1)
    
    # Calculate the cost components
    latency_degradation = latency_now - latency_target
    degraded_state = progress < 1
    latency_violation = latency_now > latency_target
    
    # Calculate the total cost
    cost = 1.0 * latency_degradation + 0.1 * degraded_state + 10.0 * latency_violation
    
    # Determine the action based on the available bandwidth and current state
    if bw_available > 0:
        # Choose the highest rate that does not exceed the available bandwidth
        action_index = min(np.where(ACTIONS_MBPS <= bw_available)[0])
    else:
        # If there's no available bandwidth, do not rebuild and maintain current state
        action_index = 0
    
    # Update the state
    state['work_done'] += work_done_rate
    
    # Return the chosen action index
    return action_index

# Constants (not defined in the file as per instructions)
ACTIONS_MBPS = np.array([20, 50, 80, 120], dtype=np.float32)
