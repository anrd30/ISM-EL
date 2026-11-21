import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    
    # Constants for actions (MBps)
    ACTIONS_MBPS = [20, 50, 80, 120]
    
    # Extracting observation fields
    fg_read_iops, fg_write_iops, fg_bw, latency_now, latency_target, hazard, hazard_uncertainty, redundancy_left, progress, work_remaining, alloc_current, bw_available = obs
    
    # Calculate available bandwidth for the rebuild
    bw_available_for_rebuild = bw_available - fg_bw
    
    # Safety filter check (if the safety filter raises the action, we should not perform any action)
    # Since we don't control the safety filter, we assume it is a function that returns True if action is safe, False otherwise
    safe_to_increase_bw = safety_filter(bw_available_for_rebuild, hazard, hazard_uncertainty, redundancy_left)
    
    # If it's safe to increase bandwidth, we choose the highest rate that we can apply without exceeding available bandwidth
    if safe_to_increase_bw:
        chosen_action_index = np.argmax([min(rate, bw_available_for_rebuild) for rate in ACTIONS_MBPS])
    else:
        # If it's not safe, we choose to not increase bandwidth (effectively index 0)
        chosen_action_index = 0
    
    # Update state with chosen action for future reference
    state['chosen_action_index'] = chosen_action_index
    
    return chosen_action_index

# Placeholder for the safety filter function
# This should be implemented based on the system's safety requirements
def safety_filter(bw_available_for_rebuild, hazard, hazard_uncertainty, redundancy_left):
    # Implement the logic based on the system's safety requirements
    # For now, we return True to simulate that it's always safe to apply an action
    return True
