import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    # Initialize state variables
    allocated = 0  # Current allocation
    progress_limit = 0.5  # Maximum allowed progress
    progress = 0.0  # Current progress

    # Initialize safety filter
    safety_filter = 0.0  # Safety filter value

    # Initialize cost variables
    latency_degradation = 0.0
    degraded_state = 0.0
    latency_violation = 0.0

    # Initialize remaining resources
    work_remaining = 0.0
    progress_remaining = 0.0
    safety_remaining = 0.0

    # Initialize other variables
    hazard = 0.0
    hazard_uncertainty = 0.0
    redundancy_left = 0.0

    # Initialize bandwidths
    allocated_bw = 0.0
    progress_bw = 0.0

    # Read the current observation
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

    # Check if safety filter is active
    if safety_filter > 0.0:
        return 0  # Safety filter overrides action

    # Calculate current bandwidth usage
    allocated_bw = alloc_current * fg_bw
    progress_bw = progress * fg_bw

    # Check safety filter constraints
    if safety_filter > 0.0:
        # Safety filter value is not 0, so no action can be taken
        return 0

    # Check if progress is within limits
    if progress > progress_limit:
        return 0  # Progress is too high

    # Calculate latency degradation
    latency_degradation = 0.1 * progress

    # Calculate degraded state
    degraded_state = 1.0

    # Calculate latency violation
    latency_violation = 0.0

    # Calculate work remaining
    work_remaining = work_remaining - progress_bw

    # Calculate remaining safety
    safety_remaining = 0.1 * (redundancy_left - progress)

    # Calculate hazard contribution
    hazard_contribution = hazard * hazard_uncertainty

    # Calculate total cost
    total_cost = latency_degradation + degraded_state + latency_violation + hazard_contribution + safety_remaining

    # Determine action based on total cost
    if total_cost < 0.5:
        # Take the lowest available rate (ACTIONS_MBPS[0])
        return 0
    elif total_cost < 1.0:
        # Take the middle available rate (ACTIONS_MBPS[1])
        return 1
    elif total_cost < 1.5:
        # Take the next highest rate (ACTIONS_MBPS[2])
        return 2
    else:
        # Take the highest available rate (ACTIONS_MBPS[3])
        return 3
