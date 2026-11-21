import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    # Unpack observation
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

    # Action ratios based on ACTIONS_MBPS = [20, 50, 80, 120]
    ratios = [1.0, 2.5, 4.0, 6.0]

    # Initialize the estimate for the allocation of action 0.
    # Start with a large value to be conservative on the first step.
    if 'alloc_0_est' not in state:
        state['alloc_0_est'] = 1.0 

    # Dynamically estimate the actual bandwidth allocation of action 0.
    # The safety filter can only raise the action, so alloc_current is an upper bound
    # for the requested action's allocation. The minimum over time converges to the true alloc_0.
    prev_a = state.get('prev_a', None)
    if prev_a is not None and alloc_current > 1e-4:
        est = alloc_current / ratios[prev_a]
        if est > 0.005:  # Ignore anomalously small or paused allocations
            if 'min_alloc_0' not in state:
                state['min_alloc_0'] = est
            else:
                state['min_alloc_0'] = min(state['min_alloc_0'], est)
            
            state['alloc_0_est'] = state['min_alloc_0']

    alloc_0_est = state['alloc_0_est']
    expected_allocs = [r * alloc_0_est for r in ratios]

    # Queueing theory model: Latency L = k / Margin
    # Margin is the remaining unused bandwidth capacity.
    B = bw_available
    M_curr = B - alloc_current
    
    # Clamp margin to prevent division by zero or negative margins during overloads
    if M_curr < 0.01:
        M_curr = 0.01
        
    k = latency_now * M_curr

    best_action = 0
    best_obj = float('inf')

    # Evaluate the expected cost for each action
    for i in range(4):
        A = expected_allocs[i]
        
        # Reject actions that consume all available bandwidth (leaves < 0.01 margin)
        if A >= B - 0.01:
            continue
            
        M_a = B - A
        L_a = k / M_a
        
        # Hard constraint: keep latency safely below the target to avoid the massive 10.0 penalty
        if L_a > 0.85 * latency_target:
            continue
            
        # The cost function is: 1.0 * latency_degradation + 0.1 * degraded_state
        # We assume latency_degradation is proportional to L_a.
        cost_rate = L_a + 0.1
        
        # We want to minimize the total cost, which is proportional to cost_rate / A
        if A < 1e-4:
            obj = float('inf')
        else:
            obj = cost_rate / A
            
        if obj < best_obj:
            best_obj = obj
            best_action = i

    # Fallback if all actions are deemed unsafe (e.g., severe bandwidth shortage)
    if best_obj == float('inf'):
        best_action = 0

    # Additional safety override: if currently violating, definitively step down the action
    if latency_now > latency_target:
        if prev_a is not None:
            best_action = min(best_action, max(0, prev_a - 1))
        else:
            best_action = 0

    state['prev_a'] = best_action
    return int(best_action)
