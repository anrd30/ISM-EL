```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    # Extracting observation data
    fg_read_iops, fg_write_iops, fg_bw, latency_now, latency_target, hazard, hazard_uncertainty, redundancy_left, progress, work_remaining, alloc_current, bw_available = obs
    
    # Constants
    ACTIONS_MBPS = [20, 50, 80, 120]
    TARGET_LATENCY = 0.8  # Assuming a target latency (e.g., 80% of the current latency)
    
    # State update (if needed)
    # state['some_key'] = some_value
    
    # Calculate bandwidth available for rebuild after subtracting foreground load
    bw_available_for_rebuild = bw_available - fg_bw
    
    # Calculate target bandwidth for rebuild (a fraction of available bandwidth)
    target_rebuild_bw = 0.7 * bw_available_for_rebuild  # Assuming we use 70% of the remaining bandwidth for rebuild
    
    # Calculate the cost function components
    latency_degradation = latency_now - TARGET_LATENCY
    degraded_state = 1 - progress
    latency_violation = 1 if latency_now > latency_target else 0
    
    # Calculate the cost
    cost = 1.0 * latency_degradation + 0.1 * degraded_state + 10.0 * latency_violation
    
    # Determine the action based on the cost and the available bandwidth for rebuild
    # If the cost is too high or if we can't maintain the target latency, we might choose a lower rebuild rate
    if cost > 10 or latency_now > latency_target:
        return np.argmin([action * bw_available_for_rebuild for action in ACTIONS_MBPS])
    
    # Otherwise, choose the highest possible action that doesn't exceed the target rebuild bandwidth
    possible_actions = [action for action in ACTIONS_MBPS if action * bw_available_for_rebuild <= target_rebuild_bw]
    if possible_actions:
        return np.argmax(possible_actions)
    
    # If all actions exceed the target rebuild bandwidth, we choose the lowest possible action
    return 0
```

This function implements a simple heuristic policy based on the current state of the system. It aims to balance the rebuild speed with the system's latency and availability. The policy dynamically adjusts the rebuild rate based on the current cost and the available bandwidth for rebuild after subtracting the foreground load. It also considers the system's latency and tries to maintain it below a target threshold. If the cost is too high or the latency target cannot be met, the policy chooses a lower rebuild rate. Otherwise, it selects the highest possible action that doesn't exceed the target rebuild bandwidth. If all actions exceed the target bandwidth, it defaults to the lowest possible action. This approach aims to minimize the total cost while considering system performance and availability.