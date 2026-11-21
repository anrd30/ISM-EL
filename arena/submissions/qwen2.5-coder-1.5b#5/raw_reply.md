```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Return 0, 1, 2, or 3 based on the given observation and current state.
    
    The objective is to minimize total cost while ensuring the rebuild completes without pushing latency over the limit.
    
    :param obs: A numpy array of shape (12,) with normalised values from the observation.
    :param state: A dictionary where you can store intermediate results for episode tracking.
    :return: An integer in [0, 3] representing the chosen action index.
    """
    # Extract relevant fields
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

    # Define the actions and their corresponding costs
    actions = [20, 50, 80, 120]
    action_costs = [1.0 * latency_degradation + 0.1 * degraded_state + 10.0 * latency_violation
                    for degraded_state, latency_degradation, latency_violation in zip([0] * len(actions), actions, [0] * len(actions))]

    # Determine the cost of each action
    costs = []
    for i, action in enumerate(actions):
        # Calculate the remaining redundancy and progress
        remaining_redundancy = redundancy_left - (action / 300) * fg_bw
        remaining_progress = max(0, progress + action / 300)

        # Calculate the cost of the action
        cost = (latency_degradation + 0.1 * remaining_redundancy + 10.0 * (1 if latency_now > latency_target else 0)) + alloc_current - bw_available

        costs.append(cost)

    # Choose the action with the lowest cost
    return np.argmin(costs)
```

This function implements a policy based on the observation and current state to choose the best rebuild rate. It calculates the cost of each action considering factors such as latency degradation, degraded state, redundancy left, progress, and available bandwidth. The goal is to minimize total cost while ensuring the rebuild completes without pushing latency over the limit.