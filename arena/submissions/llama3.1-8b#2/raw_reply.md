Here is the function `policy` as specified:
```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    
    # Calculate rebuild rate based on progress and available bandwidth
    rebuild_rate = (obs[8] * obs[11]) / (obs[9] + 1e-6)
    
    # Choose the action that minimizes cost while staying within safety filter constraints
    actions = [20, 50, 80, 120]
    min_cost = float('inf')
    best_action = -1
    
    for i in range(4):
        cost = (1.0 * obs[3] + 0.1 * (obs[8] < 1) + 10.0 * (obs[3] > obs[4])) + 0.01 * abs(actions[i] - rebuild_rate)
        if cost < min_cost:
            min_cost = cost
            best_action = i
    
    # Apply safety filter to selected action
    if obs[7] < 0.1:  # high redundancy exposure, safety filter may raise the action
        best_action = np.argmax([20, 50, 80, 120])
    
    return best_action
```
This function implements a policy that chooses a rebuild rate based on progress and available bandwidth, while trying to minimize cost and stay within safety filter constraints. The `state` dictionary is not used in this implementation, as it is intended for storing temporary variables during an episode, which are not needed here.