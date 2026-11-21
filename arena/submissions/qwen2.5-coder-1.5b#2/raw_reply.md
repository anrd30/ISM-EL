```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    # Calculate the current latency degradation based on the foreground load and bandwidth share
    fg_read_iops = obs[0]
    fg_write_iops = obs[1]
    fg_bw = obs[2]
    latency_degradation = fg_read_iops + fg_write_iops * fg_bw
    
    # Determine the remaining redundancy exposure
    redundancy_left = state.get('redundancy_left', 100)
    
    # Calculate the fraction of the rebuild done
    progress = state.get('progress', 0)
    work_remaining = state.get('work_remaining', 100 * redundancy_left - (progress / redundancy_left))
    
    # Determine the action based on the current latency degradation and remaining workload
    if latency_degradation > obs[4]:
        return 3  # Increase rebuild rate to finish faster
    elif latency_degradation < obs[4] / 2:
        return 1  # Decrease rebuild rate to reduce latency violations
    elif progress < redundancy_left * 0.95:
        return 2  # Continue rebuilding at current rate
    else:
        return 0  # Stop rebuilding immediately to prevent overloading the device
```