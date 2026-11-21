```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    # Read relevant observations
    latency = float(obs[3])
    target = float(obs[4])
    bw_avail = float(obs[11])
    
    # Initialize state
    if 'level' not in state:
        state['level'] = 1.0
        state['last_action'] = 1
        state['ema_margin'] = 0.12
    
    eps = 1e-6
    if target > eps:
        margin = (target - latency) / target
    else:
        margin = target - latency
    
    # Smooth margin to avoid reacting to single-interval noise
    alpha = 0.4
    state['ema_margin'] = alpha * margin + (1.0 - alpha) * state['ema_margin']
    ema_m = state['ema_margin']
    
    # Immediate safety: if violating or extremely close, drop to zero
    if latency > target or margin < 0.02:
        state['level'] = 0.0
        state['last_action'] = 0
        return 0
    
    # Quick reduction if close to limit
    if margin < 0.06:
        state['level'] = min(state['level'], 1.0) - 0.3
        if state['level'] < 0.0:
            state['level'] = 0.0
        action = min(state['last_action'], 1)
        # Apply headroom cap
        if bw_avail < 0.05:
            action = 0
        elif bw_avail < 0.15:
            action = min(action, 1)
        state['last_action'] = action
        return int(action)
    
    # Normal proportional control
    target_m = 0.12
    error = ema_m - target_m
    if abs(error) > 0.02:
        delta = 2.5 * error
        if delta > 1.5:
            delta = 1.5
        elif delta < -1.5:
            delta = -1.5
        state['level'] += delta
    
    # Cap by available bandwidth headroom
    if bw_avail < 0.05:
        max_level = 0.0
    elif bw_avail < 0.15:
        max_level = 1.0
    elif bw_avail < 0.30:
        max_level = 2.0
    else:
        max_level = 3.0
    
    if state['level'] > max_level:
        state['level'] = max_level
    if state['level'] < 0.0:
        state['level'] = 0.0
    if state['level'] > 3.0:
        state['level'] = 3.0
    
    # Convert continuous level to discrete action with hysteresis
    last = state['last_action']
    if state['level'] > last + 0.6:
        action = last + 1
    elif state['level'] < last - 0.6:
        action = last - 1
    else:
        action = last
    
    # Apply headroom cap directly to action
    if action > max_level:
        action = int(max_level)
    
    # Final bounds check
    if action < 0:
        action = 0
    elif action > 3:
        action = 3
    
    # Final safety check
    if latency > target or margin < 0.02:
        action = 0
        state['level'] = 0.0
    
    state['last_action'] = action
    return int(action)
```