```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    # Initialize the scaling factor M that maps action ratios to the normalized alloc_current
    if 'M' not in state:
        state['M'] = 1.0 / 120.0
    
    alloc_current = max(0.0, float(obs[10]))
    ratios = np.array([20.0, 50.0, 80.0, 120.0])
    
    # Update M based on the current applied rate to learn the exact normalized scale
    if alloc_current > 1e-4:
        Ms = alloc_current / ratios
        # Pick the M that is closest to our running estimate to handle safety filter overrides
        best_M = Ms[np.argmin(np.abs(Ms - state['M']))]
        state['M'] = 0.1 * best_M + 0.9 * state['M']
        
        # Clamp M to reasonable bounds to ensure stability
        if state['M'] < 1e-4:
            state['M'] = 1e-4
        if state['M'] > 0.1:
            state['M'] = 0.1
            
    # Estimated normalized rates for each of the 4 actions
    rates = ratios * state['M']
    
    # Maintain a short history of bw_available to protect against sudden bandwidth drops
    bw_available = float(obs[11])
    bw_hist = state.get('bw_hist', [])
    bw_hist.append(bw_available)
    if len(bw_hist) > 2:
        bw_hist.pop(0)
    state['bw_hist'] = bw_hist
    
    # Use the minimum recently observed bandwidth as a safe prediction for the next interval
    safe_bw = min(bw_hist)
    
    # Estimate queueing sensitivity C from current latency and bandwidth headroom
    # Queueing model: latency = C / (capacity - load) = C / (bw_available - alloc_current)
    latency_now = max(0.0, float(obs[3]))
    den = bw_available - alloc_current
    if den < 0.01:
        den = 0.01
    C = latency_now * den
    
    latency_target = float(obs[4])
    # Leave a 10% safety margin to strictly avoid the massive 10.0 violation penalty
    margin_target = latency_target * 0.9
    
    objs = []
    for a in range(4):
        pred_den = safe_bw - rates[a]
        
        # If the rate exceeds or is too close to available bandwidth, it is fundamentally unsafe
        if pred_den < 0.01:
            objs.append(float('inf'))
            continue
            
        # Predict latency for this action using our queueing model
        pred_lat = C / pred_den
        
        # Disqualify actions that risk violating the latency target
        if pred_lat > margin_target:
            objs.append(float('inf'))
        else:
            # Objective: minimize total remaining cost. 
            # Cost per interval is approximately (latency_degradation + 0.1).
            # Time to finish is proportional to 1 / rate.
            # Thus, we want to minimize (pred_lat + 0.1) / rate.
            objs.append((pred_lat + 0.1) / rates[a])
            
    # If all actions are deemed unsafe (e.g. heavy overload), fallback to the minimum rate (0)
    if all(obj == float('inf') for obj in objs):
        return 0
    else:
        return int(np.argmin(objs))
```