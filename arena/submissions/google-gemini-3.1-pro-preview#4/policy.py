import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    # Extract and sanitize observations
    latency_now = max(0.0, float(obs[3]))
    latency_target = max(1e-5, float(obs[4]))
    alloc_current = max(0.0, float(obs[10]))
    bw_available = max(0.0, float(obs[11]))
    
    # Action bandwidth ratios
    mbps = np.array([20, 50, 80, 120], dtype=np.float32)
    
    # 1. Dynamically learn the normalization scalar 'U' mapping mbps to alloc_current
    if 'U' not in state:
        state['U'] = None
        
    if alloc_current > 1e-4:
        if state['U'] is None:
            # First observation: assume the applied action was the one we requested
            requested = state.get('last_action', 0)
            state['U'] = alloc_current / float(mbps[requested])
        else:
            # Find the action that best explains the current allocation and update U
            best_a = int(np.argmin(np.abs(alloc_current / mbps - state['U'])))
            state['U'] = 0.5 * state['U'] + 0.5 * (alloc_current / float(mbps[best_a]))
            
    # Fallback scalar if we haven't observed any non-zero allocation yet
    U_est = state['U'] if state['U'] is not None else 0.01
    est_bw = mbps * U_est
    
    # 2. Track bandwidth history to conservatively predict future available bandwidth
    if 'bw_hist' not in state:
        state['bw_hist'] = []
    state['bw_hist'].append(bw_available)
    if len(state['bw_hist']) > 5:
        state['bw_hist'].pop(0)
        
    if len(state['bw_hist']) > 1:
        hist = np.array(state['bw_hist'])
        drops = hist[:-1] - hist[1:]
        recent_max_drop = float(np.max(drops))
    else:
        recent_max_drop = 0.0
        
    # Assume bandwidth might drop by the recent maximum drop (at least 2% margin)
    margin = max(0.02, recent_max_drop)
    pred_bw = bw_available - margin
    pred_bw = max(0.01, pred_bw)
    
    # 3. Estimate queueing delay constant C based on the current state
    # Model: latency = C / headroom
    headroom_now = bw_available - alloc_current
    if headroom_now < 0.01:
        headroom_now = 0.01
        
    C = latency_now * headroom_now
    
    # 4. Evaluate each action to minimize total expected cost
    best_action = 0
    min_cost = float('inf')
    
    for a in range(4):
        bw_a = float(est_bw[a])
        headroom_next = pred_bw - bw_a
        
        # Hard safety constraint: leave at least 5% normalized headroom
        if headroom_next <= 0.05:
            pred_lat = float('inf')
        else:
            pred_lat = C / headroom_next
            
        # Heavy penalty constraint: avoid anything coming within 10% of the latency target
        if pred_lat > latency_target * 0.9:
            total_cost = float('inf')
        else:
            # Step cost integrates latency degradation and degraded state penalty (0.1)
            step_cost = pred_lat + 0.1
            # Total cost to finish is proportional to step_cost / bandwidth
            total_cost = step_cost / bw_a
            
        if total_cost < min_cost:
            min_cost = total_cost
            best_action = a
            
    # If all actions are predicted to violate the latency target, fallback to the safest action (0)
    if min_cost == float('inf'):
        best_action = 0
        
    state['last_action'] = best_action
    return int(best_action)
