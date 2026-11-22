import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    # Read and clamp observations
    latency_now = float(obs[3])
    latency_target = float(obs[4])
    hazard = float(obs[5])
    hazard_unc = float(obs[6])
    redundancy_left = float(obs[7])
    progress = float(obs[8])
    work_remaining = float(obs[9])
    alloc_current = float(obs[10])
    bw_available = float(obs[11])
    
    eps = 1e-6
    latency_now = max(0.0, latency_now)
    latency_target = max(eps, latency_target)
    hazard = max(0.0, min(1.0, hazard))
    hazard_unc = max(0.0, min(1.0, hazard_unc))
    redundancy_left = max(0.0, min(1.0, redundancy_left))
    work_remaining = max(0.0, min(1.0, work_remaining))
    alloc_current = max(0.0, min(1.0, alloc_current))
    bw_available = max(0.0, min(1.0, bw_available))
    
    r = latency_now / latency_target
    if not np.isfinite(r):
        r = 1.0
    r = max(0.01, min(r, 10.0))
    
    # Urgency score
    urg = 0.0
    if hazard > 0.5:
        urg += (hazard - 0.5) * 2.0
    if hazard_unc > 0.5:
        urg += (hazard_unc - 0.5) * 0.4
    if redundancy_left < 0.3:
        urg += (0.3 - redundancy_left) * 2.0
    if work_remaining < 0.2:
        urg += (0.2 - work_remaining) * 2.0
    urg = max(0.0, min(1.0, urg))
    
    # Desired latency ratio (fraction of target)
    r_target = 0.80 + 0.15 * urg
    if work_remaining < 0.05:
        r_target = max(r_target, 0.90)
    if r > 1.0:
        r_target = min(r_target, 0.85)
    
    ratio = r / max(r_target, 0.1)
    if ratio > 5.0:
        ratio = 5.0
    
    # Estimate safe rebuild rate fraction using a queueing model
    util_current = (1.0 - bw_available) + alloc_current
    if util_current >= 0.99:
        safe_alloc = bw_available - 0.01 * ratio
    else:
        safe_alloc = alloc_current + (bw_available - alloc_current) * (1.0 - ratio)
    
    if safe_alloc < 0.0:
        safe_alloc = 0.0
    if safe_alloc > 1.0:
        safe_alloc = 1.0
    
    # Initialize or retrieve last action
    last = state.get('last_action', None)
    if last is None:
        if alloc_current > 0.833:
            last = 3
        elif alloc_current > 0.542:
            last = 2
        elif alloc_current > 0.292:
            last = 1
        else:
            last = 0
        state['last_action'] = last
    
    # Adjust action towards safe_alloc
    diff = safe_alloc - alloc_current
    if diff > 0.12 and last < 3 and r < 0.95:
        last += 1
    elif diff < -0.12 and last > 0:
        last -= 1
    
    # Urgency boost, but only if there is latency headroom
    if urg > 0.6 and r < 0.90 and last < 2:
        last += 1
    elif urg > 0.3 and r < 0.85 and last < 1:
        last += 1
    
    # Safety caps
    if r > 1.2:
        last = 0
    elif r > 1.0:
        last = max(0, last - 1)
    elif r > 0.95:
        last = min(last, 1)
    
    if bw_available < 0.05:
        last = 0
    elif bw_available < 0.15:
        last = min(last, 1)
    elif bw_available < 0.30:
        last = min(last, 2)
    
    # Near completion: push to finish if safe
    if work_remaining < 0.02 and r < 0.95 and bw_available > 0.2:
        last = 3
    
    # Final clamp
    if last < 0:
        last = 0
    if last > 3:
        last = 3
    state['last_action'] = last
    return int(last)
