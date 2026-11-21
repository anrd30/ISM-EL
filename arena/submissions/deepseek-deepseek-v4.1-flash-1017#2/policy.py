import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    # Extract observation fields
    fg_bw = float(obs[2])
    latency = float(obs[3])
    target = float(obs[4])
    hazard = float(obs[5])
    hazard_unc = float(obs[6])
    redundancy = float(obs[7])
    progress = float(obs[8])
    work_rem = max(0.0, float(obs[9]))
    alloc_cur = float(obs[10])
    bw_avail = float(obs[11])

    # Initialize state on first call
    if 'step' not in state:
        state['step'] = 0
        state['w'] = np.array([0.0, 0.5, 0.5], dtype=np.float64)
        state['alloc_est'] = [0.1, 0.4, 0.7, 1.0]
        state['alloc_seen'] = [False, False, False, False]
        state['last_action'] = 1
        state['prev_progress'] = progress
        state['k_est'] = None
        state['last_latency'] = latency
        state['last_fg'] = fg_bw
        state['prev_alloc'] = alloc_cur if alloc_cur > 0 else 0.1

    step = state['step']

    # First interval: pick a reasonable initial action based on current latency
    if step == 0:
        if target <= 1e-9:
            chosen = 1
        elif latency < target * 0.6:
            chosen = 3
        elif latency < target * 0.8:
            chosen = 2
        elif latency < target * 0.95:
            chosen = 1
        else:
            chosen = 0
        state['last_action'] = chosen
        state['step'] = 1
        state['prev_progress'] = progress
        state['last_latency'] = latency
        state['last_fg'] = fg_bw
        state['prev_alloc'] = alloc_cur
        return int(chosen)

    # Update model using the action that produced the current observation
    la = state['last_action']
    if alloc_cur > 1e-6:
        if not state['alloc_seen'][la]:
            state['alloc_est'][la] = alloc_cur
            state['alloc_seen'][la] = True
        else:
            if alloc_cur < state['alloc_est'][la]:
                state['alloc_est'][la] = alloc_cur

    w = state['w']
    x = np.array([1.0, fg_bw, alloc_cur], dtype=np.float64)
    pred = float(np.dot(w, x))
    err = latency - pred
    lr = 0.02
    w += lr * err * x
    # Keep weights in sensible ranges
    w[0] = np.clip(w[0], -1.0, 1.0)
    w[1] = np.clip(w[1], 0.0, 2.0)
    w[2] = np.clip(w[2], 0.0, 2.0)
    state['w'] = w

    # Estimate progress per unit allocation per interval
    dp = progress - state['prev_progress']
    if alloc_cur > 1e-6 and dp > 0:
        k_sample = dp / alloc_cur
        if state['k_est'] is None:
            state['k_est'] = k_sample
        else:
            state['k_est'] = 0.8 * state['k_est'] + 0.2 * k_sample
    state['prev_progress'] = progress

    # If rebuild is essentially done, choose the lowest action
    if work_rem <= 1e-5 or progress >= 0.9999:
        chosen = 0
        state['last_action'] = chosen
        state['step'] = step + 1
        state['last_latency'] = latency
        state['last_fg'] = fg_bw
        state['prev_alloc'] = alloc_cur
        return int(chosen)

    # Predict latency for each candidate action
    w = state['w']
    lat_preds = []
    for a in range(4):
        alloc = state['alloc_est'][a]
        pred_lat = w[0] + w[1] * fg_bw + w[2] * alloc
        lat_preds.append(max(0.0, pred_lat))

    # Conservative cap if current latency already violates target
    max_action = 3
    if latency > target:
        last_a = state['last_action']
        if latency > target * 1.1:
            max_action = 0
        else:
            max_action = max(0, last_a - 1)

    # Additional conservatism when headroom is very low
    if bw_avail < 0.1:
        max_action = min(max_action, 1)
    elif bw_avail < 0.2:
        max_action = min(max_action, 2)

    # Safety margin grows with hazard uncertainty
    margin = 0.02 + 0.1 * hazard_unc
    valid = []
    for a in range(4):
        if a > max_action:
            continue
        if lat_preds[a] <= target - margin:
            valid.append(a)

    if not valid:
        # No safe action; pick the one with lowest predicted latency within cap
        best_a = min(
            range(4),
            key=lambda a: lat_preds[a] if a <= max_action else 1e9
        )
        chosen = best_a
    else:
        # Evaluate approximate cost-to-go for each safe action
        best_cost = float('inf')
        best_a = valid[0]
        k_est = state['k_est']
        base_lat = w[0] + w[1] * fg_bw

        for a in valid:
            alloc = state['alloc_est'][a]
            lat_pred = lat_preds[a]
            # Latency degradation is the part caused by rebuild
            degradation = max(0.0, lat_pred - base_lat)
            interval_cost = degradation + 0.1

            # Estimate remaining intervals at this rebuild rate
            if k_est is not None and k_est > 1e-9 and alloc > 1e-9:
                T = work_rem / (alloc * k_est)
            else:
                # Fallback scale; only relative comparisons matter
                T = work_rem / (alloc * 0.01 + 1e-9)
            n = max(1.0, T)

            cost = n * interval_cost
            cost += 1e-6 * a  # tie-breaker: prefer lower action
            if cost < best_cost:
                best_cost = cost
                best_a = a
        chosen = best_a

    chosen = int(max(0, min(3, chosen)))

    # Persist state for next call
    state['last_action'] = chosen
    state['step'] = step + 1
    state['last_latency'] = latency
    state['last_fg'] = fg_bw
    state['prev_alloc'] = alloc_cur

    return chosen
