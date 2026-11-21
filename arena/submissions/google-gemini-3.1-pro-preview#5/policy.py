import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    # Extract observation fields
    fg_read = obs[0]
    fg_write = obs[1]
    fg_bw = obs[2]
    lat_now = obs[3]
    lat_target = obs[4]
    alloc_curr = obs[10]
    bw_avail = obs[11]
    
    # Protect against division by zero
    safe_bw_avail = max(0.01, float(bw_avail))
    ACTIONS_MBPS = np.array([20.0, 50.0, 80.0, 120.0])
    
    # Initialize state variables on the first step of the episode
    if 'history_X' not in state:
        state['history_X'] = []
        state['history_y'] = []
        state['scale'] = 1.0 / 120.0
        state['min_ratio'] = float('inf')
        state['XtX'] = np.zeros((6, 6))
        state['Xty'] = np.zeros(6)
        
    # Dynamically estimate the scaling factor from action MBps to normalized alloc_curr.
    # The safety filter only ever raises the action, so alloc_curr is always >= expected.
    # Therefore, the minimum ratio observed perfectly captures the true scale.
    if 'last_action' in state:
        expected_mbps = ACTIONS_MBPS[state['last_action']]
        ratio = float(alloc_curr) / expected_mbps
        if alloc_curr > 0.001 and ratio < state['min_ratio']:
            state['min_ratio'] = ratio
            state['scale'] = ratio
            
    scale = state['scale']
    
    # Current features for the latency model
    feat = np.array([
        fg_read,
        fg_write,
        fg_bw,
        alloc_curr,
        alloc_curr / safe_bw_avail
    ])
    
    feat_aug = np.array([
        fg_read,
        fg_write,
        fg_bw,
        alloc_curr,
        alloc_curr / safe_bw_avail,
        1.0
    ])
    
    # Update history and regression matrices
    state['history_X'].append(feat)
    state['history_y'].append(float(lat_now))
    state['XtX'] += np.outer(feat_aug, feat_aug)
    state['Xty'] += feat_aug * float(lat_now)
    
    X_mat = np.array(state['history_X'])
    y_vec = np.array(state['history_y'])
    
    # Fit a linear regression model to predict latency sensitivity
    if len(y_vec) >= 5:
        lam = 1.0  # Ridge regularization to ensure stability
        try:
            w = np.linalg.solve(state['XtX'] + lam * np.eye(6), state['Xty'])
            w_alloc_lin = max(float(w[3]), 0.05)
            w_alloc_nonlin = max(float(w[4]), 0.05)
        except np.linalg.LinAlgError:
            w_alloc_lin = 0.5
            w_alloc_nonlin = 0.5
            
        lat_std = float(np.std(y_vec[-10:]))
    else:
        w_alloc_lin = 0.5
        w_alloc_nonlin = 0.5
        lat_std = 0.05
        
    # Adaptive safety margin based on recent latency volatility
    margin = min(0.15, max(0.02, 1.5 * lat_std))
    safe_target = lat_target - margin
    
    # k-Nearest Neighbors to recall specific past failures
    def get_knn_max_latency(feat_cand, X_mat, y_vec, k=3, max_dist=0.2):
        if len(X_mat) == 0:
            return 0.0
        diff = X_mat[:, :4] - feat_cand[:4]
        # Weight allocation heavily to only match intervals with identical rebuild rates
        weights = np.array([1.0, 1.0, 1.0, 10.0])
        dist = np.sum(weights * (diff**2), axis=1)
        valid_idx = np.where(dist < max_dist)[0]
        if len(valid_idx) == 0:
            return 0.0
        k_actual = min(k, len(valid_idx))
        valid_dist = dist[valid_idx]
        idx_of_valid = np.argpartition(valid_dist, k_actual - 1)[:k_actual]
        idx = valid_idx[idx_of_valid]
        return float(np.max(y_vec[idx]))

    # Evaluate all possible actions and pick the highest safe one
    best_action = 0
    for a in range(4):
        cand_alloc = ACTIONS_MBPS[a] * scale
        delta_lin = cand_alloc - alloc_curr
        delta_nonlin = cand_alloc / safe_bw_avail - alloc_curr / safe_bw_avail
        
        # Taylor expansion around the current state to cancel out foreground load offset
        pred_lat_lin = lat_now + w_alloc_lin * delta_lin + w_alloc_nonlin * delta_nonlin
        
        feat_cand = np.array([fg_read, fg_write, fg_bw, cand_alloc, cand_alloc / safe_bw_avail])
        pred_lat_knn = get_knn_max_latency(feat_cand, X_mat, y_vec, k=3, max_dist=0.2)
        
        # Take the pessimistic prediction between the linear model and kNN memory
        if pred_lat_knn > 0:
            pred_lat = max(pred_lat_lin, pred_lat_knn)
        else:
            pred_lat = pred_lat_lin
            
        if pred_lat <= safe_target:
            best_action = a
            
    # Hard safety fallback: if we are currently violating the limit, drop to minimum allocation immediately
    if lat_now > lat_target:
        best_action = 0
        
    state['last_action'] = best_action
    return int(best_action)
