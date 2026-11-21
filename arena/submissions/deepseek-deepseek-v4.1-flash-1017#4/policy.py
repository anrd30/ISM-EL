import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    if 'n' not in state:
        state['n'] = 0
        state['theta'] = np.array([0.1, 1.0, 0.5, 0.0, 0.0], dtype=np.float64)
        state['P'] = np.eye(5, dtype=np.float64) * 1.0
        state['action_alloc'] = np.array([0.167, 0.417, 0.667, 1.0], dtype=np.float64)
        state['last_action'] = -1

    fg_bw = float(obs[2])
    latency_now = float(obs[3])
    latency_target = float(obs[4])
    hazard = float(obs[5])
    redundancy_left = float(obs[7])
    progress = float(obs[8])
    work_remaining = float(obs[9])
    alloc_current = float(obs[10])

    latency_ratio = latency_now / max(latency_target, 1e-6)

    # Update action-to-allocation estimates when the safety filter likely did not raise the action.
    if state['n'] > 0 and state['last_action'] != -1:
        a = state['last_action']
        prev_est = state['action_alloc'][a]
        if alloc_current <= prev_est + 0.05:
            state['action_alloc'][a] = 0.8 * prev_est + 0.2 * alloc_current
        for k in range(1, 4):
            if state['action_alloc'][k] < state['action_alloc'][k - 1]:
                state['action_alloc'][k] = state['action_alloc'][k - 1]

    # Recursive least-squares update for latency_ratio as a function of allocation and load.
    x = np.array(
        [1.0, alloc_current, fg_bw, alloc_current * alloc_current, alloc_current * fg_bw],
        dtype=np.float64,
    )
    y = latency_ratio
    theta = state['theta']
    P = state['P']

    Px = P.dot(x)
    denom = 0.99 + np.dot(x, Px)
    K = Px / denom
    theta = theta + K * (y - np.dot(x, theta))
    theta = np.clip(theta, -10.0, 10.0)
    P = (P - np.outer(K, np.dot(x, P))) / 0.99
    P = np.clip(P, -100.0, 100.0)
    P = (P + P.T) / 2.0

    state['theta'] = theta
    state['P'] = P
    state['n'] += 1
    n = state['n']

    # Predict latency ratio for each action.
    action_alloc = state['action_alloc']
    preds = np.zeros(4, dtype=np.float64)
    for a in range(4):
        est = action_alloc[a]
        xa = np.array(
            [1.0, est, fg_bw, est * est, est * fg_bw],
            dtype=np.float64,
        )
        preds[a] = xa.dot(theta)
    preds = np.clip(preds, 0.0, 5.0)

    # Safety margin decreases as we gather more data.
    margin = max(0.02, 0.1 / np.sqrt(n))

    # Allow more latency when much rebuild work remains; be conservative near the end.
    if work_remaining > 0.6:
        target = 0.98
    elif work_remaining > 0.3:
        target = 0.96
    elif work_remaining > 0.1:
        target = 0.93
    else:
        target = 0.90

    # If hazard is high and redundancy is low, accept slightly more latency to finish sooner.
    if hazard > 0.6 and redundancy_left < 0.4:
        target += 0.02

    target_safe = target - margin

    # Heuristic fallback for the first few intervals.
    if n < 3:
        if latency_ratio > 1.0:
            action = 0
        elif latency_ratio > 0.95:
            action = 1
        elif latency_ratio > 0.85:
            action = 2
        else:
            action = 3
    else:
        action = 0
        for a in range(3, -1, -1):
            if preds[a] <= target_safe:
                action = a
                break

        if latency_ratio > 1.0:
            action = 0
        if latency_ratio > 0.98 and action > 1:
            action = 1

    state['last_action'] = action
    return int(action)
