import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    # Initialize state variables on the first step of an episode
    if 'action' not in state:
        state['action'] = 0
    if 'buffer' not in state:
        state['buffer'] = 0.1  # Initial safety margin for bandwidth headroom

    latency_now = obs[3]
    latency_target = obs[4]
    alloc_current = obs[10]
    bw_available = obs[11]

    # 1. Adjust the safety buffer based on latency feedback (Integral Control)
    # If latency is too high, we increase the buffer to force more conservative allocations.
    # If latency is safe, we slowly decrease the buffer to allow faster rebuilding.
    if latency_now > latency_target:
        state['buffer'] += 0.15
    elif latency_now >= 0.9 * latency_target:
        state['buffer'] += 0.05
    elif latency_now >= 0.8 * latency_target:
        state['buffer'] += 0.02
    elif latency_now < 0.7 * latency_target:
        state['buffer'] -= 0.01

    # Keep the buffer within reasonable normalized bounds
    state['buffer'] = max(0.0, min(state['buffer'], 1.0))

    # 2. Calculate remaining unused bandwidth
    # bw_available is total background capacity (capacity - foreground).
    # alloc_current is the background bandwidth currently being used.
    net_headroom = bw_available - alloc_current

    # 3. Determine the next action
    # We prioritize safety first (latency violations are heavily penalized).
    if latency_now > latency_target:
        # Immediate violation: slam the brakes to the minimum action
        next_a = 0
    elif latency_now >= 0.95 * latency_target:
        # Dangerously close to violation: step down
        next_a = max(0, state['action'] - 1)
    elif net_headroom < -0.1:
        # Severely over capacity (e.g., sudden foreground load spike): slam the brakes
        next_a = 0
    elif net_headroom < state['buffer'] - 0.05:
        # Not enough headroom to satisfy our safety buffer: step down
        next_a = max(0, state['action'] - 1)
    elif net_headroom > state['buffer']:
        # Plenty of headroom available: step up to rebuild faster
        next_a = min(3, state['action'] + 1)
    else:
        # Within the hysteresis band: maintain current action
        next_a = state['action']
            
    # Save the chosen action for the next interval
    state['action'] = next_a
    
    return int(next_a)
