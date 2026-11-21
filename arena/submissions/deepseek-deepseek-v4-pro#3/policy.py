import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Deterministic PI-like controller that balances rebuild speed and latency.

    Observation indices:
      0  fg_read_iops
      1  fg_write_iops
      2  fg_bw
      3  latency_now
      4  latency_target
      5  hazard
      6  hazard_uncertainty
      7  redundancy_left
      8  progress
      9  work_remaining
      10 alloc_current
      11 bw_available
    """
    # Extract relevant fields
    lat_now   = obs[3]
    lat_targ = obs[4]
    hazard    = obs[5]
    bw_avail  = obs[11]

    # Error: positive when latency is safely below target (room to increase rebuild)
    error = lat_targ - lat_now

    # ---- Integ ral term (stored in state) ---------------------------------
    if 'integral' not in state:
        state['integral'] = 0.0

    # Update with clamping to avoid windup
    state['integral'] += error
    state['integral'] = max(-2.0, min(2.0, state['integral']))

    # ---- PI control --------------------------------------------------------
    Kp = 2.5   # proportionl gain
    Ki = 0.2   # integral gain

    u = Kp * error + Ki * state['integral']

    # Hazard boost: if the risk of a second failure is high, speed up rebuild
    if hazard > 0.5:
        u += 1.0

    # Quantise to 0..3
    action = int(round(u))
    action = max(0, min(3, action))

    # ---- Capacity limit: never pick an action that would exceed available headroom ----
    # bw_available is normalised ~[0,1]; 3 is maximum action index
    max_action_by_bw = int(np.floor(bw_avail * 3))
    action = min(action, max_action_by_bw)

    return action
