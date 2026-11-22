```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Return 0, 1,2,3  -- the rebuild rate index.
    Observations are normalised to ~[0,1]; see spec for field layout.
    """
    # --- unpack normalised observations ----------------------------------
    lat_now       = obs[3]
    lat_target   = obs[4]
    progress     = obs[8]
    work_rem     = obs[9]
    alloc        = obs[10]   # currently applied rate (normalised)
    bw_avail     = obs[11]   # headroom

    # --- map alloc_current back to an action index -----------------------
    # The four actions [20, 50, 80, 120] Mbps, normalised by max=120.
    action_levels = np.array([20.0/120.0, 50.0/120.0, 80.0/120.0, 1.0])
    cur_idx = int(np.argmin(np.abs(action_levels - alloc)))

    # --- immediate fallback on violation --------------------------------
    if lat_now > lat_target:
        # A latency violation costs 10.0, far outweighing 0.1 per step
        # for degraded state.  Drop to the lowest rate immediately.
        return 0

    # --- near completion: ease off to avoid a final violation -----------
    if progress > 0.99 or work_rem < 0.01:
        return 0

    # --- normal operation: stay safely below target --------------------
    # Our desired operating point: 90 % of the target.
    setpoint = 0.9 * lat_target
    margin = setpoint - lat_now          # >0  means we are below setpoint

    # headroom threshold: we only increase if enough free capacity exists.
    ROOMY = 0.3
    TIGHT = 0.1

    if margin > 0.05 * lat_target:          # safely below setpoint
        if bw_avail > ROOMY:
            new_idx = min(3, cur_idx + 1)
        else:
            new_idx = cur_idx                     # headroom too low – hold
    elif margin >= 0:                            # between setpoint and target
        if bw_avail < TIGHT:
            new_idx = max(0, cur_idx - 1)         # reduce to avoid crossing target
        else:
            new_idx = cur_idx
    else:                                          # above setpoint, close to target
        new_idx = max(0, cur_idx - 1)              # reduce to stay safe

    # --- store state for optional future use (currently no hysteresis) --
    state['last_idx'] = new_idx
    return new_idx
```