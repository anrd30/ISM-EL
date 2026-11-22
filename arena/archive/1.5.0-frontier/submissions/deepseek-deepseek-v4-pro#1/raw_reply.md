```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    A deterministic controller that avoids latency violations while trying to
    finish the rebuild as quickly as possible.

    The observation vector is 12-dimensional, normalised to roughly [0,1].
    See spec for field indices.
    """
    # Unpack the relevant fields
    latency_now      = obs[3]
    latency_target  = obs[4]
    hazard          = obs[5]
    alloc_current   = obs[10]
    bw_available    = obs[11]

    margin = latency_target - latency_now

    # If a violation is imminent or already present, drop to the minimum rate.
    if margin <= 0.05:
        return 0

    # Determine the desired utilisation fraction of bandwidth.
    # Conservative choices when margin is small.
    if margin < 0.2:
        target_frac = bw_available * 0.5
    else:
        target_frac = min(bw_available * 0.9, 1.0)

    # If the estimated failure hazard is high, nudge the target up slightly,
    # but never beyond a safe fraction of the headroom.
    if hazard > 0.5:
        target_frac = min(target_frac + 0.1, 1.0)
        target_frac = min(target_frac, bw_available * 0.95)

    # Discretise the continuous target to one of the four rebuild-rate levels.
    # The levels correspond to [20, 50, 80, 120] MB/s, normalised to [0,1].
    levels = np.array([0.1667, 0.4167, 0.6667, 1.0], dtype=np.float64)

    # What level is the current allocation closest to?
    cur_level = int(np.argmin(np.abs(levels - alloc_current)))
    # What level does our target correspond to?
    des_level = int(np.argmin(np.abs(levels - target_frac)))

    # Change at most one step per interval to avoid aggressive jumps that
    # could create a latency spike.
    if des_level > cur_level:
        action = min(cur_level + 1, 3)
    elif des_level < cur_level:
        action = max(cur_level - 1, 0)
    else:
        action = cur_level

    return action
```