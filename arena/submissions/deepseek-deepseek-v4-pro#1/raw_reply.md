```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    lat = float(obs[3])
    targ = float(obs[4])
    headroom = float(obs[11])
    remaining = float(obs[9])      # rebuild work left
    hazard = float(obs[5])

    # ---------------------------------------------------------
    # 1. Immediate violation?  Hard stop at lowest rebuild rate.
    # ---------------------------------------------------------
    if lat >= targ:
        return 0

    # ---------------------------------------------------------
    # 2. Exponential moving average of latency for stability.
    # ---------------------------------------------------------
    if 's_lat' not in state:
        state['s_lat'] = lat
    state['s_lat'] = 0.8 * state['s_lat'] + 0.2 * lat
    smoothed = state['s_lat']

    # Avoid division by zero – target is never truly zero.
    margin = targ - smoothed
    if targ > 0.0:
        rel_margin = margin / targ
    else:
        rel_margin = 0.0

    # ---------------------------------------------------------
    # 3. Base action from available bandwidth headroom.
    #    Map [0,1] headroom → {0,1,2,3}.
    # ---------------------------------------------------------
    base_action = int(np.clip(headroom * 4.0, 0.0, 3.0))

    # ---------------------------------------------------------
    # 4. Cap aggressiveness according to how far we are from
    #    the latency target.
    # ---------------------------------------------------------
    if rel_margin < 0.05:
        action = 0
    elif rel_margin < 0.2:
        action = min(base_action, 1)
    elif rel_margin < 0.4:
        action = min(base_action, 2)
    else:
        action = base_action

    # ---------------------------------------------------------
    # 5. If the rebuild is almost done, take fewer risks.
    # ---------------------------------------------------------
    if remaining < 0.15:
        action = min(action, 2)      # never go to index 3 (120 MBps)

    # ---------------------------------------------------------
    # 6. High hazard – try to finish faster when it is safe.
    # ---------------------------------------------------------
    if hazard > 0.7 and rel_margin > 0.2:
        action = max(action, 2)

    return int(np.clip(action, 0, 3))
```