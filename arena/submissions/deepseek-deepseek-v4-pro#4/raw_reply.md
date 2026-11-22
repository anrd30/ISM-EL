```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    # Unpack observation (all values normalised to roughly [0,1])
    # 0: fg_read_iops, 1: fg_write_iops, 2: fg_bw, 3: latency_now, 4: latency_target,
    # 5: hazard, 6: hazard_uncertainty, 7: redundancy_left, 8: progress, 9: work_remaining,
    # 10: alloc_current, 11: bw_available
    lat_now = obs[3]
    lat_target = obs[4]
    hazard = obs[5]
    progress = obs[8]
    bw_avail = obs[11]

    # 1. Never deliberately cause a latency violation – they cost 10 units.
    if lat_now > lat_target:
        return 0

    # 2. Sprint to the finish if we are very close, to avoid accumulating degraded-state cost.
    # Only do it when there is enough headroom to keep latency under control.
    if progress > 0.95 and bw_avail > 0.5:
        return 3
    if progress > 0.90 and bw_avail > 0.6:
        return 2

    # 3. Otherwise, choose a rebuild rate that respects headroom and current latency headroom.
    margin = (lat_target - lat_now) / lat_target if lat_target > 0 else 0.0

    # Base aggressiveness: we use a higher fraction of `bw_avail` when latency is well below target.
    base_factor = 0.5 + 0.5 * margin   # in [0.5, 1.0] when latency <= target

    # If the hazard (failure‑rate) is high, we may want to finish earlier, so we push a little harder.
    hazard_bonus = 0.3 * hazard         # up to 0.3 extra factor
    factor = min(base_factor + hazard_bonus, 1.0)

    agg = bw_avail * factor             # effective headroom utilisation
    # Clip: agg is already in [0, bw_avail] ⊆ [0,1]
    if agg > 0.7:
        return 3
    elif agg > 0.4:
        return 2
    elif agg > 0.15:
        return 1
    else:
        return 0
```