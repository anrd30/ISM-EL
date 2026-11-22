```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    # Unpack observation
    fg_read = float(obs[0])
    fg_write = float(obs[1])
    fg_bw = float(obs[2])
    lat = float(obs[3])
    target = float(obs[4])
    hazard = float(obs[5])
    haz_unc = float(obs[6])
    red = float(obs[7])
    progress = float(obs[8])
    work = float(obs[9])
    alloc = float(obs[10])
    bw_avail = float(obs[11])

    eps = 1e-6
    if target <= eps:
        target = 1.0

    # Initialize state
    if 'init' not in state:
        state['init'] = True
        state['sens'] = 0.5
        state['step'] = 0.25
        state['min_rate'] = [None, None, None, None]
        state['last_alloc'] = alloc
        state['last_lat'] = lat
        state['last_fg'] = fg_bw
        state['last_progress'] = progress
        state['chosen_action'] = None
        state['prev_chosen_action'] = None

    ca = state.get('chosen_action')
    prev_ca = state.get('prev_chosen_action')

    # Update minimum observed rate for the action that produced current obs
    if ca is not None:
        mr = state['min_rate'][ca]
        if mr is None or alloc < mr:
            state['min_rate'][ca] = alloc

    # Update sensitivity and step using changes since last call
    last_alloc = state['last_alloc']
    last_lat = state['last_lat']
    last_fg = state['last_fg']
    last_prog = state['last_progress']

    delta_alloc = alloc - last_alloc
    delta_lat = lat - last_lat
    delta_fg = fg_bw - last_fg
    delta_prog = progress - last_prog

    # Update latency sensitivity d(latency)/d(alloc)
    if abs(delta_alloc) > 0.02 and abs(delta_fg) < 0.15:
        s = delta_lat / delta_alloc
        if 0.02 < s < 5.0:
            state['sens'] = 0.8 * state['sens'] + 0.2 * s

    # Update action step (alloc change per action index change)
    if ca is not None and prev_ca is not None:
        da = ca - prev_ca
        if abs(da) >= 1 and abs(delta_alloc) > 0.01:
            step_est = delta_alloc / da
            if 0.02 < step_est < 1.0:
                state['step'] = 0.9 * state['step'] + 0.1 * step_est

    # Save last values for next call
    state['last_alloc'] = alloc
    state['last_lat'] = lat
    state['last_fg'] = fg_bw
    state['last_progress'] = progress
    state['prev_chosen_action'] = ca

    # Current latency ratio
    ratio = lat / target

    # Determine target ratio for safety
    target_ratio = 0.80
    if hazard > 0.6 or haz_unc > 0.5 or red < 0.4:
        target_ratio -= 0.05
    if hazard > 0.8 or red < 0.2:
        target_ratio -= 0.05
    if bw_avail < 0.2:
        target_ratio -= 0.05
    elif bw_avail > 0.6:
        target_ratio += 0.05
    if work < 0.05 or progress > 0.95:
        target_ratio = min(target_ratio, 0.6)
    target_ratio = max(0.5, min(0.9, target_ratio))

    # Near completion: finish with minimal risk
    if work < 0.02 or progress > 0.98:
        target_ratio = 0.0

    # If already violating or very close, force low
    if ratio > 0.95:
        new_action = 0
    else:
        sens = state.get('sens', 0.5)
        if sens <= 0.01:
            sens = 0.5
        step = state.get('step', 0.25)
        if step <= 0.01:
            step = 0.25

        # Estimate nominal rates for each action
        min_rate = state['min_rate']
        rates = [None, None, None, None]
        for i in range(4):
            if min_rate[i] is not None:
                rates[i] = min_rate[i]

        known = [(i, rates[i]) for i in range(4) if rates[i] is not None]
        if len(known) == 0:
            base_action = ca if ca is not None else 1
            for i in range(4):
                rates[i] = max(0.0, min(1.0, alloc + (i - base_action) * step))
        elif len(known) == 1:
            i0, y0 = known[0]
            for i in range(4):
                if rates[i] is None:
                    rates[i] = max(0.0, min(1.0, y0 + (i - i0) * step))
        else:
            known.sort()
            for i in range(4):
                if rates[i] is None:
                    left = None
                    right = None
                    for j in range(i - 1, -1, -1):
                        if rates[j] is not None:
                            left = j
                            break
                    for j in range(i + 1, 4):
                        if rates[j] is not None:
                            right = j
                            break
                    if left is not None and right is not None:
                        frac = (i - left) / (right - left)
                        rates[i] = rates[left] + frac * (rates[right] - rates[left])
                    elif left is not None:
                        rates[i] = rates[left] + (i - left) * step
                    elif right is not None:
                        rates[i] = rates[right] - (right - i) * step
                    else:
                        rates[i] = alloc
                    rates[i] = max(0.0, min(1.0, rates[i]))

        # Safety filter risk: assume it may raise the applied rate
        filter_risk = 0.0
        if hazard > 0.8 or red < 0.2:
            filter_risk = 0.2
        elif hazard > 0.6 or red < 0.4:
            filter_risk = 0.1

        # Find safe actions
        safe = []
        for i in range(4):
            pred_alloc = rates[i] + filter_risk
            pred_lat = lat + sens * (pred_alloc - alloc)
            if pred_lat < 0.0:
                pred_lat = 0.0
            pred_ratio = pred_lat / target
            if pred_ratio <= target_ratio:
                safe.append(i)

        if not safe:
            new_action = 0
        else:
            new_action = max(safe)
            # Hysteresis: keep current action unless a higher one is clearly safe
            if ca is not None and ca in safe and new_action > ca:
                pred_alloc = rates[new_action] + filter_risk
                pred_lat = lat + sens * (pred_alloc - alloc)
                pred_ratio = pred_lat / target
                if pred_ratio > target_ratio - 0.05:
                    new_action = ca

    # Final safety overrides
    if ratio > 1.0:
        new_action = 0
    if ratio > 0.9 and ca is not None and new_action > ca:
        new_action = ca

    # First action: be conservative
    if ca is None:
        new_action = 1

    new_action = int(new_action)
    if new_action < 0:
        new_action = 0
    elif new_action > 3:
        new_action = 3

    state['chosen_action'] = new_action
    return new_action
```