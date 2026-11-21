import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    A deterministic rule‑based policy that balances rebuild speed against latency
    violations.  It uses a smoothed latency margin memory and the available
    headroom from the observation to decide the rebuild rate index (0‑3).
    """

    # ------ constants, chosen empirically --------------------------------
    ALPHA_EMA = 0.3          # smoothing factor for latency margin
    MARGIN_LOW  = 0.02       # below this margin we become more cautious
    MARGIN_HIGH = 0.1        # above this margin we consider speeding up
    BW_HIGH   = 0.5           # large headroom threshold
    BW_MED    = 0.2           # moderate headroom threshold

    # ------ unpack observation --------------------------------
    fg_read_iops  = float(obs[0])
    fg_write_iops = float(obs[1])
    fg_bw         = float(obs[2])
    latency_now   = float(obs[3])
    latency_target= float(obs[4])
    hazard        = float(obs[5])
    hazard_uncert = float(obs[6])
    redundancy    = float(obs[7])
    progress      = float(obs[8])
    work_left     = float(obs[9])
    alloc_current = float(obs[10])
    bw_available  = float(obs[11])

    # ------ state initialisation on first call --------------------
    if "action" not in state:
        # Choose a safe initial action based on current conditions
        if latency_now > latency_target:
            state["action"] = 0   # already violating -> minimal rate
        elif bw_available > BW_HIGH:
            state["action"] = 2   # plenty of headroom -> moderately fast
        else:
            state["action"] = 1   # default moderate
        state["ema_margin"] = latency_target - latency_now
        return state["action"]

    # ------ memory update ------------------------------------------
    last_action = state["action"]
    error_raw = latency_target - latency_now                # positive = safe

    # Exponential moving average of the margin
    ema_margin = state.get("ema_margin", 0.0)
    ema_margin = ALPHA_EMA * error_raw + (1.0 - ALPHA_EMA) * ema_margin
    state["ema_margin"] = ema_margin

    # ------ decision logic ------------------------------------------
    # If latency is currently above target, immediately go to minimum rate
    if latency_now > latency_target:
        new_action = 0

    # If margin is critically low, reduce speed (but not below 0)
    elif ema_margin < MARGIN_LOW:
        new_action = max(0, last_action - 1)

    # Safe margin and substantial headroom → speed up (up to max 3)
    elif ema_margin > MARGIN_HIGH and bw_available > BW_HIGH:
        new_action = min(3, last_action + 1)

    # Moderately safe with some headroom and not just started → one step faster
    elif (ema_margin > 0.05 and bw_available > BW_MED
          and progress > 0.15):
        new_action = min(3, last_action + 1)

    # Otherwise hold the current rate
    else:
        new_action = last_action

    # Store and return
    state["action"] = new_action
    return new_action
