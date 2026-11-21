import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Return 0, 1, 2 or 3 based on the given observation and state.

    The observation is a 12-element float32 vector with the following fields:
        - fg_read_iops (float): foreground read load
        - fg_write_iops (float): foreground write load
        - fg_bw (float): foreground bandwidth share
        - latency_now (float): current latency
        - latency_target (float): the limit; a violation is `latency_now > latency_target`
        - hazard (float): failure-rate estimate
        - hazard_uncertainty (float): uncertainty on that estimate
        - redundancy_left (float): remaining redundancy
        - progress (float): fraction of the rebuild done
        - work_remaining (float): rebuild work left
        - alloc_current (float): the rate currently applied
        - bw_available (float): headroom: capacity minus foreground load

    The state is a dict that can be used to remember things within one episode.
    It is cleared at the start of every episode.

    The function should return an integer in [0, 3].
    """
    # Example policy logic:
    # - If latency_now > latency_target, increase alloc_current by 10%
    # - If redundancy_left < 0.5, decrease alloc_current by 5%
    # - Otherwise, keep alloc_current as is

    current_latency = obs[4]  # latency_now
    target_latency = obs[5]  # latency_target
    redundancy = obs[7]  # redundancy_left
    alloc_current = obs[10]  # alloc_current

    if current_latency > target_latency:
        alloc_current += alloc_current * 0.1  # Increase by 10%

    if redundancy < 0.5:
        alloc_current -= alloc_current * 0.05  # Decrease by 5%

    return np.argmax([alloc_current, 80, 120, 50])
