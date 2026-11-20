"""Reference submission: the greedy rule, as a worked example of the interface.

Copy this file, replace the body of `policy`, keep the signature.
"""

import numpy as np

# Observation layout (12 float32 values, all normalised unless stated):
#  0 fg_read_iops        6 hazard_uncertainty
#  1 fg_write_iops       7 redundancy_left
#  2 fg_bw               8 progress
#  3 latency_now         9 work_remaining
#  4 latency_target     10 alloc_current
#  5 hazard             11 bw_available
#
# Return an index into ACTIONS_MBPS = [20, 50, 80, 120].


def policy(obs: np.ndarray, state: dict) -> int:
    """Pick the fastest allocation that the headroom appears to allow.

    A deliberately simple reference. `state` persists within one episode only.
    """
    bw_available = float(obs[11])
    for index, fraction in ((3, 0.75), (2, 0.50), (1, 0.30)):
        if bw_available > fraction:
            return index
    return 0
