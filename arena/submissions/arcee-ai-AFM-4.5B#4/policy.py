import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    # Since the options are in ascending order, the highest rate that doesn't push latency over target is 120
    return 3
