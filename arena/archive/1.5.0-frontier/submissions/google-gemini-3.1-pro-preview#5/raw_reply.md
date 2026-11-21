`.

Let's refine the PID controller for `desired_alloc`:
```python
if 'desired_alloc' not in state:
    state['desired_alloc'] = 0.5

# Update desired_alloc based on latency
# We want to target a latency of, say, 0.8 * latency_target
target_L = latency_target * 0.8
error = target_L - latency_now

# Simple integral control
Ki = 0.5  # tune this
state['desired_alloc'] += Ki * error

# Cap desired_alloc based on bw_available
state['desired_alloc'] = min(state['desired_alloc'], bw_available * 0.9)
state['desired_alloc'] = max(state['desired_alloc'], 0.0)
```
Wait, `error` is in units of latency (normalized to [0, 1]).
`desired_alloc` is in units of alloc (normalized to [0, 1]).
If `latency_target` is around 0.5, and `latency_now` is 0.4, `error = 0.0`.
If `latency_now` is 0.2, `error = 0.2`. `desired_alloc` increases by 0.1.
This seems very reasonable!

Let's trace the PID:
If `latency_now` is low, `desired_alloc` will grow until it hits