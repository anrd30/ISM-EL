, obs[10]) # obs[10] is alloc_current
y_t = obs[3] # latency_now

# RLS equations
P = state['P']
theta = state['theta']

# Calculate gain
K = P @ x_t / (1.0 + x_t.T @ P @ x_t)
# Update weights
theta = theta + K * (y_t - x_t.T @ theta)
# Update covariance
P = (P - np.outer(K, x_t.T @ P))

state['theta'] = theta
state['P'] = P
```
Wait, RLS can blow up or become too rigid if `P` gets too small. We should add a forgetting factor `lambda = 0.99`.
```python
lam = 0.99
K = P @ x_t / (lam + x_t.T @ P @ x_t)
theta = theta + K * (y_t - x_t.T @ theta)
P = (P - np.outer(K, x_t.T @ P)) / lam
```
This will track the latency model dynamically!

Now, how do we choose the action?
For each action `a` in `[0, 1, 2, 3]`:
    `cand_alloc = est_alloc[a]`
    `x_cand = get_features(obs, cand_alloc)`