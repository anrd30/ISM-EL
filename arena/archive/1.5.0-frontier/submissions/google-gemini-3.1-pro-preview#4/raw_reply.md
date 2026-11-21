6$.
New total cost $\propto 0.5 / 100 = 0.005$.
So it's better to double the speed!
What if latency goes to 0.8? cost_rate = 0.9.
New total cost $\propto 0.9 / 100 = 0.009$. Worse!
So we should minimize `(pred_lat + 0.1) / actions_mbps[a]`.

Let's incorporate this optimization!
```python
best_a = 0
best_obj = float('inf')

for a in range(4):
    alloc_a = alloc_preds[a]
    
    if alloc_a >= bw_available * 0.9:
        continue
        
    pred_lat = c / max(1e-4, bw_available - alloc_a)
    
    if pred_lat > latency_target * 0.9:
        continue
        
    obj = (pred_lat + 0.1) / actions_mbps[a]
    if obj < best_obj:
        best_obj = obj
        best_a = a
```
This is brilliant because it balances the degradation cost and the rebuild speed!
Wait, is `latency_degradation` exactly `pred_lat`?
"latency_degradation rises as the device gets busier."
It might be `pred_lat - base_