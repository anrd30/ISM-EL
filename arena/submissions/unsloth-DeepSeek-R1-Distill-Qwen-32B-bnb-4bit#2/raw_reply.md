Okay, I need to write a Python function called policy that takes an observation array and a state dictionary, and returns an integer between 0 and 3 representing the rebuild rate action to take. The goal is to minimize the total cost, which is a combination of latency degradation, degraded state, and latency violation.

First, let's understand the problem. The system is rebuilding data onto a spare disk, and the rebuild rate affects how quickly it completes but also impacts user latency. The cost function penalizes for high latency degradation, being in a degraded state for longer, and especially for exceeding latency targets.

The observation vector has 12 elements, all normalized. I need to use these to decide the action.

So, the main factors are:
1. How much progress has been made (obs[8], progress).
2. Remaining redundancy (obs[7]).
3. Current latency (obs[3]) and target (obs[4]).
4. Available bandwidth (obs[11]).
5. Current allocation (obs[10]).

Hmm, the safety filter may increase the action, so I don't have to worry about the lower side as much as the upper.

Let me think about the cost components:
- Latency degradation: As the device gets busier, this increases. So higher bandwidth usage (higher rebuild rate) would increase this.
- Degraded state: 1 per interval until the rebuild is done. So we want to finish as quickly as possible.
- Latency violation: Heavily penalized if current latency exceeds target.

So the policy needs to balance between using enough bandwidth to finish quickly without causing latency violations.

Perhaps, if the current latency is already high (close to or over target), we should choose a lower rebuild rate. Conversely, if there's plenty of bandwidth available and latency isn't high, we can choose a higher rate.

Looking at the observation, bw_available is index 11. Maybe if bw_available is high, we can choose a higher action. Also, if progress is low, we might want to rebuild faster, but if progress is high, maybe we can slow down a bit.

Let me think about the possible actions: 0 is 20, 1 is 50, 2 is 80, 3 is 120. So higher indices mean higher rates.

So the approach could be:
- If the current latency is near or over the target, choose a lower action.
- If there's plenty of available bandwidth, choose a higher action.
- If progress is low, prioritize higher rates to finish quickly.
- If the rebuild is almost done, maybe lower the rate to minimize latency issues.

Let me think about the normalized values. Since they're in [0,1], I can compare them directly.

Possible steps:
1. Check if the current latency is above the target. If so, choose the lowest action.
2. If bw_available is high (close to 1), choose a higher action.
3. If progress is low, prioritize higher actions to finish quickly.
4. Also, consider the remaining work (work_remaining) and redundancy_left.

Wait, the redundancy_left is the remaining redundancy. Maybe if redundancy is low, we need to finish quickly, so higher actions.

But the safety filter may raise the action, so I don't have to worry about it being too low, but if I set it too high, the filter might not change it.

So the plan is:

- If current latency is above target (obs[3] > obs[4]), choose the lowest possible action (0 or 1).
- Else, if there's a lot of available bandwidth, choose a higher action.
- Also, if progress is low, prioritize higher actions.

Let me structure this as a series of conditions.

First, check if latency_now > latency_target. If so, set action to 0 (lowest rate) because we can't afford to have more latency.

Else, check the bw_available. If bw_available is high, choose higher action. But what's high? Maybe if bw_available is above a certain threshold, say 0.7, then choose 3.

If bw_available is moderate, say between 0.3 and 0.7, choose 2.

If bw_available is low, choose 1 or 0.

Wait, but also consider progress. If progress is low, we want to rebuild quickly, so choose higher action if possible.

So maybe:

if obs[3] > obs[4]:
    return 0
else:
    if progress is low (e.g., <0.5):
        if bw_available is high: choose 3
        else if bw_available is moderate: choose 2
        else: choose 1
    else:
        choose a lower action to not impact latency.

Alternatively, maybe use a weighted approach, but since the function needs to be deterministic and quick, perhaps a set of conditionals is better.

Another approach: calculate a priority based on the factors.

Alternatively, use a scoring system where each action is scored based on how it affects the cost.

But given the time constraints, perhaps a rule-based approach is better.

Let me outline the logic:

1. Check if current latency is above target. If yes, choose the lowest action (0 or 1). Maybe 0, but 1 is 50, which is higher than 20.

Wait, the actions are 20,50,80,120. So action 0 is 20, 1 is 50, etc.

If the system is already violating latency, we want to minimize the rebuild rate. So choose 0 or 1.

But 20 is the lowest. So perhaps if obs[3] > obs[4], choose 0.

But wait, the safety filter may raise the action. So if we choose 0, the filter might increase it, leading to higher latency, which is bad.

Wait, the filter never lowers the action. So if I choose a lower action, the filter might increase it, which could cause more latency. But if I set a higher action, the filter might leave it as is.

Hmm, this complicates things. So perhaps I need to be cautious about setting the action too high, because the filter could leave it as is, causing more latency.

So, perhaps when the system is near a violation, I should set a very low action, so even if the filter raises it, it doesn't go too high.

Alternatively, perhaps if the current action is causing a violation, we need to choose a lower action, even if the filter raises it.

But this is getting complicated.

Let me think differently. The cost function is latency_degradation + 0.1 * degraded_state + 10 * latency_violation. So the highest penalty is for latency_violation. So avoiding that is critical.

So the first priority is to prevent the latency from exceeding the target. So whenever possible, ensure that the action chosen doesn't cause the latency to go over.

Thus, the first condition is: if current latency is over target, choose the lowest possible action (0). But the filter may raise it, which could cause more latency. Hmm, perhaps that's a problem.

Wait, the filter may raise the action. So if I choose 0, and the filter raises it, the action becomes higher, which could cause more latency, potentially making it worse.

So perhaps when the current latency is already high, I should choose the lowest action possible, hoping that the filter doesn't raise it too much. Or perhaps even 0 is too much, but that's the lowest.

Alternatively, perhaps the filter only raises the action if the current allocation is too low. So if the current action is 0, the filter may raise it to 1 or 2 if it's needed for redundancy. But that's beyond my control.

So perhaps, when the latency is over the target, choose the lowest action (0), hoping that the filter doesn't make it worse.

Alternatively, perhaps if the current action is 0, but the filter raises it, that's worse. So perhaps in that case, the system is already in a bad state, and I can't do much except choose the lowest possible.

So perhaps the first condition is:

if obs[3] > obs[4]:
    return 0
else:
    proceed to other conditions.

Next, consider the available bandwidth. If bw_available is high, we can afford to use more bandwidth for rebuild.

So, if bw_available is high, choose higher action.

But what's high? Maybe if obs[11] is above 0.8, choose 3.

Else if above 0.6, choose 2.

Else if above 0.4, choose 1.

Else, choose 0.

But also, if progress is low, we want to rebuild quickly.

So perhaps, if progress is low, we can afford to take higher actions as long as bw_available is sufficient.

So, maybe:

if progress < 0.3:
    if bw_available > 0.8:
        return 3
    elif bw_available > 0.6:
        return 2
    elif bw_available > 0.4:
        return 1
    else:
        return 0
else:
    choose a lower action.

Wait, but what if progress is high? Then, maybe we can slow down the rebuild to reduce latency.

So, another condition:

if progress > 0.8:
    return 0
elif progress > 0.6:
    return 1
else:
    proceed as before.

Alternatively, perhaps:

if progress < 0.5:
    prioritize higher actions if bw_available is high.
else:
    lower actions.

Let me try to outline the logic:

1. If current latency > target: return 0.

2. Else if progress is high (e.g., >0.8): return 0.

3. Else if progress is between 0.5 and 0.8: return 1.

4. Else (progress <0.5):

   a. If bw_available is high (e.g., >0.8): return 3.

   b. Else if bw_available >0.6: return 2.

   c. Else if bw_available >0.4: return 1.

   d. Else: return 0.

This way, when progress is low, we try to rebuild as fast as possible, given available bandwidth. When progress is high, we slow down to avoid high latency.

But this is just one approach. Let's think about other factors.

Another factor is the remaining work (obs[9]). If there's a lot of work left, we need to rebuild faster. But I'm not sure how to incorporate that since it's normalized. Maybe if work_remaining is high, prioritize higher actions.

Alternatively, perhaps the 'alloc_current' (obs[10]) can be considered. If the current allocation is high, maybe we can lower it a bit, but I'm not sure.

Another approach is to calculate a priority based on how much the system can afford to use more bandwidth without causing latency issues.

Alternatively, perhaps the policy can be based on a combination of bw_available and progress.

Let me think of a more structured approach:

- When latency is over target, choose the lowest action.

- Else, if progress is high (e.g., >0.7), choose a lower action to minimize latency.

- Else, choose the highest possible action that doesn't cause latency issues, based on bw_available.

So, the steps:

if obs[3] > obs[4]:
    return 0
elif obs[8] > 0.7:  # progress >70%
    return 0
elif obs[8] > 0.5:  # progress >50%
    return 1
else:
    if obs[11] > 0.8:
        return 3
    elif obs[11] > 0.6:
        return 2
    elif obs[11] > 0.4:
        return 1
    else:
        return 0

Wait, but this may not cover all cases. For example, if progress is 0.6, which is between 0.5 and 0.7, the policy returns 1, but maybe the available bandwidth is very high, allowing for a higher action.

Alternatively, perhaps when progress is between 0.5 and 0.7, we can still choose a higher action if bw_available is high.

So, perhaps:

if obs[3] > obs[4]:
    return 0
else:
    if obs[8] > 0.8:
        return 0
    elif obs[8] > 0.6:
        return 1
    else:
        if obs[11] > 0.8:
            return 3
        elif obs[11] > 0.6:
            return 2
        elif obs[11] > 0.4:
            return 1
        else:
            return 0

This way, for progress above 0.6, we choose 1, and for lower progress, we prioritize higher actions based on bw_available.

Another consideration: the hazard (obs[5]) and hazard_uncertainty (obs[6]). High hazard suggests higher risk of another disk failure, so maybe we need to rebuild faster. But since the state is a dict that can be used to remember things, perhaps we can track the hazard over time, but for now, perhaps it's better to ignore it unless it's a significant factor.

Alternatively, if the hazard is high, we might want to rebuild faster to reduce redundancy exposure. So, perhaps if hazard is high, we choose higher actions.

But since the state is per episode, maybe we can track if the hazard has been consistently high, but for simplicity, perhaps not include it in this version.

So, putting it all together, the policy function can be structured as:

def policy(obs: np.ndarray, state: dict) -> int:
    if obs[3] > obs[4]:  # latency violation
        return 0
    progress = obs[8]
    if progress > 0.8:
        return 0
    elif progress > 0.6:
        return 1
    else:
        bw_available = obs[11]
        if bw_available > 0.8:
            return 3
        elif bw_available > 0.6:
            return 2
        elif bw_available > 0.4:
            return 1
        else:
            return 0

Wait, but this might not handle all cases optimally. For example, when progress is 0.5, and bw_available is 0.7, it returns 2. But if progress is 0.5, maybe higher actions are better.

Alternatively, perhaps the progress thresholds should be adjusted. Maybe 0.7 is a better threshold for when to start slowing down.

Alternatively, perhaps the policy should also consider the remaining work and the current allocation.

Another approach is to prioritize higher actions when the remaining work is high and bw_available is sufficient, and lower actions when remaining work is low.

But since the observation includes 'work_remaining' (obs[9]), perhaps we can use that.

If work_remaining is high, we need to rebuild faster.

So, perhaps:

if work_remaining is high (e.g., obs[9] > 0.7), and bw_available is high, choose higher actions.

But integrating that into the policy may complicate it.

Alternatively, perhaps the current allocation (obs[10]) can be considered. If the current allocation is high, maybe we can keep it or lower it.

But I think for simplicity, the initial approach with progress and bw_available is a good starting point.

Testing this logic:

Case 1: Latency is over target. Return 0.

Case 2: Progress is 80% or more. Return 0.

Case 3: Progress is 60-80%. Return 1.

Case 4: Progress is below 60%:

   a. If bw_available > 0.8: return 3.

   b. Else if >0.6: return 2.

   c. Else if >0.4: return 1.

   d. Else: return 0.

This seems reasonable.

But perhaps the thresholds for bw_available are too high. Maybe in practice, the system can handle higher actions with lower bw_available as long as latency isn't over.

Alternatively, perhaps the thresholds should be adjusted based on the combination of progress and bw_available.

Another consideration: the 'fg_bw' (obs[2]) is the foreground bandwidth share. If fg_bw is high, the system is busy, so maybe we should choose lower actions.

Wait, fg_bw is the foreground bandwidth share. So higher fg_bw means more foreground traffic, leaving less for rebuild. So perhaps if fg_bw is high, choose lower actions.

But in the current policy, we are using bw_available, which is the headroom. So perhaps that's already accounted for.

Because bw_available is the capacity minus foreground load. So if fg_bw is high, bw_available is low.

So the current policy using bw_available should handle that.

Another point: when the progress is low, we need to rebuild quickly. So higher actions are better, but only if bw_available allows it.

Yes, that's what the policy does.

I think this logic should handle most cases, balancing between finishing quickly when possible and avoiding latency violations.

Now, to implement this in code.

The function will:

1. Check if obs[3] > obs[4]. If yes, return 0.

2. Check progress (obs[8]):

   a. If >0.8: return 0.

   b. If >0.6: return 1.

   c. Else: proceed to check bw_available.

3. For progress <=0.6:

   a. If bw_available >0.8: return 3.

   b. Else if >0.6: return 2.

   c. Else if >0.4: return 1.

   d. Else: return 0.

So, code:

def policy(obs: np.ndarray, state: dict) -> int:
    if obs[3] > obs[4]:
        return 0
    progress = obs[8]
    if progress > 0.8:
        return 0
    elif progress > 0.6:
        return 1
    else:
        bw_available = obs[11]
        if bw_available > 0.8:
            return 3
        elif bw_available > 0.6:
            return 2
        elif bw_available > 0.4:
            return 1
        else:
            return 0

Wait, but what about when progress is 0.7? It's above 0.6, so returns 1. But if bw_available is high, it's better to use higher action.

Hmm, perhaps the progress thresholds should be adjusted to allow for higher actions when bw_available is sufficient, even if progress is above 0.6 but below 0.8.

Alternatively, perhaps the progress-based actions should be adjusted.

Maybe instead of checking progress first, we should prioritize bw_available when progress is not too high.

Alternatively, perhaps the policy should be:

If latency is over target: return 0.

Else:

   if progress is high (e.g., >0.8): return 0.

   else:

      if bw_available is high: choose higher action.

      else: choose lower action.

But this is too vague.

Alternatively, perhaps the policy should prioritize bw_available more when progress is low.

But perhaps the initial code is acceptable.

Testing it with some examples:

Example 1:

obs[3] = 0.9, obs[4] = 0.8. So latency is over. Return 0.

Example 2:

obs[3] = 0.7, obs[4] = 0.8. progress=0.5, bw_available=0.9. Returns 3.

Example 3:

progress=0.7, returns 1.

But maybe in this case, if bw_available is high, it's better to return 2 or 3.

Hmm, this suggests that the policy may not always choose the optimal action when progress is between 0.6 and 0.8 but bw_available is high.

So perhaps the progress-based actions should be modified.

Alternative approach: only when progress is above a certain threshold (like 0.9) do we slow down. Otherwise, prioritize bw_available.

But I'm not sure.

Another idea: use a weighted sum of progress and bw_available to decide the action.

But given the time constraints, perhaps the initial approach is acceptable.

Another consideration: the safety filter may raise the action, so choosing a lower action may be overridden, but choosing a higher action may stay as is.

Thus, perhaps when bw_available is high, we can choose a higher action, knowing that the filter won't lower it.

But if the current action is high and the filter raises it, that's worse.

Wait, the filter can only raise the action, not lower it. So if I choose a lower action, the filter may raise it to a higher one, which could cause more latency.

This complicates the decision-making.

So perhaps, when bw_available is high, and the system is not in a degraded state, choosing a higher action is better, but we have to be cautious about the filter.

But since the filter's behavior is unknown, perhaps the initial approach is the best.

Thus, the code as written should handle the cases as intended.
</think>

To solve this problem, we need to create a control policy for a storage system that is rebuilding data onto a spare disk. The goal is to minimize the total cost, which is a combination of latency degradation, degraded state duration, and latency violations. The policy must balance the speed of the rebuild with the impact on user-visible latency.

### Approach
The approach involves making decisions based on the current state of the system, as observed through a normalized vector of metrics. The key factors considered are:
1. **Latency Violation**: If the current latency exceeds the target, the system should prioritize minimizing the rebuild rate to avoid further penalties.
2. **Rebuild Progress**: When the rebuild is nearly complete (high progress), the system should slow down to minimize latency impact.
3. **Bandwidth Availability**: When there is sufficient available bandwidth, the system should prioritize a faster rebuild rate to complete the task quickly.

The policy uses a series of conditional checks to decide the rebuild rate:
1. If the current latency exceeds the target, choose the lowest rebuild rate.
2. If the rebuild progress is high, choose a lower rebuild rate to minimize latency impact.
3. If the rebuild progress is moderate, choose a moderate rebuild rate.
4. If the rebuild progress is low, prioritize the highest possible rebuild rate based on available bandwidth.

### Solution Code
```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    if obs[3] > obs[4]:  # Current latency exceeds target
        return 0
    progress = obs[8]
    if progress > 0.8:
        return 0
    elif progress > 0.6:
        return 1
    else:
        bw_available = obs[11]
        if bw_available > 0.8:
            return 3
        elif bw_available > 0.6:
            return 2
        elif bw_available > 0.4:
            return 1
        else:
            return 0
```

### Explanation
1. **Latency Check**: The first condition checks if the current latency exceeds the target. If true, the lowest rebuild rate (index 0) is chosen to avoid further penalties.
2. **Rebuild Progress**: The next conditions check the progress of the rebuild. If the progress is high (over 80%), the rebuild rate is slowed down to minimize latency impact. If progress is moderate (between 60% and 80%), a slightly higher rebuild rate is chosen.
3. **Bandwidth Availability**: When the rebuild progress is low, the policy prioritizes higher rebuild rates based on available bandwidth. If bandwidth is abundant, the highest rebuild rate is chosen; otherwise, it selects a rate that balances speed and latency impact.

This approach ensures that the system balances the need to complete the rebuild quickly with the need to maintain acceptable latency levels, minimizing the overall cost.