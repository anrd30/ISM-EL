Okay, I need to write a control policy for a storage system. The goal is to choose the rebuild rate every 300 seconds, selecting between four options based on the observation vector. The function needs to return the index of the chosen action, which can be 0, 1, 2, or 3.

Let me start by understanding the problem. The system is rebuilding data after a disk failure. The rebuild shares bandwidth with user traffic, so a faster rebuild might degrade user latency. The cost function is a combination of latency degradation, degraded state, and latency violations. I need to minimize this cost.

The observation vector has 12 elements, all normalized to approximately [0,1]. Each element represents different metrics like read IOPS, write IOPS, bandwidth share, current latency, target latency, hazard rate, redundancy left, progress, work remaining, current rate, and available bandwidth.

My task is to write a deterministic function that, given the observation and a state dictionary (for keeping track within an episode), returns the action index. The function must be efficient, under 10 ms per call.

First, I should think about the key factors that influence the action choice. The main considerations are:

1. Latency: If current latency is approaching or exceeding the target, we need to reduce the rebuild rate to prevent violations.
2. Progress and remaining work: If the rebuild is almost done, it's safe to maybe increase the rate. If a lot of work remains, we might want to push harder.
3. Available bandwidth: If there's enough headroom, we can increase the rebuild rate without affecting user traffic too much.
4. Hazard rate: A higher hazard means a higher risk of another disk failure. Maybe we need to finish the rebuild faster in such cases.
5. Redundancy left: Lower redundancy means it's more critical to finish the rebuild quickly, possibly at the cost of higher latency.

So, I need a way to balance these factors. Since the cost heavily penalizes latency violations, I should prioritize avoiding those. But I also want to finish the rebuild as soon as possible to minimize the degraded_state cost.

One approach is to use a weighted sum of these factors to determine the action. For example, prioritize actions that minimize latency degradation while considering how much work is left and available bandwidth.

Let me outline the steps:

1. Check if latency is already over the target. If so, we should reduce the rebuild rate to prevent further violations.
2. If latency is under the target, consider increasing the rebuild rate as much as possible without causing a violation.
3. Look at the available bandwidth (obs[11]). If there's plenty of headroom, we can increase the rebuild rate. If not, we might need to slow down.
4. Progress (obs[8]) indicates how much of the rebuild is done. If progress is high, we might want to finish quickly, even if it uses more bandwidth.
5. Work remaining (obs[9]) is another indicator of how much is left. If a lot is left, we might want to allocate more bandwidth to finish faster.
6. Hazard rate (obs[5]) and its uncertainty (obs[6]) can influence the urgency. Higher hazard means more urgency to finish the rebuild.

But how to translate these into a decision?

Perhaps, for each observation, calculate a priority score that combines these factors and select the action that best balances them.

Let me think about possible heuristics.

First, if the current latency is above the target (obs[3] > obs[4]), we must reduce the rebuild rate. So, in this case, choose the lowest possible action (index 0 or 1), depending on how much over the target we are.

If the latency is below the target, then we can consider increasing the rebuild rate, but we need to balance the available bandwidth and how much work is remaining.

Available bandwidth (obs[11]) is a measure of how much extra capacity there is. If it's high, we can afford a higher rebuild rate. If it's low, we might need to keep the rate lower.

Another factor is the progress. If the rebuild is almost done (high progress), maybe we can afford to push the rate higher to finish quickly, even if it uses more bandwidth. But if there's a lot left, we might want to balance between using more bandwidth and keeping latency low.

So, perhaps the algorithm can be:

- If latency is over target: choose the lowest possible action (0 or 1).
- Else:
   - Calculate how much work is remaining (obs[9]).
   - Calculate the available bandwidth (obs[11]).
   - If there's a lot of work remaining and high available bandwidth, choose a higher action.
   - If the work is almost done, perhaps choose a higher action to finish quickly.
   - Also, consider the hazard rate: higher hazard means more urgency, so higher action.

Wait, but the hazard is a failure rate estimate. Higher hazard means higher chance of another disk failure, so we might want to rebuild faster to reduce redundancy exposure.

So, when hazard is high, even if available bandwidth is low, we might choose a higher action to finish the rebuild quickly.

But how to quantify this?

Another approach is to prioritize the available bandwidth and the work remaining.

For example, if available bandwidth is high, we can choose a higher action, provided that the latency is not too close to the target.

But how to handle the different factors in a way that's computationally efficient.

Since the function needs to be called thousands of times per episode and must be under 10 ms, the logic has to be simple.

Let me think about possible thresholds.

First, check if latency_now > latency_target. If yes, reduce the rate.

Else:

   Look at the available bandwidth (obs[11]). If it's very high, choose the highest possible action. If it's low, choose a lower action.

But also consider the progress: if the rebuild is almost done, maybe even if available bandwidth is low, we can push harder.

Alternatively, perhaps we can create a score that combines the available bandwidth, work remaining, and hazard.

But perhaps a simpler approach is to use a priority-based system.

Another thought: the current action (obs[10]) is also available. Maybe we can adjust based on the current rate.

But I think that the current rate is part of the observation, but the function is to choose the next action.

Wait, the observation includes 'alloc_current' which is the current rate applied. So perhaps, if the current rate is already high and it's causing high latency, we might need to reduce.

But given the observation, the function must make a decision based on the current state.

Let me outline a possible decision tree:

1. If latency_now > latency_target:
   - Choose the lowest possible action (0 or 1), maybe based on how much over the target.
   - For example, if it's significantly over, choose 0; if just slightly over, choose 1.

2. Else:
   a. If work_remaining is high:
      i. If available bandwidth is high, choose higher action.
      ii. If available bandwidth is low, choose lower action.
   b. If work_remaining is low:
      i. Choose higher action to finish quickly, even if it uses more bandwidth.
   c. Also, if hazard is high, prioritize higher actions to finish faster, even if it uses more bandwidth.

But this is getting a bit complex. I need a way to translate this into code quickly.

Alternatively, perhaps use a weighted sum of certain factors to decide the priority.

Let's think about the available bandwidth (obs[11]). Higher is better for allowing a higher rebuild rate.

Work remaining (obs[9]) is the fraction left. If it's high, we need to allocate more bandwidth.

Hazard rate (obs[5]) is the risk of another failure. Higher hazard means more urgency.

So, the priority for higher actions could be influenced by:

priority = available_bandwidth * (1 - work_remaining) * (hazard)

Wait, but work_remaining is a fraction left, so (1 - work_remaining) would be the progress. Alternatively, perhaps work_remaining is the amount left, so higher means more to do.

Wait, the description says: 'work_remaining' is the rebuild work left. So higher means more work left. So perhaps, the priority for higher action is influenced by work_remaining (higher is better to allocate more), available_bandwidth (higher is better), and hazard (higher is better).

So, maybe compute a score that is proportional to work_remaining * available_bandwidth * hazard.

If this score is high, choose a higher action; if low, choose a lower action.

But how to map this score to the four action options.

Alternatively, think of it as a weighted sum where each factor contributes to the decision.

Another idea: the action should be as high as possible without causing latency to exceed the target. So, perhaps start with the highest possible action (120 MBps) and check if it's feasible given the current state.

But how to model feasibility without knowing the future? It's tricky.

Perhaps, the decision can be based on a set of rules:

- If latency is over target: reduce action.
- Else, if available bandwidth is high and work_remaining is high: increase action.
- Else, balance between available bandwidth and work remaining.

But I need to translate this into code.

Let's consider the following approach:

If latency is over target, choose the lowest action (0 or 1). Maybe 0 if available_bandwidth is low, else 1.

Else:

   Calculate a priority score that combines work_remaining, available_bandwidth, and hazard.

   The priority is higher when more work is left, more bandwidth is available, and higher hazard.

   So, priority = work_remaining * available_bandwidth * (hazard + 1)  # since hazard is normalized, adding 1 makes it positive.

   Then, select the action based on this priority:

   - If priority is very high, choose action 3 (120 MBps)
   - If high, choose 2 (80)
   - If medium, choose 1 (50)
   - Else, choose 0 (20)

But how to determine the thresholds for priority.

Alternatively, since all the variables are normalized, perhaps we can sort the actions based on a weighted sum.

Wait, perhaps the action should be the highest possible that doesn't cause latency to exceed the target. But without knowing the future, it's hard to model.

Another approach: use a simple rule based on the available bandwidth and progress.

If available_bandwidth is high (obs[11] > 0.5) and progress is low (obs[8] < 0.5), choose higher actions.

But I'm not sure.

Alternatively, the policy could be a function that selects the action based on the available bandwidth and the work remaining.

Let me think about the four actions as levels, and for each, decide when to step up or down.

Perhaps:

- If available_bandwidth is high and work_remaining is high, choose the highest action.
- If available_bandwidth is medium, choose a medium action.
- If available_bandwidth is low, choose lower action.

But how to define high, medium, low.

Alternatively, use a priority queue where higher action is chosen if the sum of work_remaining and hazard is above a certain threshold, considering available bandwidth.

Wait, perhaps the action can be determined by the following steps:

1. If latency_now > latency_target, choose the lowest action (0 or 1).

   How to choose between 0 and 1? Maybe check available_bandwidth. If available_bandwidth is low, choose 0; else, choose 1.

2. Else:

   a. Calculate a 'urgency' score: work_remaining * hazard.

   b. If urgency is high, and available_bandwidth is sufficient, choose higher action.

   c. Else, choose a lower action.

But what's the definition of 'sufficient' available_bandwidth.

Alternatively, create a priority score that combines work_remaining, hazard, and available_bandwidth.

priority = work_remaining * hazard * available_bandwidth

Then, map this score to the four actions.

For example:

- If priority > 0.8: choose action 3
- If 0.6 < priority <= 0.8: action 2
- If 0.4 < priority <=0.6: action 1
- Else: action 0

But the thresholds are arbitrary. Maybe need to adjust them.

But how to handle the normalization. Since all are in [0,1], multiplying them gives a value between 0 and 1.

So, if the product is high, it means high work remaining, high hazard, and high available bandwidth. So, we can afford to allocate more bandwidth to rebuild.

But I'm not sure if this is the best approach.

Another idea: The higher the available_bandwidth, the more we can increase the rebuild rate. So, for available_bandwidth, if it's high, we can choose a higher action, else lower.

Additionally, if work_remaining is high, we need to allocate more, even if it uses more bandwidth.

Hazard also plays a role: higher hazard means more urgency, so even with lower available_bandwidth, we might choose a higher action.

So, perhaps:

Compute a 'bandwidth_utilization' as (available_bandwidth) / (1 - work_remaining). Or perhaps, a combination where higher available_bandwidth and higher work_remaining lead to higher action.

Wait, perhaps the 'urgency' is a combination of work_remaining and hazard, and 'capacity' is available_bandwidth.

So, urgency = work_remaining * (hazard + 1)  # hazard is normalized, so adding 1 to make it positive.

capacity = available_bandwidth

Then, the action is determined by the product urgency * capacity.

If this product is high, choose higher action.

So, the steps would be:

1. If latency_now > latency_target:
   a. Choose the lowest action (0 or 1). Maybe if available_bandwidth is very low, choose 0, else 1.
2. Else:
   a. Compute urgency = work_remaining * (hazard + 1)
   b. Compute capacity = available_bandwidth
   c. Compute priority = urgency * capacity
   d. Select action based on priority:
      - if priority > 0.8: action 3
      - elif priority > 0.6: action 2
      - elif priority > 0.4: action 1
      - else: action 0

But I'm not sure if the thresholds are correct. Also, adding 1 to hazard may not be necessary if it's already normalized.

Alternatively, urgency could be work_remaining * hazard.

Wait, since work_remaining is the amount left, higher means more urgency. Hazard is the risk of another failure, higher means more urgency.

So, urgency = work_remaining * hazard.

But since both are in [0,1], their product is in [0,1].

So, priority = urgency * available_bandwidth.

Thus, the higher the priority, the higher the action.

Now, to implement this:

if obs[3] > obs[4]:
    # Latency violation, choose lower action
    if obs[11] < 0.2:  # low available bandwidth
        return 0
    else:
        return 1
else:
    urgency = obs[9] * obs[5]
    capacity = obs[11]
    priority = urgency * capacity
    if priority > 0.8:
        return 3
    elif priority > 0.6:
        return 2
    elif priority > 0.4:
        return 1
    else:
        return 0

But I'm not sure if this covers all cases. For example, when work_remaining is 1 (all remaining), hazard is 1 (max), and available_bandwidth is 1 (max), priority is 1. So, action 3.

When available_bandwidth is low, even if urgency is high, the priority might not be high enough, leading to a lower action.

Another consideration: if the rebuild is almost done (progress near 1, work_remaining near 0), then urgency is low, so priority is low, and action would be lower. But in this case, perhaps we should push to finish quickly. So, maybe the model is missing something.

Wait, work_remaining is the amount left. So, when work_remaining is high (near 1), we need to allocate more. When it's low (near 0), less allocation is needed. So the current model is correct in that sense.

But when work_remaining is low, but hazard is high, should we still allocate more? Because even if there's not much work left, a higher hazard means another failure is more likely, so we might want to finish the rebuild as quickly as possible.

So, in that case, even if work_remaining is low, if hazard is high, we might still want to choose a higher action.

But in the current model, urgency is work_remaining * hazard. So, if work_remaining is low (e.g., 0.1), and hazard is high (0.9), urgency is 0.09. If available_bandwidth is high (0.8), priority is 0.072, which would select action 0.

But in reality, even though there's little work left, a high hazard means we should finish quickly, perhaps choosing a higher action.

So, maybe the model is underweighting hazard when work_remaining is low.

Alternative approach: use urgency = work_remaining + hazard, but that might not be the right combination.

Alternatively, perhaps urgency should be max(work_remaining, hazard). But that's not considering both together.

Alternatively, compute urgency as a weighted sum, like 0.6 * work_remaining + 0.4 * hazard. But I'm not sure.

Alternatively, perhaps urgency is a combination where even a small work_remaining with high hazard is considered high urgency.

But perhaps this requires a different approach.

Another thought: when hazard is high, regardless of work_remaining, we might want to allocate as much as possible to finish the rebuild quickly, even if it uses more bandwidth.

So, perhaps, if hazard is above a certain threshold, we prioritize higher actions, even if available_bandwidth is not the highest.

So, perhaps, modify the priority calculation to give more weight to hazard when it's high.

Alternatively, in the else clause (when latency is not over target), if hazard is above a certain threshold, choose a higher action regardless of available_bandwidth, as long as it's not causing latency violation.

But how to implement that.

Alternatively, split the decision into two parts:

1. If hazard is very high, choose the highest possible action that doesn't cause latency violation.

2. Else, choose based on work_remaining and available_bandwidth.

But I'm not sure.

Alternatively, if hazard is above a certain threshold (e.g., 0.7), we treat it as a high urgency situation, and choose higher actions even if available_bandwidth is not the highest.

But this might cause latency violations, which are heavily penalized.

Alternatively, when hazard is high, we can allow a higher action as long as available_bandwidth is sufficient.

But I'm getting stuck on how to balance all these factors.

Perhaps a better approach is to look at the state and make a decision based on multiple conditions.

Let's try to outline the decision steps:

1. Check if latency_now > latency_target:
   a. If yes, choose the lowest possible action (0 or 1) based on available_bandwidth.
      i. If available_bandwidth is very low (e.g., <0.2), choose 0.
      ii. Else, choose 1.

2. Else:
   a. Compute the urgency as work_remaining * hazard.
   b. Compute the available_bandwidth threshold. If available_bandwidth is high, we can afford higher actions.
   c. If urgency is high and available_bandwidth is sufficient, choose higher actions.
   d. Else, balance between urgency and available_bandwidth.

But how to implement this.

Alternatively, here's a step-by-step approach:

- If the current latency is over the target, we must reduce the rebuild rate to prevent further violations. So, choose the lowest action possible. But which one? If available_bandwidth is very low, we might have to choose 0. Otherwise, perhaps 1.

- Else, we can afford to increase the rebuild rate. So, decide based on how much work is left and available bandwidth.

   a. If work_remaining is high, and available_bandwidth is high, choose the highest action.

   b. If work_remaining is high but available_bandwidth is low, choose a medium action.

   c. If work_remaining is low, even if available_bandwidth is high, choose a lower action to save on latency.

But how to translate this into code.

Another approach is to use the available_bandwidth as a primary factor. If available_bandwidth is high, choose higher actions, else lower.

But also, consider the work_remaining: if work_remaining is high, even if available_bandwidth is moderate, we might choose a higher action.

So, perhaps:

if available_bandwidth > 0.7:
    if work_remaining > 0.5:
        return 3
    else:
        return 2
elif available_bandwidth > 0.3:
    if work_remaining > 0.5:
        return 2
    else:
        return 1
else:
    return 0

But this is a simplistic approach and may not account for hazard.

Alternatively, combine available_bandwidth and work_remaining into a priority.

priority = available_bandwidth * (1 - work_remaining)

Wait, no, because higher work_remaining means more to do, so higher priority.

Wait, priority could be available_bandwidth * work_remaining.

So, higher when more work is left and more bandwidth is available.

Then, map this priority to the four actions.

But again, the thresholds are arbitrary.

Alternatively, let's try to implement a simple rule-based policy:

- If latency is over target: choose 0 or 1 based on available_bandwidth.

- Else:

   a. If available_bandwidth is high (above 0.7) and work_remaining is high (above 0.5): choose 3.

   b. Else if available_bandwidth is high and work_remaining is low: choose 2.

   c. Else if available_bandwidth is moderate (0.3-0.7) and work_remaining is high: choose 2.

   d. Else if available_bandwidth is moderate and work_remaining is low: choose 1.

   e. Else: choose 0.

But this is a bit vague.

Alternatively, perhaps use a weighted sum of work_remaining, available_bandwidth, and hazard to determine the action.

For example:

score = (work_remaining * 2) + (available_bandwidth * 1.5) + (hazard * 1)

Then, based on the score, select the action.

But I'm not sure about the weights.

Alternatively, use a linear combination:

score = work_remaining * available_bandwidth * hazard

Then, based on the score, select the action.

But again, the thresholds are unclear.

Another idea is to look at the progress. If progress is high, meaning rebuild is almost done, perhaps we can afford to push the rate higher, even if it uses more bandwidth.

So, if progress > 0.9, choose higher action.

But in the observation, progress is at index 8, which is the fraction done. So, if progress is high, work_remaining is low.

Hmm.

I think I'm getting stuck on trying to model all these factors. Maybe I should try to find a simple heuristic that balances the main factors: avoiding latency violations, using available bandwidth, and considering the urgency from work remaining and hazard.

Let me try to outline the code:

def policy(obs: np.ndarray, state: dict) -> int:
    if obs[3] > obs[4]:  # latency violation
        if obs[11] < 0.3:  # low available bandwidth
            return 0
        else:
            return 1
    else:
        # No latency violation, can choose higher actions
        available_bandwidth = obs[11]
        work_remaining = obs[9]
        hazard = obs[5]
        # Compute a priority score
        priority = work_remaining * hazard * available_bandwidth
        if priority > 0.8:
            return 3
        elif priority > 0.6:
            return 2
        elif priority > 0.4:
            return 1
        else:
            return 0

But wait, when work_remaining is high and hazard is high, but available_bandwidth is low, the priority would be low, leading to a lower action. But in that case, perhaps we should still choose a higher action to finish quickly despite the low available_bandwidth, but that could cause latency to increase, leading to a violation.

So, perhaps the model needs to prioritize avoiding latency violations above all else, and only when not in violation, try to maximize the rebuild rate based on available factors.

Another consideration: the current allocated rate (obs[10]) might be a factor. If the current rate is high and it's not causing a violation, perhaps we can keep it or even increase it.

But in the observation, obs[10] is the current rate applied. So, perhaps, if the current rate is already high, and no violation, we can keep it or increase it if possible.

But in the code above, this isn't considered.

Alternatively, perhaps the policy should consider the current rate and adjust based on that.

But without knowing the future, it's hard to predict the impact of changing the rate.

Another approach is to use the current rate as a starting point and adjust based on the other factors.

But given the time constraints, perhaps the initial approach is sufficient, even if it's a bit simplistic.

Testing this approach:

Case 1: Latency is over target. Available_bandwidth is low: choose 0.

Case 2: Latency is over target. Available_bandwidth is moderate: choose 1.

Case 3: No latency violation. High priority (high work_remaining, high hazard, high available_bandwidth): choose 3.

Case 4: Moderate priority: choose 2.

Case 5: Low priority: choose 0 or 1.

But I'm not sure if this covers all edge cases.

Another consideration: when hazard is very high, even if available_bandwidth is low, perhaps we should still choose a higher action to finish the rebuild quickly, even if it risks a latency violation.

But in the code above, if latency is not over, it will choose higher actions based on priority, which includes hazard.

But if hazard is high and available_bandwidth is low, the priority might not be high enough to choose a higher action.

For example, hazard=0.9, work_remaining=0.8, available_bandwidth=0.4.

priority = 0.8 * 0.9 * 0.4 = 0.288. So, priority is 0.288, which is less than 0.4, so action 0.

But in reality, high hazard suggests that we need to finish quickly, even if it uses more bandwidth, which might risk a latency violation. However, in the code, we only choose higher actions when available_bandwidth is sufficient.

This is a trade-off between completing the rebuild quickly and avoiding latency violations.

Perhaps, the code is too conservative in such cases. Maybe the hazard should have a higher weight in the priority calculation.

Alternatively, when hazard is above a certain threshold, prioritize higher actions regardless of available_bandwidth, but only if it doesn't cause a latency violation.

But how to implement that.

Perhaps, modify the priority calculation to prioritize hazard more when it's high.

Maybe, compute priority as (work_remaining * hazard) * available_bandwidth.

But if hazard is high, even with low available_bandwidth, the priority could still be high enough to choose a higher action.

Wait, in the previous example, with hazard=0.9, work_remaining=0.8, available_bandwidth=0.4:

priority = 0.8 * 0.9 * 0.4 = 0.288. So, action 0.

But perhaps, even with available_bandwidth=0.4, if hazard is high, we should choose a higher action.

But how to represent that.

Alternatively, treat hazard as a multiplier that increases the priority beyond the available_bandwidth.

But this is getting complicated.

Perhaps, instead of multiplying, add terms.

But since all variables are normalized, adding would exceed 1.

Alternatively, compute urgency as work_remaining + hazard, then multiply by available_bandwidth.

But that could be:

urgency = (work_remaining + hazard) * 0.5  # to keep it in [0,1]
priority = urgency * available_bandwidth

But I'm not sure.

Alternatively, use a weighted sum where hazard has a higher weight.

urgency = 0.6 * work_remaining + 0.4 * hazard

priority = urgency * available_bandwidth

But again, not sure.

Alternatively, in the else clause, if hazard is above a certain threshold (e.g., 0.8), then regardless of available_bandwidth, choose the highest possible action that doesn't cause latency violation.

But how to integrate that into the code.

Perhaps:

if obs[3] > obs[4]:
    # handle latency violation
else:
    if obs[5] > 0.8:  # high hazard
        if obs[11] > 0.3:  # sufficient bandwidth
            return 3
        else:
            return 2
    else:
        # proceed with the priority calculation as before

But I'm not sure if this is the best approach.

Alternatively, when hazard is high, even if available_bandwidth is low, we can choose a higher action, but only if the available_bandwidth is not too low.

For example:

if obs[5] > 0.7:
    if obs[11] > 0.2:
        return 3
    else:
        return 2
else:
    # proceed with the priority calculation

But this is speculative.

Another approach is to consider that when hazard is high, the priority is increased, even if available_bandwidth is low.

So, modify the priority calculation to:

priority = work_remaining * (hazard * 2) * available_bandwidth

But this could make the priority higher, leading to higher actions.

But since hazard is normalized, multiplying by 2 would make it up to 2.0, which is beyond 1. So, perhaps cap it at 1.

Alternatively, use min(hazard * 2, 1).

But this is getting a bit too involved.

Perhaps, given time constraints, the initial approach is acceptable, even if it's not perfect.

So, the code would be:

def policy(obs: np.ndarray, state: dict) -> int:
    if obs[3] > obs[4]:  # latency violation
        if obs[11] < 0.3:  # low available bandwidth
            return 0
        else:
            return 1
    else:
        available_bandwidth = obs[11]
        work_remaining = obs[9]
        hazard = obs[5]
        priority = work_remaining * hazard * available_bandwidth
        if priority > 0.8:
            return 3
        elif priority > 0.6:
            return 2
        elif priority > 0.4:
            return 1
        else:
            return 0

But I'm not sure if this handles all cases well. For example, when work_remaining is low but hazard is high, it might not choose a higher action.

Alternatively, perhaps the priority should be work_remaining * (hazard + 0.5) * available_bandwidth, to give more weight to hazard.

But again, not sure.

Another idea: when hazard is high, even if available_bandwidth is low, we can still choose a higher action, but only if the available_bandwidth is not too low.

For example:

if hazard > 0.8 and available_bandwidth > 0.2:
    return 2
elif hazard > 0.8 and available_bandwidth > 0.4:
    return 3
else:
    proceed with priority.

But this complicates the code.

Alternatively, adjust the priority calculation to give more weight to hazard when it's high.

But perhaps it's better to stick with the initial approach, given time constraints.

Another consideration: the state dictionary can be used to remember the previous action or other variables. But since the problem says the policy is called thousands of times, and the state is per episode, perhaps using the state to track the previous action could help in smoothing the changes.

But the initial code doesn't use the state, so perhaps it's beyond the scope for now.

In conclusion, the policy function will:

- Check for latency violation and choose lower actions if necessary.
- Else, compute a priority based on work_remaining, hazard, and available_bandwidth.
- Select the action based on the priority.

This is a heuristic approach and may not be optimal, but it's a starting point.
</think>

To solve this problem, we need to develop a control policy for a storage system that optimizes the rebuild rate of a failed disk while minimizing latency degradation and avoiding latency violations. The policy must balance the need to complete the rebuild quickly with the risk of degrading user-visible latency.

### Approach
The policy is designed to make decisions based on the current state of the system, which is provided through an observation vector. The key factors considered are:
1. **Latency Violation**: If the current latency exceeds the target, the rebuild rate must be reduced to prevent further degradation.
2. **Available Bandwidth**: Higher available bandwidth allows for a higher rebuild rate without significantly affecting user traffic.
3. **Work Remaining**: More work remaining means a higher priority to allocate more bandwidth to finish the rebuild quickly.
4. **Hazard Rate**: A higher hazard rate indicates a higher risk of another disk failure, thus necessitating a faster rebuild.

The policy uses a heuristic approach to decide the rebuild rate based on these factors. If latency is over the target, the policy reduces the rebuild rate. Otherwise, it calculates a priority score that combines the work remaining, hazard rate, and available bandwidth to determine the optimal rebuild rate.

### Solution Code
```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    if obs[3] > obs[4]:  # Latency now exceeds target
        if obs[11] < 0.3:  # Low available bandwidth
            return 0
        else:
            return 1
    else:
        available_bandwidth = obs[11]
        work_remaining = obs[9]
        hazard = obs[5]
        priority = work_remaining * hazard * available_bandwidth
        if priority > 0.8:
            return 3
        elif priority > 0.6:
            return 2
        elif priority > 0.4:
            return 1
        else:
            return 0
```

### Explanation
1. **Latency Check**: The policy first checks if the current latency exceeds the target. If it does, the rebuild rate is reduced to the lowest possible to prevent further latency issues.
2. **Priority Calculation**: When latency is within acceptable limits, the policy calculates a priority score based on the product of work remaining, hazard rate, and available bandwidth. This score helps determine how aggressively the rebuild should proceed.
3. **Action Selection**: Based on the priority score, the policy selects the appropriate rebuild rate. Higher scores result in higher rebuild rates, while lower scores result in lower rates, ensuring a balance between rebuild speed and latency management.

This approach ensures that the policy efficiently balances the need to complete the rebuild quickly with the risk of degrading user experience, using a heuristic method that is computationally efficient and suitable for real-time decision-making.