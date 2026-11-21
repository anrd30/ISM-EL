Okay, so I need to write a Python function called policy that takes an observation array and a state dictionary, and returns an action index (0, 1, 2, or 3). The goal is to minimize the cost, which is a combination of latency degradation, degraded state, and latency violation. 

Let me start by understanding the problem. The system is rebuilding data after a disk failure, and the rebuild rate affects how quickly it finishes but also impacts user latency. The actions are four possible rates: 20, 50, 80, 120 MBps, but I don't need to use the actual values, just their indices.

The observation is a 12-element float32 vector, each normalized to [0,1]. The state is a dictionary I can use to remember things within an episode.

Hmm, so the key is to balance the rebuild speed without causing too much latency. The cost function has three parts: latency_degradation (probably increases as the system is busier), degraded_state (1 per interval until rebuild is done), and latency_violation (heavy penalty if mean latency exceeds target). So, finishing quickly is good, but not if it causes a violation.

First, I need to look at the observations and decide the action. Since the observations are normalized, I can't directly know the actual values but can compare their relative positions.

Looking at the observation indices:

0: fg_read_iops
1: fg_write_iops
2: fg_bw (foreground bandwidth share)
3: latency_now
4: latency_target
5: hazard (failure rate estimate)
6: hazard_uncertainty
7: redundancy_left
8: progress (fraction done)
9: work_remaining
10: alloc_current (current rate)
11: bw_available (headroom)

So, the main factors affecting the decision are the current latency, the target, the available bandwidth, the progress of the rebuild, and the hazard of another disk failure.

I think the key factors are:

- Current latency (latency_now) compared to target (latency_target). If it's close to or over the target, we should reduce the rebuild rate to prevent violation.
- The available bandwidth (bw_available). If there's plenty of headroom, we can increase the rebuild rate.
- The progress (how much is done) and work_remaining. If a lot is left, we might want to rebuild faster, but if almost done, maybe slow down to save on cost.

Also, the hazard (probability of another disk failing) and redundancy_left might influence the decision. If redundancy is low and hazard is high, we might want to finish the rebuild faster to avoid another failure.

Wait, but the safety filter may raise the action if the redundancy exposure is too high. So, sometimes the action I choose might be overridden to a higher rate. But since the filter doesn't lower it, maybe I should focus on choosing a rate that's as high as possible without causing issues, because the filter might increase it further if necessary.

So, the strategy is to choose the highest possible action that doesn't cause the current latency to exceed the target, considering the available bandwidth and the progress.

Let me outline the steps:

1. Check if the current latency is above the target. If so, we need to reduce the rebuild rate to prevent further violations.
2. If the latency is below the target, try to maximize the rebuild rate without causing latency to go over.
3. Also, consider the available bandwidth: if there's a lot of headroom (high bw_available), we can choose a higher rate.
4. If progress is high (close to 1), maybe slow down to finish quickly but not cause too much latency.
5. Also, if the work_remaining is low, perhaps it's better to finish quickly even if it takes some bandwidth.

So, perhaps the policy should prioritize:

- If latency is close to or over target, choose the lowest action to minimize impact.
- If bandwidth is available, choose a higher action.
- Otherwise, balance between the remaining work and the available bandwidth.

Wait, but how to translate these into code?

Let me think about the variables:

- obs[3] is latency_now, obs[4] is latency_target. So, if obs[3] > obs[4], we have a violation. So, we should avoid that.

But since the observations are normalized, perhaps we can compare them directly. For example, if obs[3] is close to 1 and obs[4] is also close to 1, but maybe obs[3] is higher than obs[4].

So, in code, maybe:

if obs[3] > obs[4]:
    # current latency is over target, need to slow down
    choose the lowest possible action (index 0)
else:
    # try to choose the highest possible action without causing violation
    # based on available bandwidth and other factors.

But that's a bit simplistic. Maybe we can have a more nuanced approach.

Another idea: the available bandwidth (obs[11]) indicates how much headroom is left. If bw_available is high (close to 1), we can afford a higher rebuild rate. If it's low, we should choose a lower rate.

So, perhaps the action should be proportional to the available bandwidth. For example:

if bw_available is high, choose higher action; if low, choose lower.

But how to map this to the four actions.

Alternatively, we can create a priority based on several factors:

1. If the current latency is over the target, choose the lowest action.
2. Else, if the available bandwidth is very low, choose a lower action.
3. Else, if there's a lot of work remaining, choose a higher action.
4. Else, choose a moderate action.

But this is still a bit vague.

Another approach: the available bandwidth (obs[11]) is a key factor. If the available bandwidth is high, we can choose a higher action. If it's low, choose a lower action.

So, let's compute a value based on available bandwidth. For example:

if obs[11] > 0.8:
    choose highest action (3)
elif obs[11] > 0.6:
    choose 2
elif obs[11] > 0.4:
    choose 1
else:
    choose 0

But this is a simple step function based on available bandwidth.

But we also need to consider the current latency. So, maybe first check if current latency is over target, then choose the lowest action. Otherwise, proceed with the bandwidth-based decision.

So, the logic would be:

if obs[3] > obs[4]:
    action = 0
else:
    if obs[11] > 0.8:
        action = 3
    elif obs[11] > 0.6:
        action = 2
    elif obs[11] > 0.4:
        action = 1
    else:
        action = 0

But this might not account for the progress. For example, if the rebuild is almost done, even if the bandwidth is high, maybe it's better to finish quickly, but that's already handled by the progress in the available bandwidth, perhaps.

Wait, but work_remaining is obs[9], which is a fraction. So, if work_remaining is low, perhaps we can afford to take a higher action even if the bandwidth is a bit low, because it's only for a short time.

Alternatively, we can prioritize the rebuild rate when there's a lot of work remaining.

Hmm, maybe we can combine available bandwidth and work_remaining. For example, when there's a lot of work left (work_remaining is high), and available bandwidth is sufficient, we can choose a higher action.

Alternatively, we can compute a weighted sum of available bandwidth and work_remaining, and choose the action based on that.

But this might complicate things.

Another angle: the safety filter may raise the action, so perhaps it's better to be cautious and choose a lower action than what's optimal, as the filter can't lower it.

But the problem says the filter is applied regardless of our choice, so perhaps our policy should focus on the variables we can control.

Let me think about the cost function. The cost is 1.0 * latency_degradation + 0.1 * degraded_state + 10.0 * latency_violation.

So, the main penalties are:

- Latency violation is 10x more than the others, so it's critical to avoid that.
- The degraded state is 0.1 per interval, so it's less critical than the violation but still matters.
- Latency degradation is 1.0 per unit, so it's more important than degraded state.

So, the priority is:

1. Avoid latency violation (obs[3] > obs[4]).

2. Minimize latency degradation.

3. Minimize degraded state (finish as soon as possible).

So, the first priority is to prevent any latency violation. So, if current latency is over target, we must choose the lowest action.

If not, then we can try to maximize the rebuild rate, but considering the available bandwidth and the impact on latency degradation.

But how to model that.

Another approach: if the current action is causing high latency, we can reduce it. Otherwise, choose the highest possible action that doesn't cause the latency to go over.

But the problem is that the observation only gives the current state, not future states. So, the policy must make a decision based solely on the current observation.

So, perhaps the policy should:

1. Check if the current latency is over the target. If yes, choose the lowest action (0).

2. Else, check the available bandwidth. If available bandwidth is high, choose a higher action, else choose a lower one.

3. Also, if the progress is high (close to 1), perhaps we can afford to slow down a bit, but since the rebuild is almost done, maybe it's better to keep it as high as possible.

Wait, but if the rebuild is almost done, then the time left is short, so even a high rate may not cause much latency impact.

Alternatively, perhaps when progress is high, we can choose a higher action to finish quickly, as the time left is limited.

Hmm, this is getting a bit complex. Maybe the simplest approach is to prioritize available bandwidth and current latency.

So, let's try to code a policy that first checks if the current latency is over the target. If so, choose the lowest action. Otherwise, choose the highest possible action based on available bandwidth.

But how to map available bandwidth to action.

Alternatively, maybe we can compute a threshold based on available bandwidth. For example, if available bandwidth is above a certain level, choose a higher action, else lower.

Another thought: the available bandwidth (obs[11]) represents how much unused capacity there is. If it's high, we can afford to take more bandwidth for the rebuild. So, the action should be higher when obs[11] is higher.

So, perhaps the policy can be:

if obs[3] > obs[4]:
    action = 0
else:
    action = min(3, int(obs[11] * 3))  # maps 0-1 to 0-3, but this is a rough approach

Wait, that's a possible approach. For example, if obs[11] is 0.9, action would be 3 (since 0.9*3=2.7, int is 2, but min(3, 2) is 2? Wait, no: 0.9*3 is 2.7, int is 2, so action is 2. But maybe we can round it up or down.

Alternatively, we can use a step function:

if obs[11] > 0.75:
    action = 3
elif obs[11] > 0.5:
    action = 2
elif obs[11] > 0.25:
    action = 1
else:
    action = 0

This way, when available bandwidth is high, choose higher action.

But what about the hazard? If hazard is high (obs[5] is high), maybe we should prioritize higher rebuild rates to finish faster, thus reducing the exposure time.

So, perhaps when hazard is high, we should choose a higher action even if available bandwidth is a bit low.

Hmm, that adds another layer. So, the policy becomes:

if obs[3] > obs[4]:
    action = 0
else:
    if obs[5] > 0.5:  # high hazard
        action = 3
    else:
        if obs[11] > 0.75:
            action = 3
        elif obs[11] > 0.5:
            action = 2
        elif obs[11] > 0.25:
            action = 1
        else:
            action = 0

But this might not be sufficient. For example, if hazard is high but available bandwidth is low, choosing a high action could cause latency issues.

Alternatively, perhaps we can combine hazard and available bandwidth into a single metric.

But this is getting complicated, and I'm not sure how to balance these factors.

Another idea: the progress (obs[8]) indicates how much of the rebuild is done. If progress is low, we have a lot to do, so we might want to choose a higher action. If progress is high, maybe we can slow down.

So, perhaps:

if obs[3] > obs[4]:
    action = 0
else:
    if obs[8] < 0.5:  # more than half left
        if obs[11] > 0.75:
            action = 3
        elif obs[11] > 0.5:
            action = 2
        else:
            action = 1
    else:
        # less than half left, maybe can slow down
        if obs[11] > 0.75:
            action = 2
        elif obs[11] > 0.5:
            action = 1
        else:
            action = 0

But this is adding more conditions, and it's getting a bit unwieldy.

Wait, perhaps a better approach is to prioritize based on the following order:

1. Avoiding latency violation (obs[3] > obs[4]): choose lowest action.

2. If hazard is high, prioritize higher action to finish faster.

3. Otherwise, choose the highest possible action based on available bandwidth and progress.

But how to translate this into code.

Alternatively, perhaps we can use a weighted sum of available bandwidth and hazard to determine the action.

For example:

if obs[3] > obs[4]:
    action = 0
else:
    # calculate a score based on available bandwidth and hazard
    score = obs[11] * 0.7 + (1 - obs[5]) * 0.3
    # higher score means more available bandwidth and lower hazard, so choose higher action
    if score > 0.9:
        action = 3
    elif score > 0.7:
        action = 2
    elif score > 0.5:
        action = 1
    else:
        action = 0

But this is just a guess. Maybe the weights should be different.

Alternatively, perhaps the hazard should have a higher weight because if the hazard is high, we want to finish the rebuild as quickly as possible, even if it uses more bandwidth.

So, maybe:

score = obs[11] * 0.5 + (1 - obs[5]) * 0.5

But I'm not sure.

Alternatively, if hazard is high, we might want to prioritize higher action regardless of available bandwidth, as long as it doesn't cause a latency violation.

So, in code:

if obs[3] > obs[4]:
    action = 0
else:
    if obs[5] > 0.7:  # high hazard
        # prioritize higher action
        if obs[11] > 0.3:
            action = 3
        else:
            action = 2
    else:
        # lower hazard, prioritize available bandwidth
        if obs[11] > 0.75:
            action = 3
        elif obs[11] > 0.5:
            action = 2
        elif obs[11] > 0.25:
            action = 1
        else:
            action = 0

This way, when hazard is high, we choose a higher action if possible, even if available bandwidth is not the highest.

But I'm not sure if this is the best approach. It's getting quite involved.

Another angle: the cost function includes 10.0 * latency_violation. So, it's extremely important to avoid that. So, the first priority is to ensure that the current action doesn't cause the latency to go over the target.

So, perhaps the policy should first check if the current action would cause the latency to go over. But since we don't have the future observations, we can't know that. So, we have to base it on the current state.

So, the first condition is: if current latency is over target, choose the lowest action.

Else, we can choose a higher action, but we need to estimate whether increasing the action would cause the latency to go over.

But without knowing the future, it's hard. So, perhaps the best we can do is to choose the highest possible action that doesn't cause the current latency to be over, considering the available bandwidth and hazard.

Wait, but the available bandwidth (obs[11]) is the headroom. So, if available bandwidth is high, increasing the action is safe.

So, perhaps the policy is:

if obs[3] > obs[4]:
    action = 0
else:
    if obs[11] > 0.8:
        action = 3
    elif obs[11] > 0.6:
        action = 2
    elif obs[11] > 0.4:
        action = 1
    else:
        action = 0

But this ignores the hazard and progress.

Alternatively, perhaps when available bandwidth is high, choose the highest action, else choose the lowest possible that doesn't cause the current latency to go over.

Wait, but the hazard is another factor. High hazard means another disk failure is more likely, so we want to finish the rebuild quickly, thus choosing a higher action.

So, perhaps the policy should consider both available bandwidth and hazard.

But how?

Maybe, when hazard is high, we can tolerate a lower available bandwidth to choose a higher action.

For example:

if obs[3] > obs[4]:
    action = 0
else:
    if obs[5] > 0.5:  # high hazard
        if obs[11] > 0.4:
            action = 3
        elif obs[11] > 0.2:
            action = 2
        else:
            action = 1
    else:
        if obs[11] > 0.75:
            action = 3
        elif obs[11] > 0.5:
            action = 2
        elif obs[11] > 0.25:
            action = 1
        else:
            action = 0

This way, when hazard is high, we're more aggressive in choosing higher actions even if available bandwidth is a bit lower.

But this is getting a bit arbitrary. Maybe I should try to find a way to combine these factors into a single decision.

Alternatively, perhaps the state can be used to track some variables across intervals, but the state is cleared at the start of each episode, so it's per episode.

Wait, the state is a dictionary that's passed to the policy function. It's cleared at the start of each episode. So, I can use it to track, for example, how long the rebuild has been ongoing, or the previous actions, etc.

But for simplicity, maybe it's better to not use the state and make the policy stateless, as it's easier and faster.

So, perhaps a stateless policy that looks at the current observation and makes a decision based on that.

Another idea: the action should be as high as possible without causing the current latency to exceed the target. So, if the current action is the highest possible, but without causing a violation.

But how to determine that.

Alternatively, perhaps the policy can be based on the following:

- If current latency is over target, choose the lowest action.
- Else, if the available bandwidth is high, choose the highest action.
- Else, choose a moderate action.

But this is too vague.

Wait, perhaps I can look at the available bandwidth and the current action.

obs[10] is the current allocated rate. So, if the current action is high, but the available bandwidth is still sufficient, perhaps we can increase it.

But the problem is that the action is chosen each interval, and the safety filter may override it.

Wait, but the policy's action is just a suggestion, and the safety filter may raise it. So, perhaps the policy should choose the highest possible action that doesn't cause the current latency to be over, and let the filter handle the rest.

So, the policy can be:

if obs[3] > obs[4]:
    action = 0
else:
    # choose the highest possible action based on available bandwidth
    if obs[11] > 0.9:
        action = 3
    elif obs[11] > 0.7:
        action = 2
    elif obs[11] > 0.5:
        action = 1
    else:
        action = 0

But again, this is simplistic.

Alternatively, perhaps the available bandwidth should be the main factor, but when hazard is high, we can increase the action even if available bandwidth is a bit lower.

So, perhaps:

if obs[3] > obs[4]:
    action = 0
else:
    if obs[5] > 0.7:  # high hazard
        # prioritize higher action
        if obs[11] > 0.3:
            action = 3
        else:
            action = 2
    else:
        if obs[11] > 0.75:
            action = 3
        elif obs[11] > 0.5:
            action = 2
        elif obs[11] > 0.25:
            action = 1
        else:
            action = 0

This way, when hazard is high, we're more aggressive in choosing higher actions.

But I'm not sure if this is the best approach.

Another angle: perhaps the main factors are the available bandwidth and the progress. When there's a lot of work remaining, we should choose higher actions if possible.

So, perhaps:

if obs[3] > obs[4]:
    action = 0
else:
    if obs[9] > 0.5:  # more than half work remaining
        if obs[11] > 0.75:
            action = 3
        elif obs[11] > 0.5:
            action = 2
        else:
            action = 1
    else:
        # less than half remaining
        if obs[11] > 0.75:
            action = 2
        elif obs[11] > 0.5:
            action = 1
        else:
            action = 0

This way, when more work is left, we're more aggressive in choosing higher actions.

But again, this is just one possible approach.

Given the time constraints, perhaps I should implement a policy that first checks if the current latency is over the target, then chooses the lowest action. Otherwise, it chooses the highest possible action based on available bandwidth, considering the hazard and progress.

Alternatively, perhaps a simple approach is to choose the highest action when available bandwidth is high, else the lowest, but that might not be optimal.

Wait, perhaps a better approach is to use a weighted sum of available bandwidth and hazard, and choose the action based on that.

For example:

if obs[3] > obs[4]:
    action = 0
else:
    # compute a score that balances available bandwidth and hazard
    score = obs[11] * 0.7 + (1 - obs[5]) * 0.3
    if score > 0.95:
        action = 3
    elif score > 0.8:
        action = 2
    elif score > 0.6:
        action = 1
    else:
        action = 0

This way, higher available bandwidth and lower hazard lead to a higher score, thus a higher action.

But I'm not sure about the weights. Maybe hazard should have a higher weight since a higher hazard means we need to rebuild faster.

Alternatively, perhaps the score should be:

score = obs[11] * 0.6 + (1 - obs[5]) * 0.4

This gives more weight to available bandwidth.

But without knowing the exact dynamics of the system, it's hard to determine the best weights.

Another idea: perhaps the action should be based on the available bandwidth, but when hazard is high, we can increase the action by one level, as long as it's within the allowed range.

So, in code:

if obs[3] > obs[4]:
    action = 0
else:
    base_action = 0
    if obs[11] > 0.75:
        base_action = 3
    elif obs[11] > 0.5:
        base_action = 2
    elif obs[11] > 0.25:
        base_action = 1
    else:
        base_action = 0

    if obs[5] > 0.7:  # high hazard
        action = min(base_action + 1, 3)
    else:
        action = base_action

This way, when hazard is high, we choose a higher action, but not more than 3.

But I'm not sure if this is the best approach.

Alternatively, perhaps the policy can be a simple lookup based on available bandwidth and hazard.

But given the time, perhaps I should proceed with a policy that first checks for latency violation, then chooses the highest possible action based on available bandwidth, and uses the state to track some variables if necessary.

Wait, but the state is per episode, so perhaps I can track the previous action and adjust accordingly, but that might complicate things.

Alternatively, perhaps a simple policy that chooses the highest action possible without causing a latency violation, considering available bandwidth.

So, the code would be:

def policy(obs, state):
    if obs[3] > obs[4]:
        return 0
    else:
        if obs[11] > 0.9:
            return 3
        elif obs[11] > 0.7:
            return 2
        elif obs[11] > 0.5:
            return 1
        else:
            return 0

But this ignores other factors like hazard and progress.

Alternatively, perhaps a more balanced approach is needed.

Wait, considering that the cost function has a heavy penalty for latency violation, the first priority is to avoid that. So, the first condition is correct.

Then, among the remaining options, the next priority is to minimize latency degradation, which is 1.0 per unit. So, higher actions cause higher latency degradation, but also finish the rebuild faster, reducing the degraded_state (0.1 per interval).

So, perhaps the optimal is to find a balance where the action is as high as possible without causing the latency to go over, and considering the available bandwidth.

But without knowing the exact impact of each action on latency, it's hard to model.

Another approach is to look at the current allocated rate (obs[10]) and decide whether to increase or decrease it based on available bandwidth.

But again, without knowing the exact impact, it's tricky.

Given the time constraints, perhaps I should proceed with a policy that:

1. If current latency is over target, choose action 0.

2. Else, choose the highest possible action based on available bandwidth, with consideration for hazard.

So, the code could be:

def policy(obs, state):
    if obs[3] > obs[4]:
        return 0
    if obs[5] > 0.7:  # high hazard
        if obs[11] > 0.4:
            return 3
        else:
            return 2
    else:
        if obs[11] > 0.75:
            return 3
        elif obs[11] > 0.5:
            return 2
        elif obs[11] > 0.25:
            return 1
        else:
            return 0

This way, when hazard is high, we choose a higher action even if available bandwidth is a bit lower.

But I'm not sure if this is the best approach. It's a possible solution, but I'm not certain.

Alternatively, perhaps the policy can be based on the available bandwidth and the work remaining. If there's a lot of work left, choose a higher action if possible.

But again, without knowing the exact impact, it's hard.

Given all this, perhaps the best approach is to implement a policy that first checks for latency violation, then chooses the highest possible action based on available bandwidth, and uses hazard to adjust the action slightly.

So, the final code would look something like this.
</think>

To solve this problem, we need to develop a policy for a storage system that is rebuilding data after a disk failure. The goal is to minimize the cost, which is a combination of latency degradation, degraded state, and latency violation. The policy must balance the rebuild speed with the impact on user latency, ensuring that the system operates efficiently without causing significant delays.

### Approach
The policy is designed to make decisions based on the current state of the system, provided through an observation vector. The key factors considered are:

1. **Current Latency vs. Target Latency**: If the current latency exceeds the target, the policy must prioritize reducing the rebuild rate to prevent further latency issues.
2. **Available Bandwidth**: Higher available bandwidth allows for a higher rebuild rate without significantly impacting latency.
3. **Hazard Rate**: A higher hazard rate (probability of another disk failure) suggests the need to finish the rebuild quickly, thus preferring a higher rebuild rate.

The policy follows these steps:
1. Check if the current latency exceeds the target. If so, choose the lowest rebuild rate.
2. If the hazard rate is high, prioritize a higher rebuild rate if possible, even with slightly lower available bandwidth.
3. If the hazard rate is not high, choose the highest possible rebuild rate based on the available bandwidth.

### Solution Code
```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    if obs[3] > obs[4]:
        return 0
    if obs[5] > 0.7:  # High hazard
        if obs[11] > 0.4:
            return 3
        else:
            return 2
    else:
        if obs[11] > 0.75:
            return 3
        elif obs[11] > 0.5:
            return 2
        elif obs[11] > 0.25:
            return 1
        else:
            return 0
```

### Explanation
- **Current Latency Check**: The policy first checks if the current latency (`obs[3]`) exceeds the target latency (`obs[4]`). If it does, the policy returns the lowest rebuild rate (index 0) to minimize further latency issues.
- **Hazard Rate Consideration**: If the hazard rate (`obs[5]`) is high, the policy prioritizes a higher rebuild rate to finish quickly, even if the available bandwidth (`obs[11]`) is moderately available.
- **Available Bandwidth**: When the hazard rate is not high, the policy selects the highest possible rebuild rate based on the available bandwidth, ensuring efficient use of system resources without causing significant latency issues.

This approach ensures that the policy balances the need to rebuild data quickly with the need to maintain acceptable latency levels, thereby minimizing the overall cost.