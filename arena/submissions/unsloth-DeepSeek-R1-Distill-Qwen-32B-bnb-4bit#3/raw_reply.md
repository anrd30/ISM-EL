Okay, I need to write a policy function for a storage system where a disk has failed and the system is rebuilding data onto a spare. The goal is to minimize the total cost, which is calculated based on latency degradation, degraded state, and latency violations. 

The function I'm supposed to write is called policy, which takes an observation array and a state dictionary, and returns an action index (0, 1, 2, or 3) corresponding to the rebuild rate. 

Let me start by understanding the problem deeply. 

First, the actions are four possible rates: 20, 50, 80, 120 MBps, but in the code, I don't define them; I just return the index. So, the policy function will choose one of these indices based on the observation.

The observation is a 12-element float32 vector, all normalized to [0,1]. Each element represents different metrics:

0: fg_read_iops
1: fg_write_iops
2: fg_bw (foreground bandwidth share)
3: latency_now
4: latency_target
5: hazard
6: hazard_uncertainty
7: redundancy_left
8: progress (fraction of rebuild done)
9: work_remaining (rebuild work left)
10: alloc_current (current rate applied)
11: bw_available (headroom: capacity minus foreground load)

The cost function is:
cost = 1.0 * latency_degradation + 0.1 * degraded_state + 10.0 * latency_violation

So, we want to minimize this. Latency violation is heavily weighted, so we need to avoid that as much as possible.

The key is to balance the rebuild speed (to finish quickly) without causing latency to exceed the target, which would cause a violation.

Let me think about what each observation tells me.

- fg_bw (obs[2]) is the foreground bandwidth share. If this is high, it means the system is busy, so increasing rebuild rate might degrade latency more.
- latency_now (obs[3]) and latency_target (obs[4]): if latency_now > target, we have a violation. So, we need to check if current latency is close to or above the target.
- progress (obs[8]) and work_remaining (obs[9]): how much of the rebuild is done. If work_remaining is high, we might want to prioritize faster rebuild rates.
- bw_available (obs[11]): the headroom. If bw_available is high, we can afford to increase the rebuild rate without much impact on latency.

But there's also a safety filter that may raise the action (i.e., increase the rebuild rate) to maintain redundancy. So, if the current action is low but the filter raises it, that's out of our control. But the policy should still choose the best possible action.

So, the strategy should probably consider:

1. Avoiding latency violations at all costs, as they add 10 to the cost.
2. Finishing the rebuild as quickly as possible without causing violations.
3. Using the available bandwidth efficiently.

Let me outline possible steps:

- Check if current latency is already above target. If yes, we need to reduce the rebuild rate to prevent further violations. But wait, the cost is per interval, so if we have a violation, it's 10.0 added. So, if we are in a violation, perhaps we need to decrease the rebuild rate to allow latency to recover.

But wait, the observation is normalized. So, for example, if latency_now (obs[3]) is 0.9 and latency_target (obs[4]) is 0.8, that's a violation. Or maybe the actual comparison is based on the original values before normalization. But since all are normalized, perhaps the target is a certain normalized value, and if latency_now exceeds that, it's a violation.

Wait, no: the observation is normalized to [0,1], but the actual values are in physical terms. For example, latency_now and latency_target might have been, say, 10ms and 20ms, but normalized to 0.5 and 0.6, respectively. So, in the normalized space, if latency_now is 0.6 and target is 0.5, that's a violation.

Wait no, the description says: "latency_violation is 1 when mean latency exceeds the target." So, perhaps the actual calculation is in the original units. But the observation is normalized. Hmm, but the function receives the normalized values. So, in the policy, when we see obs[3] (latency_now) and obs[4] (latency_target), we can compare them directly to determine if a violation is occurring.

Wait, no. Because the normalization is done on each field, but the fields could have different scales. So, for example, if the original latency_now is 10ms, and target is 20ms, then in normalized terms, if the target is normalized based on its own scale, say, the maximum possible latency could be, say, 50ms. So, 20ms would be 0.4, and 10ms would be 0.2. So, the violation occurs when the actual latency is above the target, but in the normalized space, it's when obs[3] > obs[4].

Wait, but that's not necessarily the case. Because each field is normalized separately. So, for example, if the original target is 20ms, and the original latency_now is 25ms, which is higher. But if the target's normalization is based on a different range than latency_now's, their normalized values might not directly compare.

Wait, the problem statement says: "every field is normalized to approximately [0,1]. None are in physical units: compare fields against each other, not against physical constants." So, for the policy, we can compare the normalized values as they are, without considering their original scales. So, if obs[3] > obs[4], that would mean the current latency exceeds the target, leading to a violation.

So, in the policy, we can check if obs[3] > obs[4]. If so, we need to reduce the rebuild rate to avoid further violations.

But the safety filter may raise the action, so perhaps we should choose a lower action, but the filter might override it.

Wait, but the filter may raise the action, i.e., choose a higher rebuild rate, but never lower it. So, if the policy chooses a low action, the filter could increase it, which might cause a violation. So, perhaps the policy should choose a higher action, but only if it's safe.

Alternatively, perhaps the policy should choose the highest possible action that doesn't cause a violation, considering the current state.

Let me think about the factors:

- If there's a high hazard (obs[5]), meaning a higher chance of another disk failure, we might want to prioritize finishing the rebuild quickly. So, higher rebuild rate.
- If the redundancy_left (obs[7]) is low, perhaps we need to rebuild faster to restore redundancy.
- progress (obs[8]) and work_remaining (obs[9]): if work_remaining is high, we need to rebuild faster. If progress is near 1, we can perhaps slow down.

But also, the bw_available (obs[11]) indicates how much headroom is available. If bw_available is high, we can afford to use more bandwidth for rebuild, i.e., choose a higher action. If bw_available is low, increasing the rebuild rate may cause latency issues.

So, perhaps the policy should:

1. Check if current latency is already above target. If yes, reduce the rebuild rate as much as possible.
2. If not, maximize the rebuild rate (choose the highest action) as long as bw_available is sufficient.
3. Also, consider the hazard: higher hazard means more urgency to finish the rebuild, so higher actions are better.
4. Also, if work_remaining is high, prioritize higher actions.

But how to translate this into the four actions.

Let me think of a possible approach:

- If latency is already above target (obs[3] > obs[4]), then choose the lowest possible action (index 0). Because the cost of a violation is high, and we need to reduce the rebuild rate to allow latency to drop below the target.
- Else, if bw_available is high (obs[11] is high, say above 0.5), then choose the highest action possible (index 3) to finish quickly.
- If bw_available is moderate, choose a middle action.
- If hazard is high, prioritize higher actions even if bw_available is moderate.

Alternatively, perhaps a more nuanced approach:

Calculate a priority score that combines factors like bw_available, hazard, work_remaining, and current progress.

But given the time constraints (the policy must run under 10ms per call), it's better to have a simple heuristic.

Another approach: prioritize higher actions when bw_available is sufficient and hazard is high, and lower actions when bw_available is low or hazard is low.

But how to structure this.

Let me outline possible steps:

1. Check if latency is violating (obs[3] > obs[4]). If yes, choose the lowest action (0). Because we need to reduce the rebuild rate to prevent further violations.
2. Else, check if bw_available is high. If yes, choose the highest action (3) to maximize rebuild speed.
3. If bw_available is moderate, choose a middle action (1 or 2).
4. If bw_available is low, choose a lower action.

But how to define high, moderate, low for bw_available. Since it's normalized, perhaps:

- High: obs[11] > 0.7
- Moderate: 0.3 < obs[11] <= 0.7
- Low: obs[11] <= 0.3

But this is arbitrary. Alternatively, perhaps use a threshold, say 0.5.

But perhaps a better approach is to prioritize higher actions when:

- bw_available is sufficient (obs[11] is high)
- hazard is high (more urgency)
- work_remaining is high (need to finish quickly)
- progress is low (still a lot to do)

So, perhaps the policy can be:

If not in violation:

   if bw_available is high and hazard is high: choose highest action
   else if bw_available is high: choose high action
   else if hazard is high: choose higher action
   else: choose a lower action

But this is vague. Let's think of a more concrete approach.

Another idea: the action should be as high as possible without causing a violation. So, the policy can try to choose the highest action possible given the current bw_available.

But how to model that.

Alternatively, the action can be chosen based on the following:

- If the system is not in a violation, and the bw_available is high, choose the highest action.
- If bw_available is moderate, choose the next highest.
- If bw_available is low, choose a lower action.

But how to translate 'high', 'moderate', 'low' into the observation values.

Alternatively, we can compute a weighted sum of factors to decide the action.

But perhaps a simpler approach is to consider the following:

The action should be as high as possible, but constrained by the bw_available.

So, if bw_available is high, choose the highest action.

But also, if hazard is high, we might want to choose a higher action even if bw_available is moderate.

Wait, but if bw_available is moderate, increasing the rebuild rate may cause latency to increase, possibly leading to a violation.

So, perhaps a better approach is to consider the following:

- If in violation: choose lowest action.
- Else:
   - If bw_available is high and hazard is high: choose highest action.
   - Else if bw_available is high: choose high action.
   - Else if hazard is high: choose higher action.
   - Else: choose a lower action.

But this is getting complicated.

Alternatively, perhaps the action can be determined based on the following priority:

1. Avoid violation: if in violation, choose lowest action.
2. If not in violation:
   a. If bw_available is high and hazard is high: choose highest action.
   b. Else if bw_available is high: choose high action.
   c. Else if hazard is high: choose higher action.
   d. Else: choose a lower action.

But I'm not sure. Maybe a better approach is to create a scoring system for each action, where higher scores are better, and select the action with the highest score.

Each action has a certain rebuild rate, which affects the progress but also the latency.

So, for each action (0 to 3), we can calculate a score that takes into account:

- The potential to reduce work_remaining (higher action is better for this)
- The impact on latency (lower is better)
- The hazard (higher hazard means higher action is better)

But how to model this.

Alternatively, for each action, compute a 'desirability' score, which is a weighted sum of factors:

desirability = (bw_available) * (1 - hazard) * (1 - work_remaining) * (something else)

Wait, perhaps not. Let me think differently.

The goal is to choose the action that maximizes the rebuild rate without causing a violation.

So, the action should be as high as possible, given the current bw_available and hazard.

But how to translate this into the action indices.

Let me consider the bw_available as the primary factor. If bw_available is high, we can take higher actions. If it's low, we have to take lower actions.

Additionally, if the hazard is high, we might want to prioritize higher actions even if bw_available is not the highest.

So, perhaps:

If not in violation:

   if bw_available > threshold1 and (hazard > threshold2 or work_remaining > threshold3):
       choose action 3
   elif bw_available > threshold1:
       choose action 2
   elif hazard > threshold2 or work_remaining > threshold3:
       choose action 2
   else:
       choose action 1 or 0

But what are the thresholds? Since the observations are normalized, perhaps:

threshold1 = 0.5 (for bw_available)
threshold2 = 0.5 (for hazard)
threshold3 = 0.5 (for work_remaining)

But this is arbitrary.

Alternatively, perhaps the policy can be:

If not in violation:

   if bw_available > 0.5 and hazard > 0.5: choose action 3
   elif bw_available > 0.5: choose action 2
   elif hazard > 0.5: choose action 2
   else: choose action 1

But this is a simple heuristic.

Wait, but what about the current progress? If the progress is high (near 1), maybe we don't need to rebuild as fast, so we can choose a lower action.

So, perhaps we should also consider the progress. If progress is high, we can afford to slow down.

So, perhaps:

If not in violation:

   if progress is low (obs[8] < 0.5) and bw_available > 0.5 and hazard > 0.5: choose action 3
   elif progress is low and bw_available > 0.5: choose action 2
   elif progress is low and hazard > 0.5: choose action 2
   else: choose action 1

But this is getting complicated.

Alternatively, perhaps the action can be determined by the following priority:

1. If in violation, choose action 0.
2. Else, if bw_available is high and work_remaining is high, choose action 3.
3. Else if bw_available is high, choose action 2.
4. Else if hazard is high, choose action 2.
5. Else, choose action 1.

But again, this is heuristic.

Another approach: the action is chosen based on the current state, trying to maximize the rebuild rate without causing a violation.

So, the steps:

- If current latency is above target (obs[3] > obs[4]), choose action 0.
- Else:
   - If bw_available is high (obs[11] > 0.7), choose action 3.
   - Else if bw_available is moderate (obs[11] > 0.3), choose action 2.
   - Else, choose action 1 or 0 based on hazard and work_remaining.

But I'm not sure about the thresholds.

Alternatively, perhaps the action can be determined by a weighted sum of bw_available and hazard, scaled to choose the highest possible action without causing a violation.

But given the time constraints, perhaps the simplest approach is to:

- If in violation, choose the lowest action.
- Else, choose the highest possible action that doesn't cause a violation.

But how to determine that without knowing future states.

Wait, perhaps the policy can look at the current bw_available and choose the highest action that doesn't exceed the available bandwidth.

But since the actions are fixed, perhaps the highest action is chosen when bw_available is sufficient.

So, the policy could be:

if obs[3] > obs[4]:
    return 0
else:
    if obs[11] > 0.8:
        return 3
    elif obs[11] > 0.6:
        return 2
    elif obs[11] > 0.4:
        return 1
    else:
        return 0

But this is a simplistic approach, and may not consider other factors like hazard or work_remaining.

Alternatively, perhaps the policy should prioritize higher actions when the hazard is high, even if bw_available is not the highest.

So, perhaps:

if obs[3] > obs[4]:
    return 0
else:
    if obs[5] > 0.8:  # high hazard
        if obs[11] > 0.6:
            return 3
        else:
            return 2
    else:
        if obs[11] > 0.8:
            return 3
        elif obs[11] > 0.6:
            return 2
        elif obs[11] > 0.4:
            return 1
        else:
            return 0

This way, when hazard is high, the policy chooses higher actions even if bw_available is a bit lower.

But I'm not sure if this is optimal.

Another consideration: the work_remaining (obs[9]) and progress (obs[8]). If the rebuild is almost done (progress near 1), perhaps we can slow down the rebuild rate.

So, perhaps:

if obs[3] > obs[4]:
    return 0
else:
    if obs[8] > 0.9:  # almost done
        return 0
    else:
        # decide based on bw_available and hazard

But this adds more complexity.

Alternatively, perhaps the policy can be based on a combination of bw_available and work_remaining.

But given the time constraints, perhaps the best approach is to focus on the main factors: avoiding violation, using available bandwidth, and considering hazard.

So, the policy could be structured as follows:

- If current latency is above target, choose the lowest action (0).
- Else, if bw_available is very high, choose the highest action (3).
- Else, if hazard is high, choose a higher action (2 or 3) depending on bw_available.
- Else, choose a moderate action.

But how to translate this into code.

Alternatively, perhaps a more mathematical approach: compute a weighted sum of factors and select the action that maximizes the sum.

For example, for each action, compute a score:

score = bw_available * (1 - hazard) * (1 - progress) * (some function of work_remaining)

But this is getting complicated.

Another idea: prioritize higher actions when:

- bw_available is high
- hazard is high
- work_remaining is high

So, the policy could be:

if obs[3] > obs[4]:
    return 0
else:
    if obs[11] > 0.7 and obs[5] > 0.7 and obs[9] > 0.5:
        return 3
    elif obs[11] > 0.5:
        return 2
    elif obs[5] > 0.5:
        return 2
    else:
        return 1

But again, this is heuristic.

Alternatively, perhaps the policy can be based on the following:

The action is chosen as the highest possible that doesn't cause a violation, considering the available bandwidth.

But since the policy doesn't know future observations, it has to make a decision based on the current state.

Another approach: use a simple threshold-based policy.

For example:

If in violation: choose 0.

Else:

   if bw_available > 0.8 and hazard > 0.6: choose 3
   elif bw_available > 0.6: choose 2
   elif hazard > 0.6: choose 2
   else: choose 1

But this may not cover all cases.

Alternatively, perhaps the policy can be:

if obs[3] > obs[4]:
    return 0
else:
    if obs[11] > 0.7:
        if obs[5] > 0.7:
            return 3
        else:
            return 2
    else:
        if obs[5] > 0.7:
            return 2
        else:
            return 1

This way, when bw_available is high and hazard is high, choose the highest action. When bw_available is high but hazard is moderate, choose a high action. When bw_available is low but hazard is high, choose a higher action than when both are low.

But I'm not sure if this is optimal.

Another consideration: the progress and work_remaining. If the rebuild is almost done, perhaps it's better to choose a lower action to save resources, even if bw_available is high.

So, perhaps:

if obs[3] > obs[4]:
    return 0
else:
    if obs[8] > 0.9:  # 90% done
        return 0
    elif obs[11] > 0.7 and obs[5] > 0.7:
        return 3
    elif obs[11] > 0.7:
        return 2
    elif obs[5] > 0.7:
        return 2
    else:
        return 1

This adds a check for progress being high, in which case we slow down.

But I'm not sure if 0.9 is the right threshold.

Alternatively, perhaps the policy can be:

if obs[3] > obs[4]:
    return 0
else:
    if obs[11] > 0.7 and obs[5] > 0.7 and obs[8] < 0.8:
        return 3
    elif obs[11] > 0.7 and obs[8] < 0.8:
        return 2
    elif obs[5] > 0.7 and obs[8] < 0.8:
        return 2
    else:
        return 1

But this is getting too complicated.

Perhaps, given the time constraints, the best approach is to implement a policy that:

- Avoids violations by choosing the lowest action if in violation.
- Otherwise, chooses the highest possible action based on bw_available and hazard.

So, the code could be:

def policy(obs, state):
    if obs[3] > obs[4]:  # in violation
        return 0
    else:
        # prioritize higher actions when bw_available is high and hazard is high
        if obs[11] > 0.7:
            if obs[5] > 0.7:
                return 3
            else:
                return 2
        else:
            if obs[5] > 0.7:
                return 2
            else:
                return 1

But I'm not sure if this is optimal.

Alternatively, perhaps the policy can be based on the following:

The action is chosen as the highest possible without causing a violation, considering the available bandwidth and hazard.

So, perhaps:

if in violation:
    return 0
else:
    if bw_available > 0.8 and hazard > 0.8:
        return 3
    elif bw_available > 0.6:
        return 2
    else:
        return 1

But again, this is heuristic.

Another idea: the action can be determined by a simple rule that selects the highest action possible as long as the bw_available is above a certain threshold, adjusted by the hazard.

So, for example:

if in violation:
    return 0
else:
    if obs[11] > 0.6 + (obs[5] * 0.2):
        return 3
    elif obs[11] > 0.4 + (obs[5] * 0.2):
        return 2
    else:
        return 1

This way, when hazard is high, the threshold for choosing higher actions is lower.

But this is just a guess.

Alternatively, perhaps the policy can be based on the following:

Compute a 'safety' score that combines bw_available and hazard. If the safety score is high, choose a higher action.

safety_score = obs[11] * (1 - obs[5])

if in violation:
    return 0
else:
    if safety_score > 0.9:
        return 3
    elif safety_score > 0.7:
        return 2
    elif safety_score > 0.5:
        return 1
    else:
        return 0

But this is again a heuristic.

Given the time constraints, perhaps the best approach is to implement a policy that:

- Avoids violation by choosing the lowest action if in violation.
- Otherwise, chooses the highest action possible based on bw_available and hazard.

So, the code could be:

def policy(obs, state):
    if obs[3] > obs[4]:
        return 0
    else:
        if obs[11] > 0.7:
            if obs[5] > 0.7:
                return 3
            else:
                return 2
        elif obs[5] > 0.7:
            return 2
        else:
            return 1

But I'm not sure if this is optimal.

Another consideration: the progress. If the rebuild is almost done, we can afford to slow down.

So, perhaps:

if obs[3] > obs[4]:
    return 0
else:
    if obs[8] > 0.95:
        return 0
    elif obs[11] > 0.7 and obs[5] > 0.7:
        return 3
    elif obs[11] > 0.7:
        return 2
    elif obs[5] > 0.7:
        return 2
    else:
        return 1

This way, when the rebuild is almost done (progress > 0.95), we choose the lowest action to save resources.

But again, the thresholds are arbitrary.

Given all these considerations, perhaps the best approach is to implement a policy that:

- Prioritizes avoiding violations by choosing the lowest action if in violation.
- Otherwise, chooses the highest possible action based on bw_available and hazard, but also considering the progress.

So, the policy could be:

def policy(obs, state):
    if obs[3] > obs[4]:  # in violation
        return 0
    if obs[8] > 0.95:  # almost done
        return 0
    if obs[11] > 0.8:  # high bandwidth available
        if obs[5] > 0.8:  # high hazard
            return 3
        else:
            return 2
    else:
        if obs[5] > 0.8:  # high hazard
            return 2
        else:
            return 1

But I'm not sure if this covers all cases.

Alternatively, perhaps the policy can be based on a simple threshold for bw_available, and choose the highest action possible.

So, the code could be:

def policy(obs, state):
    if obs[3] > obs[4]:  # in violation
        return 0
    if obs[11] > 0.7:
        return 3
    elif obs[11] > 0.5:
        return 2
    else:
        return 1

But this ignores hazard and progress.

Alternatively, perhaps the policy can be:

if in violation: return 0
else:
    if bw_available > 0.7 and hazard > 0.5: return 3
    elif bw_available > 0.7: return 2
    elif hazard > 0.5: return 2
    else: return 1

But again, this is heuristic.

Given the time constraints, perhaps the best approach is to implement a policy that:

- Avoids violation by choosing the lowest action if in violation.
- Otherwise, chooses the highest action possible based on bw_available.

So, the code could be:

def policy(obs, state):
    if obs[3] > obs[4]:
        return 0
    else:
        if obs[11] > 0.7:
            return 3
        elif obs[11] > 0.5:
            return 2
        else:
            return 1

But this doesn't consider hazard or progress.

Alternatively, perhaps the policy can be:

def policy(obs, state):
    if obs[3] > obs[4]:
        return 0
    else:
        # Calculate a priority score
        priority = obs[11] * 2 + obs[5]
        if priority > 3.5:  # arbitrary threshold
            return 3
        elif priority > 2.5:
            return 2
        else:
            return 1

But this is just a guess.

Another approach: use a weighted sum of bw_available and hazard to decide the action.

For example:

score = obs[11] * 0.7 + obs[5] * 0.3

if score > 0.9:
    return 3
elif score > 0.7:
    return 2
elif score > 0.5:
    return 1
else:
    return 0

But this is heuristic.

Given all these possibilities, perhaps the best approach is to implement a policy that:

- Avoids violation by choosing the lowest action if in violation.
- Otherwise, chooses the highest action possible based on bw_available, with some consideration for hazard.

So, the code could be:

def policy(obs, state):
    if obs[3] > obs[4]:  # in violation
        return 0
    else:
        if obs[11] > 0.8:
            return 3
        elif obs[11] > 0.6:
            return 2
        else:
            if obs[5] > 0.7:
                return 2
            else:
                return 1

But this is just one possible approach.

Alternatively, perhaps the policy can be based on the following:

The action is chosen as the highest possible such that the bw_available is sufficient to avoid a violation, considering the hazard.

But without knowing the exact impact of each action on latency, it's hard to model.

Given the time constraints, perhaps the best approach is to implement a policy that:

- Avoids violation by choosing the lowest action if in violation.
- Otherwise, chooses the highest action possible based on bw_available, with some consideration for hazard.

So, the code could be:

def policy(obs, state):
    if obs[3] > obs[4]:
        return 0
    else:
        if obs[11] > 0.7 and obs[5] > 0.7:
            return 3
        elif obs[11] > 0.7:
            return 2
        elif obs[5] > 0.7:
            return 2
        else:
            return 1

This way, when both bw_available and hazard are high, choose the highest action. Otherwise, choose based on bw_available or hazard.

But I'm not sure if this is optimal.

Another consideration: the work_remaining. If there's a lot of work left, we need to rebuild faster.

So, perhaps:

if in violation:
    return 0
else:
    if obs[9] > 0.5:  # more than half remaining
        if obs[11] > 0.7 and obs[5] > 0.7:
            return 3
        elif obs[11] > 0.7:
            return 2
        else:
            return 1
    else:
        if obs[11] > 0.7:
            return 2
        else:
            return 1

But this is getting too complicated.

Given all these considerations, perhaps the best approach is to implement a policy that:

- Avoids violation by choosing the lowest action if in violation.
- Otherwise, chooses the highest action possible based on bw_available, adjusted by hazard and work_remaining.

But given the time constraints, perhaps the simplest approach is to implement a policy that:

- Avoids violation by choosing the lowest action if in violation.
- Otherwise, chooses the highest action possible based on bw_available.

So, the code could be:

def policy(obs, state):
    if obs[3] > obs[4]:  # in violation
        return 0
    else:
        if obs[11] > 0.7:
            return 3
        elif obs[11] > 0.5:
            return 2
        else:
            return 1

This is a simple heuristic that prioritizes using available bandwidth to rebuild as fast as possible, while avoiding violations.

But perhaps this is too simplistic and may not handle all cases well, especially when hazard is high but bw_available is moderate.

Alternatively, perhaps the policy can be:

if in violation: return 0
else:
    if bw_available > 0.7:
        return 3
    else:
        if hazard > 0.7:
            return 2
        else:
            return 1

But this ignores the progress and work_remaining.

Another idea: the action can be determined by the sum of bw_available and hazard, scaled to choose the highest possible action.

For example:

score = obs[11] + obs[5]
if score > 1.8:  # since each is up to 1, max is 2
    return 3
elif score > 1.6:
    return 2
else:
    return 1

But this is just a guess.

Given all these possibilities, perhaps the best approach is to implement a policy that:

- Avoids violation by choosing the lowest action if in violation.
- Otherwise, chooses the highest action possible based on bw_available, with some consideration for hazard.

So, the code could be:

def policy(obs, state):
    if obs[3] > obs[4]:
        return 0
    else:
        if obs[11] > 0.7:
            if obs[5] > 0.7:
                return 3
            else:
                return 2
        else:
            if obs[5] > 0.7:
                return 2
            else:
                return 1

This way, when bw_available is high and hazard is high, choose the highest action. When bw_available is high but hazard is moderate, choose a high action. When bw_available is low but hazard is high, choose a higher action than when both are low.

But I'm not sure if this is optimal.

Another consideration: the progress. If the rebuild is almost done, perhaps we can choose a lower action.

So, perhaps:

if obs[3] > obs[4]:
    return 0
else:
    if obs[8] > 0.95:
        return 0
    elif obs[11] > 0.7 and obs[5] > 0.7:
        return 3
    elif obs[11] > 0.7:
        return 2
    elif obs[5] > 0.7:
        return 2
    else:
        return 1

This adds a check for progress being high, in which case we slow down.

But again, the thresholds are arbitrary.

Given the time constraints, perhaps the best approach is to implement a policy that:

- Avoids violation by choosing the lowest action if in violation.
- Otherwise, chooses the highest action possible based on bw_available, adjusted by hazard.

So, the code could be:

def policy(obs, state):
    if obs[3] > obs[4]:
        return 0
    else:
        if obs[11] > 0.7 and obs[5] > 0.7:
            return 3
        elif obs[11] > 0.7:
            return 2
        elif obs[5] > 0.7:
            return 2
        else:
            return 1

This is a reasonable heuristic that considers both bw_available and hazard.

But perhaps the best approach is to implement a policy that uses the bw_available as the primary factor, with some consideration for hazard and progress.

But given the time, I'll proceed with this approach.
</think>

To solve this problem, we need to design a control policy for a storage system that optimizes the rebuild rate of a failed disk while minimizing the cost associated with latency degradation, degraded state, and latency violations. The policy must balance the need to complete the rebuild quickly with the need to avoid degrading user-visible latency.

### Approach
The policy is designed to minimize the total cost by balancing the rebuild rate with the system's current state. The key considerations are:
1. **Avoiding Latency Violations**: If the current latency exceeds the target, the policy should prioritize reducing the rebuild rate to prevent further violations.
2. **Using Available Bandwidth**: The policy should maximize the rebuild rate when there is sufficient available bandwidth to avoid unnecessary delays.
3. **Considering Hazard Levels**: Higher hazard levels indicate a greater risk of another disk failure, prompting a higher rebuild rate to restore redundancy quickly.

The policy uses the following approach:
1. Check