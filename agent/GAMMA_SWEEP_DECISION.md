# Gamma sweep: does reweighting the violation term fix PPO's over-conservatism?

Status: proposed, not yet reviewed by the other lanes.

Generated from four `agent.train` runs (seed 0 only, 244 iterations each,
identical to the `b2_244` clean protocol except `--reward-gamma`) and four
matching `agent.evaluate_learned` runs against `diurnal_long_holdout`. Full
output in `agent/evidence/gamma-sweep-2026-11-12/`.

## Background

`reward_gamma` is a fixed multiplier on the violation term inside the reward
(`contracts/contracts.py` `RewardConfig.gamma`, default 10.0; `b2_244` used
2.0). It is a separate mechanism from the adaptive Lagrangian multipliers
`lambda_latency`/`lambda_risk` in `agent/ppo.py`, which run in every PPO
training in this codebase regardless of gamma. The `b2_244` clean rerun
(`agent/evidence/b2-244-clean-2026-11-12/`) showed PPO losing to the
non-learning `risk_aware_rule_based` baseline at gamma=2.0: less throughput,
slower completion, a worse total reward. This sweep checks whether raising
gamma -- penalizing violations harder -- recovers that gap.

## Result

Paired difference, `ppo_lagrangian seed0` minus `risk_aware_rule_based`,
n=20 scenarios, `diurnal_long_holdout`, simulator configuration
`train-fleet,sigma=0.25`:

| gamma | total_reward diff | sem | completion_hours diff | mean_delivered_mbps diff | latency_violation_fraction diff | p_failure_mean diff |
|---|---|---|---|---|---|---|
| 1.0 | -32.27 | 14.15 | +19.08 | -22.13 | -0.004461 | +0.0001421 |
| 2.0 | -20.13 | 1.764 | +21.12 | -23.92 | -0.007621 | +0.0001487 |
| 5.0 | -25.38 | 1.898 | +25.50 | -27.07 | -0.008941 | +0.0002629 |
| 10.0 | -23.76 | 5.610 | +19.37 | -22.65 | -0.006871 | +0.0001559 |

The gamma=2.0 row reproduces the existing `b2_244`-clean evidence exactly
(checked by diff against `agent/evidence/b2-244-clean-2026-11-12/summary.md`),
confirming this sweep used the same protocol.

## Finding

At every tested gamma from 1.0 to 10.0, PPO's total reward is worse than
`risk_aware_rule_based`'s, by a broadly similar margin (-20 to -32) that does
not trend meaningfully with gamma. Completion time and delivered throughput
degrade by a similar amount at every gamma (19-26 more hours, 22-27 fewer
MB/s than the rule-based baseline, consistently). Latency violation fraction
is lower under PPO at every gamma and the reduction grows mildly with gamma
(-0.0045 at 1.0 to -0.0089 at 5.0), but that reduction is not enough to
recover the reward and throughput cost at any gamma tried.

Gamma is a fixed multiplier on the same violation signal the Lagrangian
multipliers already penalize -- raising it does not change which violations
get penalized, only by how much. Independently confirmed earlier
(`runs/b2_244_clean/seed{0,1,2}/log.csv`): `lambda_latency`/`lambda_risk`
never hit their cap of 2.0 in the gamma=2.0 clean run, so the multipliers
were not saturating. This sweep shows raising the fixed weight on top of
that does not change the outcome either. This is consistent with a separate
finding: part of the `fixed_20` holdout's measured latency violations are
forced by D1's own safety filter overriding the requested delivery rate
upward, not by the policy's own throughput choice -- a violation exogenous
to the action taken is penalized the same as one the policy could have
avoided, regardless of gamma's magnitude.

## Implication

Reweighting the violation term is not the lever that fixes PPO's
over-conservatism on this workload. This motivates charging the Lagrangian
dual update only for the marginal violation an agent's own action adds,
rather than the full violation it observes -- the approach in progress as
the L3 counterfactual-cost change to `agent/ppo.py`.

## Caveat

Single seed (0) per gamma value -- a triage sweep, not a reportable
multi-seed result. Sufficient to rule gamma out as the lever; not sufficient
on its own to characterize variance at each gamma.
