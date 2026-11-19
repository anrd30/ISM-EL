# Verdict against the pre-registration

Criterion as registered:

> On each of C and D, on the scenarios controllable under the primary definition: (1) t = mean / sem of the per-scenario gain (total reward of predictive_controller (history) minus that of risk_aware_rule_based) is greater than 2; and (2) the mean over the same scenarios of (latency_violation_fraction of the controller minus that of the rule) is not greater than 0. The criterion is met only if both conditions hold on both C and D.

## diurnal_long_popC

| definition | n | better / worse | mean gain | sem | t | violation diff (sem) | share of optimum | condition 1 (t > 2) | condition 2 (viol <= 0) |
|---|---|---|---|---|---|---|---|---|---|
| primary | 16 | 16 / 0 | +9.669 | 1.407 | 6.87 | -0.00008 (0.00011) | 84.1% | True | True |
| clip_0.01 | 13 | 13 / 0 | +9.983 | 1.696 | 5.89 | -0.00010 (0.00014) | 83.0% | True | True |
| clip_0.25 | 16 | 16 / 0 | +9.669 | 1.407 | 6.87 | -0.00008 (0.00011) | 84.1% | True | True |
| fixed20_cost | 16 | 16 / 0 | +9.669 | 1.407 | 6.87 | -0.00008 (0.00011) | 84.1% | True | True |
| all | 20 | 20 / 0 | +8.764 | 1.222 | 7.17 | +0.00369 (0.00238) | 71.4% | True | False |

## diurnal_long_popD

| definition | n | better / worse | mean gain | sem | t | violation diff (sem) | share of optimum | condition 1 (t > 2) | condition 2 (viol <= 0) |
|---|---|---|---|---|---|---|---|---|---|
| primary | 16 | 16 / 0 | +17.293 | 3.860 | 4.48 | -0.00219 (0.00057) | 45.6% | True | True |
| clip_0.01 | 13 | 13 / 0 | +17.544 | 4.254 | 4.12 | -0.00234 (0.00060) | 46.8% | True | True |
| clip_0.25 | 16 | 16 / 0 | +17.293 | 3.860 | 4.48 | -0.00219 (0.00057) | 45.6% | True | True |
| fixed20_cost | 16 | 16 / 0 | +17.293 | 3.860 | 4.48 | -0.00219 (0.00057) | 45.6% | True | True |
| all | 20 | 20 / 0 | +15.237 | 3.282 | 4.64 | -0.00075 (0.00090) | 43.2% | True | True |

## Verdict (primary definition, every fresh population)

**PASS**: diurnal_long_popC: t = 6.87 (condition 1 met), violation difference -0.00008 (condition 2 met); diurnal_long_popD: t = 4.48 (condition 1 met), violation difference -0.00219 (condition 2 met)
