# Scenario split of the evaluation harness (holdout and altB)

`baselines.evaluate` now writes, per episode, `filter_forced`, `fixed20_clip_fraction`,
`fixed20_latency_cost`, `exogenous_violations`, `latency_cost` and `data_loss_probability`, and per run
three summary blocks (all, controllable, filter-forced), a Pareto table and a recovery-time distribution
(`pareto.csv`, `recovery_time_distribution.csv`). Definitions are in `baselines/evaluate.py`.

## Commands

    python -m baselines.evaluate --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \
        --seeds 20 --eval-fleet --hazard-error-sigma 0.25 --diagnostics --out agent/evidence/scenario-split-2026-11-19/holdout
    python -m baselines.evaluate --workload-csv data_pipeline/profiles/profile_diurnal_long_altB.csv \
        --seeds 20 --eval-fleet --hazard-error-sigma 0.25 --diagnostics --out agent/evidence/scenario-split-2026-11-19/altB

`profile_diurnal_long_altB.csv` is not on this branch; the file used is the one of
`agent/residual-q-altb-check-2026-11-19` (sha256 `bf19b148bdd47546d1029e8365dfc530dae6635f402ebfa814782ac7073acd7e`,
recorded in the `altB/summary.md` workload section).

## Scenario definition (fixed before any result)

`filter_forced`: the safety filter raises a `fixed_20` rollout's applied action on at least 10% of the
intervals of that scenario. Controllable is the rest. Sensitivity definitions: thresholds 1% and 25%, and
`fixed_20` latency cost above 50.

| Profile | primary (10%) | 1% | 25% | `fixed_20` cost > 50 |
| --- | --- | --- | --- | --- |
| holdout | 3 9 17 19 | 2 3 9 16 17 18 19 | 3 9 17 19 | 9 17 19 |
| altB | 3 9 17 19 | 2 3 9 16 17 18 19 | 3 9 17 19 | 9 17 19 |

The pre-registered expectation was {9, 17, 19} on the holdout and {9, 17} on altB. The primary definition
flags seed 3 in both (clip fraction 0.42; `fixed_20` latency cost 21.8 on the holdout and 0.8 on altB) and
altB seed 19 (clip fraction 0.54; cost 145.5). The definition is kept as registered.

## Results

`ORACLE_lookahead` minus `risk_aware_rule_based`, paired by seed, standard error across scenarios:

| Profile | Block | n | Rule mean reward | Gap | t |
| --- | --- | --- | --- | --- | --- |
| holdout | all | 20 | -330.20 | +11.850 +/- 1.496 | 7.92 |
| holdout | controllable | 16 | -61.10 | +13.753 +/- 1.472 | 9.34 |
| holdout | filter-forced | 4 | -1406.56 | +4.241 +/- 1.858 | 2.28 |
| altB | all | 20 | -172.92 | +29.246 +/- 2.924 | 10.00 |
| altB | controllable | 16 | -78.58 | +33.165 +/- 2.716 | 12.21 |
| altB | filter-forced | 4 | -550.29 | +13.571 +/- 4.571 | 2.97 |

- **Concentration of cost.** Holdout seeds 9, 17, 19 carry 84.2% of the rule's total reward cost and 97.7%
  of its latency cost; altB seeds 9, 17, 19 carry 61.8% and 80.4%.
- **Equal outcomes.** `risk_aware_rule_based` and `workload_only_throttle` have identical episode rewards on
  every scenario of both profiles.
- **Latency cost on holdout seeds 9, 17, 19** (rule, oracle, `fixed_20`, `fixed_50`): 2744 / 2744 / 2743 / 2743,
  1350 / 1350 / 1350 / 1350, 1338 / 1338 / 1336 / 1336.
- **Exogenous violations per episode** (mean over scenarios, as counted; see the limit below): holdout 8.8
  for the rule and the oracle (8.9 for `fixed_20`), altB 4.4 for every policy.

## Limit of `exogenous_violations`

The count uses the hazard field of the observation, which is clipped at 1.8e-4 per hour. On holdout seeds
9 and 17 and altB seeds 9 and 17 the reported hazard is above that, the recovered risk floor is lower than the
environment's, and the count is a lower bound. For `fixed_20`, whose violations all equal the exogenous ones:
holdout 68 and 20 counted against 179 and 91 realised, altB 66 and 9 against 91 and 42. On every other
scenario the count equals the realised violations of `fixed_20`
(`baselines/test_scenario_split.py`). The counterfactual takes the hazard and the remaining work from the
policy's own trajectory.

## Files

`holdout/` and `altB/`: `episodes.csv`, `summary.md`, `pareto.csv`, `recovery_time_distribution.csv`.
