# T-lane decision record: a deterministic residual Q-policy; wins on its own population, reverses on an independent one

Status: the holdout win reverses on an independent disk population. On
`profile_diurnal_long_holdout.csv`, which shares its 44 disks with the training data, the policy
beats `SeasonalForecastRule` by +3.83 ± 1.07 (t 3.57). Zero-shot on `profile_diurnal_long_altB.csv`
(150 disks, none of the 44, a disjoint 14-day window; section 3b) it loses to
`SeasonalForecastRule` by -49.93 ± 8.09 (t -6.17) and to `risk_aware_rule_based` by
-45.47 ± 7.51 (t -6.05), and each of the three models loses on its own.
Diagnosis (section 3b): the bandwidth level matches, so this is not a load-scale shift. altB
differs in its IOPS inputs (67.8% of its states have a feature outside the training range, 59.6%
from `fg_read_iops`) and in day-to-day repeatability (the load one day earlier predicts the
next interval with RMSE 13.45 MB/s against 2.74 on the training profile, worse than the last
value at 9.02). The policy's same-time-yesterday features carry a property of the 44 selected
disks that altB does not have. Not yet reviewed by the other lanes. Nothing here changes an
existing default.

Code: `agent/residual_q.py`; `--q-model` added to `agent/evaluate_learned.py` and
`agent/evaluate_ood.py`; tests in `agent/test_residual_q.py`. Evidence:
`agent/evidence/residual-q-2026-11-14/`.

## Decision

- **The policy.** `ResidualQPolicy` takes the action `SeasonalForecastRule` (the hand-built
  forecast rule) would take, and may move it one index down or one index up.
- **Safety mask.** Moves that would exceed the causal latency ceiling of
  `sim.latency_filter` are masked. The ceiling is computed from the last observed
  foreground load; keeping the rule's action is never masked.
- **Deterministic by construction.** The policy takes the move with the largest predicted
  advantage when that advantage exceeds a margin, and keeps the rule's action otherwise.
  The greedy action of the value function is the policy, so there is no sampled-versus-
  argmax distinction (`FORECAST_PPO_DECISION.md`, "Correction"). With a margin above every
  prediction it is `SeasonalForecastRule` exactly (tested).
- **Learning.** Fitted Q on a one-step decision with exact paired returns.
  - The foreground load is exogenous, and an interval's latency depends only on that
    interval's allocation. So once an interval has executed, the one-interval value of each
    move at the realised load is known exactly, on the identical scenario.
  - Each move is scored as `-(alpha * D + gamma * V) + progress_value * delivered`: the
    contract's latency terms with contract coefficients, plus the rule's own price for
    rebuild progress.
  - Targets are the advantages of moving down and moving up over keeping the rule's action,
    so everything the moves share cancels in the target. Two gradient-boosted tree
    regressors fit them from the residual features built on `SeasonalForecastRule`.
- **Data.** 80 rollouts of the rule itself on the first 7 training days, with a random
  allowed move on 10% of intervals. The final model is refitted on those rollouts plus 24
  rollouts of training days 8 to 10.
- **Selection.** The margin is chosen from {0.1, 0.25, 0.5} by the paired gap over
  `SeasonalForecastRule` on 40 scenarios of training days 8 to 10.
- **Training conditions.** Train-split fleets, `hazard_error_sigma=0.25`, contract reward
  coefficients. Nothing from any evaluation workload was used in design or selection.
- **Holdout use.** `profile_diurnal_long_holdout.csv` was evaluated once, after all three
  main models were trained.

How this differs from the earlier failures:

- **vs. the per-interval correction network** (`train_residual.py`): that network
  corrected the plain rule, chose by argmax with no margin, had no latency mask, and was
  validated on its own training days. This corrects the forecast rule, deviates only above
  a margin selected on later days, and cannot request an action above the causal ceiling.
- **vs. PPO** (`PPO_COLLAPSE_DECISION.md`, `FORECAST_PPO_DECISION.md`): no policy gradient,
  no Lagrange multiplier, no stochastic policy. The paired target removes the scenario
  variance that dominated every PPO return (below).

## Evidence

### 1. Variance decomposition (committed `forecast-ppo-2026-11-11/episodes.csv` and `-12-stochastic/episodes.csv`)

| Quantity | Value |
| --- | --- |
| Between-scenario sd of `risk_aware_rule_based` return (holdout) | **706.3** |
| Share of return variance explained by scenario (rule, hand-built rule, 3 PPO seeds) | ~100% |
| sd across scenarios of the paired gap, PPO minus rule (argmax / sampled) | 6.03 / 10.32 |
| Between-seed sd within a scenario (argmax / sampled) | 0.48 / 6.60 |
| Typical 17 scenarios only: scenario / policy / policy x scenario share | 49% / 24% / 27% |

- **What the 6.03 is:** it is often quoted as "between-scenario sd", but it is the sd of the
  paired gap, which is already net of scenario effects.
- **What an unpaired return carries:** the scenario sd of 706, against a target effect of
  about 8.
- **This method's target is paired** move by move on the identical interval, so that
  component is zero by construction.

### 2. Where the headroom is (training days only; 30 scenarios, seeds 6000 to 6029)

On `profile_diurnal_long_train.csv`, the oracle beats `SeasonalForecastRule` by +12.24 per
episode:

| Part of the episode | Oracle edge over the forecast rule | Forecast rule edge over the rule |
| --- | --- | --- |
| first 25 h (no same-time-yesterday value exists yet) | +5.30 | -0.01 |
| after 25 h | +6.93 | +2.13 |

### 3. Holdout, once (`holdout/`)

Configuration `v1,eval-fleet,sigma=0.25`, 20 paired scenarios (0 to 19). Rule -330.196,
`SeasonalForecastRule` +3.685, oracle +11.850.

| Policy | vs `SeasonalForecastRule` | Scenarios better/worse | vs rule | Violation fraction |
| --- | --- | --- | --- | --- |
| `residual_q` seed 0 (margin 0.1) | +3.375 | 17 / 3 | +7.060 | 0.0464 |
| `residual_q` seed 1 (margin 0.25) | +3.254 | 18 / 2 | +6.939 | 0.0460 |
| `residual_q` seed 2 (margin 0.1) | +4.850 | 19 / 1 | +8.535 | 0.0462 |
| **seeds averaged per scenario** | **+3.827 ± 1.071 (t = 3.57)** | 18 / 2 | **+7.512 ± 1.155** | |
| `SeasonalForecastRule` | | | +3.685 | 0.0462 |
| `risk_aware_rule_based` | | | | 0.0444 |

- **Standard error:** sems are across the 20 scenarios after averaging the three seeds
  within each scenario. The between-seed sd within a scenario is 1.23.
- **Headroom:** +7.512 is 63% of the gain of `OracleLookahead` (+11.850), a one-step greedy reference and not an upper
  bound; see the correction below for the exact optimum and the controllable scenarios.
- **Worst scenario:** the lowest seed-averaged scenario gap is -0.51.

### 3b. Independent disk population, zero-shot (`altB-2026-11-19/`)

`profile_diurnal_long_altB.csv`: 150 Tencent volumes chosen by the same diurnal rule, none of the 44
of `profile_diurnal_long`, over the 14-day window 2020-06-23 to 2020-07-07, which shares no sample time
with it. The window and the volume set were fixed and written before any reward on it existed. The
three main models were evaluated as they are, without retraining, with the same configuration as
section 3 (`v1,eval-fleet,sigma=0.25`, 20 scenarios). Seeds averaged per scenario.

| Policy | vs `SeasonalForecastRule` | Scenarios better/worse | vs rule | Violation fraction |
| --- | --- | --- | --- | --- |
| `residual_q` seed 0 (margin 0.1) | -55.22 | 2 / 18 | -50.76 | 0.0319 |
| `residual_q` seed 1 (margin 0.25) | -36.38 | 4 / 16 | -31.92 | 0.0289 |
| `residual_q` seed 2 (margin 0.1) | -58.20 | 2 / 18 | -53.74 | 0.0326 |
| **seeds averaged per scenario** | **-49.93 ± 8.09 (t = -6.17)** | 2 / 18 | **-45.47 ± 7.51 (t = -6.05)** | |
| `SeasonalForecastRule` | | | +4.46 ± 2.01 (t = 2.22, 15 / 20 scenarios) | 0.0221 |
| `risk_aware_rule_based` | | | | 0.0226 |

Oracle headroom on both populations (per-seed paired differences, 20 scenarios):

| | original profile (44 disks) | altB (150 disks) |
| --- | --- | --- |
| oracle vs `SeasonalForecastRule` | +8.165 ± 1.362 (t = 6.00) | +24.787 ± 2.371 (t = 10.46) |
| oracle vs `risk_aware_rule_based` | +11.850 ± 1.496 (t = 7.92) | +29.246 ± 2.924 (t = 10.00) |
| `SeasonalForecastRule` vs `risk_aware_rule_based` | +3.685 ± 0.276 (t = 13.36, 20 / 20) | +4.459 ± 2.008 (t = 2.22, 15 / 20) |

- **Headroom.** altB has 2.5 to 3 times the exploitable headroom of the original profile on both
  bases. The narrow ceiling of section 3 is a property of that one disk population.
- **Where the loss is.** Of the -49.93, -50.25 is the latency terms and +0.32 the degraded-state
  term. The policy takes 4.38 more violations per episode than the rule (18 of 20 scenarios worse),
  which is -43.83 at `gamma = 10`; the rest is the degradation term (-6.41). It moves away from the
  rule's action on about 20% of intervals (47.3 up, 46.0 down per episode), and 65% of the gap
  accrues before step 288 (-32.29 before, -17.64 after, both significant). On the holdout the violation
  fraction was 0.0462 for both policies; here it is 0.0312 against 0.0221.
- **Not a load-scale shift.** Foreground MB/s: mean 69.8 / 70.2 / 63.9, p90 94.4 / 95.0 / 94.0, p99
  107.7 / 107.3 / 108.2, max 116.8 / 116.3 / 120.8 (train / holdout / altB). No interval of any profile is
  above the 182.2 MB/s threshold. The median of altB is lower (63.8 against 78.0 and 78.9).
- **Feature coverage.** Against the min and max of the training feature vectors, 1.0% of holdout states
  and 67.8% of altB states have at least one feature outside the range. `fg_read_iops` is outside in
  59.6% (mean read IOPS 28,126 against 4,536; write IOPS 57,524 against 5,794), `level_change_vs_yday` in
  10.8%, the 12 load-change features in 1.0% to 3.4% each. Trees treat a value outside the training range
  as the nearest extreme bin.
- **Repeatability.** RMSE of the same-time-yesterday load against the next interval: 2.74 / 2.01 / 13.45 MB/s
  (train / holdout / altB), against 10.12 / 10.36 / 9.02 for the last value. Correlation of step changes
  with the step changes one day earlier: 0.970 / 0.987 / 0.585. The same-time-yesterday features
  (`residual_features.py`, "Why same time yesterday") rest on the first two figures.
- **What this does not separate.** The two differing input groups, the IOPS features and the
  yesterday features, were not ablated, so the loss is not attributed to one of them. The figures
  do not distinguish "learned something specific to the 44 disks" from "relied on inputs whose
  relation to the target differs". No feature-scaling change is indicated, because the bandwidth
  scale matches.
- **Harness.** With `margin = 1e9` the policy on altB is identical to `SeasonalForecastRule` in actions and
  per-step rewards on all 20 seeds, and in every column returned by `evaluate()` except the policy name.

### 4. Mechanism (holdout, per-policy totals from `holdout/episodes.csv`)

Reward gap versus `SeasonalForecastRule`, split into the rebuild-time term (`beta` x
intervals) and the latency term (`alpha * D + gamma * V` summed):

| Policy | Gap | Rebuild time | Latency | Intervals | Violations per episode | Delivered MB/s |
| --- | --- | --- | --- | --- | --- | --- |
| seed 0 | +3.38 | +1.92 | +1.46 | -19.2 | -0.20 | 86.1 |
| seed 1 | +3.25 | +0.27 | +2.98 | -2.8 | -0.25 | 83.3 |
| seed 2 | +4.85 | +1.89 | +2.96 | -18.9 | -0.30 | 86.1 |
| `SeasonalForecastRule` | | | | | | 82.9 |

- **What the policy does:** it finishes faster *and* violates less. It speeds up where the
  forecast rule's uncertainty-padded belief is too cautious, and steps down ahead of
  violations the forecast rule walks into.
- **Where it moves** (inner validation days, margin 0.5): 50→80 MB/s and 80→120 MB/s up
  moves were 224 of 455 moves, 80→50 MB/s down moves 72. 334 of the 455 moves fell in the
  first 25 h, where the forecast rule is still the plain rule.

### 5. Replication on the three rolling-origin folds (`sfold1/` to `sfold3/`)

Retrained from scratch on each `profile_diurnal_long_sfold<k>_train.csv`, same protocol,
evaluated on `profile_diurnal_long_sfold<k>_holdout.csv`. All four partitions are windows of the same
14 days of the same 44 disks. These folds test which days are held out within one 44-disk population;
they do not test generalisation across populations. Section 3b shows the result they replicate did not
hold on another population:

| Fold | vs `SeasonalForecastRule` | t | Scenarios better | Per seed | vs rule | Oracle vs rule |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | +3.87 ± 1.09 | 3.53 | 18/20 | +3.79 / +3.01 / +4.80 | +7.27 | +11.66 |
| 2 | +5.87 ± 1.58 | 3.71 | 18/20 | +5.43 / +6.50 / +5.69 | +9.45 | +13.37 |
| 3 | +4.13 ± 1.22 | 3.40 | 17/20 | +4.63 / +3.66 / +4.12 | +7.82 | +11.85 |
| main split | +3.83 ± 1.07 | 3.57 | 18/20 | +3.38 / +3.25 / +4.85 | +7.51 | +11.85 |

- **Consistency:** every one of the 12 seed-by-partition combinations beats
  `SeasonalForecastRule`.
- **Ceiling:** every result is below the gain of its own `OracleLookahead`, which is a one-step greedy reference, not a
  ceiling (the exact hindsight optimum is higher by about +1 on the holdout).
- **Latency:** the violation fraction equals the forecast rule's within 0.0003 on every fold.
- **Not cross-validation:** the folds overlap by about 75%, so this is a
  boundary-sensitivity check, as in `FORECAST_PPO_DECISION.md`.
- **Not independent evidence of generalisation:** every fold shares the 44 disks and the window of
  `profile_diurnal_long`. Agreement across the folds measured sensitivity to the cut point within that
  population. It did not predict the altB result.

### 6. Out of distribution (`ood/`)

Main models, `agent/evaluate_ood.py` (synthetic days generated independently, the 6 h/day
phase shift, the degraded fleet), plus the real Tencent holdouts from
`agent/evidence/workload-selection-check-2026-11-12/`. Seeds averaged per scenario.

| Condition | Q vs rule | Q vs `SeasonalForecastRule` | Violation fraction (Q / forecast rule / rule) |
| --- | --- | --- | --- |
| holdout, phase shift | **+4.88 ± 2.51** | **+8.07 ± 2.47** | 0.038 / 0.038 / 0.035 |
| holdout, degraded fleet (0.9) | **+6.33 ± 1.19** | **+2.83 ± 1.08** | 0.096 / 0.096 / 0.092 |
| synthetic `idle` | -0.02 | -0.02 | 0 / 0 / 0 |
| synthetic `diurnal`, days differ | -14.20 ± 2.88 | -9.68 ± 1.90 (20/20 worse) | 0.028 / 0.026 / 0.025 |
| synthetic `bursty` | -55.45 ± 8.75 | -17.63 ± 6.39 | 0.032 / 0.021 / 0.018 |
| synthetic `peak` | -11.86 ± 1.33 | -11.08 ± 1.21 | 0.162 / 0.161 / 0.161 |
| real `bursty_long` holdout | -27.22 ± 4.54 | -23.29 ± 4.68 (18/20 worse) | 0.032 / 0.028 / 0.028 |
| real `idle_long` holdout | -0.03 | -0.03 | 0 / 0 / 0 |

- **No unsafe failure.** On sustained peak load the violation fraction is 0.162 against
  0.161 for both rules. Compare 0.977 and -9,283 for forecast-PPO.
- **Phase shift:** it survives, and beats the forecast rule by more than at home.
- **Loads without a usable daily cycle:** it loses to both rules, by 10 to 55 against the
  rule. That is far less than forecast-PPO's -293 to -350 on the same real bursty volumes,
  but it is a loss.

## Correction (2026-11-20): the basis of "63% of the oracle headroom" and the controllable-scenario figures

- **The basis.** The 63% is 7.512 / 11.850: the gain over the rule divided by the gain of `OracleLookahead`, over all
  20 holdout scenarios. `OracleLookahead` knows the next interval's load but plans one interval at a time
  (`baselines/policies.py`, its docstring), so it is a reference and not an upper bound. The exact hindsight optimum
  (`DP_hindsight_optimum`, `agent/OPTIMUM_DECISION.md`) gains +12.863 over the rule on the same scenarios, so the
  residual Q-policy captures 58% of it.
- **The scenarios.** Three holdout seeds (9, 17, 19) carry 84% of the rule's total cost and every policy pays nearly
  the same there, because the safety filter forces the allocation (`agent/SCENARIO_SPLIT_DECISION.md`). On the 16
  scenarios controllable under the registered definition (the filter raises a `fixed_20` rollout's applied action on
  fewer than 10% of the intervals; seeds 3, 9, 17, 19 excluded):

| Quantity (holdout, 3 models averaged per scenario) | All 20 | Controllable 16 |
| --- | --- | --- |
| residual Q minus rule | +7.512 +/- 1.155 (t 6.50) | +8.834 +/- 1.216 (t 7.26) |
| residual Q minus `SeasonalForecastRule` | +3.827 +/- 1.071 (t 3.57) | +4.753 +/- 1.234 (t 3.85) |
| `OracleLookahead` minus rule | +11.850 | +13.753 |
| hindsight optimum minus rule | +12.863 | +14.836 |
| residual Q as a share of the `OracleLookahead` gain | 63% | 64% |
| residual Q as a share of the optimum's gain | 58% | 60% |

- **Reading.** The share is about 60% on the controllable scenarios too, so the correction changes the label ("oracle
  headroom" is a one-step reference) and not the size. The result on the independent population (altB, above) is
  unaffected: -49.93 +/- 8.09 against the forecast rule on all scenarios.
- Sources: this file's holdout evidence for the residual Q-policy; `agent/evidence/scenario-split-2026-11-19/holdout/`
  (scenario flags, `OracleLookahead`) and `agent/evidence/optimum-2026-11-19/holdout_dp/` (optimum) of the
  evaluation-fixes branches (#34, #35).

## Verdict

- **On the population it was trained on, the policy beats `rule_seasonal_forecast`; on an
  independent population it loses to it** (section 3b: -49.93 ± 8.09, t -6.17). The claims below
  hold for the 44-disk `profile_diurnal_long` population and are not established beyond it.
  - **Main holdout:** +3.83 ± 1.07 (t 3.57, 18 of 20 scenarios). That is +7.51 over
    `risk_aware_rule_based`, 63% of the gain of `OracleLookahead` (58% of the exact hindsight optimum's gain; see the
    correction below).
  - **Replication:** the result repeats on all three rolling-origin folds (+3.87, +5.87,
    +4.13; 12 of 12 seed-partitions positive).
  - **Mechanism:** it finishes the rebuild faster and violates the latency limit less, both
    measured.
- **It is not a general controller.** It is unsafe nowhere tested, but it loses to both
  rules on workloads without a usable daily cycle, including real bursty Tencent volumes, and on
  altB, whose aggregate is selected by the same daily-cycle rule but repeats from day to day far less
  (step-change correlation 0.585 against 0.970). The claim is scoped to the 44-disk population of
  `profile_diurnal_long`.
- **What did not work, and why:**
  - **PPO-family methods** (`PPO_COLLAPSE_DECISION.md`, `L3_DECISION.md`,
    `BARRIER_DECISION.md`, `FORECAST_PPO_DECISION.md`): their returns carry the scenario
    variance (sd 706) against an effect of about 8, and their stochastic policies did not
    hold up under sampling.
  - **The per-interval correction network on the plain rule** (`train_residual.py`):
    validated on its own training days, with no margin and no mask, so it overfit and
    deviated too freely.
  - **The three-parameter policy search** (`search_seasonal.py`): its parameters did not
    transfer.
  - **Within this method,** a margin of 0 (deviate whenever the predicted advantage is
    positive) was less consistent on the first inner-validation check (34 of 40 scenarios
    better, against 37 of 40 at 0.25), and is excluded from the grid.

## Limits

- **One-step reduction.** A move changes the future only through the remaining rebuild
  work, which the progress price values approximately. Reported results are full episode
  rewards from the environment, so this can cost performance but cannot inflate a number.
- **Value-based RL in its simplest form.** This is fitted Q on a one-step decision with
  full counterfactual feedback, made possible by the exogenous load. A multi-step
  bootstrapped variant was not tried.
- **Workload scope.** It loses out of distribution on non-periodic load (section 6). A
  support guard (keep the rule's action when the state is outside the training data's
  range) is the obvious next step. It was not added here, because the out-of-distribution
  sets have now been seen and would have to be replaced by fresh ones to test it honestly.
- **Inner-validation selection.** The margin grid was trimmed from {0, 0.1, 0.25, 0.5, 1.0}
  to {0.1, 0.25, 0.5} after the first inner-validation check (training days 8 to 10 only).
  Validation gaps (+6.9 to +10.5) are maxima over the grid and biased upward. The holdout
  gaps are lower.
- **Seeds.** Seeds vary the rollout and exploration randomness and the tree seeds, not the
  data window. Between-seed sd within a scenario is 1.2.
- **One training population.** Every trained model, every fold and the holdout use the same 44
  disks. Generalisation to unseen disks was tested once (section 3b) and failed. A claim of
  generalisation needs a test on a population no model or selection has touched; the altB profile has
  now been evaluated and cannot serve as that test again.
- **Holdout history.** The holdout window has shaped earlier lanes' work
  (`FORECAST_PPO_DECISION.md`, Limits). This method was designed from the training days,
  the inner validation days and the committed decision records, and evaluated on the
  holdout once. The rolling folds and the out-of-distribution sets were evaluated after
  that single holdout evaluation, not used to change the method.
- **Serialisation.** `model.pkl` is a pickle of scikit-learn regressors; load only files
  this module wrote.
- **Simulator.** The latency model is uncalibrated (`sim/LIMITATIONS.md`).

## Commands

Training (each seed about 6 minutes on one core with `OMP_NUM_THREADS=1`):

    python -m agent.residual_q --seeds 0 1 2 \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_train.csv --out runs/residual_q/main
    python -m agent.residual_q --seeds 0 1 2 \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_sfold<k>_train.csv --out runs/residual_q/sfold<k>

Holdout and folds:

    python -m agent.evaluate_learned --q-model runs/residual_q/main/seed0 runs/residual_q/main/seed1 runs/residual_q/main/seed2 \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \
        --seeds 20 --eval-fleet --hazard-error-sigma 0.25 --diagnostics --out runs/eval_residual_q
    (and the same with --workload-csv data_pipeline/profiles/profile_diurnal_long_sfold<k>_holdout.csv)

Out of distribution:

    python -m agent.evaluate_ood --q-model runs/residual_q/main/seed0 runs/residual_q/main/seed1 runs/residual_q/main/seed2 \
        --holdout-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv --seeds 20 --shifted-sigmas \
        --extra-csv agent/evidence/workload-selection-check-2026-11-12/profile_bursty_long_holdout.csv \
                    agent/evidence/workload-selection-check-2026-11-12/profile_idle_long_holdout.csv \
        --out runs/eval_ood_residual_q

The three main models are committed under `training/main/seed<k>/`. Pass those directories
to `--q-model` to re-evaluate without retraining.
