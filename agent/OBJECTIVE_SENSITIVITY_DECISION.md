# Decision record: how much of the result depends on the objective's coefficients

Status: proposed. Not yet reviewed by the other lanes. Nothing here changes a default. The coefficients alpha, beta, gamma,
the latency target and the unloaded latency are [ASSUMED] (`contracts/contracts.py`, `sim/config.py`); this record measures what
the conclusions do when they change.

Code: `agent/objective_sensitivity.py`; tests in `agent/test_objective_sensitivity.py`. Evidence:
`agent/evidence/objective-sensitivity-2026-11-20/` (`runs/` episodes, `analysis.csv`, `headline.md`, `exceptions.md`).

## Decision (method)

- **Every setting is re-run, not rescored.** For each setting of one factor around the default (alpha 1, beta 0.1,
  gamma 10, latency_unloaded_ms 2.0, latency_target_ms 15.0) the `SimConfig` is rebuilt and, under it,
  `DP_hindsight_optimum` is solved again and `risk_aware_rule_based`, `rule_seasonal_forecast (hand-built)`,
  `predictive_controller (history)` and `ORACLE_lookahead` are run again, on the same 20 scenarios of each of the four
  populations (A holdout, B altB, C popC, D popD; evaluation fleet, sigma 0.25, history rows 576).
- **Grid.** beta in {0.01, 0.03, 0.1, 0.3, 1.0}, gamma in {1, 3, 10, 30, 100}, alpha in {0.3, 1, 3},
  latency_target_ms in {12, 15, 18}, latency_unloaded_ms in {1.5, 2.0, 2.5}: 15 distinct settings (the default once).
- **Progress value.** Every policy that uses `progress_value` has `beta / 25` (0.004 at the default). The fixed-0.004
  variant is reported beside it for the four settings with beta other than 0.1.
- **Selector coefficients verified.** `LatencyRule.score` reads `alpha` and `gamma` from `config.reward`, and its
  ceiling from `config.latency_target_ms`; a test checks that the rule built for a setting carries them.
- **Excluded.** The residual Q-policy: it was trained at the default objective and has no price to rescale.
- **Statistics.** Paired by seed, mean +/- sem across scenarios, `t` = mean / sem; reported on the 16 scenarios controllable
  under the primary definition and on all 20.

## Evidence

**1. Ordering stability.** Pooled ordering of the five policies by mean reward under each setting, against the same
population's ordering at the default ("same order" when equal), and the controller's gain over the rule with `t`:

| setting | A primary | A all | B primary | B all | C primary | C all | D primary | D all |
|---|---|---|---|---|---|---|---|---|
| alpha=0.3 | same order; ctrl +14.11, t 10.1 | same order; ctrl +12.10, t 8.2 | same order; ctrl +10.68, t 3.3 | DP>oracle>ctrl>rule>seas; ctrl +8.44, t 2.9 | same order; ctrl +8.73, t 3.0 | same order; ctrl +9.13, t 3.8 | same order; ctrl +22.89, t 4.3 | same order; ctrl +18.74, t 4.0 |
| alpha=3 | DP>oracle>ctrl>seas>rule; ctrl +21.27, t 12.3 | DP>oracle>ctrl>seas>rule; ctrl +18.22, t 9.1 | same order; ctrl +24.27, t 5.3 | same order; ctrl +21.69, t 5.1 | same order; ctrl +10.41, t 10.6 | same order; ctrl +9.41, t 9.7 | same order; ctrl +24.39, t 4.6 | same order; ctrl +22.78, t 4.8 |
| beta=0.01 | DP>oracle>ctrl>seas>rule; ctrl +7.54, t 5.2 | DP>oracle>ctrl>seas>rule; ctrl +5.64, t 3.8 | DP>oracle>ctrl>rule>seas; ctrl +0.48, t 0.2 (t<=2) | DP>oracle>rule>ctrl>seas; ctrl -0.12, t -0.1 (t<=2) | same order; ctrl +1.64, t 10.5 | same order; ctrl +1.46, t 9.3 | same order; ctrl +6.59, t 2.4 | same order; ctrl +4.68, t 1.9 (t<=2) |
| beta=0.03 | DP>oracle>ctrl>seas>rule; ctrl +9.29, t 6.4 | DP>oracle>ctrl>seas>rule; ctrl +7.71, t 5.7 | same order; ctrl +10.51, t 3.3 | same order; ctrl +9.20, t 3.4 | same order; ctrl +4.09, t 9.6 | same order; ctrl +3.68, t 9.1 | same order; ctrl +15.02, t 4.2 | same order; ctrl +13.33, t 4.4 |
| beta=0.3 | same order; ctrl +56.29, t 16.8 | same order; ctrl +50.06, t 11.6 | same order; ctrl +9.30, t 2.1 | DP>oracle>ctrl>rule>seas; ctrl +6.83, t 1.6 (t<=2) | DP>ctrl>oracle>seas>rule; ctrl +6.85, t 2.2 | DP>ctrl>oracle>seas>rule; ctrl +7.80, t 3.0 | same order; ctrl +31.96, t 6.7 | same order; ctrl +26.58, t 5.7 |
| beta=1 | same order; ctrl +150.55, t 39.1 | same order; ctrl +134.55, t 14.8 | same order; ctrl +34.39, t 5.1 | same order; ctrl +31.28, t 4.7 | same order; ctrl +31.03, t 4.8 | same order; ctrl +30.65, t 5.6 | same order; ctrl +36.74, t 6.2 | same order; ctrl +33.55, t 6.3 |
| default | same order; ctrl +15.15, t 10.1 | same order; ctrl +13.01, t 8.3 | same order; ctrl +13.84, t 3.7 | same order; ctrl +11.02, t 3.2 | same order; ctrl +9.67, t 6.9 | same order; ctrl +8.76, t 7.2 | same order; ctrl +17.29, t 4.5 | same order; ctrl +15.24, t 4.6 |
| gamma=1 | DP>oracle>ctrl>seas>rule; ctrl +20.21, t 25.8 | same order; ctrl +17.94, t 12.7 | same order; ctrl +3.42, t 2.5 | same order; ctrl +2.81, t 2.2 | same order; ctrl +2.04, t 3.3 | same order; ctrl +1.85, t 3.6 | same order; ctrl +4.06, t 3.9 | same order; ctrl +3.28, t 3.6 |
| gamma=100 | DP>oracle>ctrl>seas>rule; ctrl +56.27, t 4.3 | DP>oracle>ctrl>seas>rule; ctrl +40.32, t 3.0 | DP>oracle>ctrl>rule>seas; ctrl +15.31, t 0.7 (t<=2) | DP>oracle>ctrl>rule>seas; ctrl +7.23, t 0.4 (t<=2) | same order; ctrl +27.44, t 2.9 | same order; ctrl +23.50, t 3.0 | same order; ctrl +72.44, t 2.8 | same order; ctrl +53.48, t 2.4 |
| gamma=3 | same order; ctrl +19.49, t 19.7 | same order; ctrl +17.39, t 12.6 | same order; ctrl +4.48, t 2.7 | same order; ctrl +3.63, t 2.3 | same order; ctrl +3.06, t 3.6 | same order; ctrl +2.73, t 4.0 | same order; ctrl +10.04, t 4.5 | same order; ctrl +8.38, t 4.4 |
| gamma=30 | same order; ctrl +25.48, t 6.3 | same order; ctrl +21.30, t 5.7 | same order; ctrl +42.57, t 6.5 | same order; ctrl +38.80, t 6.3 | same order; ctrl +18.02, t 4.9 | same order; ctrl +15.87, t 5.1 | same order; ctrl +48.54, t 4.6 | same order; ctrl +44.84, t 5.0 |
| latency_target_ms=12 | DP>oracle>ctrl>seas>rule; ctrl +33.08, t 17.6 | DP>oracle>ctrl>seas>rule; ctrl +28.82, t 11.5 | same order; ctrl +27.67, t 5.3 | same order; ctrl +25.20, t 5.5 | same order; ctrl +10.28, t 5.6 | same order; ctrl +9.02, t 5.7 | same order; ctrl +11.39, t 3.0 | same order; ctrl +11.25, t 3.7 |
| latency_target_ms=18 | same order; ctrl +11.01, t 6.8 | same order; ctrl +9.51, t 6.4 | same order; ctrl +9.91, t 2.9 | DP>oracle>ctrl>rule>seas; ctrl +7.79, t 2.6 | same order; ctrl +5.58, t 2.4 | same order; ctrl +6.23, t 3.1 | same order; ctrl +25.29, t 4.6 | same order; ctrl +21.85, t 4.7 |
| latency_unloaded_ms=1.5 | same order; ctrl +8.71, t 5.7 | same order; ctrl +7.43, t 5.5 | DP>oracle>ctrl>rule>seas; ctrl +10.65, t 2.9 | DP>oracle>ctrl>rule>seas; ctrl +7.89, t 2.4 | same order; ctrl +7.24, t 2.1 | same order; ctrl +7.60, t 2.7 | same order; ctrl +15.17, t 4.1 | same order; ctrl +13.02, t 4.2 |
| latency_unloaded_ms=2.5 | DP>oracle>ctrl>seas>rule; ctrl +33.08, t 17.6 | DP>oracle>ctrl>seas>rule; ctrl +28.82, t 11.5 | same order; ctrl +27.67, t 5.3 | same order; ctrl +25.20, t 5.5 | same order; ctrl +10.28, t 5.6 | same order; ctrl +9.02, t 5.7 | same order; ctrl +11.39, t 3.0 | same order; ctrl +11.25, t 3.7 |

**Every exception** (a pooled ordering that differs from the default's, or a controller `t` not above 2):

- alpha=0.3, population B, all (n=20): pooled order DP>oracle>ctrl>rule>seas (default DP>oracle>ctrl>seas>rule); controller minus rule +8.445 +/- 2.866, t 2.95, better/worse 13/7
- alpha=3, population A, primary (n=16): pooled order DP>oracle>ctrl>seas>rule (default DP>ctrl>oracle>seas>rule); controller minus rule +21.268 +/- 1.728, t 12.31, better/worse 16/0
- alpha=3, population A, all (n=20): pooled order DP>oracle>ctrl>seas>rule (default DP>ctrl>oracle>seas>rule); controller minus rule +18.222 +/- 1.997, t 9.12, better/worse 20/0
- beta=0.01, population A, primary (n=16): pooled order DP>oracle>ctrl>seas>rule (default DP>ctrl>oracle>seas>rule); controller minus rule +7.539 +/- 1.446, t 5.21, better/worse 16/0
- beta=0.01, population A, all (n=20): pooled order DP>oracle>ctrl>seas>rule (default DP>ctrl>oracle>seas>rule); controller minus rule +5.643 +/- 1.498, t 3.77, better/worse 19/1
- beta=0.01, population B, primary (n=16): pooled order DP>oracle>ctrl>rule>seas (default DP>oracle>ctrl>seas>rule); controller minus rule +0.476 +/- 2.592, t 0.18, better/worse 9/7
- beta=0.01, population B, all (n=20): pooled order DP>oracle>rule>ctrl>seas (default DP>oracle>ctrl>seas>rule); controller minus rule -0.123 +/- 2.138, t -0.06, better/worse 11/9
- beta=0.01, population D, all (n=20): pooled order DP>oracle>ctrl>rule>seas (default DP>oracle>ctrl>rule>seas); controller minus rule +4.679 +/- 2.401, t 1.95, better/worse 14/6
- beta=0.03, population A, primary (n=16): pooled order DP>oracle>ctrl>seas>rule (default DP>ctrl>oracle>seas>rule); controller minus rule +9.294 +/- 1.444, t 6.43, better/worse 16/0
- beta=0.03, population A, all (n=20): pooled order DP>oracle>ctrl>seas>rule (default DP>ctrl>oracle>seas>rule); controller minus rule +7.715 +/- 1.359, t 5.68, better/worse 20/0
- beta=0.3, population B, all (n=20): pooled order DP>oracle>ctrl>rule>seas (default DP>oracle>ctrl>seas>rule); controller minus rule +6.829 +/- 4.155, t 1.64, better/worse 11/9
- beta=0.3, population C, primary (n=16): pooled order DP>ctrl>oracle>seas>rule (default DP>oracle>ctrl>seas>rule); controller minus rule +6.846 +/- 3.113, t 2.20, better/worse 12/4
- beta=0.3, population C, all (n=20): pooled order DP>ctrl>oracle>seas>rule (default DP>oracle>ctrl>seas>rule); controller minus rule +7.797 +/- 2.608, t 2.99, better/worse 16/4
- gamma=1, population A, primary (n=16): pooled order DP>oracle>ctrl>seas>rule (default DP>ctrl>oracle>seas>rule); controller minus rule +20.212 +/- 0.784, t 25.79, better/worse 16/0
- gamma=100, population A, primary (n=16): pooled order DP>oracle>ctrl>seas>rule (default DP>ctrl>oracle>seas>rule); controller minus rule +56.271 +/- 13.127, t 4.29, better/worse 16/0
- gamma=100, population A, all (n=20): pooled order DP>oracle>ctrl>seas>rule (default DP>ctrl>oracle>seas>rule); controller minus rule +40.320 +/- 13.449, t 3.00, better/worse 19/1
- gamma=100, population B, primary (n=16): pooled order DP>oracle>ctrl>rule>seas (default DP>oracle>ctrl>seas>rule); controller minus rule +15.310 +/- 20.625, t 0.74, better/worse 10/6
- gamma=100, population B, all (n=20): pooled order DP>oracle>ctrl>rule>seas (default DP>oracle>ctrl>seas>rule); controller minus rule +7.233 +/- 17.389, t 0.42, better/worse 11/9
- latency_target_ms=12, population A, primary (n=16): pooled order DP>oracle>ctrl>seas>rule (default DP>ctrl>oracle>seas>rule); controller minus rule +33.076 +/- 1.883, t 17.57, better/worse 16/0
- latency_target_ms=12, population A, all (n=20): pooled order DP>oracle>ctrl>seas>rule (default DP>ctrl>oracle>seas>rule); controller minus rule +28.820 +/- 2.512, t 11.47, better/worse 20/0
- latency_target_ms=18, population B, all (n=20): pooled order DP>oracle>ctrl>rule>seas (default DP>oracle>ctrl>seas>rule); controller minus rule +7.792 +/- 3.022, t 2.58, better/worse 12/8
- latency_unloaded_ms=1.5, population B, primary (n=16): pooled order DP>oracle>ctrl>rule>seas (default DP>oracle>ctrl>seas>rule); controller minus rule +10.648 +/- 3.610, t 2.95, better/worse 10/6
- latency_unloaded_ms=1.5, population B, all (n=20): pooled order DP>oracle>ctrl>rule>seas (default DP>oracle>ctrl>seas>rule); controller minus rule +7.887 +/- 3.319, t 2.38, better/worse 11/9
- latency_unloaded_ms=2.5, population A, primary (n=16): pooled order DP>oracle>ctrl>seas>rule (default DP>ctrl>oracle>seas>rule); controller minus rule +33.076 +/- 1.883, t 17.57, better/worse 16/0
- latency_unloaded_ms=2.5, population A, all (n=20): pooled order DP>oracle>ctrl>seas>rule (default DP>ctrl>oracle>seas>rule); controller minus rule +28.820 +/- 2.512, t 11.47, better/worse 20/0

- **Counts.** Of 60 setting-by-population cases, 11 (primary) and 13 (all scenarios) have a pooled ordering different
  from the default's. In the primary class, 8 are a swap of the controller and the look-ahead oracle and 3 a swap of
  the seasonal rule and the rule; in the all-scenario class 7 and 5, plus one other (B, beta = 0.01: the rule above
  the controller). `DP_hindsight_optimum` is first in all 120 cases.
- **The default ordering on A is itself a near-tie.** The controller is above the oracle by 0.22 (+15.147 against +14.931);
  in every other setting on A the oracle is above the controller.
- **Controller against the rule.** The controller's gain is positive in 119 of 120 cases (primary and all scenarios; the
  exception is B, beta = 0.01, all scenarios, -0.12). Its `t` is above 2 in 114 of 120 cases. The six cases with `t` not
  above 2: B beta = 0.01 (primary +0.48, t 0.18; all -0.12, t -0.06), B gamma = 100 (primary +15.31, t 0.74; all +7.23,
  t 0.42), B beta = 0.3 (all +6.83, t 1.64), D beta = 0.01 (all +4.68, t 1.95). Population B is the weak one: its
  controllable gain at the default is +13.84 with t 3.7.
- **Scenario by scenario.** The per-scenario ordering equals the same scenario's default ordering in 56.7% (primary) and
  54.5% (all) of the scenario-by-setting cases on average over the non-default settings (`analysis.csv`,
  `share_scenarios_same_order_as_default_scenario`): the pooled ordering is stable, the per-scenario ordering is not,
  because the policies within about one reward unit of each other swap.

**2. Headroom depends on the objective.** `DP_hindsight_optimum` minus the rule, across the 15 settings (default in
parentheses):

| Population | Controllable, % of the rule's cost | Controllable, absolute | All scenarios, % of the rule's cost |
| --- | --- | --- | --- |
| A | 17.0 to 64.1 (25.8) | 8.30 to 151.62 (16.04) | 2.4 to 15.9 (4.1) |
| B | 19.7 to 69.1 (43.8) | 9.49 to 105.85 (34.40) | 4.7 to 15.3 (13.1) |
| C | 8.7 to 40.0 (20.0) | 2.07 to 49.28 (11.50) | 1.7 to 6.9 (4.5) |
| D | 23.8 to 75.7 (42.3) | 16.27 to 155.23 (37.90) | 4.7 to 15.6 (10.1) |

The oracle's share of that headroom is 0.91 to 1.00 (A), 0.84 to 1.00 (B), 0.19 to 1.00 (C), 0.89 to 1.00 (D) on the
controllable scenarios, and the controller's share 0.80 to 0.99 (A), 0.05 to 0.58 (B), 0.46 to 0.91 (C), 0.19 to 0.60 (D).
So the headroom is not a constant of the environment: over the grid it moves by a factor of 3 to 7 in percent of
cost and 10 to 24 in absolute reward, and the oracle's share of it is above 0.9 in most settings (minimum 0.19, population C).

**3. The two latency parameters are one axis.** Only the ratio of the unloaded latency to the target matters:
latency_target_ms = 12 with unloaded 2.0 and latency_target_ms = 15 with unloaded 2.5 have the same ratio (1/6) and give
identical results in every population (for example A, controller +33.08, t 17.6, in both).

**4. The fixed-0.004 variant** (settings with beta other than 0.1; controllable scenarios, controller gain over the rule
with progress value beta / 25 against 0.004): A beta 0.3 +56.29 against +31.93, beta 1 +150.55 against +90.69;
B beta 0.01 +0.48 against +14.87, beta 1 +34.39 against +3.54; C beta 0.3 +6.85 against +26.28, beta 1 +31.03 against +84.42;
D beta 0.01 to 1 +6.59, +15.02, +31.96, +36.74 against +16.76, +16.88, +18.48, +22.64. The sign of the controller's gain is the same in
all 16 cases; the rule `beta / 25` is the one fixed in advance and is the headline.

## Decisions made during execution

- **Latency axes.** The grid includes both latency parameters as specified; the equivalence of target 12 and unloaded 2.5
  was found in the results and is reported, not removed.
- **`t` with zero variance** is reported as infinite (the sign of the mean), not NaN.
- **Ordering is by mean reward** over the scenario class.
- **Scenario flags** come from the `fixed_20` rollout of the same setting and scenario; the primary definition uses the clip
  fraction, which does not depend on the reward coefficients, so the 16 controllable scenarios are the same in every
  setting.

## Limits

- **One factor at a time.** Interactions between coefficients were not explored.
- **20 scenarios per population** and a handful of hindsight-near ties: the oracle and controller orderings flip
  inside about one reward unit.
- **The controllable class has 16 scenarios.** Population B's controller gain has a large standard error in several settings.
- **The progress-value rule `beta / 25`** is a rule of the rule's design, not a tuned value; the fixed variant shows it matters
  for the controller's size of gain.
- **The residual Q-policy** is not in this record.
