# Decision record: the predictive controller on two fresh populations (pre-registered)

Status: proposed. Not yet reviewed by the other lanes. Nothing here changes an existing default. The criterion, the
populations and the protocol were committed before any evaluation on C or D: `agent/PREREGISTRATION_2026-11-19.md`,
commit `e32dcfd2a7f0be7a2994d14b68d62e632f62d2a9` (pushed to `origin/eval/fresh-populations-2026-11-19`). Created by a
local merge of the exact-optimum branch (#35) into the predictive-controller branch (#37); the PR is based on #37.
Depends on contract 1.5.0 (#36, a proposal needing sign-off).

Code: `agent/evaluate_predictive.py`, `agent/preregistered_verdict.py`,
`data_pipeline/tencent/build_profile_altB.py` (extended: `--population altB|C|D`); tests in
`agent/test_evaluate_predictive.py`, `agent/test_preregistered_verdict.py`,
`data_pipeline/tests/test_fresh_populations.py`. Evidence: `agent/evidence/fresh-populations-2026-11-19/`.

## Verdict, verbatim against the pre-registration

Criterion as registered:

> On each of C and D, on the scenarios controllable under the primary definition: (1) t = mean / sem of the
> per-scenario gain (total reward of predictive_controller (history) minus that of risk_aware_rule_based) is greater
> than 2; and (2) the mean over the same scenarios of (latency_violation_fraction of the controller minus that of the
> rule) is not greater than 0. The criterion is met only if both conditions hold on both C and D.

**PASS** (primary definition, 16 controllable scenarios of 20 on each population): `diurnal_long_popC`: t = 6.87
(condition 1 met), violation difference -0.00008 (condition 2 met); `diurnal_long_popD`: t = 4.48 (condition 1 met),
violation difference -0.00219 (condition 2 met).

| Population | n | Better / worse | Gain over the rule | t | Violation difference (sem) | Share of the optimum |
| --- | --- | --- | --- | --- | --- | --- |
| C | 16 | 16 / 0 | +9.669 +/- 1.407 | 6.87 | -0.00008 (0.00011) | 84.1% |
| D | 16 | 16 / 0 | +17.293 +/- 3.860 | 4.48 | -0.00219 (0.00057) | 45.6% |

Sensitivity variants (both conditions hold on both populations under each): clip fraction at least 1% (13
scenarios) C +9.983 (t 5.89), D +17.544 (t 4.12); at least 25% (16) as primary; `fixed_20` cost above 50 (16) as
primary. **All scenarios (not the criterion): C +8.764 +/- 1.222 (t 7.17, 20/0), violation difference +0.00369:
condition 2 is not met on all scenarios of C; D +15.237 +/- 3.282 (t 4.64, 20/0), violation difference -0.00075.**
On C the controller's violation fraction exceeds the rule's in the four filter-forced scenarios (+0.0025, +0.0082,
+0.044, +0.0205 for seeds 3, 9, 17, 19) while its reward is higher there (+6.48, +0.93, +5.27, +7.89).

## Decision

- **Populations** (`PREPROCESSING.md` B5c; sidecars list the volumes):
  - C: `profile_diurnal_long_popC.csv`, the diurnal rule, 14 volumes (p95 100.476 MB/s, mean 65.82), window
    2020-07-22 04:00 to 2020-08-05 03:55 UTC (5311 volumes with a gap-free run; 194 excluded as volumes of A and altB;
    pool 4813 after the eligibility rule). sha256 in the manifest.
  - D: `profile_diurnal_long_popD.csv`, 336 volumes in the order of `numpy.random.default_rng(20261119)` (p95 100.415,
    mean 74.76), window 2020-06-08 04:00 to 2020-06-22 03:55 UTC (3818 volumes with a run; 208 excluded; pool 3455), no
    ranking by daily-cycle strength.
  - Both are disjoint from A and altB (and from each other) in window and volumes (tests).
- **One evaluation per population,** `python -m agent.evaluate_predictive --workload-csv <csv> --seeds 20 --eval-fleet
  --hazard-error-sigma 0.25 --q-model <3 residual-Q run dirs> --out <dir>`, history_rows = 576.

## Evidence

Gain over `risk_aware_rule_based`, primary definition (16 scenarios), paired by scenario; learned seeds averaged first:

| Policy | C gain +/- sem (t; better/worse) | D gain +/- sem (t; better/worse) |
| --- | --- | --- |
| `rule_seasonal_forecast (hand-built)` | +3.454 +/- 0.361 (9.57; 16/0) | -21.429 +/- 6.049 (-3.54; 2/14) |
| residual Q-policy, zero-shot (3 models) | -9.100 +/- 2.360 (-3.86; 3/13) | -62.575 +/- 11.126 (-5.62; 0/16) |
| `rule+history` | +0.506 +/- 1.561 (0.32; 7/9) | +3.509 +/- 1.819 (1.93; 11/3) |
| `predictive_controller[seasonal]+history` | +9.692 +/- 1.423 (6.81; 16/0) | +17.274 +/- 4.748 (3.64; 13/3) |
| controller without history | +2.882 +/- 3.275 (0.88; 10/6) | +15.450 +/- 3.099 (4.99; 14/2) |
| **`predictive_controller (history)`** | **+9.669 +/- 1.407 (6.87; 16/0)** | **+17.293 +/- 3.860 (4.48; 16/0)** |
| `ORACLE_lookahead` (not deployable) | +9.912 +/- 1.664 | +36.832 +/- 4.046 |
| `DP_hindsight_optimum` (not deployable) | +11.498 +/- 1.662 | +37.904 +/- 4.074 |

- **Against the seasonal rule.** The controller beats `rule_seasonal_forecast` by +6.215 +/- 1.165 (t 5.34, 16/0) on C and
  +38.723 +/- 6.728 (t 5.76, 15/1) on D.
- **The earlier winners fail on new populations.** On D, whose volumes are not selected for a daily cycle, the
  seasonal rule loses 21 to the baseline and the residual Q-policy loses 63 in every scenario; on C the residual
  Q-policy loses 9.
- **What carries the gain.** On D the history is not needed for the gain (the controller without history gains
  +15.45) and the belief is (the unchanged rule with the history gains +3.51). On C the history matters: without it
  the gain is +2.88 (t 0.88); with it +9.67.
- **Headroom.** The share of the optimum is 84% on C and 46% on D; the look-ahead oracle captures 86% and 97%. On D
  most of the headroom over the rule is not reached.

## Decisions made during execution

- **D's N rule.** "Add disks in sampled order until the aggregate p95 reaches 100 MB/s" was implemented as the rule the
  other profiles use: the largest prefix of the sampled order whose aggregate p95 is at most 100 x 1.005 MB/s
  (`tune_n`, mode `not_above`), recorded in the pre-registration. Alternative rejected: stopping at the first prefix
  at or above the target, which would take a different N.
- **C has 14 volumes.** The diurnal ranking puts the highest-bandwidth volumes first, so the p95 ceiling is reached
  after 14; recorded, not adjusted.
- **Comparator code from the residual-Q branch.** `agent/residual_q.py`, its tests, the three committed models and the
  `--q-model` additions to `evaluate_learned.py` and `evaluate_ood.py` were taken verbatim from
  `agent/residual-q-policy` (not merged into `main`), so the zero-shot comparator runs here.
- **The verdict script was written after the evaluation started,** from the registered text; it is pinned by tests on
  synthetic frames and on the committed reference-population evidence (holdout t 10.11, altB t 3.68). Nothing in the
  evaluation depends on it.
- **No re-run, no changed setting, seed, policy or definition** after the pre-registration commit.

## Limits

- **Same data source.** C and D are other 14-day windows and other volumes of the same Tencent trace; the sampling
  interval and the simulator's latency model (uncalibrated, `sim/LIMITATIONS.md`) are shared. The result is not a
  statement about other traces or a real array.
- **Four forced scenarios per population,** the same seeds (3, 9, 17, 19) because the fleet and failure streams depend
  on the scenario seed; the controllable subsets are 16 scenarios each.
- **All-scenario violation fraction on C.** The registered criterion is on controllable scenarios; on all scenarios of
  C the controller's mean violation fraction is above the rule's by 0.0037.
- **Needs the history field.** The headline requires contract 1.5.0; without history the controller gains +2.88 on
  C (not significant) and +15.45 on D.
- **Share of the optimum on D is 46%.** The gain is significant, and a large part of the headroom is not reached.
