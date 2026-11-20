# Decision record: a predictive-distribution controller inside the rule's selector

Status: proposed. Not yet reviewed by the other lanes. Depends on contract 1.5.0 (`agent/PRE_FAILURE_HISTORY_DECISION.md`,
a proposal needing sign-off); without the history field the controller runs, with a weaker result (below). Nothing here
changes an existing default.

Code: `agent/predictive_controller.py` (`PredictiveController`, `PredictiveControllerNoHistory`, `RuleWithHistory`);
tests in `agent/test_predictive_controller.py`. Evidence: `agent/evidence/predictive-controller-2026-11-19/`.

## Decision

- **The controller.** `PredictiveController` is `RiskAwareRuleBased` with one change, the belief about the next
  interval's foreground load. The selector (progress-value expected cost), the risk floor and all parameters are the
  rule's.
- **Belief.** Three forecasters (last value; AR(3) refit every 12 steps on the last 1500 samples; same time yesterday
  plus the level change over the day) are scored by their squared error over the last 48 observations; the lowest is
  chosen. The belief is the chosen forecast plus each of that forecaster's last 288 realised errors, clipped at zero.
  Before 24 errors exist the belief is the forecast plus {-20, -10, 0, 10, 20} MB/s. Every constant is fixed in the
  module and was not tuned on the evaluation scenarios.
- **History.** With `SimConfig.history_rows` the harness hands the controller the two days before the episode
  (`set_history`); they are replayed through the forecasters so the error pools are full at the first decision.
- **Comparison arms** (same scenarios): `rule+history` (the unchanged rule with its belief buffers pre-filled),
  `predictive_controller[seasonal]+history`, `predictive_controller[adaptive] (no history)`.

## Evidence

**1. Reproduction on the original scenarios** (history_rows = 0, start rows unchanged, the two days before the start
row built by wrap-around with the script `reproduce_prototype_scenarios.py`, which reads the environment's private
start row; excluded seeds {9, 17, 19} holdout, {9, 17} altB; gap over `risk_aware_rule_based`, `v1,eval-fleet,sigma=0.25`):

| Profile | Arm | n | Gain over the rule | Better | Share of the optimum |
| --- | --- | --- | --- | --- | --- |
| holdout | controller with history | 17 | +13.78 +/- 1.40 (t 9.9) | 17/17 | 94.5% |
| holdout | `rule+history` | 17 | +3.96 +/- 1.36 | 13/17 | 27.1% |
| holdout | controller, no history | 17 | +5.88 +/- 1.84 | 15/17 | 40.3% |
| altB | controller with history | 18 | +13.11 +/- 3.42 (t 3.8) | 13/18 | 41.2% |
| altB | `rule+history` | 18 | +0.68 +/- 1.43 | 10/18 | 2.1% |
| altB | controller, no history | 18 | +8.92 +/- 3.64 | 11/18 | 28.1% |

The targets (+13.78 holdout, +13.11 altB, 94.5%, 41.2%) reproduce to the stated precision. The port's belief equals the
earlier implementation's exactly (largest absolute difference 0 over 924 steps for each forecaster mode).

**2. Protocol: the contract's history field, scenarios with start rows moved so the history never wraps**
(`agent/evaluate_predictive.py`, `history_rows = 576`; ten holdout start rows and one altB start row differ from the
original scenarios, `agent/PRE_FAILURE_HISTORY_DECISION.md`; every arm, the rule, the optimum and `OracleLookahead`
included, is evaluated on these scenarios). Gain of `predictive_controller (history)` over `risk_aware_rule_based`:

| Definition of controllable | Holdout n | Gain | Better / worse | Share of optimum | altB n | Gain | Better / worse | Share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all scenarios | 20 | +13.01 +/- 1.58 (t 8.26) | 20 / 0 | 94.1% | 20 | +11.03 +/- 3.44 (t 3.20) | 14 / 6 | 37.0% |
| primary (10%) | 16 | +15.15 +/- 1.50 (t 10.11) | 16 / 0 | 94.4% | 16 | +13.84 +/- 3.76 (t 3.68) | 12 / 4 | 40.2% |
| clip fraction at least 1% | 13 | +14.88 +/- 1.67 (t 8.90) | 13 / 0 | 94.2% | 13 | +12.43 +/- 4.07 (t 3.06) | 10 / 3 | 37.9% |
| `fixed_20` cost above 50 | 17 | +14.83 +/- 1.44 (t 10.28) | 17 / 0 | 94.3% | 17 | +13.89 +/- 3.54 (t 3.93) | 13 / 4 | 41.4% |

- **Violations.** The paired difference of the latency violation fraction is within 0.002 of zero on every
  profile and block (holdout all +0.002, controllable -0.001; altB -0.002).
- **Against the other arms** (primary definition, gain over the rule): holdout `rule_seasonal_forecast` +3.99 +/- 0.20,
  `rule+history` +5.55 +/- 1.44, controller without history +7.17 +/- 1.78, seasonal-only with history +15.15 +/- 1.50
  (identical to the adaptive controller there), `OracleLookahead` +14.93, optimum +16.04. altB: `rule_seasonal_forecast`
  +6.44 +/- 1.51, `rule+history` +0.78 +/- 1.61, controller without history +9.08 +/- 4.05, seasonal-only with history
  +10.98 +/- 4.01, `OracleLookahead` +33.17, optimum +34.40.
- **What carries the gain.** The belief and the history together: the unchanged rule given the same history gains
  +3.96 (holdout) and +0.68 (altB) in the reproduction, and the controller without history +5.88 and +8.92, against
  +13.78 and +13.11 for both together. On the holdout the controller matches `OracleLookahead` (94.4% against 93.1% of
  the optimum), which is possible because the oracle is a one-step greedy reference
  (`baselines/policies.py`, `OracleLookahead`).

## Decisions made during execution

- **Targets reproduced on the original scenarios first, with a labelled private read.** The contract field cannot
  rebuild the wrapped history for start rows below 576, so the reproduction script reads `_offset` and `_trace`; the
  controller and the protocol runs use the field only. Alternative rejected: reporting only protocol numbers, which
  would have left the target comparison undecided (the scenarios differ).
- **Evaluation script lives with the fresh-population evaluation.** `agent/evaluate_predictive.py` needs the scenario
  split and the optimum (#34, #35); this branch stacks on the contract proposal only. The evidence here was produced
  on a local merge of #34, #35, #36 and this branch.
- **No-history arm as a class that does not implement `NeedsHistory`,** so the harness gives it nothing; an instance
  attribute or a flag did not opt out of the runtime protocol check.
- **Parameters unchanged from the earlier implementation.** No tuning was done for these scenarios.

## Limits

- **Selected load.** The holdout and altB aggregates are selected for daily-cycle strength; the seasonal forecaster
  is useful on them because of that selection. A later pre-registered
  evaluation tests it on populations not selected that way.
- **History field needs the contract change.** The result with history is not available without contract 1.5.0.
- **altB is noisy.** The controller is better in 12 of 16 controllable altB scenarios and 13 of 18 in the
  reproduction; its standard error is 3.4 to 3.8.
- **Hand-chosen constants.** Window lengths and offsets are fixed, not derived; their sensitivity was not measured.
- **Two profiles, 20 scenarios each.**
