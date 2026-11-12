# T-lane decision record: a learned controller that beats the rule, and why it is not a network

Status: proposed. Not yet reviewed by the other lanes. The adopted controller is now the PPO
agent in `agent/FORECAST_PPO_DECISION.md`, which beats this controller; this record stays as
the finding that the daily forecast is the exploitable signal, and as its comparator.

Follows `agent/PPO_COLLAPSE_DECISION.md`, which concluded that constrained PPO trails
`risk_aware_rule_based` by more than the headroom on offer. Evidence:
`agent/evidence/seasonal-search-2026-11-10/`.

## Decision

- **Adopted controller:** `SeasonalForecastRule` (`agent/seasonal_rule.py`). This is
  `risk_aware_rule_based` with its one-hour load belief replaced by a forecast from the
  load observed at the same time on the previous day of the same episode.
- **Learning:** three parameters (`progress_scale`, `spread`, `mix`) chosen by the
  cross-entropy method on episode return (`agent/search_seasonal.py`).
- **Information set:** the same as the rule's, meaning observations from the current
  episode only. The forecast is unavailable for the first 24 hours and 1 hour of each
  episode, and the policy is the rule until then.
- **Contract:** reward coefficients are the contract's. No `gamma` override, unlike `b2_244`.
- **Fleet split:** training draws fleets from the train-split table. `--eval-fleet` is used
  at evaluation only. This avoids the open question in `PPO_COLLAPSE_DECISION.md` rather
  than answering it.
- **Validation is time-forward:** the last 3 days of `profile_diurnal_long_train.csv` are
  held back for model selection. Nothing from the holdout is used before the final
  evaluation.
- **Comparator:** the same search over the rule's progress price only (`--base rule`).
  Gain beyond it is attributable to the forecast, not to retuning the baseline.

## Why the forecast

- `profile_diurnal_long` aggregates 44 volumes selected for the strongest daily cycle
  (sidecar: "selected extremes, not a representative sample").
- Next-interval load forecast error (RMSE, MB/s), training half: last value 10.1,
  one-hour median 9.6, same time yesterday 2.7, yesterday plus level change 2.0.
- Step changes correlate at 0.97 with the step changes one day earlier. Days are not
  copies: day-to-day RMSE is 1.7 to 3.6 MB/s.
- The rule's belief has no time-of-day term, which leaves this structure unused.

## Evidence

Configuration `v1,eval-fleet,sigma=0.25`, holdout `profile_diurnal_long_holdout.csv`
(sha256 `94fe6bd6…` (Corrected: this is the LF hash of the committed file; an earlier note quoted the CRLF hash of the same content from a Windows test run that temporarily rewrote it before the LF version was restored. The file content is unchanged.)), 20 paired scenarios (0 to 19).

**Baseline reproduction first:** every row of the baseline harness is identical to
`agent/evidence/b2-3seed-2026-11-07/summary.md`. Rule -330.2, oracle -318.3, so the
oracle headroom is +11.9.

| Policy | Gap vs rule | sem | Scenarios better |
| --- | --- | --- | --- |
| `seasonal_search` seed 0 | +3.27 | 0.25 | 20 / 20 |
| `seasonal_search` seed 1 | +2.86 | 0.24 | 20 / 20 |
| `seasonal_search` seed 2 | +3.68 | 0.25 | 20 / 20 |
| pooled, 60 pairs | **+3.27** | **0.15** | 60 / 60 |
| `rule_progress_search`, 3 seeds | -0.14 to -0.17 | 0.05 to 0.10 | 5 to 7 / 20 |
| hand-built `SeasonalForecastRule` (all parameters neutral) | +3.69 | 0.28 | 20 / 20 |

- **Headroom:** the pooled gain is 28% of the oracle's.
- **Against the retuned rule:** +3.42, +3.00 and +3.86 (sem 0.24 to 0.26).
- **Strata:** every seed is positive in every stratum, the pre-registered
  `initial_budget_infeasible` split and the post-hoc magnitude split alike.
- **Tails:** smallest per-scenario gain +0.56. On the three extreme scenarios every seed
  gains +0.56 to +2.59. Contrast `PPO_COLLAPSE_DECISION.md`, where one PPO seed lost
  -513 there.
- **Latency:** violation fraction 0.045 against the rule's 0.044. Share of episodes
  within the latency budget 0.85, the same as the rule.
- **Completion:** every episode completed, in 41.4 to 41.9 h against the rule's 43.6 h.
- **Selected parameters:** `progress_scale` 0.49 to 0.80, `spread` 0.82 to 0.98, `mix`
  0.82 to 0.98.
- **Validation gaps** (training days 8 to 10, best generation): +3.8, +3.1 and +4.7. The
  comparator's were +1.4, +1.4 and +2.3.

### What this does and does not show

- **The gain is the forecast.** The searched parameters do not beat the untuned
  hand-built variant (+3.27 pooled against +3.69).
- **The progress-price retune did not transfer.** Both searches lowered the progress
  price, which helped on the validation days, and the retuned rule alone is -0.15 on the
  holdout.
- **Report it as:** a forecast from yesterday's load beats the rule on this workload, and
  policy search over a three-parameter forecasting policy reproduces that gain on three
  of three seeds. Do not report that learning found it.

## Rejected alternatives

All development numbers here use the placeholder risk model (`hazard=v0`) and are not
reportable.

- **Per-interval correction network trained with PPO** (`agent/train_residual.py
  --method ppo`). The policy outputs -1, 0 or +1 action steps on top of the rule.
  - It never moved the deterministic policy usefully away from the rule. Exploring the
    faster correction under high load draws violation penalties, its probability falls
    in every state, and the gradient that would raise it again is proportional to that
    probability.
  - This is the same starvation mechanism as in `PPO_COLLAPSE_DECISION.md`.
- **The same network trained on counterfactual outcomes** (`--method counterfactual`,
  one-step action values from the simulator's latency model at the realised load).
  - On validation scenarios from the *same* training days: +9.2 to +14.5.
  - On the holdout: +3.5, -0.04 and -5.9.
  - On later training days, with the time-forward split: -0.4 to -9.7.
  - Gradient-boosted trees on the same targets, restricted to any of four feature
    subsets, also lost on later days (-1.8 to -7.5).
  - Conclusion: per-interval corrections learned on 7 days do not transfer. The code is
    kept so these figures can be reproduced.
- **The development run of this controller** under `hazard=v0` gave +3.45, +3.47 and +3.53
  on the same holdout window.

## Limits

- **Holdout reuse:** the holdout window was evaluated twice in development under
  `hazard=v0`, first for the rejected network and then for this controller. The `hazard=v1`
  configuration was evaluated once.
- **Workload:** one workload family, selected for its daily cycle. On a workload without
  one, the forecast has nothing to use and the policy reduces to roughly the rule.
- **Forecast warm-up:** the forecast needs a day plus an hour of the current episode. The
  rule's mean episode on the holdout is 44 hours, so it is active for less than half of it.
- **Search budget:** three parameters, three seeds, 8 generations (default 12), 12
  candidates of 24 episodes each. None of this is tuned.
- **Training file:** rebuilt as the first 2,880 rows of `profile_diurnal_long.csv`
  (sha256 `65680adf…` (Corrected: this is the LF hash of the committed file; an earlier note quoted the CRLF hash of the same content from a Windows test run that temporarily rewrote it before the LF version was restored. The file content is unchanged.)). The holdout rebuilt the same way matched the original byte for
  byte, but the original training file was not available to compare.
- **Simulator:** the latency model is uncalibrated (`sim/LIMITATIONS.md`). Every figure
  is a property of the model.
