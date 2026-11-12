# Verification of `agent/SEASONAL_SEARCH_DECISION.md` (branch `agent/T9-seasonal-policy-search`)

Status: independent read-only check. The decision record was not edited. This file makes no
recommendation about merging.

Checked commit: `9576b65` (`origin/agent/T9-seasonal-policy-search`, one commit ahead of
`origin/main` at `f689ec6`). Files read in full: `SEASONAL_SEARCH_DECISION.md`,
`seasonal_rule.py`, `search_seasonal.py`, `residual_features.py`, `residual_policy.py`,
`train_residual.py`, `evaluate_learned.py`, `test_learned.py`, and under
`evidence/seasonal-search-2026-11-10/`: `README.md`, `summary.md`, `strata.csv`,
`baseline_reproduction.md`, `episodes.csv`, all six `search.json` and the seed-0 seasonal
`log.csv`. Contrast documents: `agent/PPO_COLLAPSE_DECISION.md` (read from the main
checkout; it is not on the T9 branch). `agent/PPO_COLLAPSE_ADDENDUM.md` is not on disk.

## Scope limit that applies to every row below

`hazard/artifacts/v1/` is gitignored and absent in the clean worktree, so no `hazard=v1`
number in the evidence could be re-run. Every CONFIRMED below rests on the committed evidence
files and on the code, not on a reproduction of the simulation. Two checks that do not need
the artifacts were possible and were done: the SHA-256 values recorded in the evidence match
the local `profile_diurnal_long_train.csv` (`65680adf…` (Corrected: this is the LF hash of the committed file; an earlier note quoted the CRLF hash of the same content from a Windows test run that temporarily rewrote it before the LF version was restored. The file content is unchanged.)) and `profile_diurnal_long_holdout.csv`
(`94fe6bd6…` (Corrected: this is the LF hash of the committed file; an earlier note quoted the CRLF hash of the same content from a Windows test run that temporarily rewrote it before the LF version was restored. The file content is unchanged.)), and all figures were re-derived from `episodes.csv`.

## Claim verification

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| a | Holdout is never read before the final evaluation | CONFIRMED (code); partly COULD NOT VERIFY (development history) | The string `holdout` does not occur in `train_residual.py`, `search_seasonal.py`, `residual_features.py`, `residual_policy.py` or `seasonal_rule.py`. Training and search read only `--workload-csv`, whose default is the train file, and write `trained_on_sha256`. `evaluate_learned.py` is the only module that takes the holdout, and it refuses a checkpoint or search result whose `trained_on_sha256` equals the evaluation file's. The record states the holdout window was also evaluated twice under `hazard=v0` during development; that history is not in the repository and cannot be checked. |
| b | The train/validation split is strictly time-forward | CONFIRMED (mechanism), with two caveats | `split_time_forward` takes the first `N - 864` rows for fitting and the last 864 (`val_days=3`, 288 steps per day) for validation, rebases the validation `t_seconds`, and is tested for contiguity (`test_split_time_forward_is_contiguous_and_rebased`). `SeasonalForecastRule` reads only observations from the current episode (`test_features_are_causal` covers the residual features). Caveats: (1) the rule's progress price and `REF_RATE_MBPS` were tuned on "the training half" of `profile_diurnal_long` (comment in `train_residual.py`), which appears to include the validation days; (2) the environment wraps the trace (`trace_wraps` in `sim/recovery_env.py`), so a roughly 530-step validation episode wraps within the 864-row slice. Wrapping stays inside the validation slice and does not touch fit rows. `val_gap` in `search.json` is a maximum over generations and is biased upward (stated in the module docstring). |
| c | `--eval-fleet` is applied at evaluation only | CONFIRMED for the search and evaluation runs; COULD NOT VERIFY for the rejected residual networks | All six `search.json` files have `"eval_fleet": false`, and `search_seasonal.py` has no `--eval-fleet` flag at all. The recorded commands in `README.md` pass `--eval-fleet` only to `baselines.evaluate` and `agent.evaluate_learned`. `train_residual.py` refuses `--eval-fleet` unless `--allow-eval-fleet-training` is also given. No invocation or checkpoint of a residual-network run is committed, so how those were trained is not checkable. |
| d | The searched parameters (+3.27 pooled) do not beat the untuned hand-built variant (+3.69) | CONFIRMED | `summary.md` and `episodes.csv`: hand-built +3.685 (sem 0.276); seeds +3.273, +2.862, +3.684. Per-pair, searched minus hand-built is -0.412 (sem 0.072, 60 pairs), so on this window the search is slightly worse than its starting point. `search.json` best generations are 7, 2 and 2 of 8, i.e. seeds 1 and 2 kept a generation near the initial distribution. The search's initial `mix` is `sigmoid(1.1) = 0.75`, not the hand-built 1.0, so the search does not start at the hand-built variant. |
| e | The rejected PPO-correction network fails by "the same starvation mechanism" as `PPO_COLLAPSE_DECISION.md` | COULD NOT VERIFY as stated (similar in description, different in detail) | See the comparison below the table. |
| f | Headline +3.27 pooled, 60 pairs | CONFIRMED, with a statistical caveat | Recomputed from `episodes.csv`: mean paired gap 3.2731, 60 of 60 pairs positive, minimum 0.562, rule -330.20, oracle -318.35, headroom 11.850 (28% share = 27.6%). The record's sem 0.15 (0.1466) treats the 60 pairs as independent, but the three seeds share the same 20 scenarios. Averaging the three seeds per scenario gives sem 0.242 on 20 scenarios, which is the honest uncertainty. The gain is still large against that. |

Further figures checked and matching the record: selected parameters (`progress_scale` 0.49–0.80,
`spread` 0.82–0.98, `mix` 0.82–0.98); validation gaps +3.78, +3.07, +4.66 against comparator
+1.36, +1.37, +2.34; gain over the retuned rule +3.42, +3.00, +3.86; completion 41.4–41.9 h against
43.6 h; violation fraction 0.0455 / 0.0454 / 0.0457 against 0.0444; 20 of 20 completed; gains on
the three extreme scenarios (9, 17, 19) between +0.56 and +2.59 for every seed.

### Mechanism comparison for row e

- **`PPO_COLLAPSE_DECISION.md` (constrained PPO, 4 actions):** early exploration sampled the 120 MB/s
  action in all load states, it violated 67–72% of the time, and its probability fell to about
  0.1% and did not recover after the pressure relaxed. The record attributes the pressure to the
  reward's `gamma` term, not the multiplier. The seed-1 tail failure on the extreme scenarios is a
  different thing: it is over-aggressive allocation (66–79 MB/s requested), not starvation.
- **Residual PPO (`train_residual.py`, `--method ppo`):** the policy chooses -1, 0 or +1 steps on
  top of the rule. The stated reason it did not move is that exploring the faster correction under
  high load draws violation penalties, its probability falls in every state, and the gradient that
  would raise it where it is right is proportional to that probability.
- **What is common:** a rarely profitable action is penalised in aggregate while the policy is
  state-blind, its probability falls, and the policy-gradient signal to restore it shrinks with its
  probability. In that sense the two descriptions are the same family.
- **What differs or is missing:** the residual policy starts at the rule (no-change probability
  about 0.69) and corrects by one index, rather than choosing from a full tier. The "gradient is
  proportional to probability" argument appears only in the residual description, and neither record
  tests it. The residual-PPO runs are not committed (no logs, no checkpoint, numbers under
  `hazard=v0`), so the statement rests on the docstring and decision-record prose. The earlier
  record itself says the cause of PPO's failure to find a selective rule "is consistent with the
  evidence without being demonstrated by it".

## Test run

Clean worktree at `9576b65` (detached), Python 3.11. `pip install -r requirements.txt` was not run:
the branch has no `requirements.txt`, and the existing environment already provides the packages.

Command: `python -m pytest -q -p no:cacheprovider`

Result: **1711 passed, 30 skipped, 4 warnings** in 254 s (a repeat gave 1711 passed, 30 skipped in
278 s). `agent/test_learned.py` alone: **18 passed**, 0 failed, 0 skipped. No failures.

The 30 skips, by reason (from `pytest -rs`):

- 11: `hazard/artifacts/v1/` or the v1 fleet table absent (gitignored; generated by `python -m hazard.fit_v1`).
- 9: slow tests (`--runslow` or `BASELINES_RUN_SLOW`).
- 7: `ISM_ENV_FACTORY` not set (adapter conformance has no environment to check).
- 3: `backblaze_filtered.parquet` or `backblaze_drive_index.parquet` not built.

`agent/test_learned.py` uses `v0_config` throughout, so its 18 passes do not exercise the v1 risk model.
Because the worktree lacks the untracked `profile_diurnal_long_{train,holdout}.csv` of the main checkout,
the evidence files were checked by hash against the main checkout's copies, not by re-running them.

## What this adds to the project

Three results now sit next to each other.

1. **Constrained PPO** (`PPO_COLLAPSE_DECISION.md`) trails the rule by about -11 reward on the
   held-out window after its defects were fixed, against a total oracle headroom of 11.85, and the
   largest action was starved early in training. It is evaluated on the time-forward holdout window
   with held-out fleets, so "out of distribution" here means a later time window and a different fleet
   table. Its observation is the contract's 12 fields with no load history, so it had no access to
   the daily structure at all. One of its three seeds lost -513 on the three extreme scenarios through
   over-aggressive allocation, which is a separate failure from the starvation.
2. **A hand-built controller with a one-day-lagged forecast** beats the rule by +3.3 to +3.7 reward on
   every one of 20 scenarios, including the three extremes (+0.56 to +2.59), with violation fraction
   0.0455 against 0.0444 (within noise), and recovers 28% of the oracle headroom. That is a gain of
   about 1% of the rule's reward. The tuned parameters add nothing over the hand-built version
   (-0.41, sem 0.07), so the gain is the forecast. The record's own wording, "do not report that
   learning found it", is right.
3. **Two attempts to learn a correction on top of the rule** did not help. The supervised
   counterfactual network, given the seasonal features, reached +9.2 to +14.5 on validation scenarios
   drawn from its own training days and then +3.5, -0.04 and -5.9 on the holdout and -0.4 to -9.7 on
   later training days: ordinary overfitting to 7 days. The PPO-trained correction network did not
   move away from the rule at all. That is a failure to learn, not a failure to generalise, and it is
   described in prose only. "Both fail to generalise past the training days" is therefore accurate for
   the counterfactual network and not for the PPO one.

**Does this support the thesis "exploiting known structure generalises where learning the same thing
end-to-end does not, at this data scale"?** Partly. It supports a narrower statement: on this one
workload family, in this uncalibrated simulator, a controller that is given the daily cycle as a
parameter-free forecast beats the rule and beats PPO, while the learned alternatives tried were worse
than the rule and failed to improve on it with 7 to 10 days of data. It does not yet isolate "structure
versus learning", for these reasons:

- **Information set.** PPO's policy never saw yesterday's load. The only learned model given the
  lagged features is the residual network, and it was trained on 7 days. The comparison is between
  different inputs as much as different methods.
- **The workload was selected for the property exploited.** `profile_diurnal_long` aggregates 44 volumes
  chosen for the strongest daily cycle, and its sidecar says "selected extremes, not a representative
  sample". A forecast built on daily periodicity is expected to win there.
- **The searched variant is not "learning".** Three parameters, 8 generations, and slightly worse than
  the untuned start. The positive result rests on the hand-built forecast, which was designed after
  inspecting the training half (RMSE and correlation figures in the record).
- **Holdout exposure.** The holdout window was evaluated under `hazard=v0` while the controller and
  the rejected network were being developed, and the development run gave +3.45 to +3.53. The `v1`
  evaluation is the first under that configuration, not the first on that window.
- **Statistical unit.** The three seeds reuse the same 20 scenarios; the effective sample is 20.
- **Contrast with PPO is not like for like.** The PPO runs trained with `--eval-fleet`, the open
  contamination question in `PPO_COLLAPSE_DECISION.md`, and at `gamma=2.0`; the seasonal search did
  neither.
- **Simulator.** The latency model is uncalibrated, so every figure is a property of the model.

Evidence that would strengthen the thesis:

- A workload without a daily cycle (or the same trace with the day-to-day phase scrambled) where the
  forecast has nothing to use. The prediction is that the controller falls back to roughly the rule's
  score; if it still gains, the gain is not from periodicity.
- PPO and the residual network trained with the lagged features available, and with more training
  days, to give a learning curve against the number of days. If they close the gap with data, the
  thesis becomes "data scale", not "structure versus learning".
- More than one time-forward holdout window, and a fresh holdout not used in development.
- The residual-network logs, checkpoints and invocations committed, so the "overfitting" and
  "failure to learn" findings can be audited.
- Re-running the PPO results without `--eval-fleet` in training.

Evidence that would break it:

- A learned model with the same inputs and enough training days matching or exceeding the
  hand-built controller on the holdout.
- The seasonal gain disappearing on a second window, or on the periodicity-scrambled control.
- The gain shrinking under a calibrated latency model.
