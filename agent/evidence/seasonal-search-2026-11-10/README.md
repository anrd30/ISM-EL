# Seasonal-forecast rule with policy search, fitted risk model, held-out window

Run on 2026-11-10 in a cloud workspace, Python 3.13, xgboost 3.4.1 (the ensemble was fitted
with 3.2.0). `hazard/artifacts/v1/` matched every SHA-256 in `hazard/manifest.json`.

## Commands

Baseline reproduction, before any learned policy was run:

    python -m baselines.evaluate --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \
        --seeds 20 --eval-fleet --hazard-error-sigma 0.25 --diagnostics --out runs/baseline_v1

Searches, three seeds per base (`--generations 8`; the default is 12):

    python -m agent.search_seasonal --base seasonal --seeds 0 1 2 --generations 8 \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_train.csv \
        --hazard-error-sigma 0.25 --out runs/search
    python -m agent.search_seasonal --base rule --seeds 0 1 2 --generations 8 \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_train.csv \
        --hazard-error-sigma 0.25 --out runs/search

Holdout evaluation, run once:

    python -m agent.evaluate_learned \
        --search-result runs/search/{seasonal,rule}/seed{0,1,2}/search.json \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \
        --seeds 20 --eval-fleet --hazard-error-sigma 0.25 --diagnostics --out runs/eval_search

## Contents

| File | What it is |
| --- | --- |
| `baseline_reproduction.md` | the baseline harness on this setup; every row is identical to `agent/evidence/b2-3seed-2026-11-07/summary.md` |
| `summary.md`, `strata.csv`, `episodes.csv` | the holdout evaluation, as written by `agent.evaluate_learned` |
| `search/<base>/seed<k>/search.json`, `log.csv` | each search's selected parameters, provenance and per-generation log |

Absolute workspace paths in these files were rewritten to repo-relative paths. No value
was changed.

## Data provenance

- **Holdout:** `profile_diurnal_long_holdout.csv`, SHA-256 `2c83b9db…a59b1`. Byte-identical
  to the file every earlier evaluation used. It was rebuilt from the committed
  `profile_diurnal_long.csv` as the last 1,152 rows, with `t_seconds` rebased and CRLF line
  endings, and the hash matched.
- **Training:** the first 2,880 rows of `profile_diurnal_long.csv`, rebuilt the same way,
  SHA-256 `96cfe789…072a4`. The original training file was not available to compare
  against.
- **Fleets:** training used the train-split table (no `--eval-fleet`). Evaluation used the
  held-out table.

## Reportability

There is no workload leakage and no fleet-table overlap between training and
evaluation. Contract reward coefficients throughout. This holdout configuration
(`v1,eval-fleet,sigma=0.25`) was evaluated once.

The holdout *window* was evaluated twice before, under the placeholder risk model
(`hazard=v0`), during development: once for the rejected network and once for this
controller. `agent/SEASONAL_SEARCH_DECISION.md` states this.

## Finding

Paired against `risk_aware_rule_based` (-330.2) on 20 scenarios. Oracle -318.3, so the
oracle headroom is +11.9.

| Policy | Gap | sem | Scenarios better |
| --- | --- | --- | --- |
| `seasonal_search` seed 0 | +3.27 | 0.25 | 20 / 20 |
| `seasonal_search` seed 1 | +2.86 | 0.24 | 20 / 20 |
| `seasonal_search` seed 2 | +3.68 | 0.25 | 20 / 20 |
| pooled, 60 pairs | +3.27 | 0.15 | 60 / 60 |
| `rule_progress_search`, 3 seeds | -0.14 to -0.17 | 0.05 to 0.10 | 5 to 7 / 20 |
| hand-built `SeasonalForecastRule` | +3.69 | 0.28 | 20 / 20 |

- The pooled gain is 28% of the oracle headroom.
- Seasonal search beats the retuned rule of the same seed by +3.42, +3.00 and +3.86
  (sem 0.24 to 0.26).
- The smallest per-scenario gain of any seasonal seed is +0.56.
- On the three extreme scenarios (9, 17, 19) every seed gains +0.56 to +2.59.
- Latency violation fraction is 0.045 against the rule's 0.044. The share of episodes
  meeting the latency budget is 0.85, the same as the rule.
- Every episode completed the rebuild. Completion time is 41.4 to 41.9 h, against the
  rule's 43.6 h.
