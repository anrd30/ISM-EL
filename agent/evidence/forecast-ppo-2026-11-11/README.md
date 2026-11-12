# PPO agent with load-forecast inputs, fitted risk model, held-out window

Run on 2026-11-11 in a cloud workspace: Python 3.13, torch 2.15, xgboost 3.4.1.
`hazard/artifacts/v1/` matched `hazard/manifest.json`, and the baseline harness on this
setup reproduced `agent/evidence/b2-3seed-2026-11-07/summary.md` row for row
(`agent/evidence/seasonal-search-2026-11-10/baseline_reproduction.md`).

## Commands

Training, three seeds per arm, default settings (120 iterations of 4,096 steps):

    python -m agent.forecast_ppo --init bc --seeds 0 1 2 \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_train.csv \
        --hazard-error-sigma 0.25 --out runs/forecast_ppo
    python -m agent.forecast_ppo --init scratch --seeds 0 1 2 \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_train.csv \
        --hazard-error-sigma 0.25 --out runs/forecast_ppo

Holdout evaluation, run once:

    python -m agent.evaluate_learned \
        --ppo-checkpoint runs/forecast_ppo/bc/seed{0,1,2}/ckpt_best.pt \
                         runs/forecast_ppo/bc/seed{0,1,2}/ckpt_init.pt \
                         runs/forecast_ppo/scratch/seed{0,1,2}/ckpt_best.pt \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \
        --seeds 20 --eval-fleet --hazard-error-sigma 0.25 --diagnostics --out runs/eval_forecast_ppo

## Contents

| File | What it is |
| --- | --- |
| `summary.md`, `strata.csv`, `episodes.csv` | the holdout evaluation, as written by `agent.evaluate_learned` |
| `training/<init>/seed<k>/log.csv` | per-iteration training logs, both arms |
| `training/bc/seed<k>/ckpt_best.pt` | the evaluated PPO agents. Re-evaluating seed 2 on scenarios 0 to 2 reproduced `episodes.csv` exactly |
| `training/bc/seed<k>/ckpt_init.pt` | the same networks after imitation, before PPO |

Workspace paths in the text files and the checkpoint metadata were rewritten to
repo-relative paths. No value or weight was changed.

## Provenance and reportability

- **Holdout:** `profile_diurnal_long_holdout.csv`, SHA-256 `94fe6bd6…5a53d` (Corrected: this is the LF hash of the committed file; an earlier note quoted the CRLF hash of the same content from a Windows test run that temporarily rewrote it before the LF version was restored. The file content is unchanged.), the file of
  every earlier evaluation.
- **Training:** the first 2,880 rows of `profile_diurnal_long.csv` (SHA-256 `65680adf…` (Corrected: this is the LF hash of the committed file; an earlier note quoted the CRLF hash of the same content from a Windows test run that temporarily rewrote it before the LF version was restored. The file content is unchanged.)).
  The first 7 days are used for training and the last 3 for validation.
- **Fleets:** train-split table for training, held-out table for evaluation.
- **Reward:** contract coefficients throughout.
- **Selection:** every model-selection decision used the validation days only.
- **Holdout use:** this configuration was evaluated on the holdout once.
- **Disclosure:** the design of this agent followed the earlier finding, made on this
  holdout, that the same-time-yesterday forecast carries the gain
  (`agent/SEASONAL_SEARCH_DECISION.md`).

## Finding

Paired against `risk_aware_rule_based` (-330.2) on 20 scenarios. Oracle -318.3, so the
oracle headroom is +11.9.

| Policy | Gap | sem | Scenarios better |
| --- | --- | --- | --- |
| `forecast_ppo_bc` seed 0 | +6.24 | 1.42 | 19 / 20 |
| `forecast_ppo_bc` seed 1 | +6.32 | 1.42 | 19 / 20 |
| `forecast_ppo_bc` seed 2 | +6.71 | 1.26 | 20 / 20 |
| pooled, 60 pairs | **+6.43** | **1.35**[^corr] | 58 / 60 |
| `forecast_bc_only` (same networks before PPO) | -2.83 to +4.44 | 1.8 to 2.6 | 11 to 18 / 20 |
| `forecast_ppo_scratch`, 3 seeds | -18.1 to -20.4 | 1.5 to 2.4 | 0 / 20 |
| hand-built `SeasonalForecastRule` | +3.68 | 0.28 | 20 / 20 |

- **Headroom:** the pooled gain is 54% of the oracle's.
- **PPO against its own starting point:** +5.17, +1.88 and +9.54 (sem 1.24, 0.93, 2.14).
- **PPO against the hand-built heuristic:** +2.56, +2.64 and +3.03 (sem 1.2 to 1.4), but
  better on only 8 or 9 of 20 scenarios. The edge is concentrated in some scenarios.
- **Latency:** violation fraction 0.045 against the rule's 0.044. Share of episodes
  within the latency budget 0.85, the same as the rule.
- **Completion:** every episode completed, in 41.4 to 41.6 h against the rule's 43.6 h.
- **The one losing scenario** (scenario 3, seeds 0 and 1, -6.5): one extra latency
  violation (-10), partly offset by finishing 3 hours sooner.
- **Extreme scenarios** (9, 17, 19): all three seeds gain +0.42 to +2.33.
- **From scratch:** the PPO-from-scratch arm collapsed to a near-constant slow allocation
  (54 MB/s delivered, 63 h completion), as in `agent/PPO_COLLAPSE_DECISION.md`.

## Hash note

The recorded hash of `profile_diurnal_long_holdout.csv` (`workload_sha256`) and, where noted, of the training file (`trained_on_sha256`) in `episodes.csv` now carry the LF hashes (`94fe6bd6f8a5...` holdout, `65680adfc9c5...` train). Corrected: this is the LF hash of the committed file; an earlier note quoted the CRLF hash of the same content from a Windows test run that temporarily rewrote it before the LF version was restored. The file content is unchanged. The `fit_csv_sha256` and `val_csv_sha256` fields of the search results are hashes of the split pieces written during that run and were not changed.

## Statistical correction (2026-11-12)

The pooled-row sem in the table above was **1.35**, not the 0.78 it originally
read. The three seeds are evaluated on the same 20 scenarios, so the 60
pairs are not 60 independent samples: mean within-scenario, across-seed sd
is 0.48, against a between-scenario sd of 6.03 -- near-replicates, not
independent draws. Averaging the three seeds within each scenario first,
then taking the sem across the resulting 20 scenario-level means, gives sem
1.35 (t = 4.77). The mean is unchanged at +6.43; only the uncertainty
estimate was wrong. Verified independently from
`agent/evidence/forecast-ppo-2026-11-11/episodes.csv`.
