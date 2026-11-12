# `b2_244`, `--eval-fleet` contamination question resolved — clean re-run

Resolves the open question in `agent/evidence/b2-3seed-2026-11-07/README.md`
("Reportability: conditional, one question open"): the original `b2_244` run
passed `--eval-fleet` to both training and evaluation, so training sampled
fleets from the held-out-split feature table rather than the train-split one.
Whether that counted as leakage was left to the RL-agent lane to rule on.

This re-run drops `--eval-fleet` from both steps instead of ruling on it, so
the question doesn't need a ruling: training now draws fleets from the
train-split table throughout, matching every other evaluated checkpoint in
this project.

Training, three seeds:

    python -m agent.train --iterations 244 --rollout-steps 8192 --seeds 0 1 2 \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_train.csv \
        --hazard-error-sigma 0.25 --reward-gamma 2.0 \
        --out runs/b2_244_clean

Evaluation, all three checkpoints against every baseline on identical scenarios:

    python -m agent.evaluate_policy \
        --checkpoint runs/b2_244_clean/seed{0,1,2}/ckpt_final.pt \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \
        --seeds 20 --out runs/eval244/b2_3seed_clean --diagnostics \
        --hazard-error-sigma 0.25

Run on 2026-11-12. Workload sha256 `94fe6bd6f8a5cb163cba7a9fafac1274d4ec6511113920a63d4993f03295a53d`
(the committed LF holdout file — no CRLF-provenance risk here, this is a fresh run).

Contents: `summary.md` and `episodes.csv` as written by the evaluation
harness; `seed{0,1,2}/log.csv` the per-iteration training logs.

## Finding

Paired difference, `ppo_lagrangian` seed vs. `risk_aware_rule_based`, total
reward, n=20 scenarios (from `summary.md`):

| seed | mean_diff | sem |
|---|---|---|
| 0 | -20.13 | 1.764 |
| 1 | -16.98 | 1.471 |
| 2 | -39.50 | 3.463 |

All three seeds lose to the rule baseline by several SEMs — not noise. Every
seed also shows negative `latency_degradation_mean` and
`latency_violation_fraction` deltas (i.e. fewer violations than the rule) but
positive `mean_diff` on `completion_hours` and negative on
`mean_delivered_mbps` (slower, less throughput) and a small positive
`p_failure_mean` delta. Net: `ppo_lagrangian` trades throughput and speed for
a marginally safer, much slower delivery, and the contract reward scores
that trade as a net loss against the hand-built rule.

This is a clean, unambiguous number: no `--eval-fleet` reportability caveat
applies to it.
