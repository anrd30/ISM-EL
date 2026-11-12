# Forecast PPO agent outside its training distribution

Produced by `agent/evaluate_ood.py` from the three committed checkpoints in
`agent/evidence/forecast-ppo-2026-11-11/training/bc/`. Configuration: fitted risk model,
held-out fleet table, `hazard_error_sigma=0.25` unless the condition says otherwise, 20
paired scenario seeds (0 to 19) per condition.

## Command

    python -m agent.evaluate_ood \
        --ppo-checkpoint agent/evidence/forecast-ppo-2026-11-11/training/bc/seed0/ckpt_best.pt \
                         agent/evidence/forecast-ppo-2026-11-11/training/bc/seed1/ckpt_best.pt \
                         agent/evidence/forecast-ppo-2026-11-11/training/bc/seed2/ckpt_best.pt \
        --holdout-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \
        --seeds 20 --out runs/eval_ood

Needs `hazard/artifacts/v1/` and the holdout CSV; does not need the Tencent pipeline
outputs. The synthetic and phase-shifted workloads are generated deterministically into
`runs/eval_ood/workloads/` and are not committed. About 10 minutes on two CPU cores.

## Reproducibility

- **Run twice from the committed checkpoints:** `gaps.csv` and all 960 rows of
  `episodes.csv` were identical, apart from the output path in `workload_source`.
- **Paths:** workspace paths in `episodes.csv` were rewritten to repo-relative paths. No
  value was changed.

## Conditions

| Condition | What changes from training | Same-time-yesterday vs last-value forecast RMSE (MB/s) |
| --- | --- | --- |
| `synthetic_<profile>_4day` | 4 independently generated synthetic days of each `workload.generate` profile; no day repeats | idle 1.96 vs 1.39, diurnal 3.62 vs 2.42, bursty 47.0 vs 34.5, peak 11.4 vs 5.7 |
| `holdout_phase_shift` | holdout with day d shifted 6 h x d, so yesterday's same time is a different time of day | 34.7 vs 10.4 |
| `holdout_sigma_0`, `holdout_sigma_0.5` | risk-estimate error changed from 0.25 | as holdout, 2.0 vs 10.3 |
| `holdout_degraded_fleet` | every fleet pinned at degradation level 0.9 | as holdout |

- **Synthetic profiles:** the committed 24 h profiles are deliberately not used. An
  episode lasts about 44 h and the simulator loops a trace past its end, so on a one-day
  trace yesterday is an exact copy of today and any daily forecast is perfect by
  construction.
- **Sigma conditions:** sigma changes only the true failure draws, and no drive failed in
  any of these scenarios. These two conditions are therefore uninformative.

## Finding

Gap = total reward minus `risk_aware_rule_based`. Three PPO seeds:

| Condition | Rule | PPO agent gap | Hand-built forecast rule gap | Oracle gap |
| --- | --- | --- | --- | --- |
| holdout (sigma 0 / 0.5, identical to the main result) | -330.2 | +6.24 / +6.32 / +6.71 | +3.69 | +11.85 |
| holdout, degraded fleet | -634.8 | +4.16 / +4.28 / +4.81 | +3.50 | +9.49 |
| holdout, phase shift | -285.8 | -35.40 / -10.47 / -17.71 | -3.18 | +19.41 |
| synthetic diurnal | -186.4 | -23.43 / -2.54 / -13.86 | -4.52 | +1.05 |
| synthetic bursty | -311.3 | -196.4 / -174.3 / -190.3 | -37.82 | +186.8 |
| synthetic peak | -2952 | -9283 (all seeds) | -0.78 | +5.41 |
| synthetic idle | -33.4 | 0 | 0 | 0 |

- **Where it holds:** the agent's gain holds only where the daily pattern holds and load
  stays near the training range. It survives a risk-side shift (degraded fleets).
- **Where it fails:** it loses on every workload without a usable daily pattern.
- **Synthetic `peak`:** mean load is 147 MB/s, above the training maximum of about
  117 MB/s, and the agent violates latency on 97.7% of intervals.
- **Synthetic `bursty`:** its violation fraction is 0.068 to 0.084, against the rule's
  0.018.
- **The hand-built forecast rule** degrades far less, because it keeps the rule's latency
  model.
