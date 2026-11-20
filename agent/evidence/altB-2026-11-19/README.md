# altB: zero-shot evaluation of the residual Q-policy on an independent disk population

Profile `data_pipeline/profiles/profile_diurnal_long_altB.csv` (4032 rows, 14 days, six-column replay
schema, `t_seconds` from 0) with sidecar `profile_diurnal_long_altB.meta.json`, built by
`data_pipeline/tencent/build_profile_altB.py`. The residual Q-policy models
(`agent/evidence/residual-q-2026-11-14/training/main/seed{0,1,2}`) were evaluated on it without any
retraining. No model was fitted, selected or changed in this folder.

## What the profile is

| Item | Value |
| --- | --- |
| Window | 2020-06-23 04:00:00 UTC to 2020-07-07 03:55:00 UTC (`start_ts` 1592884800, `end_ts` 1594094100) |
| Reference window (`diurnal_long`) | 2020-07-07 04:00:00 UTC to 2020-07-21 03:55:00 UTC. The two windows share no sample time; altB ends one step before it starts |
| Window rule | the start covered by a gap-free run of 4032 samples in the most volumes among starts whose window shares no sample time with the reference window: 6046 volumes (the reference window is covered by 6469) |
| Volumes | 150 selected from 5622 ranked candidates (`disk_ids.csv`, role `selected`) |
| Excluded | the 44 volumes of `profile_diurnal_long.meta.json` (`disk_ids.csv`, role `excluded_reference_disk`) removed from the pool before ranking |
| Overlap with the 44 | 0 volumes. Without the exclusion the same window selects 109 volumes, 40 of them from the 44 |
| Construction | the diurnal rule of `build_profiles.py` (highest `daily_strength`, the median-bytes-per-I/O eligibility rule, N the largest whose aggregate p95 is at most 100 MB/s x 1.005), `validate_replay_frame` before writing |
| Aggregate | p95 99.48 MB/s, mean 63.90 MB/s, max 120.79 MB/s |
| sha256 | `profile_diurnal_long_altB.csv` `bf19b148bdd47546d1029e8365dfc530dae6635f402ebfa814782ac7073acd7e`; `.meta.json` `1253581950a01bf938cddf28aa967f1b3215b3376b987d74308e0de61954e6a0` |

The profile is evaluated whole. No train/holdout split of altB exists in the repository; the sidecar's
`suggested_split` is metadata only.

## Order of operations

From file modification times (local time, 2026-11-19, UTC+05:30) and the session's command order:

1. 22:08:42 `build_profile_altB.py` written.
2. Before step 3, two things were inspected, neither containing a reward: the per-start coverage table of
   14-day windows (disk counts and time overlap with the reference window only), and a dry run of the
   builder (`--dry-run`) that reported the selection with and without the exclusion (150 volumes with it;
   109 volumes, 40 shared with the 44, without it).
3. 22:09:54 `profile_diurnal_long_altB.csv`, `profile_diurnal_long_altB.meta.json` and the `B5b` section of
   `data_pipeline/PREPROCESSING.md` written. The window and the volume set are fixed from here on.
4. 22:14:59 `episodes.csv`, `strata.csv` and `summary.md` of the zero-shot evaluation written
   (`runs/check_residual_q_altB/`), 5 minutes after step 3.
5. 22:44 to 22:49 the diagnostics in `diagnostics/` were computed.

No altB reward existed when the window and the volume set were fixed. The exclusion of the 44 volumes was
added after the dry run showed the overlap, which was also before any reward existed.
`evaluate_learned` did not refuse the profile; `--no-strict-holdout` was not passed.

## Evaluation command

    python -m agent.evaluate_learned \
        --q-model agent/evidence/residual-q-2026-11-14/training/main/seed0 \
                  agent/evidence/residual-q-2026-11-14/training/main/seed1 \
                  agent/evidence/residual-q-2026-11-14/training/main/seed2 \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_altB.csv \
        --seeds 20 --eval-fleet --hazard-error-sigma 0.25 --diagnostics --out runs/check_residual_q_altB

Files: `episodes.csv`, `strata.csv`, `summary.md`. Pooling: the three models are averaged within each of the
20 scenarios, and the standard error is across scenarios.

## Result

| Policy | vs `rule_seasonal_forecast` | Scenarios better / worse | vs `risk_aware_rule_based` | Violation fraction |
| --- | --- | --- | --- | --- |
| `residual_q` seed 0 (margin 0.1) | -55.22 | 2 / 18 | -50.76 | 0.0319 |
| `residual_q` seed 1 (margin 0.25) | -36.38 | 4 / 16 | -31.92 | 0.0289 |
| `residual_q` seed 2 (margin 0.1) | -58.20 | 2 / 18 | -53.74 | 0.0326 |
| **seeds averaged per scenario** | **-49.93 +/- 8.09 (t = -6.17)** | 2 / 18 | **-45.47 +/- 7.51 (t = -6.05)** | |
| `rule_seasonal_forecast` | | | +4.46 +/- 2.01 (t = 2.22, 15 / 20 scenarios) | 0.0221 |
| `risk_aware_rule_based` | | | | 0.0226 |

## Oracle headroom, both populations

Per-seed paired differences, 20 scenarios, standard error across scenarios.

| | original profile (44 disks) | altB (150 disks) |
| --- | --- | --- |
| oracle vs `rule_seasonal_forecast` | +8.165 +/- 1.362 (t = 6.00) | +24.787 +/- 2.371 (t = 10.46) |
| oracle vs `risk_aware_rule_based` | +11.850 +/- 1.496 (t = 7.92) | +29.246 +/- 2.924 (t = 10.00) |
| `rule_seasonal_forecast` vs `risk_aware_rule_based` | +3.685 +/- 0.276 (t = 13.36, 20 / 20) | +4.459 +/- 2.008 (t = 2.22, 15 / 20) |

altB has about 2.5 to 3 times the exploitable headroom of the original profile on both bases. The narrow
ceiling recorded for the original profile belongs to that one disk population.

## Diagnostics (`diagnostics/`)

All computed without training. `altB_diagnostics.py` writes the `a_` to `d_` files; `altB_extras.py` writes
`e_predictability.csv`.

**Harness check (`d_harness_sanity.csv`).** `ResidualQPolicy` with `margin=1e9` on altB: actions and
per-step rewards are identical to `SeasonalForecastRule` on all 20 seeds, and the frames returned by
`evaluate()` are identical in every column except the policy name.

**Load scale (`a_load_scale.csv`), foreground MB/s.** The latency-violation threshold is
`violation_rho_threshold` x `BW_TOTAL_MBPS` = 182.2 MB/s.

| Profile | rows | mean | p50 | p90 | p99 | max | share of intervals above 182.2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2880 | 69.8 | 78.0 | 94.4 | 107.7 | 116.8 | 0 |
| holdout | 1152 | 70.2 | 78.9 | 95.0 | 107.3 | 116.3 | 0 |
| altB | 4032 | 63.9 | 63.8 | 94.0 | 108.2 | 120.8 | 0 |

The upper tail and the share of intervals where load plus an allocation exceeds the threshold match
(80 MB/s: 3.1% / 3.3% / 3.4%; 120 MB/s: 68% / 68% / 52%). The median of altB is 14 MB/s lower.

**Feature coverage (`b_feature_coverage.csv`, `b_feature_coverage_any.csv`).** Feature vectors of
`SeasonalForecastRule` rollouts on 20 scenarios, against the min and max of the feature vectors of 80 rule
rollouts (random allowed move on 10% of intervals) on the whole training profile (38195 states).

| Profile | states | any feature outside min/max | mean features outside per state |
| --- | --- | --- | --- |
| holdout | 9742 | 1.0% | 0.014 |
| altB | 9447 | 67.8% | 1.06 |

Per feature, altB share outside the training min/max (the holdout share is below 0.3% for each):
`fg_read_iops` 59.6%; `level_change_vs_yday` 10.8%; `load_change_lag1..12` 1.0% to 3.4%;
`yday_change_ahead1..6` 0.4% to 1.2%; `fg_bw`, `bw_available`, `progress`, `work_remaining`,
`thr_minus_*` below 0.4%. `fg_read_iops` is higher at the same bandwidth: mean read IOPS 28126 against
4536 on train, mean write IOPS 57524 against 5794.

**Predictability of the load (`diagnostics/e_predictability.csv`).**

| Profile | RMSE, same time yesterday (MB/s) | RMSE, last value (MB/s) | correlation of step changes with the step changes one day earlier |
| --- | --- | --- | --- |
| train | 2.74 | 10.12 | 0.970 |
| holdout | 2.01 | 10.36 | 0.987 |
| altB | 13.45 | 9.02 | 0.585 |

On altB the load one day earlier predicts the next interval worse than the last observed value.

**Where the loss is (`c_decomposition_summary.csv`, `c_per_episode.csv`).** Residual Q minus
`SeasonalForecastRule`, paired per seed, models averaged per scenario, 20 scenarios:

| Quantity | Mean | sem | t |
| --- | --- | --- | --- |
| total gap | -49.93 | 8.09 | -6.17 |
| latency terms (`alpha * D + gamma * V`) | -50.25 | 8.12 | -6.19 |
| degraded-state term (`beta * C`) | +0.32 | 0.20 | 1.59 |
| steps 0 to 287 (total) | -32.29 | 5.88 | -5.49 |
| steps 288 onward (total) | -17.64 | 4.09 | -4.32 |
| violations per episode, residual Q minus rule | +4.38 | 0.70 | 6.27 (18 of 20 scenarios worse) |
| violation term (`gamma = 10`) | -43.83 | | |
| degradation term (remainder) | -6.41 | | |

Moves away from the rule's action per episode: 47.3 up, 46.0 down (of about 469 intervals); 26.4 up and 25.6
down before step 288. Violation fraction 0.0312 against 0.0221 for the rule. Episode length 469.2 against
472.4 intervals; mean delivered 86.0 MB/s against 85.5. On the holdout the violation fraction was 0.0462
for both.
