# Latency ceiling: policies evaluated with and without `SimConfig.latency_filter`

Record: `sim/LATENCY_FILTER_DECISION.md`. Nothing under the earlier evidence directories was
changed or overwritten.

## What was run

Three variants of each condition: no filter, `ceiling_oracle_fg` (`latency_filter=True`, reads the load of the interval being
decided; not deployable) and `ceiling_causal_fg` (`latency_filter_causal=True`, reads the previous interval's load, the `fg_bw`
the observation carries). `report.md` puts the three side by side.

Policies: `risk_aware_rule_based`, `rule_seasonal_forecast (hand-built)`, and the three committed
`forecast_ppo_bc` checkpoints (`agent/evidence/forecast-ppo-2026-11-11/training/bc/seed{0,1,2}/ckpt_best.pt`).
The checkpoints were trained without the filter and were not retrained. Each condition ran on the
same 20 scenario seeds (0..19) under each variant. Held-out fleet
table, `hazard=v1`.

| Condition | Workload | sigma | Fleet |
| --- | --- | --- | --- |
| `home` | `profile_diurnal_long_holdout.csv` | 0.25 | sampled |
| `peak`, `bursty`, `idle`, `diurnal_generic` | `profile_{peak,bursty,idle,diurnal}.csv` | 0.25 | sampled |
| `sigma_0`, `sigma_0.5` | `profile_diurnal_long_holdout.csv` | 0.0, 0.5 | sampled |
| `degraded_fleet` | `profile_diurnal_long_holdout.csv` | 0.25 | degradation pinned at 1.0 |

## What was not run, and why

`agent/evaluate_ood.py` does not exist in this checkout or on any local ref, including the tip of
`origin/agent/T9-seasonal-policy-search`. The conditions above are therefore a set chosen here from the
assets that are present, not the conditions that file defines. In particular **no phase-shift condition was
run**: its definition is not available. The no-filter numbers are produced by this harness under the same set of
conditions, so each variant is internally comparable, but none of these numbers is comparable with an
out-of-distribution table produced elsewhere.

## Harness check

The `home` condition with the filter off reproduces `agent/evidence/forecast-ppo-2026-11-11/episodes.csv`
exactly: the largest per-scenario difference in total reward is 0 for every policy
(first table of `report.md`).

## Files

| File | What it is |
| --- | --- |
| `report.md` | every condition, no filter / oracle-fg / causal-fg, side by side |
| `episodes_filter0.csv`, `episodes_filter1.csv`, `episodes_filter2.csv` | per-episode rows for no filter, oracle-fg and causal-fg (`baselines.evaluate.compare` columns plus `condition`, `latency_filter` set to `none`, `oracle` or `causal`) |
| `extreme_diagnostic.txt` | per-step filter activity under both ceilings on the three extreme `home` scenarios and three `peak` scenarios |
| `scripts/run_condition.py`, `scripts/build_report.py`, `scripts/diag_extreme.py` | the drivers; they import `agent.forecast_ppo`, `agent.seasonal_rule` and `agent.train_residual`, so they run from a tree that contains the T9 branch |
| `t9_cli_flag.patch` | the `--latency-filter` and `--latency-filter-causal` flags for `agent/evaluate_learned.py` and `agent/train_residual.py`, which exist only on that branch; the flag is already in this branch's `agent/train.py`, `agent/evaluate_policy.py` and `baselines/evaluate.py` |

## Commands

From a tree holding the T9 branch, `sim/latency_filter.py` and the changed `sim/config.py` and
`sim/recovery_env.py`, with `hazard/artifacts/v1/` and the train and holdout profile files present:

    python scripts/run_condition.py --condition home --latency-filter 0 --out OUT   # 0 none, 1 oracle-fg, 2 causal-fg
    python scripts/run_condition.py --condition home --latency-filter 1 --out OUT
    python scripts/run_condition.py --condition home --latency-filter 2 --out OUT
    (the same for each condition)
    python scripts/build_report.py OUT agent/evidence/forecast-ppo-2026-11-11/episodes.csv REPORT_DIR

The oracle-fg results were re-run after the causal flag was added and matched the earlier run exactly (largest per-scenario reward difference 0).
