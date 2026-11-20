# Decision record: the exogenous cost trap in the latency constraint, measured

Status: proposed. Not yet reviewed by the other lanes. Nothing here changes an existing default. Stacked on
`eval/scenario-split-2026-11-19` (#34): `exogenous_violations` and the scenario flags come from that change.

Code: `agent/exogenous_trap.py` (`x_table`, `x_summary`, `carriers`, `eq3_check`, `training_x`, and the `compare`
subcommand); tests in `agent/test_exogenous_trap.py`. Evidence: `agent/evidence/exogenous-cost-trap-2026-11-19/`.

## Definitions

- `C`: latency violation fraction of an episode. `d = 0.05` (`LATENCY_VIOLATION_FRACTION_MAX`).
- `X`: the fraction of intervals in which the smallest action the safety filter allows would also have violated
  latency (`exogenous_violations / steps`, the policy's own trajectory).
- `f = X / C`, the exogenous share of the violations (equation (3)). At the smallest action every violation is
  exogenous (`f = 1`); a more aggressive action adds violations the smallest action would not cause, and `f` falls.
- `C'`: the latency cost restricted to the violations the agent's action adds beyond the smallest action
  (`marginal_cost_steps` of `agent.train --marginal-violation-cost`, `agent/L3_DECISION.md`).

## Evidence

**1. X against d.**

| Where | X | C | f | Scenarios or iterations with X above d |
| --- | --- | --- | --- | --- |
| PPO training, pooled over 3 seeds x 244 iterations (`l3_stats.csv`, train workload and configuration) | **0.0560** | 0.0664 | 0.843 | 54.8% of the iterations (53.3%, 58.2%, 52.9% by seed) |
| `risk_aware_rule_based`, `diurnal_long_train`, train-fleet, sigma 0.25 (harness) | 0.0490 | 0.0504 | 0.972 | seeds 11, 18 |
| `fixed_20`, same | 0.0405 | 0.0405 | 1.000 | seeds 11, 18 |
| `fixed_20`, holdout, eval-fleet | 0.0277 | 0.0277 | 1.000 | seeds 9, 17, 19 |
| `risk_aware_rule_based`, holdout, eval-fleet | 0.0438 | 0.0444 | 0.988 | seeds 9, 17, 19 |
| `fixed_20`, altB | 0.0119 | 0.0119 | 1.000 | seeds 9, 17 |
| `risk_aware_rule_based`, altB | 0.0182 | 0.0226 | 0.807 | seeds 9, 17 |

- **The training distribution is above d.** Along the policies PPO visited, `X = 0.0560 > 0.05`. With
  `C >= X`, the constraint `C <= d` has no feasible policy on average over the training iterations; per seed
  `X = 0.0568, 0.0561, 0.0551`.
- **Per scenario, a few scenarios carry it.** The scenario carries `X` when the safety filter forces a high
  allocation: the rule has `X = 0.482` and `0.451` in training-workload seeds 11 and 18 (95% of the summed `X`);
  `X = 0.459, 0.229, 0.187` in holdout seeds 9, 17, 19 (99%); `0.225, 0.112` in altB seeds 9, 17 (92%).
  Averages below `d` (holdout, altB) hide scenarios whose `X` is several times `d`.

**2. Equation (3), `f = X / C` over the fixed rates** (`eq3_check.csv`):

| Profile | f(20) | f(50) | f(80) | f(120) |
| --- | --- | --- | --- | --- |
| training workload | 1 | 1 | 0.689 | 0.087 |
| holdout | 1 | 1 | 0.583 | 0.076 |
| altB | 1 | 1 | 0.311 | 0.038 |

`f = 1` at the smallest action in every scenario with `C > 0` (to 1e-12). `f` is non-increasing in the rate on all
three profiles and strictly decreasing only from 50 MB/s up: 50 MB/s adds no violation beyond those of 20 MB/s on
any of the three, so `f(50) = f(20) = 1`. The statement "falls with aggressiveness" holds weakly, not strictly.

**3. The figures of the analysis, traced** (what can be reproduced from which file):

| Figure | Source | Reproduces? |
| --- | --- | --- |
| 0.843 = 335,685 / 398,170 | `real_violations` (sum 398,170) and `min_action_violations` (sum 335,685) of `runs/l3_244/seed{0,1,2}/l3_stats.csv`; 335,685 also equals `real_violations - marginal_cost_steps`. Copied to `evidence/.../l3_stats/` | **Yes**, from the copied files. The originals are under the gitignored `runs/`; before this change no committed file held them (`agent/evidence/l3-2026-11-12/` has only `log.csv` and `episodes.csv`) |
| lambda_latency 1.20 to 1.68 (clean baseline) | `lambda_latency` of `agent/evidence/b2-244-clean-2026-11-12/seed{0,1,2}/log.csv` | **Partly.** Final values 1.664, 1.305, 1.196; maxima 1.682, 1.403, 1.215. "1.20 to 1.68" matches the smallest final value and the largest maximum; the range of the final values is 1.20 to 1.66 |
| lambda_latency driven to 0 by `C'` | `lambda_latency`, last row, `agent/evidence/l3-2026-11-12/seed{0,1,2}/log.csv` | **Yes**: 0.0000 in all three seeds (maxima 0.165, 0.147, 0.150) |
| the second multiplier unchanged | `lambda_risk`, last row, the same files against `b2-244-clean` | **Approximately**: L3 0.546, 0.367, 0.606; clean 0.554, 0.302, 0.615. Same order of magnitude, differences up to 0.065, not identical |

**4. `C'`-corrected PPO against the predictive controller and the optimum** (holdout, train-fleet, sigma 0.25,
history_rows 576, 20 scenarios; filter-forced seeds under that configuration {7, 9, 11, 17, 18}; 15 controllable;
gain over `risk_aware_rule_based`, learned seeds averaged within each scenario):

| Policy | Gain, controllable (primary) | t | Better / worse | Share of the optimum |
| --- | --- | --- | --- | --- |
| `predictive_controller (history)` | +14.550 +/- 1.510 | 9.64 | 15 / 0 | 93.9% |
| PPO without `C'`, deterministic | -27.751 +/- 1.502 | -18.47 | 0 / 15 | |
| PPO without `C'`, sampled | -72.790 +/- 2.921 | -24.92 | 0 / 15 | |
| PPO with `C'` (L3), deterministic | -18.926 +/- 1.409 | -13.43 | 0 / 15 | |
| PPO with `C'` (L3), sampled | -83.326 +/- 3.456 | -24.11 | 0 / 15 | |
| `ORACLE_lookahead` (not deployable) | +14.328 +/- 1.509 | 9.50 | 15 / 0 | 92.5% |
| `DP_hindsight_optimum` (not deployable) | +15.492 +/- 1.505 | 10.29 | 15 / 0 | 100% |

`C'` lifts the deterministic PPO by +8.8 on the controllable scenarios and lowers the sampled PPO by 10.5; neither
approaches the controller or the rule. Sensitivity variants (`gaps.csv`) give the same ordering. The controller's
violation fraction is 0.001 lower than the rule's on controllable scenarios and 0.002 higher on all scenarios.

## Decision

- The exogenous share is measured per episode by the harness (`exogenous_violations`) and in training by
  `l3_stats.csv`; both are reported against `d`, overall and per scenario.
- The figures 0.843 and lambda-to-0 are reproduced from committed or copied files; the lambda range is stated as
  1.196 to 1.664 final (1.215 to 1.682 maximum), not as 1.20 to 1.68.

## Decisions made during execution

- **Copy `l3_stats.csv` into the evidence.** The only record of the 335,685 and 398,170 sums was under the
  gitignored `runs/`; the three files (small) are committed so the figure is traceable. Alternative rejected:
  citing `runs/` paths a clean clone does not have.
- **Direct X on the training workload by the harness,** with the rule and the fixed rates under the training
  configuration (train-fleet, sigma 0.25), because the PPO run's own counters exist only for the policy PPO trained.
- **The comparison arm configuration.** Train-fleet, sigma 0.25, history_rows 576, so that the controller has its
  history; PPO ignores the history. The start rows of ten holdout seeds differ from the earlier L3 evaluation, so
  the numbers differ from `L3_DECISION.md` (deterministic, history off).
- **Equation (3) reported as non-increasing,** not strictly decreasing, because 50 MB/s adds no violation over 20.

## Limits

- **`C'` is a different constraint.** It bounds the violations the agent's own action adds, not the violation rate;
  the system's latency budget `C <= d` is not enforced by it. Using it removes the infeasibility that
  `X > d` causes, at the price of not constraining the exogenous violations.
- **The diagnostic needs a counterfactual model.** `X` requires the minimum-action violation at each interval, which
  here is computed from the simulator's own latency, filter and hazard formulas on the policy's trajectory; on a real
  system it needs a validated model of the foreground load, the latency and the filter.
- **Slater's condition.** `C(pi) >= X(pi)` holds because a larger allocation does not lower latency violation; where
  `X > d` on the trajectories reachable by every policy, no policy is strictly feasible and the dual problem has no
  finite optimum. On the training distribution `X` averages 0.056, above `d`, so the condition fails on average; it
  can hold per scenario for the controllable scenarios (holdout `X = 0.0277`). Whether `X` depends on the policy
  through the hazard and the remaining work (a faster rebuild lowers later exposure) was not mapped; `X` is measured
  on each policy's own trajectory.
- **lambda did not reach its cap.** In the clean runs `lambda_latency` ends at 1.2 to 1.7, below the cap of 2.0
  (`agent/L3_DECISION.md`); the divergence the analysis predicts for `X > d` is bounded by the cap and the
  estimator, not shown as unbounded growth.
- **One training distribution, three seeds.** The PPO figures use the runs of `agent/PPO_COLLAPSE_DECISION.md` and
  `L3_DECISION.md`; the comparison is on the holdout only.
