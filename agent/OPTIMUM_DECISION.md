# Decision record: the exact hindsight optimum and the value of next-interval information

Status: proposed. Not yet reviewed by the other lanes. Nothing here changes an existing default. Stacked on
`eval/scenario-split-2026-11-19` (the scenario split of `agent/SCENARIO_SPLIT_DECISION.md`).

Code: `baselines/dp_optimum.py` (`solve_hindsight`, `DPHindsightOptimum`), `agent/voi_sweep.py`
(`NoisyOracle`); tests in `baselines/test_dp_optimum.py` and `agent/test_voi_sweep.py`. Evidence:
`agent/evidence/optimum-2026-11-19/`.

## Decision

- **A non-deployable reference.** `DP_hindsight_optimum (not deployable)` solves one scenario exactly by backward
  induction over (step, remaining work) on a 1 GB work grid with linear interpolation of the value. It reads the
  realised foreground trace, the trace offset and the initial reported hazard from the environment (as
  `OracleLookahead` reads the next load) and executes the plan through the observed remaining work.
- **No restated formulas.** The risk floor is `sim.safety_filter.apply_safety_filter`, the latency is
  `sim.config.latency_mean_ms`, the reward terms and the cost are `contracts.contracts.compute_reward` with
  `latency_degradation`, `degraded_cost` and `latency_violation`, the delivered rate and the work update are the
  environment's expressions.
- **Value of information.** `NoisyOracle` replaces the oracle's belief with a forecast `true + sigma * z` and the
  predictive distribution `forecast + sigma * q_i` on 41 normal quantiles (calibrated by construction), seeded.
  `sigma = 0` is `OracleLookahead`.
- **Reporting.** Gain over `risk_aware_rule_based` and its share of the optimum's gain, on the scenarios
  controllable under the primary definition, each sensitivity variant, and an explicit excluded-seed list.

## Evidence

- **Planned value equals executed reward.** Holdout seeds 0 and 2: planned and executed cost 44.2280 and 47.0427,
  equal to 1e-13 (`baselines/test_dp_optimum.py`). Under the v0 configuration, `diurnal` seeds 0 to 6 match to 1e-6.
- **Optimum above `OracleLookahead`.** Over all 20 scenarios the optimum exceeds `OracleLookahead` by +1.013 +/-
  0.071 (holdout) and +1.132 +/- 0.094 (altB).
- **Controllable-scenario headroom.** Optimum minus `risk_aware_rule_based`, total reward:

| Definition of controllable | Holdout n | Gain | Share captured by `OracleLookahead` | altB n | Gain | Share |
| --- | --- | --- | --- | --- | --- | --- |
| primary (10%) | 16 | +14.84 | 92.7% | 16 | +34.40 | 96.4% |
| clip fraction at least 1% | 13 | +14.34 | 92.4% | 13 | +32.81 | 96.3% |
| clip fraction at least 25% | 16 | +14.84 | 92.7% | 16 | +34.40 | 96.4% |
| `fixed_20` cost above 50 | 17 | +14.58 | 92.6% | 17 | +33.53 | 96.3% |
| explicit seeds {9, 17, 19} / {9, 17} excluded | 17 | +14.58 | 92.6% | 18 | +31.79 | 96.2% |

- **Value of information** (share of the optimum's gain over the rule, by forecast sd in MB/s):

| Profile, definition | 0 | 2 | 5 | 7 | 10 | 15 |
| --- | --- | --- | --- | --- | --- | --- |
| holdout, excluded {9, 17, 19} (n 17) | 92.6 | 90.8 | 84.6 | 65.0 | 20.3 | -61.3 |
| holdout, primary (n 16) | 92.7 | 91.0 | 84.9 | 65.4 | 24.6 | -58.8 |
| holdout, `fixed_20` cost above 50 (n 17) | 92.6 | 90.8 | 84.6 | 65.0 | 20.3 | -61.3 |
| holdout, clip at least 1% (n 13) | 92.4 | 90.6 | 84.7 | 62.8 | 16.2 | -67.9 |
| altB, excluded {9, 17} (n 18) | 96.2 | 95.1 | 91.8 | 86.0 | 71.3 | 31.2 |
| altB, primary (n 16) | 96.4 | 95.4 | 91.9 | 86.2 | 71.7 | 31.4 |
| altB, `fixed_20` cost above 50 (n 17) | 96.3 | 95.2 | 91.9 | 86.3 | 71.8 | 32.1 |
| altB, clip at least 1% (n 13) | 96.3 | 95.1 | 90.8 | 84.6 | 69.7 | 23.7 |

- **Reading.** On controllable scenarios the next-interval look-ahead captures 92% to 96% of the exact optimum, so
  the optimum is not much above the oracle; the headroom over the deployed rule is 14 to 15 (holdout) and 32 to 34
  (altB). A forecast with an error sd of 7 MB/s keeps 65% (holdout) and 86% (altB) of it; at 15 MB/s the
  holdout gain is negative. Information about the next interval is worth the most when its error is below about
  5 MB/s on these profiles.

## Decisions made during execution

- **Targets reproduced as stated.** Holdout 92.6 / 90.8 / 84.6 / 65.0 / 20.3 / -61.3; altB 96.2 / 95.1 / 91.8 / 86.0 /
  71.3 / 31.2, on the explicit excluded-seed lists. The altB list {9, 17} is not any of the registered definitions
  (the `fixed_20` cost variant flags {9, 17, 19} there); it is kept as an explicit option for the reproduction.
- **Floor from the scalar filter on the grid.** The prototype's vectorised floor restated the filter; the port calls
  `apply_safety_filter` at every grid point instead (12001 calls per scenario, under 1 s). Alternative rejected:
  keeping the vectorised copy.
- **Cost per interval from the contract helpers** for the four possible delivered rates, indexed by the floored
  action, instead of vectorised formulas.
- **Execution reads the work from the observation** (de-normalised, rounded to the nearest grid point). The
  hazard and trace come from the environment at plan time, which is the non-deployable part.

## Limits

- **Constant hazard.** The plan holds the initial reported hazard fixed. A drive failure during the episode changes
  the hazard and can end it; the plan does not anticipate failures, so executed equals planned only without them.
- **Grid effect.** The policy reads the action at the nearest grid point while the planned value interpolates. Where
  a risk-floor boundary falls between grid points the two differ: v0 `diurnal` seed 7 executes 0.83 below its plan
  (`baselines/test_dp_optimum.py`); holdout seed 9 differs by 0.08.
- **Hindsight.** The optimum is for one realised trace; it is a reference, not an attainable policy.
- **Terminal condition.** The horizon is 1200 steps with a large cost on unfinished work, so every plan completes
  the rebuild inside it.
- **Seeds of the noisy belief.** Episode `i` uses the stream `10000 + i`; the table depends on consecutive seeds
  from 0.
- **Two profiles, 20 scenarios each**; the controllable subsets have 13 to 18 scenarios.
