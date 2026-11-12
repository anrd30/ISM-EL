# Three-seed run of the `reward-gamma=2.0` configuration, held-out window

Training, three seeds:

    python -m agent.train --iterations 244 --rollout-steps 8192 --seeds 0 1 2 \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_train.csv \
        --hazard-error-sigma 0.25 --eval-fleet --reward-gamma 2.0 \
        --out runs/b2_244

Evaluation, all three checkpoints against every baseline on identical scenarios:

    python -m agent.evaluate_policy \
        --checkpoint runs/b2_244/seed{0,1,2}/ckpt_final.pt \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \
        --seeds 20 --out runs/eval244/b2_3seed --diagnostics \
        --hazard-error-sigma 0.25 --eval-fleet

Run on 2026-11-07. Seed 0 finished 09:30 UTC, seeds 1 and 2 at 16:38 and 16:38 UTC.
244 logged iterations and 1,998,848 environment steps per seed. `lambda_max=2` clipped
0 of 244 latency and 0 of 244 reliability updates on seed 0 (`train.txt`).

Contents: `summary.md` and `episodes.csv` as written by the evaluation harness; `train.txt`
the per-iteration driver output for seed 0; `seed{0,1,2}/log.csv` the training logs.

## Reportability: conditional, one question open

Unlike `upperbound-2026-11-06/`, there is no train/evaluation workload leakage here and
no stale checkout. The workload split is time-forward and enforced: training used the
train half, evaluation the holdout half, and `agent/evaluate_policy.py` refuses a
checkout trained on the evaluation workload unless `--no-strict-holdout` is passed.
`hazard_error_sigma=0.25` has committed provenance.

**But training passed `--eval-fleet`, and so did evaluation.** `agent/train.py` line 209
selects `v1_eval_config()` when that flag is set, which uses the held-out-split fleet
feature table rather than the train-split one. Fleets are sampled fresh per episode from
per-(model, column) pools, so training and evaluation shared a fleet *distribution*, not
fleet instances. Whether that is contamination is unresolved, and
`upperbound-2026-11-06/README-DO-NOT-REPORT.md` lists the mirror-image mistake as one of
its three disqualifying reasons.

**No number here may be published until the RL-agent lane rules on that.** If the ruling
is that it is contamination, re-run without `--eval-fleet` and re-evaluate; the run is
about 28 minutes per seed.

## Finding

Paired against `risk_aware_rule_based` on 20 scenarios (`summary.md`):

| Policy | Reward | Paired gap | sem |
| --- | --- | --- | --- |
| `risk_aware_rule_based` | -330.2 | — | — |
| `ORACLE_lookahead` (not deployable) | -318.3 | +11.9 | — |
| `ppo_lagrangian` seed 0 | -341.3 | -11.12 | 1.294 |
| `ppo_lagrangian` seed 1 | -414.3 | -84.07 | 48.23 |
| `ppo_lagrangian` seed 2 | -346.3 | -16.13 | 1.74 |

Seed 1's gap and its standard error are both an order of magnitude off the other two.
**The unweighted aggregate of these three numbers must not be reported.** The 20
scenarios are not homogeneous: the holdout sidecar discloses them as "selected extremes,
not a representative sample", and recomputing per-episode from `episodes.csv` shows 17
scenarios with a baseline reward between -54.46 and -69.89 and 3 (scenario seeds 9, 17,
19) at -2782.69, -1389.94 and -1385.13 — a 20x to 50x magnitude difference that dominates
any unweighted mean.

Split on that line:

| | typical, 17 | extreme, 3 |
| --- | --- | --- |
| seed 0 | -10.98 +- 1.51 | -11.93 +- 1.44 |
| seed 1 | -8.33 +- 1.67 | **-513.31 +- 197.03** |
| seed 2 | -15.18 +- 1.94 | -21.53 +- 2.03 |

On the 17 representative scenarios the result is reproducible: pooled -11.50 +- 1.05 over
51 pairs, mean-of-seed-means -11.50 with standard deviation 3.45 across seeds.

On the 3 extremes, seed 1 alone fails, by -142.41, -814.02 and -583.50. The mechanism is
in `episodes.csv`: on those scenarios seed 1 requests 78.6, 71.2 and 66.0 MB/s against
seed 0's 27.6, 30.8, 37.7 and seed 2's 20.4, 21.4, 23.6, and carries a correspondingly
higher `latency_degradation_mean` (2.133, 1.543, 1.139 against seed 0's 1.795, 0.828,
0.791). Every policy completed every episode — `terminal_failure` and `horizon_expired`
are 0 throughout — so this is accumulated per-step cost, not a termination event.
`risk_budget_violated` is `True` on scenarios 9 and 17 for every policy including the
rule, so the extremity is a property of the environment, not of any controller.

Seed 1 is the best of the three seeds on representative load and the worst by far on the
extremes, from one slightly-more-aggressive policy character. **Typical-case performance
is reproducible across seeds; tail robustness is not**, and nothing in the training setup
selects for it.

The subset split was chosen after seeing the results. Both halves are reported above
rather than one being dropped, but a pre-registered scenario set would be stronger.

Full context, the rejected alternatives, and the limits: `agent/PPO_COLLAPSE_DECISION.md`.

Hash note: `episodes.csv` (`workload_sha256` column, `diurnal_long_holdout` rows) and `summary.md` now carry the LF hash `94fe6bd6f8a5...` of `profile_diurnal_long_holdout.csv`. Corrected: this is the LF hash of the committed file; an earlier note quoted the CRLF hash of the same content from a Windows test run that temporarily rewrote it before the LF version was restored. The file content is unchanged.
