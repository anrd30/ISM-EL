# Do not report these numbers

These logs come from a deliberately contaminated run. No number derived from them may be
published. See `agent/T_DECISION.md`, section "An upper-bound training run, and why its
numbers are not reportable", for the three contamination sources.

In short: training and evaluation shared one workload window (on purpose), the
evaluation used train-split fleets rather than `v1_eval_config()`'s held-out split (by
accident), and the checkout was 10 commits behind `main`, predating changes to
`sim/scenario.py`, `sim/config.py`, `hazard/v1.py` and `sim/stress.py` (by accident).

The ranking between policies is internally valid, because every policy ran against the
same contaminated simulator on identical episodes. The absolute values are not
comparable with anything produced on current `main`.

Command:

    python -m agent.train --iterations 122 --rollout-steps 8192 --seeds 0 1 2 \
        --workload-csv data_pipeline/profiles/profile_diurnal.csv \
        --hazard-error-sigma 0.25 --out <scratch>/ppo-realdiurnal-UPPERBOUND

Run on 2026-11-06.
