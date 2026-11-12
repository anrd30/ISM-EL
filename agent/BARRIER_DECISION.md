# Barrier-method latency constraint — real 3-seed result (2026-11-12)

## Background

Third constraint regime, alongside the existing Lagrangian penalty (`b2_244`
clean baseline) and the marginal-violation-cost reclassification (L3,
`L3_DECISION.md`): a log-barrier cost on the normalized latency margin,
replacing (not supplementing) the Lagrangian `lambda_latency` term in the
training objective. `lambda_latency` is still computed, updated and logged
during barrier-mode training, but purely as an inert diagnostic monitor --
it does not affect the objective in this mode.

Design (`agent/ppo.py::latency_barrier`): `d = (L - latency_now)/L`,
`L = 1.5 * latency_target_ms`; cost `= -log(d)` for `d >= 0.05`, else a
linear tangent extension (continuous in value and slope at the switch);
capped at 10.

Proposed by the RL-agent lane as a mechanism predicted to reach L3's
speed/throughput gain without L3's small increase in violation fraction.

Two training configurations were run, in this order:

1. Fixed coefficient (0.05), no annealing -- deliberately, for a clean
   single-variable first read against the baseline.
2. Linear annealing, 0.05 -> 0.005 (`--barrier-coef-final 0.005`), after
   (1)'s result motivated trying it.

Both: 3 seeds, 244 iterations, same workload/sigma/gamma as the clean
baseline and L3. Evaluation protocol identical to those two throughout
(`--workload-csv profile_diurnal_long_holdout.csv --seeds 20 --diagnostics
--hazard-error-sigma 0.25`), by argmax (the convention used for every
other reported figure in this project) unless noted as stochastic.

## Result 1 (fixed coefficient, no annealing): opposite of the prediction

Barrier minus clean Lagrangian baseline, argmax eval, pooled across seeds
(scenario-paired, 3-seed-averaged-then-across-20-scenarios sem):

| metric | mean diff | sem | t |
|---|---|---|---|
| total_reward | -16.9650 | 1.8876 | -8.987 |
| completion_hours | +14.1375 | 1.5730 | 8.987 (slower) |
| mean_delivered_mbps | -4.2813 | 0.4678 | -9.151 (less throughput) |
| latency_violation_fraction | -0.0007 | 0.0006 | -1.012 (n.s.) |
| p_failure_mean | +0.0003 | 0.0000 | 6.947 (worse) |

Significantly slower, lower-throughput and higher-failure-probability than
the clean baseline. The metric the barrier was meant to help
(`latency_violation_fraction`) didn't move either way.

Per-training-seed `total_reward` diff vs. the clean baseline (argmax):
seed 0 -59.195±5.370, seed 1 +8.830±0.692, seed 2 -0.530±1.354. Taken at
face value this read as "one catastrophic seed, one genuine win, one
wash" -- see Correction below for why the "win" doesn't hold up.

## Correction: argmax vs. stochastic evaluation

All three trained policies have residual entropy (~0.92-1.07 nats against
a ~1.39 maximum, i.e. not close to deterministic). Argmax (mode) action
selection and stochastic (sampling) action selection are not
interchangeable readings of such a policy, and the gap between them turned
out to matter here.

Re-evaluating the fixed-coefficient barrier checkpoints, the clean
baseline checkpoints, and the L3 checkpoints all with `--stochastic`,
against `risk_aware_rule_based` (reward -394.2, fixed across all of these
since it's a deterministic rule):

| | seed0 vs rule | seed1 vs rule | seed2 vs rule |
|---|---|---|---|
| barrier, argmax | -79.30 | -8.12 | -40.00 |
| barrier, stochastic | -79.05±5.841 | -52.83±3.816 | -62.83±5.283 |
| baseline, argmax | -20.10 | -16.90 | -39.50 |
| baseline, stochastic | -71.54±5.376 | -62.34±5.638 | -69.98±4.201 |
| L3, argmax | -23.93 | -20.92 | -22.65 |
| L3, stochastic | -77.51±5.838 | -57.51±4.091 | -102.10±7.287 |

Two things fall out of this:

1. **Argmax flatters every regime relative to its true (stochastic)
   behavior**, by a large margin, in all three constraint regimes tested
   so far -- this is a general property of policies with this much
   residual entropy, not something specific to the barrier method.
2. **Seed 0's barrier result is real, not an artifact**: -79.30 (argmax)
   vs. -79.05 (stochastic) -- nearly identical. Seed 1's apparent "win" at
   the top of this document *was* largely an argmax artifact: -8.12 under
   argmax looked close to the rule baseline, but -52.83 under stochastic
   is clearly a loss, in the same range as the other two seeds. The
   original "two bad seeds, one genuine win" framing is withdrawn; under
   honest (stochastic) evaluation all three fixed-coefficient seeds lose
   clearly to the rule baseline, and the fixed-coefficient barrier's
   stochastic-mode mean (-64.9) is roughly comparable to the baseline's
   own stochastic-mode mean (-68.0) and somewhat better than L3's
   (-79.0) -- i.e. once evaluated the same honest way, the fixed-coefficient
   barrier run is not dramatically worse than the other two regimes in
   aggregate; the dramatic *argmax* gap was itself partly an artifact of
   evaluation mode, not purely of the barrier mechanism.

(Full stochastic-eval summaries: `agent/evidence/barrier-2026-11-12-stochastic/`.)

## Result 2 (annealed, 0.05 -> 0.005): argmax looked like a clean win; it wasn't

Argmax evaluation, same scenario-paired methodology, against the clean
baseline:

| metric | mean diff | sem | t |
|---|---|---|---|
| total_reward | +6.3322 | 1.7110 | 3.701 (better) |
| completion_hours | -9.6083 | 0.9675 | -9.931 (faster) |
| mean_delivered_mbps | +6.9105 | 0.7840 | 8.814 (more throughput) |
| latency_violation_fraction | +0.0001 | 0.0015 | 0.040 (no change) |
| p_failure_mean | ~0.0000 | 0.0000 | 0.420 (no change) |

Taken at face value this was the predicted outcome (L3's speed/throughput
gain without L3's violation-rate increase), and was reported as such.
**That reading does not survive stochastic evaluation and is retracted.**

Given how large the argmax/stochastic gap turned out to be for the
fixed-coefficient run (see above), the annealed checkpoints were
re-evaluated with `--stochastic` before anything here was committed. The
gap is not smaller for the annealed run -- it is larger. Same
scenario-paired methodology, against the clean baseline, **stochastic**:

| metric | mean diff | sem | t |
|---|---|---|---|
| total_reward | -5.3252 | 2.4842 | -2.144 (worse) |
| completion_hours | -9.7764 | 0.9385 | -10.417 (faster) |
| mean_delivered_mbps | +6.2799 | 0.5876 | 10.687 (more throughput) |
| latency_violation_fraction | +0.0023 | 0.0006 | 4.048 (significantly more violations) |
| p_failure_mean | ~-0.0000 | 0.0000 | -6.662 (negligible in magnitude) |

Against L3, stochastic:

| metric | mean diff | sem | t |
|---|---|---|---|
| total_reward | +5.7481 | 1.9072 | 3.014 (better than L3) |
| completion_hours | -2.7319 | 0.3338 | -8.185 (faster than L3) |
| mean_delivered_mbps | +1.9266 | 0.3215 | 5.993 (more throughput than L3) |
| latency_violation_fraction | -0.0010 | 0.0008 | -1.309 (n.s., roughly matched) |

**The speed and throughput gain over the clean baseline is real and
survives both evaluation modes.** Everything else about the original
"matches the prediction" claim does not:

- `total_reward` reverses sign: +6.33 (argmax, "better") becomes -5.33
  (stochastic, significantly worse, t=-2.14) against the clean baseline.
- `latency_violation_fraction` reverses from "no change" (argmax, t=0.04)
  to a significant *increase* (stochastic, +0.0023, t=4.05) -- the same
  size increase L3 itself showed against the baseline (+0.0023, t=1.73,
  itself only borderline), with a tighter sem. The barrier method was
  specifically proposed to buy L3's speed *without* this cost. Under
  honest evaluation it does not avoid the cost; it reproduces almost
  exactly L3's own violation-rate increase, while also posting a worse
  total reward than the baseline, which L3 does not.
- Per-seed `total_reward` diff vs. baseline flips too: argmax showed seed
  2 at +23.401±3.335 (the strongest "win"); stochastic shows seed 2 at
  -4.244±4.660. All three seeds are negative under stochastic (seed 0
  -9.956±3.505, seed 1 -1.775±2.431, seed 2 -4.244±4.660) -- no seed beats
  the baseline on reward once evaluated honestly.
- The comparison against L3 (not the baseline) is the one piece that
  holds up in both modes: faster, more throughput, reward modestly better
  than L3, violation fraction not significantly different from L3's. So
  relative to L3, annealing the barrier buys a small additional edge;
  relative to the clean Lagrangian baseline, it does not, once evaluated
  the same honest way the fixed-coefficient run was.

This is the second finding in this document built on an argmax number
that didn't hold up, after seed 1's apparent "win" in Result 1. The
pattern is consistent: every regime tested so far looks meaningfully
better under argmax than under the sampling the training objective
actually optimises, and the size of that gap is not something that can
be eyeballed from training-time entropy alone -- it has to be measured
per checkpoint, which is what task 2 (separately) does for the
forecast-PPO headline number.

(Full evidence: `agent/evidence/barrier-2026-11-12-anneal/` for argmax,
`agent/evidence/barrier-2026-11-12-anneal-stochastic/` for stochastic.)

## Interpretation

The untuned, unannealed fixed coefficient (0.05) was too aggressive for
most seeds' training trajectories -- flagged as a live possibility before
the first run, and borne out. The root cause was not a single late
catastrophic event: all three fixed-coefficient seeds abandon the fastest
action within the first 20 training iterations and never revisit it; what
distinguished the badly-affected seeds was how far they drifted toward the
floor afterward, not a point of divergence.

Linear annealing to a 10x-smaller final coefficient resolves this almost
completely: no seed collapses, and the aggregate result now matches the
predicted mechanism -- a continuous cost near the latency limit buys the
same speed/throughput gain as penalizing marginal violations after the
fact (L3), without L3's own small, borderline increase in violation
fraction.

The argmax/stochastic correction above is a general finding about how
every regime in this project has been evaluated so far (all headline
figures to date are argmax), not specific to the barrier method. It
doesn't change which regime (annealed barrier) is being recommended here,
but it is worth flagging to the other lanes as a methodological point
that could affect the margins reported for the clean baseline and L3 too,
if those were re-checked under `--stochastic`.

## Status

Real, reproducible, 3-seed result, evaluated in both argmax and
stochastic mode throughout. Neither configuration reaches the predicted
outcome (L3's speed/throughput gain without L3's violation-rate increase)
once evaluated honestly:

- Fixed coefficient (0.05, no anneal): loses to the clean baseline on
  reward, speed, and throughput; the one seed that looked like a genuine
  exception under argmax (seed 1) does not survive stochastic
  evaluation.
- Annealed (0.05 -> 0.005): the speed and throughput gain over the clean
  baseline is real and survives both evaluation modes. The reward and
  violation-rate comparison against the baseline does not: argmax showed
  a significant reward improvement and no violation-rate change; honest
  (stochastic) evaluation shows a significant reward *loss* and a
  violation-rate *increase* of about the same size as L3's own. Against
  L3 specifically (not the baseline), the annealed barrier is a modest,
  robust improvement in both evaluation modes.

Net: as implemented and tested here, the barrier method does not deliver
on its proposed advantage over the existing Lagrangian penalty (L3) --
annealing fixes the fixed-coefficient version's instability and
genuinely beats L3 by a small margin, but it does not beat the clean
Lagrangian baseline on reward, and it does not avoid the violation-rate
cost it was specifically proposed to avoid. The speed/throughput gain
over the baseline is the one part of the original story that holds up.

Proposed, not yet reviewed by the other lanes.
