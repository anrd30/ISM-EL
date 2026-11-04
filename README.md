# ISM-EL — A Closed-Loop, Distribution-Aware Risk Scheduler for Self-Healing Storage

Working title: **Predict, Quantify, Control**.

A storage group that has lost a drive must rebuild the missing data. The rebuild
competes with user traffic for the same bandwidth, and until it finishes the
group is down to a single tolerated failure. This project builds a controller
that sets the rebuild bandwidth every five minutes using the measured failure
hazard of the surviving drives, under an explicit latency constraint and an
explicit additional-drive-failure exposure budget. Conditioning on the full
distribution of remaining recovery time, not only its mean, is a planned
extension: the current interface exposes only scalar
`work_remaining`/`progress`, not a recovery-time distribution.

RVCE, VII semester, Information Storage Management experiential-learning
project.

---

## 1. The problem

When a drive in a RAID or erasure-coded group fails, the system enters a
degraded state and starts reconstructing the lost data onto a spare. Two things
are true at the same time:

- **The group is exposed.** An 8+2 group tolerates two drive failures. After the
  failure that starts the rebuild it tolerates one more, which leaves no further
  redundancy; a third failure, the second during the rebuild, is data loss. The
  longer the rebuild takes, the longer the window in which additional failures can
  exhaust the group's tolerance.
- **The rebuild hurts users.** Reconstruction reads from every surviving drive
  and writes to the spare. That bandwidth is taken from foreground traffic, so
  tail latency rises while the rebuild runs.

Rebuilding fast shortens the exposure window and hurts latency. Rebuilding
slowly protects latency and lengthens the exposure window. A scheduler has to
choose, repeatedly, under a load that changes through the day.

## 2. What exists today, and the gap

Prior work falls into three groups:

| Strand | What it does | What it does not do |
| --- | --- | --- |
| SMART-based failure prediction | Predicts which drive will fail | Controls nothing; the prediction is not wired to any actuator |
| Fixed or heuristic rebuild throttling | Caps rebuild bandwidth, or backs off when latency rises | Ignores how risky the specific surviving drives are, and reacts after latency has already degraded |
| Learned (RL) rebuild schedulers | Learns a bandwidth policy from load | Uses a mean or a point estimate of recovery time, and no per-drive hazard |

The gap this project targets: nobody closes the loop with **per-drive risk** and
the **whole recovery-time distribution** rather than its mean. A mean recovery
time hides the tail, and the tail is exactly where data loss happens.

## 3. What this project proposes

One closed loop with three parts.

```
          SMART features                 load profile (5-min samples)
                |                                   |
                v                                   v
   +------------------------+            +------------------------------+
   |  PREDICT               |            |  QUANTIFY                    |
   |  hazard(features, H)   |----------->|  simulator: recovery time    |
   |  -> p(fail in 30 d),   |  per-drive |  + latency model             |
   |     uncertainty        |  hazard    |                              |
   +------------------------+            +--------------+---------------+
                                                        | state (12 fields)
                                                        v
                                         +------------------------------+
                                         |  CONTROL                     |
                                         |  policy -> rebuild bandwidth |
                                         |  (20 / 50 / 80 / 120 MB/s)   |
                                         |  under latency + risk limits |
                                         +--------------+---------------+
                                                        |
                                                        v
                                            applied allocation, next interval
```

- **Predict.** A hazard model turns a drive's SMART history into the probability
  that it fails within the next 30 days, with an uncertainty estimate.
- **Quantify.** The simulator advances the rebuild under a replayed foreground
  load and records, per episode, the completion time and the latency the users
  would have seen. The evaluation harness saves one row per episode, so the
  recovery-time sample is kept; today's summary reports its mean, and the
  distribution with its tail is expected output 2 below, not yet produced.
- **Control.** A policy picks one of four rebuild rates every control interval.
  A risk safety filter clips requests that would exceed the exposure budget;
  latency violations are measured and reported rather than rejected by the
  environment.

Built so far: the simulator, the safety filter, the non-learned controllers,
and a fitted hazard model (`hazard/v1.py`, A3). `hazard/v0.py`'s unfitted
placeholder is still the model the simulator actually calls; wiring the
simulator to v1's feature set is A5, not yet done. The learned controller has
no training script. Section 8 has the per-task status.

## 4. Expected output

1. A controller that, on held-out load profiles, finishes rebuilds faster than
   fixed-rate and heuristic baselines **at equal or better** latency-violation
   and additional-drive-failure exposure numbers.
2. The recovery-time distribution per profile and per policy, reported with its
   tail, not only its mean.
3. A calibrated hazard model: when it says 2%, about 2% of those drives fail.
4. An honest limitations section. The latency model is an idealisation and is
   not calibrated against hardware; the numbers are properties of the model, not
   measurements of a real array.

## 5. Design and methodology

### 5.1 The frozen contract

`contracts/contracts.py` (version 1.4.0) and `contracts/INTERFACES.md` hold every
shared constant and every shared data shape. **It is frozen**: changing it
requires agreement across lanes, because all four lanes compile against it.

| Constant | Value | Meaning |
| --- | --- | --- |
| `CONTROL_INTERVAL_S` | 300 | One decision every 5 minutes; equals the trace sampling interval |
| `ACTIONS_MBPS` | 20, 50, 80, 120 | The four rebuild rates the policy may choose |
| `BW_TOTAL_MBPS` | 200 | Bandwidth shared by foreground traffic and rebuild |
| `GROUP_DATA_DRIVES` / `GROUP_PARITY_DRIVES` | 8 / 2 | 8+2 group, tolerates 2 failures, 9 survivors at episode start |
| `REBUILD_WORK_GB` | 12,000 | One 12 TB drive's worth of reconstruction |
| `HORIZON_DAYS` / `HORIZON_HOURS` | 30 / 720 | The hazard model's prediction horizon |
| `RISK_BUDGET_EPS` | 0.005 | Maximum tolerated additional-drive-failure exposure probability |
| `LATENCY_VIOLATION_FRACTION_MAX` | 0.05 | Evaluation limit: an action's latency-violation share must stay at or below 5% to be scored as meeting the latency budget |
| `LATENCY_LIMIT_FACTOR` | 1.5 | Latency limit = 1.5 x the service target |
| `OBS_DIM` | 12 | Observation fields handed to the policy |

### 5.2 The latency model

Foreground and rebuild traffic share one budget:

```
rho      = (foreground_mbps + effective_allocation_mbps) / 200
latency  = latency_unloaded_ms / (1 - rho)        # single-queue form
```

With `latency_unloaded_ms = 2.0`, `latency_target_ms = 15.0` and a limit factor
of 1.5, the latency limit is 22.5 ms, which is first exceeded at `rho = 0.9111`
(about 182 MB/s of combined load). Utilisation is clamped at `rho_max = 0.98`.

The form matters more than the numbers. The marginal latency cost of one more
MB/s of rebuild grows as the load rises, while the benefit (a shorter exposure
window) does not. That is what makes the decision load-dependent and gives a
controller something to learn. Under a linear latency model the best policy
collapses to "always rebuild at maximum". `sim/config.py` and
`sim/LIMITATIONS.md` state this assumption and its consequences in full.

### 5.3 The safe-action ladder

How many of the four actions stay inside the latency budget depends on the load
profile. Measured by `build_profiles.latency_ladder` with the simulator's own
functions, over the four committed Tencent profile CSVs (one 24-hour,
288-interval replay each):

| Profile | Safe actions | Violation fraction at 20 / 50 / 80 / 120 MB/s |
| --- | --- | --- |
| idle | 4 | 0.0 / 0.0 / 0.0 / 0.0 |
| diurnal | 3 | 0.0 / 0.0 / 0.024 / 0.698 |
| bursty | 2 | 0.0 / 0.007 / 0.177 / 0.295 |
| peak | 1 | 0.0 / 0.729 / 1.0 / 1.0 |

A monotone 4/3/2/1 ladder is deliberate: every profile keeps at least one legal
action, and the profiles differ enough that a controller must condition on load.
The ladder rests on a single 24-hour window per profile; the closest pass is
diurnal at 80 MB/s, violating 2.4% against the 5% allowed. `sim/LIMITATIONS.md`
section 9 reports the same ladder for the *synthetic* generator instead, averaged
over 20 seeds; the safe-action counts agree, the violation fractions do not, and
the two tables must not be quoted interchangeably.

### 5.4 Risk filter

The latency rule and the risk rule are separate. For a candidate allocation, the
filter computes the remaining completion time from `work_remaining` at that
allocation (not a fixed horizon) and bounds `P = 1 - exp(-hazard * T_remaining)`,
the probability of at least one additional drive failure before that work
finishes — an exposure probability, not a data-loss probability. An action is
feasible only while `P < 0.005`. Because `P` falls as the allocation rises, the
filter only ever moves an action up; when no action satisfies the budget it
applies the largest allocation, the minimum-risk choice, and records the breach in
`info["violated_risk_budget"]`. Foreground contention can push the delivered rate
below the requested allocation, which the simulator reports separately (see
`sim/LIMITATIONS.md`). Nothing in the latency rule substitutes for this check, or
the reverse.

## 6. Datasets

Neither dataset is stored in this repository.

| Dataset | Used for | Shape |
| --- | --- | --- |
| [Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data), 2022 Q1-Q4 | the hazard model | 365 daily CSVs (364 used), 179 columns, one row per drive per day; about 28 GB of CSVs on disk |
| Tencent Cloud Volume workload traces | the foreground load profiles | 16,296 headerless per-disk files, 300-second samples, 60 days, about 7.3 GB on disk |

What was selected from each, and why, is in `data_pipeline/A1_DECISION.md`
(Backblaze) and `data_pipeline/PREPROCESSING.md` (both).

**Backblaze selection (A1):** calendar year 2022; four drive models
(ST4000DM000, ST8000NM0055, ST12000NM0008, TOSHIBA MG07ACA14TA); 1,770 failures
over 33,052,284 drive-days, a pooled annual failure rate of 1.95%. One raw file,
`2022-12-10.csv`, is excluded because every row in it is dated 2022-12-11.

**Tencent selection (B5):** four load profiles (idle, diurnal, bursty, peak),
each one 288-row 24-hour replay chosen from the real traces, with a provenance
sidecar per profile.

---

## 7. Repository layout

```
contracts/          Frozen interface: constants, observation/action shapes,
                    replay-file and disk-index schemas, validators.
                    contracts.py, INTERFACES.md. Do not edit without sign-off.

sim/                The simulator (QUANTIFY). config.py holds the latency model
                    and every design constant; recovery_env.py is the episode;
                    safety_filter.py clips requests exceeding the exposure budget;
                    scenario.py loads replay traces and samples fleets;
                    stress.py builds stress scenarios. LIMITATIONS.md is
                    required reading before quoting any simulator number.

hazard/             The risk interface (PREDICT). v0.py is a constant annual
                    failure rate (2.11%) times a bounded SMART multiplier, not
                    fitted to data; it is still the model the simulator
                    actually calls. v1.py is the fitted alternative (A3): a
                    25-model XGBoost bootstrap ensemble with isotonic
                    calibration, built by fit_v1.py from A2's train/val split
                    and tested by test_v1.py. v1's artifacts
                    (artifacts/v1/, about 218 MB) are gitignored like
                    data_pipeline's generated outputs; manifest.json
                    hash-pins them and A3_DECISION.md records the fit's
                    evidence and limits.

baselines/          Non-learned controllers to compare against: the four fixed
                    rates, a risk-blind workload throttle, a risk-aware rule, and
                    an oracle look-ahead that is an upper reference rather than a
                    comparator. evaluate.py is the paired evaluation harness.

workload/           Synthetic load-profile generator producing the same four
                    profile shapes. It is the simulator's default foreground
                    source; a real profile is used only when SimConfig is given
                    a workload_csv.

data_pipeline/      Everything that turns raw public datasets into the tables
                    the other lanes consume.
  backblaze/
    inspect_schema.py   Read-only: column sets, per-model failure counts.
    extract.py          B1. Raw daily CSVs -> backblaze_filtered.parquet.
    timelines.py        B2. Per-drive rolling features + drive index.
    labels.py           A2. 30-day labels + train/val/test split by drive.
  tencent/
    inspect_schema.py   Read-only.
    extract_load.py     B3. 16,296 headerless files -> one clean load table
                        plus the per-disk summary and run tables.
    build_disk_index.py B4. Per-disk subscription metadata index joined by disk_uid.
    build_profiles.py   B5. The four 24-hour load profiles + plot + sidecars.
  profiles/             The four profile CSVs, their sidecars, the comparison
                        plot. These ARE committed and are hash-pinned.
  check_outputs.py      B6. The acceptance checker (see section 10).
  manifest.json         sha256 of every generated output, row counts for the
                        tabular ones (the sidecars and the plot have none), and
                        the digests of all 364 Backblaze source files.
  PREPROCESSING.md      What each stage did, with the counts from its last run.
  A1_DECISION.md        Data window, drive models, SMART columns, evidence.
  A2_DECISION.md        Label rule, split rule, seed, per-split counts.
  tests/                Unit tests for every stage, plus the leakage test.

CLAUDE.md           Project context and conventions for AI coding assistants.
                    Section 2 records lane ownership; section 10 the code
                    standards; section 12 the A-lane notes.
```

Generated data lives under `data_pipeline/data/` and is **not** in git
(`data_pipeline/data/` and `*.parquet` are ignored). Only the profile CSVs, the
plot, the sidecars and `manifest.json` are committed, so the checker can verify
a clone without the bulk data present.

## 8. Status

As of 2026-11-04, on `main`: 396 tests pass under `data_pipeline/tests`, which is
the pipeline suite only; `contracts/`, `sim/`, `baselines/`, `workload/` and
`hazard/` carry their own tests and run from the repo root. Across the whole
repo, `python -m pytest -q` passes 1,535 tests with 16 pre-existing skips
unrelated to the hazard lane. The acceptance checker reports `RESULT: PASS` in
`--committed-only` mode on a clean clone, and in full mode once the
Backblaze/Tencent outputs have been regenerated locally (they are gitignored,
so a clean clone's full-mode run reports `INCOMPLETE` until then, not `PASS`).
`hazard/v1.py`'s own artifacts are gitignored the same way, so a clean clone's
`hazard` tests run against a small synthetic fixture built in-process, not the
real fitted model, until `python -m hazard.fit_v1` is run locally.

| Area | Task | Status |
| --- | --- | --- |
| Interface | Frozen contract v1.4.0 | Done |
| Simulator | Episode, latency model, safety filter, scenarios, stress | Done |
| Baselines | Fixed-rate and heuristic controllers, evaluation harness | Done |
| Risk (placeholder) | `hazard/v0.py` constant-rate + SMART multiplier | Done, still the model the simulator calls |
| Risk (fitted) | `hazard/v1.py` -- 25-model XGBoost bootstrap ensemble, isotonic calibration | Done, PR #11; not yet wired into the simulator |
| Data B0 | Repo scaffold, ignore rules, schema inspection | Done |
| Data B1 | Backblaze extract, 4 models, 2022 | Done |
| Data B2 | Per-drive timelines, drive index, leakage test | Done |
| Data B3 | Tencent load extract | Done |
| Data B4 | Tencent disk index | Done |
| Data B5 | Four load profiles, latency ladder | Done |
| Data B6 | Acceptance checker | Done |
| Risk A1 | Data window, drive models, SMART columns | Done, PR #7 |
| Risk A2 | 30-day labels, drive-level train/val/test split | Done, PR #8 |
| Risk A3 | Fit the hazard model on the training split | Done, PR #11 |
| Risk A4 | Evaluate: calibration, PR-AUC, recall vs `hazard/v0.py` | Not started |
| Risk A5 | Replace `hazard/v0.py` with the fitted model | Not started |
| Risk A6 | Risk-model slides | Not started |
| Control | RL controller training | No training script in the repo yet |
| Report | Literature review, figures, report, deck | Tracked outside this repo |

### What A1, A2 and A3 produced

| Output | Rows | Grain |
| --- | --- | --- |
| `backblaze_filtered.parquet` | 33,052,284 | one row per drive per day, selected models and columns |
| `backblaze_timelines.parquet` | 33,052,284 | the same rows plus backward-looking features (7- and 30-day windows) |
| `backblaze_drive_index.parquet` | 92,525 | one row per drive: model, first/last seen, failure date |
| `backblaze_labels.parquet` | 33,052,284 | `label`: 1 positive, 0 negative, -1 censored |
| `backblaze_drive_split.parquet` | 92,525 | `split`: train / val / test |

Label totals: 52,387 positive, 30,370,344 negative, 2,629,553 censored.
Split: 1,770 failed drives divided 1,239 / 266 / 265.

**Two rules for anyone consuming this data.** Train, validate and score only on
rows where `label != -1`; censored rows are unknown outcomes, not healthy
drives. And use the stored split: never re-split, and never split by row or by
date, because consecutive rows of one drive are near-copies.

**What A3 produced.** A 25-model bootstrap ensemble (`hazard/artifacts/v1/`,
about 218 MB, gitignored; hashes in `hazard/manifest.json`), fit on the train
split (21,291,333 rows, 36,871 positive) and calibrated on val (4,567,066
rows, 7,762 positive). Ensemble validation log-loss 0.010024, Brier 0.001585 --
a real improvement over a constant-at-base-rate prediction (log-loss 0.01254,
Brier 0.001697), but a modest one; these are sanity checks, not evidence of
real-world quality, since val was also used for calibration. Whether v1
actually beats `hazard/v0.py` on the untouched test split is A4's question.

## 9. Known limitations and open items

- The latency model is an idealisation of one shared queue. It is not
  calibrated against any device. `sim/LIMITATIONS.md` is the authority.
- The safe-action ladder rests on one 24-hour window per profile.
- The Backblaze selection is one calendar year and two vendors, with no vendor
  held out, so transfer to an unseen vendor is untested.
- The train/val/test split is not time-forward; the splits overlap in dates.
- Per-model failed-drive counts in val and test are 52 to 95, so per-model
  metrics will be noisy and need error bars.
- `hazard/v0.py` still carries the Q1-2022 annual rate (2.11%) while the current
  extract's pooled rate is 1.95%. Its baseline was also set from data that
  includes the test drives, so A4 must either re-derive it from training drives
  or disclose the head start.
- 414 of the 6,817 Tencent candidate volumes (6.1%) report a median below 512
  bytes per I/O, which is not physically plausible. The cause is unresolved; those
  volumes are excluded from profile selection. The share is of the candidates that
  share the gap-free 24-hour window, not of all 16,296 volumes.
- `ACTIONS_MBPS` is a prototype action set, illustrative rather than measured
  device limits, so every result quoted against it has to say so
  (`contracts/INTERFACES.md` section 4).
- `RISK_BUDGET_EPS` bounds exposure, never data loss. An episode starts with one
  tolerated failure left, so exposure is one further failure and loss needs two.
  A data-loss figure has to be computed separately and labelled as such.
- Every one of A3's 25 boosters hit its round cap; early stopping never fired,
  so the fit's true plateau was never observed.
- A3's calibration has almost no resolution across most of the low-risk input
  space: an all-zero input and an input with one SMART raw count set as high
  as 50,000 (the rest absent) calibrate to the identical output, because both
  raw scores fall under isotonic's lowest step.
- `sim/scenario.py` still only generates the 5 synthetic features of
  `hazard.v0.FEATURE_NAMES`, so v1's 57 inputs cannot be fed by the simulator
  yet; that wiring is A5.
- `A1_DECISION.md`, `A2_DECISION.md` and `A3_DECISION.md` all still read
  "Status: proposed".

## 10. Getting started

**Requirements.** Python 3.11 or newer (the pipeline uses
`hashlib.file_digest`), then:

```
pip install -r data_pipeline/requirements.txt
```

**Clone and verify without the bulk data.** This is the normal state for a fresh
clone, and it checks everything that is committed:

```
python -I data_pipeline/check_outputs.py --committed-only
python -m pytest data_pipeline/tests -q
```

Checks that need the generated parquet files report `[SKIP] ... not tracked in
git`. That is expected; `RESULT: PASS` is what matters.

**Regenerate the data.** Download the raw datasets into
`data_pipeline/data/raw/` as `backblaze_2022Q1` ... `backblaze_2022Q4` and
`tencent/`, then run from inside `data_pipeline/`:

```
python -I backblaze/extract.py          # B1  (reads all 364 daily CSVs)
python -I backblaze/timelines.py        # B2
python -I backblaze/labels.py           # A2
python -I tencent/extract_load.py       # B3  (memory-hungry)
python -I tencent/build_disk_index.py   # B4
python -I tencent/build_profiles.py     # B5
```

Every stage rewrites its own section of `PREPROCESSING.md` and updates
`manifest.json`. B1 and A2 additionally compare their measured counts with the
values recorded in their own source and abort on any difference; B2 to B5 abort
only on a structural problem, and it is the checker below that verifies their
outputs against `manifest.json` and `PREPROCESSING.md`:

```
python -I data_pipeline/check_outputs.py
```

**Fit the hazard model (optional, A3).** Needs `data_pipeline`'s outputs
already regenerated. From the repo root:

```
pip install -r hazard/requirements.txt
python -m hazard.fit_v1
```

About 95 minutes on a laptop GPU (CUDA); falls back to CPU, much slower, if
none is found. Writes `hazard/artifacts/v1/` (gitignored, about 218 MB) and
regenerates `hazard/manifest.json`. Without this, `hazard`'s own tests still
pass against a small synthetic fixture built in-process.

**Run everything:**

```
python -m pytest -q
```

## 11. Working agreements

- **One branch per task**, named `<lane>/<task>-<slug>`, for example
  `data/B5-profiles` or `risk/A2-horizon-label`. Open a pull request into
  `main`; do not push to `main` directly.
- **The contract is frozen.** A change to `contracts/` needs agreement from the
  lanes that consume it, and a version bump plus a changelog entry in
  `INTERFACES.md`.
- **Every generated output is hash-pinned.** If you regenerate a file, the
  manifest entry changes in the same commit. Never hand-edit a hash.
- **The acceptance checker must pass in both modes before a merge**, together
  with the full test suite.
- **Code standards (`CLAUDE.md` section 10).** Prints, logs, comments and
  docstrings state facts only: no narration, no personal names, no "next step"
  advice. Explanation belongs in the markdown documents.
- **Decisions are recorded, not remembered.** A choice about data or method goes
  into a decision record with its numbers, its rejected alternatives and its
  limits.

## 12. Key documents

| Document | What it answers |
| --- | --- |
| `contracts/INTERFACES.md` | Every shared constant and shape, and why each was chosen |
| `sim/LIMITATIONS.md` | What the simulator does not model |
| `data_pipeline/PREPROCESSING.md` | What every pipeline stage did, with its last run's counts |
| `data_pipeline/A1_DECISION.md` | Which Backblaze data, and the evidence for it |
| `data_pipeline/A2_DECISION.md` | The label rule and the split, with per-split counts |
| `hazard/A3_DECISION.md` | What model was fit, how, and its real evidence and limits |
| `CLAUDE.md` | Lane ownership, conventions, code standards, A-lane notes |

## 13. Data attribution

Backblaze Drive Stats data is published by Backblaze and used here under their
terms: Backblaze is cited as the source, the data is not resold, and
responsibility for its use rests with this project. The Tencent Cloud Volume
workload traces are used under their own publication terms.
