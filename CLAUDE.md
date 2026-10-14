# ISM Phase-2 — Project Context for Claude Code

Read this file fully before touching any code. It is the complete context for the project and for Aryan's own workstream. Everything marked **[STATED]** is taken directly from the team's planning PDF/PPT. Everything marked **[PROPOSED]** is a recommendation this file makes that the team has not confirmed — treat it as a default to follow unless Aryan says otherwise, not as settled fact. Everything marked **[OPEN]** is an unresolved question that blocks part of the work.

Repo owner for this checkout: **Aryan Rai (USN 1RV23CS052)**, who owns the **data pipeline** lane only (tasks B1–B6). Do not implement another teammate's lane (Aniruddh's RL/simulator code, Aakarsh's risk model, Aditi's literature/report/deck) — only scaffold the interfaces this lane must hand to them.

---

## 1. The project, in one paragraph [STATED]

Title (PPT): *"A Closed-Loop, Distribution-Aware Risk Scheduler for Self-Healing Storage."* Title (PDF plan): *"Risk-Aware Workload-Adaptive Self-Healing Storage"* — these two titles disagree; treat the PDF as the operative plan for Phase 2. When a storage group (RAID / erasure-coded) loses a disk, it enters **degraded state** and must **rebuild** (reconstruct) the lost data from survivors. Rebuild traffic competes with real application ("foreground") traffic for the same disks/network. Rebuilding fast shortens the vulnerable window but slows applications; rebuilding slow protects applications but leaves the group exposed longer, raising the chance that a second failure causes permanent data loss. Today's systems use a fixed rebuild rate or a workload-only throttle, both of which ignore how likely a further failure actually is right now.

**Proposed solution:** a controller that, every control interval, picks a rebuild bandwidth from a small action set using foreground load, latency, rebuild progress, and a **calibrated failure-risk estimate**. It minimizes foreground latency degradation **subject to an explicit reliability budget** (constrained RL / CMDP), with an independent safety filter that clips unsafe actions. It is compared against three non-learned baselines and validated with an ablation that removes the risk input.

**Claim scope (slide 14):** the contribution is this formulation plus a measured advantage over a strong risk-aware rule-based baseline — *not* "RL for storage" (already published) and *not* a new rebuild mechanism. The team commits to a falsification test: if removing the hazard input and retraining performs the same, they report that honestly.

**Status as of 2026-10-14: nothing is implemented.** The PPT is a Phase-1 proposal; the PDF is a Phase-2 work-assignment plan. No code, data extracts, or results exist yet in this repo. Never write code comments, docstrings, or commit messages that imply a result has been measured, a model has been trained, or a claim has been verified, unless it actually has in this repo.

---

## 2. Team and ownership [STATED, PDF pp.1–4]

| Person | USN | Lane | Tasks |
|---|---|---|---|
| Aniruddh Shreyas | 1RV23CS037 | RL agent and simulator — builds **all v0 fallbacks** | T1–T10 |
| Aakarsh Prabhu | (TBD) | Risk model | A1–A6 |
| **Aryan Rai (you)** | **1RV23CS052** | **Data pipeline** | **B1–B6** |
| Aditi Rajesh | (TBD) | Literature, report, figures | C1–C6 |

Mentor: Dr. Rajashree Shettar, Dept. of CSE, RVCE.

**How delegation works [STATED, PDF p.1]:** "Every delegated piece has a working version before it is handed out." Aniruddh builds v0 of the hazard model (T3), the workload generator (T4), and the simulator (T2) first. Aakarsh's A-tasks and Aryan's B-tasks are **upgrades** to those v0s, not dependencies the project waits on. If an upgrade lands, results get stronger; if it slips, v0 runs the experiments and the project still finishes. **This means your B-lane work is important for result quality and credibility, but the project does not block on it** — except that A2–A5 (the risk model) block on your B1/B2 output, and the risk model *is* on a tighter, blocking path.

**The four team rules [STATED, PDF p.6]** — these govern how you write code and commits:
1. **Deliverables are file-shaped, never idea-shaped.** A CSV with named columns, a file exposing a named function, a markdown table with named fields. Completion must be checkable in under a minute.
2. **Nothing enters the run path without a passing adapter test.** An artifact meets the frozen contract or v0 stays in place.
3. **Silent fallback.** Anything not delivered by its date reverts to v0 without discussion — don't be precious about unfinished work; make sure v0 is never broken by your changes.
4. **No chasing.** One status check a week, in writing. A task that slips twice reverts to v0, and its owner moves to something with a later deadline.

---

## 3. The frozen interface [STATED, PDF pp.4–5]

This is written once by Aniruddh in week 1 (`contracts/INTERFACES.md`, `contracts/contracts.py`) and never changed casually after. **Aryan's B-lane code must produce data that fits into this contract, even though Aryan does not write the contract file itself.** As of 2026-10-14 the contract has not been frozen yet — see §6 Open Questions.

**Observation vector** (fixed order, normalized to roughly [0,1] before reaching the policy):

| # | Field | Unit | Note | Fed by |
|---|---|---|---|---|
| 0 | `fg_read_iops` | IOPS | from trace or generator | **you (B3)** |
| 1 | `fg_write_iops` | IOPS | | **you (B3)** |
| 2 | `fg_bw` | MB/s | combined foreground bandwidth | **you (B3, converted)** |
| 3 | `latency_now` | ms | measured this interval | Aniruddh (simulator) |
| 4 | `latency_target` | ms | constant per episode | Aniruddh |
| 5 | `hazard` | per hour | calibrated, horizon H fixed | Aakarsh (fed by your B1/B2) |
| 6 | `hazard_uncertainty` | per hour | 0 if model gives none | Aakarsh |
| 7 | `redundancy_left` | count | further failures tolerable | Aniruddh (simulator) |
| 8 | `progress` | fraction 0–1 | | Aniruddh (simulator) |
| 9 | `work_remaining` | GB | | Aniruddh (simulator) |
| 10 | `alloc_current` | MB/s | last action applied | Aniruddh (simulator) |
| 11 | `bw_available` | MB/s | hard ceiling this interval | Aniruddh (simulator) |

**Action:** index into `{20, 50, 80, 120}` MB/s (prototype only; illustrative, not measured system limits).

**Reward:** `r = -alpha*D - beta*C - gamma*V`, where `D` is normalized latency degradation (`D = max(0, (L - L0)/L0)`), `C` is the degraded-state cost for the interval, `V` is a constraint-violation penalty. Coefficients live in a config file, never hardcoded.

**Required `info` keys every step, computed independently of the reward** (so a reward penalty is never mistaken for proof a constraint held): `latency_p95`, `latency_p99`, `violated_latency` (bool), `p_failure_before_complete`, `violated_risk_budget` (bool), `alloc_clipped_by_filter` (bool), `recovery_complete` (bool).

**Termination:** recovery completes, a terminal failure state is reached, or the episode horizon expires. All three must be distinguishable in `info`.

**Hazard signature:** `hazard(features, horizon) -> (p, uncertainty)`. Horizon `H` is frozen once and never tuned afterward. **Note the unit conflict:** field 5 above says "per hour" (a rate), but the signature returns `p`, a probability over horizon H. This is an open contradiction in the source documents — flag it to Aniruddh/Aakarsh, don't silently resolve it yourself.

---

## 4. Your lane: the data pipeline (B1–B6) [STATED, PDF p.3]

> "Turn the two public datasets into clean files the rest of the project reads. Mechanical and fully bounded: no modelling judgment required." — **this framing understates it.** B2's rolling-window choice and B5's profile-selection thresholds genuinely are judgment calls; make them, write them down, and flag them for confirmation rather than treating them as settled.

### B1 — Backblaze extract
Download **one quarter**, project only the needed columns, align the SMART schema across daily files (columns vary quarter to quarter — never concatenate without aligning first), subset to the chosen drive models. Output **one** Parquet or CSV file.
**Do not:** concatenate Backblaze quarters without aligning the SMART schema first — stated explicitly as the one thing not to do.

### B2 — Per-drive timelines
Group by `serial_number`, sort by `date`, emit a per-drive sequence with rolling differences and missingness indicators. **This is what Aakarsh's risk model fits on** — it is the most important handoff in your lane.

### B3 — Tencent extract
Parse load traces into canonical replay files with **exactly** these columns: `t_seconds, read_iops, write_iops, read_bw_kbps, write_bw_kbps, disk_usage_mb`.

### B4 — Join the metadata
Attach `disk_attr, disk_type, vm_cpu, vm_memory, disk_capacity` by `disk_uid`.
**Note:** B3 says "exactly" six columns and B4 says to attach five more — these two instructions conflict as literally written. **[PROPOSED]** resolve by keeping B3's six-column replay files untouched and writing B4's metadata into a *separate* disk-index file, joined on `disk_uid` only when something downstream needs it.

### B5 — Four intensity profiles
Select real trace windows that are genuinely **idle, diurnal, bursty, and peak**. One file each, 24 hours at the 300-second sampling interval → **288 rows per file.**

### B6 — PREPROCESSING.md
Every filter, every dropped row, every imputation, with counts before and after.

### Lane acceptance [STATED]
"The files load with the exact column names above, row counts match PREPROCESSING.md, and the four profiles are visibly different when plotted. Checkable in under a minute."

### Fallback if you slip [STATED]
The v0 synthetic workload generator (Aniruddh's T4) covers all four profiles. Real traces make the evaluation more credible; they are not required for it to run. **This means B3–B5 are not on the critical path** — but B1–B2 effectively are, because Aakarsh's risk-model lane (A1–A6) depends on them and his lane closes at week 5.

---

## 5. Source datasets — exact schemas [STATED, PDF/PPT]

### Backblaze Drive Stats
- One row = one drive on one day.
- Stable core columns: `date, serial_number, model, capacity_bytes, failure`.
- `failure=1` marks the **last operational day** before the drive failed; `failure=0` otherwise.
- `smart_N_raw` and `smart_N_normalized` columns exist per SMART attribute; **the set of columns changes between quarterly files.** Always reindex to one aligned column list per quarter before any multi-quarter operation.
- Example rows:
  ```
  date,serial_number,model,capacity_bytes,failure,smart_1_normalized,smart_1_raw
  2022-01-01,ZLW18P9K,ST14000NM001G,14000519643136,0,73,20467240
  2022-01-01,ZLW0EGC7,ST12000NM001G,12000138625024,0,84,228715872
  2022-01-01,1050A084F97G,TOSHIBA MG07ACA14TA,14000519643136,0,100,0
  ```
- Source: Backblaze official Drive Stats (backblaze.com/cloud-storage/resources/hard-drive-test-data). Quarterly files can be multi-gigabyte — select columns and models before loading fully into memory.

### Tencent Cloud Virtual Disk Traces
- ~16,000 cloud virtual disks sampled from 10 warehouses in a Shanghai data center. ~1.3 GB compressed total. Source: github.com/Tencent/Cloud-Volume-Workload-Traces (README).
- **Subscription table** (metadata, one row per disk): `disk_uid, disk_attr, disk_type, user_type, vm_cpu, vm_memory, disk_capacity`.
  Example: `disk_uid=0a0a7b84-2279-405e-9cbc-b395c725e01a, disk_attr=0, disk_type=1, user_type=0, vm_cpu=4, vm_memory=32, disk_capacity=50`
- **Load trace table** (time series, one row per sample): `timestamp, read_IOPS, read_bandwidth, write_IOPS, write_bandwidth, disk_usage`. Up to two months per disk, **300-second (5-minute) sampling interval.**
  Example: `timestamp=1593748800, read_IOPS=11, read_bandwidth=50 KB/s, write_IOPS=7, write_bandwidth=12 KB/s, disk_usage=3360 MB`
- **Units:** bandwidth is reported in **KB/s**, not MB/s. The observation contract's `fg_bw` is MB/s. A conversion must happen somewhere in the pipeline — see Open Questions.

### What neither dataset contains [STATED, slide 19]
Rebuild start/end times, rebuild bandwidth allocated, application latency during rebuild, or counterfactual outcomes under alternative actions. This is *why* the simulator (Aniruddh's T2) exists — it is not optional scaffolding, it is load-bearing. Anything your code derives that looks like a "recovery duration" or "latency during rebuild" must come from the simulator, never be invented from these two datasets, and must be labeled simulated wherever it appears.

---

## 6. Open questions — resolve before coding depends on the answer [OPEN]

Do not guess silently on these; surface them to Aryan to raise with the team. Build the parts of your lane that don't depend on the answer first (see §8 execution order).

1. **Hazard unit contradiction:** "per hour" (field 5) vs. a probability `p` (hazard signature). Not your call, but your B2 output's column choices should stay agnostic to this.
2. **Numeric constants undefined:** horizon `H`, risk budget `ε`, latency limit `L_max`, control-interval length, episode horizon. None have values yet. Slide 15: "must be fixed before implementation and not tuned after seeing results."
3. **"Distribution-aware" has no home:** the title claims the controller uses the full remaining-recovery-time distribution, but no field in the 12-field observation vector carries it. Not yours to fix, but don't invent one either.
4. **Tencent unit conversion:** `_kbps` → MB/s for `fg_bw`. **[PROPOSED]** your B3/B4 output should store raw KB/s values (do not pre-convert), and the MB/s conversion happens in the adapter that feeds the simulator — keeps your files a faithful, re-usable record of the source data. Confirm this with Aniruddh before his adapter test (T10) is written.
5. **Load scaling:** the Tencent example disk peaks around 50 KB/s; the simulator's action set is 20–120 **MB/s**. A single real disk's trace, even unit-converted, will not meaningfully load a simulated storage group at that scale. **This is a real open design question, not just a units bug** — raise it explicitly; don't silently multiply by an invented factor.
6. **A1 not yet decided:** which Backblaze quarter, which 1–2 drive models, which SMART columns. This blocks B1. Get it from Aakarsh in writing before starting B1.
7. **Repo/CI specifics:** Python version, package manager, where large data files live (should NOT be committed to git — see `.gitignore`).

---

## 7. Technical background (so generated code and comments are accurate)

- **Degraded state / redundancy_left:** if a group tolerates N failures and k have occurred, `redundancy_left = N - k`. At 0, one more failure loses data permanently.
- **Censoring (relevant to Aakarsh's A2, but shapes what your B2 timelines must expose):** a drive that disappears from the dataset before the label horizon *without* a recorded failure is **censored** — unknown fate — not healthy. Your B2 output should expose `first_seen`/`last_seen` per drive so A2 can tell this apart from a genuinely healthy, fully-observed drive.
- **Leakage:** never let a feature use information from a later date than the row it describes. All rolling-window features must be **trailing** (use only past days), never centered. This is the single most important correctness property of your B2 code — write a test for it (see §9).
- **PR-AUC, recall, calibration:** these are Aakarsh's concern for A4, but they explain *why* your censoring/missingness columns matter — failure is a rare class (~1,193 of 36,849 disks in one cited study), so naive accuracy is useless and the label construction (A2) depends on clean timeline data from you.

---

## 8. Execution order for THIS lane, right now

You are on branch `data/B0-setup`, repo is otherwise empty. The interface contract is **not frozen yet** (open question #2/#6). Work in this order so you make progress without guessing at values nobody has confirmed:

1. **B0 — scaffold (do now, this session).** Folder layout, `.gitignore`, env/requirements file, empty `PREPROCESSING.md` with its section headers, a `manifest.json` stub, and a `tests/` skeleton. No data processing logic yet. See the files already created alongside this one.
2. **Inspect, don't finalize.** Write (but do not run destructively) a small script to inspect the Tencent subscription/load schema once you have a sample file, and a Backblaze schema-inspection script once a quarter is downloaded. These just print/report column lists — useful regardless of what A1 decides.
3. **Wait on A1** (quarter + drive models + SMART columns) before writing B1's real filtering logic. You can write the B1 script's structure and its schema-alignment test now, with the filter parameters as named constants at the top of the file (not yet filled in), clearly marked `# TODO: confirm with Aakarsh (A1)`.
4. **Wait on contract freeze** (units, scaling, `t_seconds` origin) before finalizing B3/B4 output units — same approach: write the parser structure now, mark the unit-conversion boundary clearly, do not pre-convert KB/s → MB/s yourself (see Open Question 4).
5. Once A1 lands: B1 → B2 (ship to Aakarsh ASAP — his lane is tighter than yours).
6. Then B3 → B4 → B5 (lower urgency; has a fallback).
7. B6 (`PREPROCESSING.md`) is written incrementally in every PR, not as a final step.

---

## 9. Engineering conventions for this lane

- **Language/tools:** Python, pandas, pyarrow (Parquet). Use `-I` isolation awareness: never trust ambient `sys.path`.
- **Outputs:** Parquet for large processed tables (B1, B2), small CSVs for the four 288-row profile files (B5) so they're easy to diff and inspect in PRs.
- **No global statistics across drives for imputation.** Leave genuine gaps as `NaN` plus a `<col>_missing` indicator column. Forward-fill, if used at all, is **within one drive, from past rows only.**
- **Every script is idempotent and re-runnable**, writing counts (rows in, rows out, rows dropped and why) to both stdout and `PREPROCESSING.md`.
- **Tests to write as you build (not an afterthought):**
  - `tests/test_schema_alignment.py` — every daily Backblaze file reindexed to one column list; missing columns become NaN and are counted, never silently dropped.
  - `tests/test_no_leakage.py` — cut the B2 timeline at a date `d`, recompute features; every feature for rows `≤ d` must be byte-identical whether computed on the full dataset or data truncated at `d`. If anything changes, a feature is reading the future.
  - `tests/test_profile_grid.py` — each B5 profile file has exactly 288 rows at exactly 300-second spacing with no gaps.
  - `check_outputs.py` at the pipeline root — runs all of the above plus column-name/dtype checks in under a minute (this is the literal lane acceptance test).
- **Never commit raw or large processed data.** `.gitignore` excludes `data/raw/` and `data/processed/`; commit only scripts, `PREPROCESSING.md`, `manifest.json` (file, rows, sha256), and the four small profile CSVs.
- **Commit messages** start with the task ID: `B1: align SMART schema across daily files`. Branch names: `data/<task-id>-<slug>` (you're already on `data/B0-setup`).
- **Never fabricate:** no placeholder "results," no invented row counts, no SMART column choices presented as decided when they're still open. If a value is unconfirmed, write `# TODO: confirm with <person> (<task-id>)` rather than a plausible-looking guess.

---

## 10. Code and comment standards

Code is read by a compiler/interpreter and by engineers, never by a person
being talked to. Keep it that way:

- **No narration in `print`/`log` output.** Runtime output reports facts the
  code computed — counts, paths, pass/fail — never advice, next steps, or
  who to ask. `print("rows_in=91234 rows_out=90112 dropped=1122")`, not
  `print("Next, confirm with X...")`.
- **No names of teammates, or references to who should do what, anywhere in
  `.py`/`.json`/config files.** Ownership and coordination live in
  `CLAUDE.md` and `PREPROCESSING.md`, not in code.
- **Docstrings and comments explain WHAT the code does and WHY, technically**
  (a one-line purpose, a `Usage:` block, a non-obvious invariant) — not
  project status, scaffolding apologies ("currently a stub"), or
  motivational asides. A skipped test's `reason=` names the missing
  dependency in one short phrase, nothing more.
- **A TODO is a single short line naming the missing input**, e.g.
  `# TODO: quarter/models/SMART columns`, never a sentence addressed to a
  person.
- Citing a spec section in a comment (e.g. "see CLAUDE.md §6") is fine when
  it points at a real, load-bearing constraint — keep these rare and short.

Applies to every file in `data_pipeline/`, present and future.

## 11. What NOT to build in this lane

- Do not implement `RecoveryEnv`, the hazard model, PPO, baselines, or the safety filter — that's Aniruddh's T2–T9.
- Do not implement the horizon label `Y_H`, survival/hazard fitting, or calibration metrics — that's Aakarsh's A2–A5. Your job stops at clean timelines with censoring info exposed.
- Do not write the literature verification table or the Phase-2 deck — that's Aditi's C1–C6.
- Do not change `contracts/INTERFACES.md` or `contracts/contracts.py` once they exist — those are Aniruddh's, frozen after week 1, changeable only with all four teammates' sign-off.
