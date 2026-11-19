# Decision record: pre-failure load history in the reset info (contract 1.5.0)

Status: proposal, needs sign-off of all four teammates (a contract change). Not yet reviewed by the other lanes.
Nothing changes while `SimConfig.history_rows` is 0, the default.

Code: `contracts/contracts.py` (version 1.5.0, `PRE_FAILURE_HISTORY_ROWS`, `RESET_HISTORY_INFO_KEYS`),
`contracts/INTERFACES.md` (section 6 and the changelog), `sim/config.py` (`history_rows`),
`sim/recovery_env.py` (reset), `baselines/policies.py` (`NeedsHistory`), `baselines/evaluate.py` (hand-over);
tests in `sim/test_pre_failure_history.py`. Evidence: `agent/evidence/pre-failure-history-2026-11-19/`.

## Decision

- **What a deployed controller has.** A system that rebuilds after a drive failure has monitored its load before
  the failure. The simulator starts every episode at a random trace row and exposes only the current load, so a
  policy that forecasts from history begins with none. `SimConfig.history_rows = N` makes the reset `info` carry the
  `N` trace rows before the start row (foreground bandwidth in MB/s, read IOPS, write IOPS), oldest first, ending at
  row `trace_start_row - 1`; the reset observation carries row `trace_start_row`. The contract constant is
  `PRE_FAILURE_HISTORY_ROWS = 576` (two days at 300 s).
- **Hand-over.** `baselines.policies.NeedsHistory.set_history(fg_bw, read_iops, write_iops)` is called by the
  harness after `env.reset()` and `policy.reset()` and before the first policy call. Policies without it are
  unchanged. A history policy evaluated with `history_rows = 0` raises `ValueError`.
- **No wrap.** A drawn start row below `history_rows` is mapped onto `[history_rows, len(trace))` by
  `history_rows + drawn * (len(trace) - history_rows) // history_rows`; every other start row is unchanged, and the
  same random draw is consumed, so the fleet, hazard and failure streams of every seed are unchanged. The reset
  `info` records `trace_start_row_drawn` and `trace_start_row_remapped`.
- **Default off.** With `history_rows = 0` the reset and step outputs, `info` keys and values are unchanged; the
  contract's observation, action set, reward and required `info` keys do not change.

## Evidence

- **Off-by-one pinned.** `history[-1]` is trace row `start - 1`, `history[0]` is row `start - 576`, and the reset
  observation's foreground bandwidth is row `start`, on all 20 seeds of the holdout.
- **Flag off is identical.** The full test suite passes unchanged; `fixed_50` and `fixed_120` on
  `profile_diurnal_long` seeds 0 to 2 reproduce `baselines/evidence/diurnal-long-sigma0.25/episodes.csv` (the
  committed file stores floats to 15 significant digits; equality is to a relative 1e-12). A policy without
  `set_history` run with `history_rows = 576` gives the same episodes on every seed whose start row is unchanged.
- **Seeds whose start row changes** (seeds 0 to 19, `history_rows = 576`, the evaluation configuration):

| Profile (rows) | Seeds with a changed start row | Rule reward, those seeds, off / on |
| --- | --- | --- |
| holdout (1152) | 10: 0, 1, 6, 7, 8, 9, 13, 15, 17, 18 | -465.70 / -473.32 |
| altB (4032) | 1: 17 | -581.37 / -1676.22 |

  The rule and `fixed_50` were re-run on the history-on configuration for all 20 seeds of both profiles
  (`history_on_vs_off.csv`); rewards of the unchanged seeds are identical. Every later arm that uses the history
  shares the history-on scenarios.

## Decisions made during execution

- **Remap instead of redrawing.** Drawing `integers(history_rows, n)` changes every seed's start row, and a
  redraw consumes a different number of values for the seeds that wrap. The deterministic map keeps 10 of 20 (holdout)
  and 19 of 20 (altB) scenarios identical and changes the rest in a recorded way. Alternative rejected: shrinking
  the draw range.
- **History as `info` arrays plus a protocol,** not as a new observation field: the contract keeps the 12-field
  observation vector.
- **Optional keys only on reset.** Step `info` is unchanged.

## Limits

- **Short traces.** A 1152-row trace (the holdout) leaves 576 start rows when `history_rows = 576`; half the
  scenarios move. A trace not longer than `history_rows` is rejected at reset.
- **The map is not uniform.** Remapped start rows are spread over `[history_rows, n)` by a stretch of
  `[0, history_rows)`; the distribution of start rows is the original one except for those seeds.
- **The history is the replay trace itself.** For a replay file the "history" is earlier rows of the same
  selected aggregate,, which has a strong daily cycle by construction (selected for daily-cycle strength).
- **JSON.** The arrays are not JSON-serialisable; a logger must convert them.
