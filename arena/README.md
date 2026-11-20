# Recovery Scheduling Arena

Rank language models by the quality of the **control policy they write**, scored against
a **provably optimal reference**.

A model is given a storage recovery-bandwidth scheduling problem and writes a Python
function. The arena runs it against the exact optimum computed by dynamic programming,
and reports what fraction of the achievable gain it captured.

Spec version **1.0.0**. Every result records the version it was produced under.

---

## Three commands

```bash
git clone <this repo> && cd <repo> && git checkout arena/llm-controller-arena
pip install -r requirements.txt
python -m arena.run verify
```

`verify` builds the reference values (the first run solves the dynamic program, a few
minutes) and checks determinism. **If it does not print `VERIFY PASS`, stop** — results
from that checkout are not comparable with anyone else's.

Then:

```bash
python -m arena.run eval --submission arena/submissions/_template.py --name greedy-reference
python -m arena.run report
```

No large downloads. The arena uses the heuristic v0 hazard model so it runs from a clean
clone; nothing needs the 200 MB fitted artifacts.

---

## The score

For each scenario, with policy return `J`, rule return `J_rule`, optimum `J_star`:

```
G = (J - J_rule) / (J_star - J_rule)
```

| G | meaning |
|---|---|
| `1.0` | optimal |
| `0.0` | matched the reference rule |
| `< 0` | worse than the rule |
| `> 1` | impossible — a bug in the reference, reported not clipped |

`G` is dimensionless, so the reward's weights cannot flatter a submission.

A scenario counts only if the optimum beats the rule by more than `EPSILON_REL` times
the rule's own cost. The test is relative, not absolute, so rescaling the reward cannot
change which scenarios count. Below that
the safety filter forces the allocation and no policy can differ. The excluded count is
always printed.

The standard error is taken **across scenarios**. Seeds sharing a scenario are
near-replicates; pooling them as independent understates the error by about 1.7x.

The reference submission scores about **-11**. It is deliberately naive, to show the
interface and the scale without giving away a good policy.

---

## Writing a submission

One file, one function:

```python
def policy(obs: np.ndarray, state: dict) -> int:
    ...
    return action_index   # 0..3, indexing ACTIONS_MBPS = [20, 50, 80, 120]
```

Rules, all enforced:

- **Deterministic.** The same `(obs, state)` must give the same action. `verify` checks it.
- **10 ms per call.** Exceeding it fails the episode.
- **`state` is cleared on every reset**, so episodes are independent.
- `numpy` and the standard library only. No files, no network, no simulator internals,
  nothing about future load.

Put it at `arena/submissions/<model-name>/policy.py`.

---

## Tracks

**zero-shot** — the model sees the prompt once and submits once.

**iterative** — up to 5 rounds. After each round it receives its score **on the
development profile only**, plus any error. It never sees a held-out score.

Run both under identical conditions. The difference between them is a result.

---

## What is held out

| profile | role | shown to the model |
|---|---|---|
| `profile_diurnal_long_train.csv` | development, the only feedback source | yes |
| `profile_diurnal_long_holdout.csv` | evaluation | no |

The headline number is the held-out score. The **dev gap** — held-out minus development
— is printed for every submission: a model that scores well on development and badly on
held-out wrote a policy for the prompt, not for the problem.

---

## What is profiled

| column | what it catches |
|---|---|
| `G`, `sem` | the headline |
| `better` | scenarios beating the rule, out of the controllable ones |
| `viol` | violation fraction against the 0.05 budget — **a policy can score well and still breach** |
| `hours` | mean completion time, the coefficient-free view |
| `dev_gap` | overfitting to the prompt |
| `max_ms` | worst-case decision latency |
| `status` | `ok` or `error`: how often a model produces runnable code at all |

Rank on `G`, but read `viol` next to it. A submission that reaches `G = 0.8` while
breaching the budget has not solved the task.

---

## Collecting results

Each participant commits `arena/results/<model-name>.json`. `python -m arena.run report`
builds the leaderboard from whatever is in that folder. No server, no database.

---

## Rules that keep the arena honest

1. **The spec is frozen before the first run.** Any change bumps `SPEC_VERSION` and
   archives old results rather than migrating them.
2. **One configuration**, in `arena/config.py`: seeds, `EPSILON_REL`, the budgets, the
   profiles. Nothing varies run to run.
3. **Never clip `G > 1`.** It means the reference is wrong, and that must surface.
4. **Every result records the spec version, the submission path, and the timestamp.**

---

## Status

Working: scoring, reference generation with a validity check, the submission interface
with budget and determinism enforcement, the CLI, the leaderboard.

Not yet built, and wanted before this is distributed widely:

- **Sandboxing.** Submissions are imported in-process. LLM-generated code is untrusted;
  this needs a subprocess with an import allowlist, no network and a memory cap.
- **Hash-pinned populations**, so results from different machines are provably comparable.
- **The generation harness** for calling models (API and local) — see `arena/prompts/`.
- More held-out populations, including the stress set.
