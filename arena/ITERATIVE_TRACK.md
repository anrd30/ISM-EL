# Iterative track

Zero-shot is not how anyone uses these models. This track gives a model ten attempts,
shows it how each one scored, and measures what that buys.

## Why the baseline question comes first

`arena/submissions/_sine/policy.py` is `1 if math.sin(obs[2]) > 0.3 else 2`. It has been
described as a nonsense policy. It is not. `obs[2]` is foreground bandwidth, normalised
to roughly [0, 1], and sine is monotonic over that range, crossing 0.3 at 0.305. So the
policy is exactly:

    if foreground load > 0.305: rebuild at 50 MB/s, else rebuild at 80 MB/s

That is a load-aware threshold controller with a reasonable threshold. Measured
directly, `_sine` scores -0.893 on the holdout and a plain threshold at 0.30 on the same
field scores -0.901; the difference is noise. Actual nonsense, a uniform random action
every step, scores -142.19.

So the floor claim is not "every model loses to nonsense". Two claims are true and both
belong in the paper:

- Every model tested is beaten by a constant policy that returns 50 MB/s and reads none
  of its twelve observations (median G -1.246 against -2.62 for the best frontier model).
- Models are far above uniform random, so they are producing control, just bad control.

An exhaustive sweep of the single-threshold family puts its best member at median
G -0.377, which does not reach the tuned rule at 0.0, let alone the optimum at 1.0.
Excluding `ssd_latency`, where the filters saturate, the best threshold lands near +0.03:
it ties the reference controller and captures essentially none of the remaining gap to
optimal. The environments are therefore not solvable by picking one number, and do not
need to be made harder.

## The three arms

Everything below uses the same currency, **simulator evaluations**, because that is the
resource that actually supplies information. Tokens and wall-clock are reported
separately, as costs the search arm does not pay at all.

| arm | command | what it isolates |
|---|---|---|
| random search | `python -m arena.search_baseline` | what blind sampling gets for the same budget |
| no memory | `python -m arena.iterate --memory none` | reacting to a score with no recollection |
| full memory | `python -m arena.iterate --memory full` | reasoning across rounds |

The gap between the two `iterate` arms is what memory is worth. Without the no-memory
arm, an improvement cannot be distinguished from re-rolling with a hint. Without the
search arm, the whole track proves nothing, because with four actions and a couple of
thresholds blind sampling is a strong competitor.

## Running it

Control arm first, since it needs no model and the cache makes the curve free afterwards:

```bash
python -m arena.search_baseline scan            # ~460 members x 9 environments, parallel
python -m arena.search_baseline curve --budget 30
```

Then the model, on the GPU. Pin the card by UUID, not by index (see `run_gpu.sh`):

```bash
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-....
python -m arena.iterate --model hf:Qwen/Qwen2.5-Coder-7B-Instruct \
    --name qwen2.5-coder-7b --rounds 10 --repeats 5 --temperature 0.7
python -m arena.iterate --model hf:Qwen/Qwen2.5-Coder-7B-Instruct \
    --name qwen2.5-coder-7b --rounds 10 --repeats 5 --temperature 0.7 --memory none
```

Ollama works the same way with `--model ollama:qwen2.5-coder:7b`.

Weights are loaded once per process and reused across every round and repeat, so run a
model's two arms as two commands rather than interleaving them.

Budget note: a round costs one dev evaluation plus eight held-out evaluations, about
90 seconds of CPU at 20 seeds each. Ten rounds times five repeats is roughly an hour of
scoring per arm on top of generation. `--holdout-every 2` halves that if the curve does
not need every round.

## What the run records

One JSON per repeat under `arena/iterations/<name>-<memory>/`, plus every reply and every
policy file. Each round carries the dev result, the eight held-out results, prompt and
completion tokens, generation seconds and evaluation seconds.

Three things to read out of it:

- **Rounds to competence.** Where does dev G cross the constant-50 floor, and the sine
  floor, if it ever does.
- **Tokens to competence.** The same curve against cumulative tokens. This is where the
  model is compared against search, which spends none.
- **Overfitting to the feedback signal.** Dev and held-out are plotted together. The
  model never sees held-out, so if dev keeps improving after held-out turns over, the
  loop is fitting the development workload rather than learning control. That is a
  result, not a bug.

## What the model is and is not told

Feedback is the development score, the violation fraction, the rebuild time, the action
histogram, the slowest decision, and a note if the policy was constant. If the policy
crashed or blew the step budget, the harness error goes back verbatim and the model is
asked to repair it.

It is never told anything about the eight held-out environments, by name or by hint, and
`arena/test_iterate.py` asserts that the whole conversation contains none of their
profile names. `arena/prompts/task_prompt.md` is unchanged and must stay that way: it is
the experimental stimulus shared with the zero-shot track.
