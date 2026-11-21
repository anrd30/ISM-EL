# Arena run, spec 1.4.0

First run where both backends share one generation budget (8192 tokens), so Ollama
and HuggingFace results are comparable. Everything before 1.4.0 is archived under
`arena/results_pre140/` and `arena/results_spec120/` and must not be mixed in: the
budget differed between backends, and a model truncated mid-thought was recorded as
producing no policy.

- Run: 2026-11-20, 20:01 to 22:49 IST, one GPU (pinned by UUID), 10 models x 5 repeats.
- Temperature 0.7, seed = repeat index, so repeats measure output variance.
- Scored on `profile_diurnal_long_holdout.csv`, 20 seeds, never shown to any model.
- `G = (J_policy - J_rule) / (J_star - J_rule)`. G = 1 matches the DP optimum,
  G = 0 matches the tuned rule, negative is worse than the rule.

## Attempt accounting

| stage | count |
|---|---|
| generation attempts (10 models x 5) | 50 |
| replies containing `def policy` | 44 |
| submissions that ran to completion and scored | 27 |

The 6 that wrote no function: BioMistral-7B on all 5 repeats, DeepSeek-R1-32B on
repeat 5. BioMistral therefore has no result file and does not appear in the
leaderboard at all -- it is a 0/5, not a low score.

## Leaderboard

| model | runs | valid | G_mean | G_sd | G_best | G_worst | viol | over_budget | degenerate |
|---|---|---|---|---|---|---|---|---|---|
| `_sine-nonsense` | 1 | 1 | **-0.89** | - | -0.89 | -0.89 | 0.022 | 0/1 | 0/1 |
| unsloth/Qwen2.5-Coder-14B-4bit | 5 | 5 | -8.14 | 0.00 | -8.14 | -8.14 | 0.015 | 0/5 | 5/5 |
| Qwen/Qwen2.5-3B-Instruct | 5 | 2 | -9.88 | 2.46 | -8.14 | -11.61 | 0.026 | 0/2 | 1/2 |
| `_greedy-template` | 1 | 1 | -11.01 | - | -11.01 | -11.01 | 0.051 | 1/1 | 0/1 |
| unsloth/DeepSeek-R1-Distill-32B-4bit | 4 | 3 | -27.50 | 37.72 | -3.38 | -70.97 | 0.044 | 1/3 | 0/3 |
| qwen2.5-coder-7b (ollama) | 5 | 5 | -29.96 | 45.40 | -1.90 | -110.17 | 0.040 | 1/5 | 2/5 |
| microsoft/Phi-3-mini-128k | 5 | 3 | -45.28 | 64.33 | -8.14 | -119.57 | 0.037 | 1/3 | 2/3 |
| arcee-ai/AFM-4.5B | 5 | 4 | -89.61 | 161.54 | -6.44 | -331.90 | 0.186 | 1/4 | 1/4 |
| llama3.1-8b (ollama) | 5 | 3 | -97.94 | 95.99 | -10.33 | -200.54 | 0.165 | 2/3 | 0/3 |
| qwen2.5-coder-1.5b (ollama) | 5 | 2 | -143.05 | 180.47 | -15.44 | -270.66 | 0.248 | 2/2 | 1/2 |
| BioMistral/BioMistral-7B | 5 | 0 | - | - | - | - | - | - | - |

`valid` = produced runnable code. `over_budget` = violation fraction above 0.05.
`degenerate` = returned one action on every decision, i.e. a constant, not a controller.

## What the numbers mean

**No model beats a nonsense policy.** `arena/submissions/_sine/policy.py` is four
lines, `1 if math.sin(obs[2]) > 0.3 else 2`, and scores -0.89. The best model scores
-8.14, nine times worse. The sine function has no access to anything the models did
not have.

**The top model wins by refusing to act.** Action counts for
Qwen2.5-Coder-14B#1 are `[28891, 0, 0, 0]`: the slowest rate on every decision,
all five repeats, sd 0.00. The rebuild takes 120.4 hours instead of 54.1. The value
-8.14 recurs across Phi-3, qwen-7b and Qwen-3B because it *is* the score of
"always slowest". Several models independently converge on it.

**Modulating is punished.** Every model with a violation fraction above 0.05 scores
worse than the ones that sit still. The 10x weight on `latency_violation` means one
aggressive choice costs more than hours of delay.

## The caveat, stated before a reviewer states it

Score against mean completion time:

| policy | hours | G |
|---|---|---|
| Qwen-14B (constant slowest) | 120.4 | -8.14 |
| qwen2.5-coder-7b#3 | 63.1 | -1.90 |
| `_sine-nonsense` | 54.1 | -0.89 |
| `_greedy-template` | 38.4 | -11.01 |

G is close to a U-shape in a single scalar, mean bandwidth, with an optimum near
50-60 MB/s. The sine policy oscillates between actions 1 and 2 and lands in that
band by accident. As scored today the arena largely measures *did you pick an
average rate near the sweet spot*, which is a real finding about this environment
but is not yet a measurement of control quality.

The discriminating test is held-out populations where the sweet spot moves
(populations C, D, altB and the Toshiba hazard config). A policy that wins by a
lucky constant rate should collapse there; a policy that reads `bw_available`
should not. That is the next thing to build, and until it is run no claim about
model capability from this table is safe.

## Adding a model

Nothing here is a dependency of scoring: a hand-written policy needs only numpy
and the simulator.

    # local weights on a GPU
    python -m arena.generate --model hf:<org>/<model> \
        --out arena/submissions/<name>#1 --temperature 0.7 --seed 1
    # a local Ollama server
    python -m arena.generate --model ollama:<model> \
        --out arena/submissions/<name>#1 --temperature 0.7 --seed 1
    # a hosted API (key from the environment)
    python -m arena.generate --model api:openai/<model> --out arena/submissions/<name>#1

    python -m arena.run eval --submission arena/submissions/<name>#1/policy.py --name "<name>#1"
    python -m arena.run report --by-model

Rules for a result to belong in this table:

- Run `python -m arena.run verify` first. If it says FAIL the checkout is not comparable.
- Five repeats at temperature 0.7, seeds 1..5, named `<model>#<k>`. One run is an
  anecdote: G_sd here reaches 180.
- Do not edit `arena/prompts/task_prompt.md`. It is the experimental stimulus; changing
  it invalidates every score above.
- Do not change anything in `arena/config.py` without bumping `SPEC_VERSION` and
  re-running everything.
- A model that writes no function is a 0/N, not a missing row. Record it.

`run_all.sh` drives the whole roster; `rescore.sh` re-scores existing submissions
without calling any model.
