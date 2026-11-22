# Paper draft

`main.tex` builds with `IEEEtran.cls`, which ships with Overleaf and TeX Live and is
deliberately not vendored here. Upload `main.tex` to Overleaf and it compiles as-is.

## Rules for this draft

Every number is either traced to a committed result file or wrapped in `\TODO`. Do not
clear a `\TODO` by estimating. This project has retracted three findings that were
asserted before they were measured -- an oracle misread as a performance ceiling, an
aggregate ratio reported as a per-step counterfactual, and a sensitivity table built by
rescoring episodes under an objective they were not produced for. The macro is there to
make the remaining gaps visible, not tidy.

## Where each number comes from

| Claim in the draft | Source |
|---|---|
| random $-99.37$, constants, threshold family $-0.377$ | `arena/reference/search_baseline.json` |
| `_sine` $= -0.625$, threshold at 0.30 $= -0.641$ | same, plus `arena/test_evolve.py` proving the two are one function |
| iterative table, best-on-development selection | `arena/iterations/*/repeat*.json` |
| random search at matched budget | `python -m arena.search_baseline curve` |
| population search controls converge to $-1.588$ | `arena/evolve_runs/control-*/run.json` |
| 232 of 460 behaviourally constant, 19 beat the best constant | `arena/reference/search_baseline.json` |
| filter saturation on the SSD environment | per-scenario arrays in the 1.6.0 results |

## Open before submission

- Rescore the small-model roster at spec 1.6.0 so one table can hold every model.
- Land the clipped-decision instrumentation and report the fraction per environment.
- Finish the language-model arm of the population search.
- Decide whether the fitted-hazard environment is reported or declared absent.
- Bibliography: FunSearch, AlphaEvolve, Eureka, OR-Library, the Tencent trace, Backblaze.
