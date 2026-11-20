# Decision record: completion time against latency violations, without reward coefficients

Status: proposed. Not yet reviewed by the other lanes. Nothing here changes a default.

Code: `agent/frontier.py` (`python -m agent.frontier run|analyse`), `baselines/dp_optimum.py` (`violation_mu`); tests in
`agent/test_frontier.py`. Evidence: `agent/evidence/frontier-2026-11-20/` (`points/` one file per parameter, `frontier_by_scenario.csv`,
`frontier_summary.csv`, `frontier_area_0_to_0.1.csv`, `frontiers.png`).

## Decision

- **Frontier objective.** Per scenario, the hindsight optimum is solved with the interval cost `1 + mu * V` (one unit of time per
  interval and a price `mu` on the violation indicator `V`; no `alpha`, `beta` or `gamma`) for `mu` on a log grid of 31 points from 1e-2
  to 1e4; each point is the executed (completion hours, latency violation fraction). The horizon is 2400 steps and unfinished work
  has a very large terminal cost, so every plan completes.
- **Policy frontiers.** `risk_aware_rule_based` and `predictive_controller (history)` are run over 12 log-spaced `progress_value`
  from 1e-4 to 1e-1, on the same scenarios (evaluation fleet, sigma 0.25, history rows 576, seeds 0 to 19, populations A to D).
- **Constrained optimum.** Per scenario `H(0.05)`: the least completion hours among the executed points whose violation fraction is at
  most `LATENCY_VIOLATION_FRACTION_MAX` = 0.05.
- **Limit of the Lagrangian sweep.** A sweep over a price finds only points on the convex hull of the achievable set; with discrete
  actions a non-hull point can be missed, so the optimum's `H(0.05)` is an upper bound on the true constrained minimum, within the
  resolution of the `mu` grid.

## Evidence

Mean completion hours at violation fraction at most 0.05 (scenarios with at least one feasible point), controllable scenarios
under the primary definition (16 of 20 per population); the all-scenario figures are in `frontier_summary.csv`:

| Population | n | Optimum | Controller | Rule | Rule minus controller (sem) | Controller's curve below the rule's everywhere |
| --- | --- | --- | --- | --- | --- | --- |
| A | 16 | 36.292 | 36.526 | 36.740 | 0.214 (0.026) | 16 of 16 |
| B | 16 | 34.557 | 35.578 | 35.604 | 0.026 (0.099) | 0 of 16 |
| C | 16 | 34.484 | 34.859 | 35.135 | 0.276 (0.044) | 15 of 16 |
| D | 16 | 39.438 | 41.500 | 41.755 | 0.255 (0.182) | 0 of 16 |

- **All scenarios:** 17 (A), 18 (B), 17 (C) and 17 (D) of 20 scenarios have a feasible point; the others are the filter-forced
  scenarios whose violation fraction stays above 0.05 for every policy and every price. Means over the feasible scenarios: A 36.314 /
  36.549 / 36.760, B 34.505 / 35.537 / 35.560, C 34.652 / 35.024 / 35.299, D 39.476 / 41.471 / 41.686 (optimum / controller / rule).
- **Dominance.** The controller's achievable curve is at or below the rule's at every violation level on all 16 controllable scenarios
  of A and on 15 of 16 of C. On B and D it is not: the curves cross. At the 0.05 limit the controller is ahead of the rule on A
  and C (about 0.2 to 0.3 hours, 6 to 8 standard errors), and not distinguishable on B (0.026 +/- 0.099) and D (0.255 +/- 0.182).
- **Area between each curve and the optimum's** (hours x violation fraction, mean over controllable scenarios, range 0 to 0.1, where
  both policies' sweeps reach): controller 0.020 / 0.186 / 0.050 / 0.309 against rule 0.181 / 0.207 / 0.121 / 0.369 on A / B / C / D;
  the controller's curve is closer to the optimum's in every population.
- **Distance from the optimum at the limit.** Controller 0.23, 1.02, 0.38 and 2.06 hours above the optimum on A, B, C, D; the rule
  0.45, 1.05, 0.65 and 2.32.

## Decisions made during execution

- **Frontier plan horizon 2400 steps and terminal cost 1e9.** With the 1200-step horizon and a 1e4 terminal cost a plan at a high
  price preferred not to finish; the longer horizon and larger terminal cost keep every plan complete.
- **Areas over 0 to 0.7 are in `frontier_summary.csv`; areas over 0 to 0.1 are reported here.** The policy sweeps stop at a
  progress value of 0.1, so beyond a violation fraction of about 0.07 their curves are flat while the optimum's keep falling; the
  0 to 0.7 areas (e.g. D, controller 2.385 against rule 1.826) mostly measure that, not the region of interest.
- **Executed points only.** Planned values are not used; a point is the executed (hours, violation) of the plan on the scenario.

## Limits

- **Hindsight reference.** The optimum knows the future trace and the initial hazard; no deployable policy can be on its curve.
- **Policy frontiers are traced by one parameter.** The sweep of `progress_value` is one family of policies; a different
  parameterisation could reach a different frontier.
- **Hull only.** See the limit of the Lagrangian sweep above.
- **20 scenarios per population;** the controllable class has 16.
