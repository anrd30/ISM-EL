"""Post-registration comparator: the residual Q-policy retrained on a population's own training split.

On the holdout split of one population (the last 4 of the 14 days) the arms are: the residual Q-policy retrained on that
population's training split (3 models), the residual Q-policy zero-shot from population A (the 3 models trained on the
A training split), ``risk_aware_rule_based``, ``rule_seasonal_forecast (hand-built)``, ``predictive_controller (history)``,
``ORACLE_lookahead`` and ``DP_hindsight_optimum``. Evaluation fleet, sigma 0.25, history rows 576, seeds 0 to 19, the same
scenarios for every arm. Learned models with several seeds are averaged within each scenario first.

This evaluation was run after the pre-registered evaluation of ``agent/PREREGISTRATION_2026-11-19.md`` and does not replace it.

Usage::

    python -m agent.evaluate_residual_q_per_population --holdout-csv <csv> --retrained <run_dir> --zero-shot <run_dir> --out <dir>
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Sequence

from agent.evaluate_predictive import CONTROLLER, OPTIMUM, ORACLE, RULE, SEASONAL, gap_table
from agent.predictive_controller import PredictiveController
from agent.residual_q import ResidualQPolicy
from agent.seasonal_rule import SeasonalForecastRule
from baselines.dp_optimum import DPHindsightOptimum
from baselines.evaluate import _build_config, compare, forced_seed_sets, to_markdown
from baselines.policies import OracleLookahead, RiskAwareRuleBased
from contracts.contracts import PRE_FAILURE_HISTORY_ROWS

__all__ = ["main"]

RETRAINED = "residual-Q (retrained on its own training split)"
ZERO_SHOT = "residual-Q (zero-shot transfer from A)"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.evaluate_residual_q_per_population", description=__doc__.split("\n\n")[0])
    parser.add_argument("--holdout-csv", type=Path, required=True)
    parser.add_argument("--retrained", type=Path, required=True, help="directory holding seed*/model.pkl of the retrained models")
    parser.add_argument("--zero-shot", type=Path, required=True, help="directory holding seed*/model.pkl of the models trained on A")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    ns = argparse.Namespace(hazard_error_sigma=0.25, eval_fleet=True, latency_filter=False, latency_filter_causal=False)
    config = dataclasses.replace(_build_config(ns), history_rows=PRE_FAILURE_HISTORY_ROWS)
    arms = [RiskAwareRuleBased(config), SeasonalForecastRule(config), PredictiveController(config, name=CONTROLLER)]
    retrained = [ResidualQPolicy.from_dir(d, config, f"residual_q retrained {d.name}") for d in sorted(args.retrained.glob("seed*"))]
    zero_shot = [ResidualQPolicy.from_dir(d, config, f"residual_q zero-shot {d.name}") for d in sorted(args.zero_shot.glob("seed*"))]
    arms += retrained + zero_shot + [OracleLookahead(config), DPHindsightOptimum(config)]
    episodes = compare(arms, profiles=[args.holdout_csv], seeds=list(range(args.seeds)), config=config)
    args.out.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.out / "episodes.csv", index=False, lineterminator="\n")
    treatments = {
        SEASONAL: [SEASONAL],
        CONTROLLER: [CONTROLLER],
        RETRAINED: [a.name for a in retrained],
        ZERO_SHOT: [a.name for a in zero_shot],
        ORACLE: [ORACLE],
        OPTIMUM: [OPTIMUM],
    }
    table = gap_table(episodes, treatments, {RULE: [RULE], SEASONAL: [SEASONAL]})
    table.to_csv(args.out / "gaps.csv", index=False, lineterminator="\n", float_format="%.6g")
    text = "\n".join([
        f"# Post-registration: residual Q retrained per population, `{args.holdout_csv.name}`", "",
        "Run after the pre-registered evaluation; it does not replace it. Gains are paired by scenario seed; learned models "
        "with several seeds are averaged within each scenario first; `sem` is across scenarios.", "",
        to_markdown(forced_seed_sets(episodes)), "", to_markdown(table)])
    (args.out / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
