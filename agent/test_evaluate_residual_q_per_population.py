"""Labels and argument handling of the per-population residual-Q comparator."""

from __future__ import annotations

import pytest

from agent.evaluate_residual_q_per_population import RETRAINED, ZERO_SHOT, main


def test_the_comparators_are_labelled_by_what_they_are() -> None:
    assert ZERO_SHOT == "residual-Q (zero-shot transfer from A)"
    assert RETRAINED == "residual-Q (retrained on its own training split)"


def test_the_holdout_and_both_model_directories_are_required() -> None:
    with pytest.raises(SystemExit):
        main(["--holdout-csv", "x.csv"])
