"""v1 failure-hazard model: a calibrated bootstrap ensemble of gradient-boosted trees.

Satisfies the frozen ``contracts.contracts.HazardModel`` interface. The
ensemble is fitted by ``hazard/fit_v1.py`` on the training split of the
Backblaze timelines and loaded from ``hazard/artifacts/v1/``.

Prediction
----------
Each of the ensemble's boosters returns a raw probability for a drive-day. The
booster's own isotonic map (stored as threshold arrays and applied with linear
interpolation, clipped at both ends) turns it into a calibrated probability.
``p_horizon`` is the mean of the calibrated outputs and ``uncertainty`` is
their population standard deviation, clipped to the contract's ``[0, 0.5]``.
Both refer to the label horizon of the fit, 720 hours (30 days); any other
``horizon_hours`` raises ``ContractViolation``.

Features
--------
The input mapping holds the value columns recorded in ``meta.json`` (SMART
levels and trailing 7- and 30-day differences) and, optionally, ``model``, the
drive model name as a string. A value column that is absent, ``None``, a
``bool``, non-numeric, NaN or infinite is passed to the trees as NaN, and its
``<column>_missing`` flag is set to 1; no value is imputed and a missing value
is never replaced by zero. A ``model`` that is absent or not one of the
fitted categories is passed as a missing category. Keys that are not feature
names are ignored. Negative values are valid inputs, since differences can be
negative.

The fit used ``<column>_missing`` equal to "the value is NaN" for every row, so
the flags are derived from the values here and a flag key in the input is not
read.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from contracts.contracts import (
    HORIZON_HOURS,
    ContractViolation,
    hazard_p_to_per_hour,
    hazard_uncertainty_to_per_hour,
)

__all__ = ["ARTIFACT_DIR", "FittedHazardV1", "build_feature_frame", "feature_columns"]

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "v1"

MODEL_KEY = "model"
_UNCERTAINTY_MAX = 0.5
_BELOW_ONE = math.nextafter(1.0, 0.0)


@dataclass(frozen=True)
class _Ensemble:
    meta: dict
    boosters: tuple[xgb.Booster, ...]
    thresholds: tuple[tuple[np.ndarray, np.ndarray], ...]


@lru_cache(maxsize=None)
def _load(directory: Path) -> _Ensemble:
    """Load and cache the boosters, calibrator thresholds and metadata of ``directory``."""
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    boosters = []
    for name in meta["boosters"]:
        booster = xgb.Booster()
        booster.load_model(directory / name)
        booster.set_param({"device": "cpu"})
        boosters.append(booster)
    arrays = np.load(directory / meta["calibrators"])
    thresholds = tuple((arrays[f"x_{i:02d}"], arrays[f"y_{i:02d}"]) for i in range(len(boosters)))
    return _Ensemble(meta=meta, boosters=tuple(boosters), thresholds=thresholds)


def feature_columns(directory: Path = ARTIFACT_DIR) -> tuple[str, ...]:
    """Return the model input columns in training order (values, flags, ``model``)."""
    return tuple(_load(directory).meta["feature_columns"])


def _numeric(value: object) -> float:
    """Return ``value`` as a finite float, or NaN if it is not a usable number."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return math.nan
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(x):
        return math.nan
    # A value finite in float64 can still overflow float32 (the dtype the model
    # frame is built in), silently becoming +-inf, which np.isnan would not
    # catch; recheck finiteness after the same narrowing the frame will do.
    with np.errstate(over="ignore"):
        narrowed = float(np.float32(x))
    return x if math.isfinite(narrowed) else math.nan


def build_feature_frame(features: Mapping[str, float], directory: Path = ARTIFACT_DIR) -> pd.DataFrame:
    """Return the one-row model input for ``features`` (see the module docstring)."""
    if not isinstance(features, Mapping):
        raise ContractViolation(f"features must be a mapping, got {type(features).__name__}")
    meta = _load(directory).meta
    values = np.array([_numeric(features.get(name)) for name in meta["value_columns"]], dtype=np.float32)
    row = dict(zip(meta["value_columns"], values))
    row.update(zip(meta["flag_columns"], np.isnan(values).astype(np.float32)))
    name = features.get(MODEL_KEY)
    row[MODEL_KEY] = pd.Categorical(
        [name if isinstance(name, str) and name in meta["model_categories"] else None],
        categories=meta["model_categories"],
    )
    return pd.DataFrame(row)[meta["feature_columns"]]


@dataclass(frozen=True)
class FittedHazardV1:
    """Fitted v1 implementation of ``contracts.contracts.HazardModel``.

    Calling an instance returns ``(p_horizon, uncertainty)`` over 720 hours as
    builtin floats: the mean and the standard deviation of the calibrated
    outputs of the ensemble's boosters. The artifacts are read once per
    directory and cached.

    Attributes:
        directory: Folder holding ``meta.json``, the boosters and the calibrators.
    """

    directory: Path = ARTIFACT_DIR

    def __post_init__(self) -> None:
        _load(self.directory)

    def __call__(
        self, features: Mapping[str, float], horizon_hours: float = HORIZON_HOURS
    ) -> tuple[float, float]:
        """Return ``(p_horizon, uncertainty)`` for one drive.

        Raises:
            ContractViolation: if ``features`` is not a mapping, if
                ``horizon_hours`` is not a finite positive real number, or if
                it differs from the fitted horizon of 720 hours.
        """
        hazard_p_to_per_hour(0.0, horizon_hours)
        if float(horizon_hours) != HORIZON_HOURS:
            raise ContractViolation(
                f"horizon_hours must equal the fitted horizon {HORIZON_HOURS}, got {horizon_hours!r}"
            )
        ensemble = _load(self.directory)
        frame = build_feature_frame(features, self.directory)
        matrix = xgb.DMatrix(frame, enable_categorical=True)
        calibrated = np.array(
            [
                np.interp(float(booster.predict(matrix)[0]), x, y)
                for booster, (x, y) in zip(ensemble.boosters, ensemble.thresholds)
            ],
            dtype=np.float64,
        )
        p = min(max(float(calibrated.mean()), 0.0), _BELOW_ONE)
        uncertainty = min(float(calibrated.std()), _UNCERTAINTY_MAX)
        hazard_uncertainty_to_per_hour(uncertainty, p, horizon_hours)
        return p, uncertainty
