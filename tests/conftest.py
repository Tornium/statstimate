# Copyright (C) 2026 tiksan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import sys  # noqa: F401
import types  # noqa: F401

import numpy as np
import pytest

import xgboost  # noqa: F401
from statstimate import Estimator, ModelSettings
from statstimate.feature_engineering import REQUIRED_STATS, change_feature_names, snapshot_feature_names

BASE = (*REQUIRED_STATS, "networth", "bestdamage")


class StubSnapshot:
    """log stat score = 1 + 0.5 * log1p(energy_est). Deterministic, no XGBoost needed."""

    def __init__(self, names: list[str]) -> None:
        self.column = names.index("energy_est")
        self.calls: list[np.ndarray] = []

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert features.dtype == np.float32
        self.calls.append(features)
        return 1.0 + 0.5 * np.log1p(features[:, self.column].astype(np.float64))


class StubChange:
    """log growth = 0.01 * gap_days."""

    def __init__(self, names: list[str]) -> None:
        self.column = names.index("gap_days")
        self.calls: list[np.ndarray] = []

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert features.dtype == np.float32
        self.calls.append(features)
        return 0.01 * features[:, self.column].astype(np.float64)


@pytest.fixture
def settings() -> ModelSettings:
    return ModelSettings(
        base_features=BASE,
        features=tuple(snapshot_feature_names(BASE)),
        change_features=tuple(change_feature_names(BASE)),
        keep_days=3.0,
        model_version="test-1",
    )


@pytest.fixture
def estimator(settings: ModelSettings) -> Estimator:
    return Estimator.from_predictors(
        settings, StubSnapshot(list(settings.features)), StubChange(list(settings.change_features))
    )


@pytest.fixture
def player() -> dict[str, float]:
    return {
        "daysbeendonator": 400,
        "xantaken": 1200,
        "refills": 500,
        "energydrinkused": 300,
        "useractivity": 3_000_000,
        "statenhancersused": 10,
        "lsdtaken": 20,
        "exttaken": 40,
        "attackswon": 900,
        "networth": 4_740_570_924_889,
        "bestdamage": 5000,
    }
