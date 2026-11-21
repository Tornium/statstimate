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

import json
import math
from pathlib import Path

import numpy as np
import pytest

from statstimate import Estimator, ModelSettings, PreviousStats
from statstimate.feature_engineering import snapshot_features
from statstimate.schema import stats_matrix

from .conftest import StubChange, StubSnapshot


def _snapshot_log(estimator: Estimator, stats: dict[str, float]) -> float:
    features = snapshot_features(stats_matrix([stats], estimator.input_fields), estimator.input_fields)

    return float(StubSnapshot(list(estimator.settings.features)).predict(features)[0])


def test_snapshot_without_previous(estimator: Estimator, player: dict[str, float]) -> None:
    estimate = estimator.estimate(player)
    assert estimate.method == "snapshot"
    assert estimate.model_version == "test-1" == estimator.version
    assert estimate.log_stat_score == pytest.approx(_snapshot_log(estimator, player))
    assert estimate.stat_score == pytest.approx(math.exp(estimate.log_stat_score))


def test_recent_previous_is_kept(estimator: Estimator, player: dict[str, float]) -> None:
    estimate = estimator.estimate(player, PreviousStats(stat_score=85_000, personal_stats=player, days_ago=3))
    assert estimate.method == "keep previous"
    assert estimate.stat_score == pytest.approx(85_000)


def test_older_previous_uses_the_change_model(estimator: Estimator, player: dict[str, float]) -> None:
    older = dict(player, xantaken=1000)
    estimate = estimator.estimate(player, PreviousStats(stat_score=85_000, personal_stats=older, days_ago=52))
    assert estimate.method == "change model"
    assert estimate.stat_score == pytest.approx(85_000 * math.exp(0.01 * 52), rel=1e-6)


def test_change_model_receives_the_snapshot_estimates(settings: ModelSettings, player: dict[str, float]) -> None:
    snapshot, change = StubSnapshot(list(settings.features)), StubChange(list(settings.change_features))
    estimator = Estimator.from_predictors(settings, snapshot, change)
    older = dict(player, xantaken=1000)
    estimator.estimate(player, PreviousStats(stat_score=85_000, personal_stats=older, days_ago=52))
    row = dict(zip(settings.change_features, change.calls[0][0], strict=True))
    assert row["snap_cur"] == np.float32(_snapshot_log(estimator, player))
    assert row["snap_prev"] == np.float32(_snapshot_log(estimator, older))
    assert row["prev_log_y"] == np.float32(math.log(85_000))


def test_batch_matches_single_calls(estimator: Estimator, player: dict[str, float]) -> None:
    older = dict(player, xantaken=900, refills=450)
    cases = [
        (player, None),
        (player, PreviousStats(85_000, older, 1)),
        (player, PreviousStats(85_000, older, 30)),
        ({}, None),
        (older, PreviousStats(40_000, {}, 400)),
    ]
    batch = estimator.estimate_many([stats for stats, _ in cases], [previous for _, previous in cases])
    singles = [estimator.estimate(stats, previous) for stats, previous in cases]
    assert batch == singles
    assert [e.method for e in batch] == ["snapshot", "keep previous", "change model", "snapshot", "change model"]


def test_empty_batch(estimator: Estimator) -> None:
    assert estimator.estimate_many([]) == []


def test_model_inputs(estimator: Estimator, player: dict[str, float]) -> None:
    inputs = estimator.model_inputs(player, PreviousStats(85_000, player, 52))
    assert inputs["method"] == "change model"
    assert inputs["snapshot_features"].shape == (len(estimator.settings.features),)
    assert inputs["change_features"].shape == (len(estimator.settings.change_features),)
    assert estimator.model_inputs(player)["change_features"] is None


@pytest.mark.parametrize(
    "previous",
    [
        PreviousStats(0, {}, 10),
        PreviousStats(-5, {}, 10),
        PreviousStats(100, {}, -1),
        PreviousStats(float("nan"), {}, 10),
    ],
)
def test_invalid_previous_is_rejected(estimator: Estimator, player: dict[str, float], previous: PreviousStats) -> None:
    with pytest.raises(ValueError):
        estimator.estimate(player, previous)


def test_previous_length_mismatch(estimator: Estimator, player: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        estimator.estimate_many([player, player], [None])


def test_missing_model_directory_gives_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model-settings.json"):
        Estimator(model_path=tmp_path)


def test_settings_are_checked_before_loading_models(tmp_path: Path, settings: ModelSettings) -> None:
    raw = settings.to_dict()
    raw["features"] = raw["features"][:-1]
    (tmp_path / "model-settings.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="snapshot features"):
        Estimator(model_path=tmp_path)
