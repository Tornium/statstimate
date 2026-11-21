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

import numpy as np
import pytest

from statstimate import ModelSettings
from statstimate.estimator import choose_method
from statstimate.schema import stats_matrix


def test_missing_and_none_become_zero_and_unknown_fields_are_ignored() -> None:
    matrix = stats_matrix([{"a": 5, "b": None, "zzz": 1}, {}], ["a", "b", "c"])
    assert matrix.tolist() == [[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert matrix.dtype == np.float64


def test_numpy_scalars_are_accepted() -> None:
    assert stats_matrix([{"a": np.int64(7), "b": np.float32(1.5)}], ["a", "b"]).tolist() == [[7.0, 1.5]]


@pytest.mark.parametrize("bad", [True, "12", float("nan"), float("inf"), [1]])
def test_bad_values_are_rejected(bad: object) -> None:
    with pytest.raises(ValueError):
        stats_matrix([{"a": bad}], ["a"])


def test_non_mapping_record_is_rejected() -> None:
    with pytest.raises(ValueError):
        stats_matrix([[1, 2]], ["a"])  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("days", "expected"),
    [(None, "snapshot"), (0, "keep previous"), (3, "keep previous"), (3.01, "change model"), (400, "change model")],
)
def test_rule(days: float | None, expected: str) -> None:
    assert choose_method(days, keep_days=3) == expected


def test_settings_round_trip(settings: ModelSettings) -> None:
    assert ModelSettings.from_dict(settings.to_dict()) == settings


def test_settings_reject_wrong_feature_order(settings: ModelSettings) -> None:
    raw = settings.to_dict()
    raw["features"] = list(reversed(raw["features"]))
    with pytest.raises(ValueError, match="snapshot features"):
        ModelSettings.from_dict(raw)


def test_settings_reject_missing_key(settings: ModelSettings) -> None:
    raw = settings.to_dict()
    del raw["keep_days"]
    with pytest.raises(ValueError, match="keep_days"):
        ModelSettings.from_dict(raw)


def test_settings_reject_bad_json() -> None:
    with pytest.raises(ValueError):
        ModelSettings.from_json("{nope")


def test_settings_without_version_use_the_default(settings: ModelSettings) -> None:
    raw = {k: settings.to_dict()[k] for k in ("features", "change_features", "base_features", "keep_days")}
    loaded = ModelSettings.from_json(json.dumps(raw))
    assert loaded.model_version == "unknown" and loaded.snapshot_file == "snapshot.ubj"
