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

"""XGBoostPredictor logic, tested against a stand-in booster so it runs without XGBoost."""

from __future__ import annotations

import types

import numpy as np
import pytest

from statstimate import predictor


class FakeBooster:
    def __init__(self, names: list[str] | None, n_features: int, rounds: int, best: str | None) -> None:
        self.feature_names = names
        self._n_features, self._rounds, self._best = n_features, rounds, best
        self.params: dict[str, object] = {}
        self.predict_calls: list[tuple[object, ...]] = []

    def load_model(self, raw: bytearray) -> None:
        assert isinstance(raw, bytearray)

    def set_param(self, params: dict[str, object]) -> None:
        self.params.update(params)

    def num_features(self) -> int:
        return self._n_features

    def num_boosted_rounds(self) -> int:
        return self._rounds

    def attr(self, key: str) -> str | None:
        return self._best if key == "best_iteration" else None

    def inplace_predict(self, data: np.ndarray, iteration_range: tuple[int, int]) -> np.ndarray:
        self.predict_calls.append((data, iteration_range))
        return np.ones(len(data), dtype=np.float32)


@pytest.fixture
def make(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    def _make(names=("a", "b"), n_features=2, rounds=100, best="49", **kwargs):  # type: ignore[no-untyped-def]
        booster = FakeBooster(list(names) if names is not None else None, n_features, rounds, best)
        fake = types.SimpleNamespace(Booster=lambda: booster, core=types.SimpleNamespace(XGBoostError=RuntimeError))
        monkeypatch.setattr(predictor, "xgboost", fake)
        return predictor.XGBoostPredictor(b"model", ["a", "b"], **kwargs), booster

    return _make


def test_uses_trees_up_to_the_best_iteration(make) -> None:  # type: ignore[no-untyped-def]
    model, booster = make(rounds=100, best="49")
    assert model.trees_used == 50 and model.iteration_range == (0, 50)
    out = model.predict(np.zeros((3, 2), dtype=np.float32))
    assert out.dtype == np.float64 and out.shape == (3,)
    assert booster.predict_calls[0][1] == (0, 50)


def test_uses_every_tree_when_the_file_is_trimmed(make) -> None:  # type: ignore[no-untyped-def]
    model, _ = make(rounds=50, best="49")
    assert model.trees_used == 50 and model.iteration_range == (0, 0)
    model, _ = make(rounds=50, best=None)
    assert model.iteration_range == (0, 0)


def test_feature_name_mismatch_is_rejected(make) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        make(names=("b", "a"))


def test_feature_count_mismatch_is_rejected(make) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        make(names=None, n_features=3)
