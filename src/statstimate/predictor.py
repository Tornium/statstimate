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

import typing

import numpy as np
import numpy.typing as npt

try:
    import xgboost
except ImportError:
    raise ImportError(
        "statstimate needs XGBoost to load its models. Install it with one of:\n"
        '  pip install "statstimate[cpu]"   # CPU-only XGBoost (servers, most users)\n'
        '  pip install "statstimate[gpu]"   # full XGBoost'
    ) from None


class XGBoostPredictor:
    """
    Wrapper around ``xgboost.Booster`` loaded from a raw .ubj model.
    """

    def __init__(
        self,
        raw_model: bytes,
        expected_features: typing.Sequence[str],
        threads: int | None = None,
        device: str | None = None,
        model_name: str = "model",
    ) -> None:
        booster = xgboost.Booster()

        try:
            booster.load_model(bytearray(raw_model))
        except xgboost.core.XGBoostError as e:
            raise ValueError(f"The {model_name} could not be loaded: {e}") from None

        if threads is not None:
            booster.set_param({"nthread": int(threads)})
        if device is not None:
            booster.set_param({"device": device})

        expected = list(expected_features)
        model_features = booster.feature_names
        if model_features is not None and list(model_features) != expected:
            raise ValueError(f"The {model_name}'s features don't match model-settings.json")
        if booster.num_features() != len(expected):
            raise ValueError(
                f"The {model_name} expects {booster.num_features()} features "
                f"but model-settings.json lists {len(expected)}"
            )

        total_rounds = booster.num_boosted_rounds()
        best_iteration = booster.attr("best_iteration")
        self.trees_used = (
            int(best_iteration) + 1
            if best_iteration is not None and int(best_iteration) + 1 < total_rounds
            else total_rounds
        )
        self.iteration_range = (0, self.trees_used) if self.trees_used < total_rounds else (0, 0)
        self.booster = booster

    def predict(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        out = self.booster.inplace_predict(features, iteration_range=self.iteration_range)
        return np.asarray(out, dtype=np.float64).reshape(-1)
