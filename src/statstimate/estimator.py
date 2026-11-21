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

import os
import typing
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .feature_engineering import change_features, snapshot_features
from .model_settings import ModelSettings
from .predictor import XGBoostPredictor
from .schema import (
    EstimatedStats,
    EstimateMethod,
    PersonalStats,
    PreviousStats,
    check_previous,
    stats_matrix,
)

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable


MODEL_DIR_ENV = "STATSTIMATE_MODEL_DIR"


def _model_directory(model_dir: str | os.PathLike[str] | None) -> Traversable:
    if model_dir is not None:
        return Path(model_dir)

    env = os.environ.get(MODEL_DIR_ENV)
    if env:
        return Path(env)

    return resources.files("statstimate").joinpath("models")


def _read_model(location: Traversable, name: str) -> bytes:
    file_location = location.joinpath(name)
    if not file_location.is_file():
        raise ValueError(
            f"Model file {name!r} could not be found in {location}. Pass Estimator(model_dir=...) or set "
            f"{MODEL_DIR_ENV} to a folder containing model-settings.json, snapshot.ubj and change.ubj."
        )

    return file_location.read_bytes()


def choose_method(days_ago: float | None, keep_days: float) -> EstimateMethod:
    """
    Pick the estimation method.

    * no earlier stat score => "snapshot"
    * earlier stat socre at most ``keep_days`` old => "keep previous" (returned unchanged)
    * otherwise => "change model"
    """

    if days_ago is None:
        return "snapshot"
    elif days_ago <= keep_days:
        return "keep previous"

    return "change model"


@dataclass(frozen=True)
class EstimationRun:
    methods: typing.list[EstimateMethod]
    log_estimates: npt.NDArray[np.float64]
    snapshot_inputs: npt.NDArray[np.float32]
    change_inputs: npt.NDArray[np.float32] | None


class Estimator:
    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        device: typing.Optional[str] = None,
        threads: typing.Optional[int] = None,
    ) -> None:
        """
        Initialize a stat estimator.

        Load the model once and reuse it. Each time the model is loaded takes some time and some memory.

        Parameters
        ----------
        model_path : str | os.PathLike[str] | None
            Path to the directory with the model files. Defaults to the ``STATSTIMATE_MODEL_DIR`` environment
            variable and then the models bundled with the library.

        device : str | None
            XGBoost device for predictions, such as ``"cpu"`` or ``"cuda"``. ``None`` keeps XGBoost's default
            (the CPU). A GPU only helps for large batches and needs the full ``xgboost`` package.

        threads : int | None
            The number of threads to be used when estimating stats. For a web server with one worker per core,
            use 1. Using ``None`` utilizes XGBoost's default value of all cores.
        """

        model_directory = _model_directory(model_path)

        settings_file = _read_model(model_directory, "model-settings.json").decode("utf-8")
        settings = ModelSettings.from_json(settings_file)

        snapshot_file = _read_model(model_directory, settings.snapshot_file)
        change_file = _read_model(model_directory, settings.change_file)
        snapshot = XGBoostPredictor(
            snapshot_file, settings.features, threads=threads, device=device, model_name="snapshot model"
        )
        change = XGBoostPredictor(
            change_file, settings.change_features, threads=threads, device=device, model_name="change model"
        )

        self._setup(settings, snapshot, change)

    @classmethod
    def from_predictors(
        cls, settings: ModelSettings, snapshot_model: XGBoostPredictor, change_model: XGBoostPredictor
    ) -> Estimator:
        """
        Build an estimator around other model backends (such as ONNX, tests, etc).

        Each predictor's ``predict`` takes a float32 matrix in the column order of ``settings.features`` or
        ``settings.change_features`` and returns the log stat score for the snapshot model or the log growth
        since the earlier stat score the change model.
        """

        settings.validate()

        estimator = cls.__new__(cls)
        estimator._setup(settings, snapshot_model, change_model)

        return estimator

    def _setup(self, settings: ModelSettings, snapshot: XGBoostPredictor, change: XGBoostPredictor) -> None:
        self.settings = settings
        self._snapshot = snapshot
        self._change = change

    @property
    def version(self) -> str:
        return self.settings.model_version

    @property
    def input_fields(self) -> tuple[str, ...]:
        return self.settings.base_features

    def estimate(self, personal_stats: PersonalStats, previous_stats: PreviousStats | None = None) -> EstimatedStats:
        """
        Estimate one player's stat score.

        Parameters
        ----------
        personal_stats : mapping of str to number
            The player's current personal stats.
        previous_stats : PreviousStats | None
            An earlier real stat score for this player with the personal stats from that time, if there is one.
        """

        return self.estimate_many([personal_stats], [previous_stats])[0]

    def estimate_many(
        self,
        personal_stats: typing.Sequence[PersonalStats],
        previous_stats: typing.Sequence[PreviousStats | None] | None = None,
    ) -> list[EstimatedStats]:
        """
        Estimate many players at once. This is much faster than calling ``estimate`` in a loop.

        ``previous``, if given, must have one entry per player (``None`` where there's no earlier stat score).
        """

        run = self._run(personal_stats, previous_stats)

        return [
            EstimatedStats(
                stat_score=float(np.exp(log_estimate)),
                log_stat_score=float(log_estimate),
                method=method,
                model_version=self.version,
            )
            for log_estimate, method in zip(run.log_estimates, run.methods, strict=True)
        ]

    def model_inputs(
        self, personal_stats: PersonalStats, previous_stats: typing.Optional[PreviousStats] = None
    ) -> dict[str, typing.Any]:
        """
        The exact model inputs for one player, for debugging and for golden test cases.

        Returns ``{"method", "snapshot_features", "change_features"}``. The features are float32 arrays in model
        column order; ``change_features`` is ``None`` unless the change model runs.
        """

        run = self._run([personal_stats], [previous_stats])

        return {
            "method": run.methods[0],
            "snapshot_features": run.snapshot_inputs[0],
            "change_features": None if run.change_inputs is None else run.change_inputs[0],
        }

    def _run(
        self,
        personal_stats: typing.Sequence[PersonalStats],
        previous_stats: typing.Sequence[PreviousStats | None] | None,
    ) -> EstimationRun:
        count = len(personal_stats)
        previous_list: typing.Sequence[PreviousStats | None] = (
            previous_stats if previous_stats is not None else [None] * count
        )

        if len(previous_list) != count:
            raise ValueError(f"Got {count} personal-stats records but {len(previous_list)} previous entries")

        for i, entry in enumerate(previous_list):
            if entry is not None:
                check_previous(entry, i)

        base = self.settings.base_features
        current_stats = stats_matrix(personal_stats, base)
        snapshot_inputs = snapshot_features(current_stats, base)
        snapshot_current = (
            np.asarray(self._snapshot.predict(snapshot_inputs), dtype=np.float64).reshape(-1) if count else np.zeros(0)
        )

        methods: typing.List[EstimateMethod] = [
            choose_method(None if entry is None else float(entry.days_ago), self.settings.keep_days)
            for entry in previous_list
        ]
        log_estimates = snapshot_current.copy()

        for i, method in enumerate(methods):
            entry = previous_list[i]

            if method == "keep previous" and entry is not None:
                log_estimates[i] = np.log(float(entry.stat_score))

        rows = [i for i, method in enumerate(methods) if method == "change model"]
        change_inputs = None

        if rows:
            entries = [entry for entry in (previous_list[i] for i in rows) if entry is not None]
            previous_stats = stats_matrix([entry.personal_stats for entry in entries], base)
            previous_log = np.log(np.array([float(entry.stat_score) for entry in entries]))
            gap = np.array([float(entry.days_ago) for entry in entries])
            snapshot_previous = np.asarray(
                self._snapshot.predict(snapshot_features(previous_stats, base)), dtype=np.float64
            ).reshape(-1)

            change_inputs = change_features(
                current_stats[rows],
                previous_stats,
                base,
                previous_log_stat_score=previous_log,
                gap_days=gap,
                snapshot_current=snapshot_current[rows],
                snapshot_previous=snapshot_previous,
            )
            growth = np.asarray(self._change.predict(change_inputs), dtype=np.float64).reshape(-1)
            log_estimates[rows] = previous_log + growth

        return EstimationRun(methods, log_estimates, snapshot_inputs, change_inputs)
