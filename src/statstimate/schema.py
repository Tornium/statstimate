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

import math
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np
import numpy.typing as npt

EstimateMethod = typing.Literal["snapshot", "keep previous", "change model"]
PersonalStats = Mapping[str, int | float | None]


@dataclass(frozen=True)
class PreviousStats:
    """
    An earlier stat score and data for the same player.

    Only use a real stat score (such as from a stat spy) here. Using an earlier estimate
    would cause the error in the estimates to accumulate.

    Attributes
    ----------
    stat_score: float
        The earlier stat score.
    personal_stats : PersonalStats
        The player's personal stats at the time of that stat score.
    days_ago : float
        Days between that stat score and the current personal stats.
    """

    stat_score: float
    personal_stats: PersonalStats
    days_ago: float


@dataclass(frozen=True)
class EstimatedStats:
    """
    The result of an estimate.

    Attributes
    ----------
    stat_score : float
        The estimated stat score of a user
    log_stat_score : float
        Natural log of the estimated stat score (what the models predict).
    method : EstimateMethod
        Which method produced the estimate.
    model_version : str
        The version of the model files used.
    """

    stat_score: float
    log_stat_score: float
    method: EstimateMethod
    model_version: str


def _to_float(value: object, field: str, row: int) -> float:
    if value is None:
        return 0.0
    elif isinstance(value, bool) or not isinstance(value, (Real, np.integer, np.floating)):
        raise ValueError(f"Record {row}: field {field!r} must be a number but was {type(value).__name__}")

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Record {row}: field {field!r} must be finite but was {value!r}")

    return number


def stats_matrix(records: typing.Sequence[PersonalStats], fields: typing.Sequence[str]) -> npt.NDArray[np.float64]:
    """
    Convert personal-stats records to an array of shape (len(records), len(fields)).

    Missing fields and ``None`` become 0 (the convention used in training). Unknown fields are ignored.
    """

    stats_data = np.zeros((len(records), len(fields)), dtype=np.float64)
    for i, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Record {i}: expected a mapping of stat name to value but got {type(record).__name__}")

        for j, name in enumerate(fields):
            if name not in record:
                continue

            stats_data[i, j] = _to_float(record[name], name, i)

    return stats_data


def check_previous(previous: PreviousStats, row: int) -> None:
    """
    Check the stats provided for the previous stats to ensure the values are correct. If some
    value is incorrect, a `ValueError` will be raised.
    """

    if not isinstance(previous, PreviousStats):
        raise ValueError(f"Record {row}: previous must be a statstimate.Previous but was {type(previous).__name__}")
    elif _to_float(previous.stat_score, "stat_score", row) <= 0:
        raise ValueError(f"Record {row}: previous stat score must be > 0 but was {previous.stat_score!r}")
    elif _to_float(previous.days_ago, "days_ago", row) < 0:
        raise ValueError(f"Record {row}: previous days_ago must be >= 0 but was {previous.days_ago!r}")
