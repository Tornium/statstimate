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

import functools
import typing

import numpy as np
import numpy.typing as npt

# The following stats are required from the personal stats as they're used to compute some
# "reflected" features.
REQUIRED_STATS: typing.Tuple[str, ...] = (
    "daysbeendonator",
    "xantaken",
    "refills",
    "energydrinkused",
    "useractivity",
    "statenhancersused",
    "lsdtaken",
    "exttaken",
    "attackswon",
)

# These are the "reflected" features that are engineered from the above stats.
ENGINEERED_FEATURES: tuple[str, ...] = (
    "sqrt_donator_days",
    "sqrt_min_lifetime_energy",
    "sqrt_max_lifetime_energy",
    "whale_index",
    "xanax_per_day",
    "energy_est",
    "xan_per_active_day",
    "refills_per_active_day",
    "energy_per_active_day",
    "ext_per_xan",
    "attacks_per_active_day",
)

# These are the "reflected" features engineered from the change in stats to be used for the
# change model instead of the snapshot model.
CHANGE_EXTRA_FEATURES: tuple[str, ...] = (
    "prev_log_y",
    "gap_days",
    "energy_est",
    "whale_index",
    "d_energy_est",
    "d_energy_per_day",
    "snap_cur",
    "snap_prev",
    "snap_growth",
    "snap_minus_prev_label",
)


_ENERGY_INDEX = ENGINEERED_FEATURES.index("energy_est")
_WHALE_INDEX = ENGINEERED_FEATURES.index("whale_index")


def snapshot_feature_names(base_features: typing.Sequence[str]) -> list[str]:
    """
    The names of the features ordered for the snapshot model.
    """

    return [*base_features, *ENGINEERED_FEATURES]


def change_feature_names(base_features: typing.Sequence[str]) -> list[str]:
    """
    The names of the features ordered for the change model.
    """

    return [*base_features, *(f"d_{c}" for c in base_features), *CHANGE_EXTRA_FEATURES]


def check_base_features(base_features: typing.Sequence[str]) -> None:
    """
    Raises ``ValueError`` if the base features lack a stat the engineered features need.
    """

    _stat_columns(tuple(base_features))


@functools.lru_cache(maxsize=32)
def _stat_columns(base_features: tuple[str, ...]) -> dict[str, int]:
    missing = [s for s in REQUIRED_STATS if s not in base_features]

    if missing:
        raise ValueError(f"Base features are missing stats required by the engineered features: {missing}")

    return {name: base_features.index(name) for name in REQUIRED_STATS}


def engineer_features(stats: npt.ArrayLike, base_features: typing.Sequence[str]) -> npt.NDArray[np.float32]:
    """
    Compute the engineered features for the snapshot model.

    Parameters
    ----------
    stats : array-like of shape (n, len(base_features))
        Personal stats in ``base_features`` order.
    base_features : sequence of str
        The model's base features.

    Returns
    -------
    numpy.ndarray of float32 of shape (n, len(ENGINEERED_FEATURES))
    """

    s = np.asarray(stats, dtype=np.float64)
    columns = _stat_columns(tuple(base_features))

    def stat(name: str) -> npt.NDArray[np.float64]:
        return s[:, columns[name]]

    donator_days = np.maximum(stat("daysbeendonator"), 1.0)
    xanax = stat("xantaken")
    refills = stat("refills")
    cans = stat("energydrinkused")
    active_days = stat("useractivity") / 86400 + 1

    donator_refills = np.minimum(refills, donator_days)
    fixed_energy = xanax * 250 + donator_refills * 150 + np.maximum(refills - donator_days, 0) * 100
    sqrt_donator_days = np.sqrt(donator_days)
    sqrt_min = np.sqrt(fixed_energy + cans * 5)
    sqrt_max = np.sqrt(fixed_energy + cans * 45)
    # Same operation order as training: ((min + max) / 2) * sqrt_donator_days * 1.005 ** SEs
    whale_index = (sqrt_min + sqrt_max) / 2 * sqrt_donator_days * 1.005 ** stat("statenhancersused")
    energy = fixed_energy + cans * 20 + stat("lsdtaken") * 50

    features = [
        sqrt_donator_days,
        sqrt_min,
        sqrt_max,
        whale_index,
        xanax / donator_days,
        energy,
        xanax / active_days,
        refills / active_days,
        energy / active_days,
        stat("exttaken") / (xanax + 1),
        stat("attackswon") / active_days,
    ]
    return np.column_stack(features).astype(np.float32)


def snapshot_features(stats: npt.ArrayLike, base_features: typing.Sequence[str]) -> npt.NDArray[np.float32]:
    """
    The snapshot model's input with shape (n, len(snapshot_feature_names(base_features))).
    """

    stats_array = np.asarray(stats, dtype=np.float64)
    return np.hstack([stats_array.astype(np.float32), engineer_features(stats_array, base_features)])


def change_features(
    current_stats: npt.ArrayLike,
    previous_stats: npt.ArrayLike,
    base_features: typing.Sequence[str],
    *,
    previous_log_stat_score: npt.ArrayLike,
    gap_days: npt.ArrayLike,
    snapshot_current: npt.ArrayLike,
    snapshot_previous: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    """
    The change model's input with shape (n, len(change_feature_names(base_features))).

    Parameters
    ----------
    current_stats, previous_stats : array-like of shape (n, len(base_features))
        Personal stats now and at the time of the earlier label.
    base_features : sequence of str
        The model's base features.
    previous_log_stat_score : array-like of shape (n,)
        Natural log of the earlier label.
    gap_days : array-like of shape (n,)
        Days between the earlier label and now.
    snapshot_current, snapshot_previous : array-like of shape (n,)
        The snapshot model's predictions (log stat score) for the current and the earlier personal stats.
    """

    current = np.asarray(current_stats, dtype=np.float64)
    previous = np.asarray(previous_stats, dtype=np.float64)
    current32, previous32 = current.astype(np.float32), previous.astype(np.float32)
    engineered_current = engineer_features(current, base_features)
    engineered_previous = engineer_features(previous, base_features)

    previous_log = np.asarray(previous_log_stat_score, dtype=np.float64).astype(np.float32)
    gap = np.asarray(gap_days, dtype=np.float64).astype(np.float32)
    snap_current = np.asarray(snapshot_current, dtype=np.float64).astype(np.float32)
    snap_previous = np.asarray(snapshot_previous, dtype=np.float64).astype(np.float32)

    energy_change = engineered_current[:, _ENERGY_INDEX] - engineered_previous[:, _ENERGY_INDEX]
    extra = [
        previous_log,
        gap,
        engineered_current[:, _ENERGY_INDEX],
        engineered_current[:, _WHALE_INDEX],
        energy_change,
        energy_change / np.maximum(gap, np.float32(1)),
        snap_current,
        snap_previous,
        snap_current - snap_previous,
        snap_current - previous_log,
    ]

    out = np.hstack([current32, current32 - previous32, np.column_stack(extra)])
    if out.dtype != np.float32:
        raise AssertionError(f"Change features were promoted to {out.dtype}")

    return out
