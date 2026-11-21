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

import math

import numpy as np
import pytest

from statstimate.feature_engineering import (
    CHANGE_EXTRA_FEATURES,
    ENGINEERED_FEATURES,
    REQUIRED_STATS,
    change_feature_names,
    change_features,
    check_base_features,
    engineer_features,
    snapshot_feature_names,
    snapshot_features,
)
from statstimate.schema import stats_matrix

from .conftest import BASE


def test_feature_names_order() -> None:
    assert snapshot_feature_names(["a", "b"]) == ["a", "b", *ENGINEERED_FEATURES]
    assert change_feature_names(["a", "b"]) == ["a", "b", "d_a", "d_b", *CHANGE_EXTRA_FEATURES]


def test_missing_required_stat_is_rejected() -> None:
    with pytest.raises(ValueError, match="xantaken"):
        check_base_features([s for s in REQUIRED_STATS if s != "xantaken"])
    with pytest.raises(ValueError):
        engineer_features(np.zeros((1, 2)), ["a", "b"])


def test_engineered_values_match_hand_calculation(player: dict[str, float]) -> None:
    got = dict(zip(ENGINEERED_FEATURES, engineer_features(stats_matrix([player], BASE), BASE)[0], strict=True))

    donator_days, xanax, refills, cans = 400.0, 1200.0, 500.0, 300.0
    active_days = 3_000_000 / 86400 + 1
    # 100 refills are beyond the donator days, so they only refill 100 energy
    fixed = xanax * 250 + min(refills, donator_days) * 150 + max(refills - donator_days, 0) * 100
    sqrt_min, sqrt_max = math.sqrt(fixed + cans * 5), math.sqrt(fixed + cans * 45)
    energy = fixed + cans * 20 + 20 * 50
    expected = {
        "sqrt_donator_days": math.sqrt(donator_days),
        "sqrt_min_lifetime_energy": sqrt_min,
        "sqrt_max_lifetime_energy": sqrt_max,
        "whale_index": (sqrt_min + sqrt_max) / 2 * math.sqrt(donator_days) * 1.005**10,
        "xanax_per_day": xanax / donator_days,
        "energy_est": energy,
        "xan_per_active_day": xanax / active_days,
        "refills_per_active_day": refills / active_days,
        "energy_per_active_day": energy / active_days,
        "ext_per_xan": 40 / (xanax + 1),
        "attacks_per_active_day": 900 / active_days,
    }
    for name, value in expected.items():
        assert got[name] == np.float32(value), name


def test_zero_donator_days_are_clipped_to_one() -> None:
    engineered = engineer_features(stats_matrix([{"xantaken": 10}], BASE), BASE)[0]
    assert engineered[ENGINEERED_FEATURES.index("sqrt_donator_days")] == 1.0
    assert engineered[ENGINEERED_FEATURES.index("xanax_per_day")] == 10.0


def test_snapshot_features_shape_and_dtype(player: dict[str, float]) -> None:
    features = snapshot_features(stats_matrix([player, {}], BASE), BASE)
    assert features.shape == (2, len(BASE) + len(ENGINEERED_FEATURES))
    assert features.dtype == np.float32
    assert features[0, BASE.index("networth")] == np.float32(4_740_570_924_889)


def test_change_differences_are_float32(player: dict[str, float]) -> None:
    older = dict(player, networth=4_740_570_000_000, xantaken=1100)
    current, previous = stats_matrix([player], BASE), stats_matrix([older], BASE)
    features = change_features(
        current,
        previous,
        BASE,
        previous_log_stat_score=[np.log(80_000.0)],
        gap_days=[52.0],
        snapshot_current=[11.2],
        snapshot_previous=[11.0],
    )
    names = change_feature_names(BASE)
    row = dict(zip(names, features[0], strict=True))
    assert features.dtype == np.float32 and features.shape == (1, len(names))
    assert row["d_networth"] == np.float32(4_740_570_924_889) - np.float32(4_740_570_000_000)
    assert row["d_xantaken"] == 100
    assert row["gap_days"] == 52
    assert row["prev_log_y"] == np.float32(np.log(80_000.0))
    assert row["snap_growth"] == np.float32(11.2) - np.float32(11.0)
    assert row["snap_minus_prev_label"] == np.float32(11.2) - np.float32(np.log(80_000.0))
    energy_now = engineer_features(current, BASE)[0, ENGINEERED_FEATURES.index("energy_est")]
    energy_then = engineer_features(previous, BASE)[0, ENGINEERED_FEATURES.index("energy_est")]
    assert row["d_energy_est"] == energy_now - energy_then
    assert row["d_energy_per_day"] == (energy_now - energy_then) / np.float32(52)


def test_gap_below_one_day_uses_one_for_the_rate(player: dict[str, float]) -> None:
    current = stats_matrix([player], BASE)
    previous = stats_matrix([dict(player, xantaken=1199)], BASE)
    features = change_features(
        current,
        previous,
        BASE,
        previous_log_stat_score=[10.0],
        gap_days=[0.5],
        snapshot_current=[11.0],
        snapshot_previous=[11.0],
    )
    row = dict(zip(change_feature_names(BASE), features[0], strict=True))
    assert row["d_energy_per_day"] == row["d_energy_est"]
