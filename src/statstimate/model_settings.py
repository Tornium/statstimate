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

import json
import typing
from dataclasses import dataclass

from .feature_engineering import change_feature_names, check_base_features, snapshot_feature_names


@dataclass(frozen=True)
class ModelSettings:
    """
    Settings of a model stored in model-settings.json.

    Required keys: ``base_features``, ``features``, ``change_features``, ``keep_days``.
    Optional keys: ``model_version``, ``snapshot_file``, ``change_file``.
    """

    base_features: tuple[str, ...]
    features: tuple[str, ...]
    change_features: tuple[str, ...]
    keep_days: float

    model_version: str = "unknown"
    snapshot_file: str = "snapshot.ubj"
    change_file: str = "change.ubj"

    @classmethod
    def from_dict(cls, raw_data: dict):
        try:
            settings = cls(
                base_features=tuple(raw_data["base_features"]),
                features=tuple(raw_data["features"]),
                change_features=tuple(raw_data["change_features"]),
                keep_days=float(raw_data["keep_days"]),
                model_version=str(raw_data.get("model_version", "unknown")),
                snapshot_file=str(raw_data.get("snapshot_file", "snapshot.ubj")),
                change_file=str(raw_data.get("change_file", "change.ubj")),
            )
        except KeyError as e:
            raise ValueError(f"The settings file is missing the key {e.args[0]!r}") from None
        except (TypeError, ValueError) as e:
            raise ValueError(f"The settings file has an invalid value: {e}") from None

        settings.validate()
        return settings

    @classmethod
    def from_json(cls, text: str):
        raw_data = json.loads(text)

        if not isinstance(raw_data, dict):
            raise ValueError("The settings file must contain a JSON object")

        return cls.from_dict(raw_data)

    def validate(self) -> None:
        check_base_features(self.base_features)

        if list(self.features) != snapshot_feature_names(self.base_features):
            raise ValueError(
                "The snapshot features in the settings file don't match this library's features; "
                "the model was made for a different version of statstimate"
            )
        elif list(self.change_features) != change_feature_names(self.base_features):
            raise ValueError(
                "The change features in the settings file don't match this library's features; "
                "the model was made for a different version of statstimate"
            )
        elif not self.keep_days >= 0:
            raise ValueError(f"keep_days must be >= 0 but was {self.keep_days}")

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "model_version": self.model_version,
            "base_features": list(self.base_features),
            "features": list(self.features),
            "change_features": list(self.change_features),
            "keep_days": self.keep_days,
            "snapshot_file": self.snapshot_file,
            "change_file": self.change_file,
        }
