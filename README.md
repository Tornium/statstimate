# Statstimate
Tornium's stat estimation library to estimate a user's stats from their personal stats.

This library provides the models and a python reference implementation to generate stat estimates. You can re-use the models in other languges using the [XGBoost](https://xgboost.readthedocs.io/en/stable/) library.

## Installation
To support both CPU and GPU based estimation, the statstimate library is split into two groups of dependencies:
- `statstimate[cpu]`: uses the `xgboost-cpu` library to provide CPU-only estimation
- `statstimate[gpu]`: uses the `xgboost` library to provide mostly GPU-only estimation

These can then be installed with `pip install statstimate[cpu]` and `pip install statstimate[gpu]` respectively.

> [!NOTE]
> The `xgboost-cpu` library currently only supports x86_64 (amd64) Linux and Windows, so any other platforms will need to utilize `statstimate[gpu]`

## Usage
TBA

### Accuracy (model 2026-11-13)

| Method | Typical miss | Within ±25% |
|----------------------------------------------|---|---|
| Estimation (personal stats only) | 26% | 67% |
| Change model (earlier label > 3 days old) | 9% | 93% |
| Show previous (earlier label ≤ 3 days old) | 4% | 96% |

The error of the stat score grows with the age of the earlier stat score value: about 5% at 1 day, 9% at 15–60 days, 12% past 180 days.

## Model files
The wheel bundles `models/model-settings.json`, `snapshot.ubj`, and `change.ubj`. To use other files (such as an older model), pass `Estimator(model_dir=...)` or set `STATSTIMATE_MODEL_DIR`.

## Contributing
TBA

To install library with development dependencies to your virtual environment, you can run `pip install -e ".[cpu]" --group dev`. To only install the dev dependencies, you can run `pip install --group dev` to install the `dev` dependency group.

## License
Copyright 2026 tiksan

This project is licensed under GPLv3; see [LICENSE.md](LICENSE.md) for more details.
