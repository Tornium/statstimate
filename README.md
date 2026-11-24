# Statstimate
Tornium's stat estimation library for Torn City. It estimates a user's stats from their personal stats and can get much more accurate when you also have an earlier stat score for that player (such as one from a stat spy).

This repository provides two components:
- The models: two XGBoost models plus a settings file, published under `models/`. Any language with [XGBoost](https://xgboost.readthedocs.io/en/stable/
) bindings can load them.
- A Python reference implementation (`statstimate` on PyPI) that loads the models, builds their inputs, and picks the right estimation method for each player.

For every player, statstimate picks one of three methods:
- `snapshot`: Estimates the stat score from the current personal stats alone if no previous stat score is provided.
- `keep previous`: Returns the earlier stat score unchanged if the earlier stat score is at most three days old.
- `change model`: Estimates how much the previous stat score grew since it was recorded from how the personal stats changed if the earlier stat score is more than three days old.

## Installation
To support both CPU and GPU based estimation, the statstimate library is split into two groups of dependencies:
- `statstimate[cpu]`: uses the `xgboost-cpu` library to provide CPU-only estimation
- `statstimate[gpu]`: uses the `xgboost` library to provide mostly GPU-only estimation

These can then be installed with `pip install statstimate[cpu]` and `pip install statstimate[gpu]` respectively. Installing only statstimate without an extra doesn't install the XGBoost dependency preventing the library from being imported. Additionally, the library requires at least 3.10 and the correct versions of dependencies (as older versions can produce incorrect estimates).

> [!NOTE]
> The `xgboost-cpu` library currently only supports x86_64 (amd64) Linux and Windows, so any other platforms will need to utilize `statstimate[gpu]`

### Nix
The repository also provides a [Nix flake](https://wiki.nixos.org/wiki/Flakes) which provides the `package.nix` (an overlay that adds `statstimate` to every Python package set) and a development shell.

```sh
nix build github:Tornium/statstimate
```

```nix
nixpkgs.overlays = [ statstimate.overlays.default ];
```

By default, Nix will use the CPU-only version of XGBoost. You can configure it to use the GPU version of the library by configuring `cudaSupport = true`.

## Usage
### Estimate one player 
```python
from statstimate import Estimator
 
# Loading the models takes time and memory: create one Estimator and reuse it.
estimator = Estimator()
 
personal_stats = {
    "useractivity": 3_000_000,
    "xantaken": 1200,
    "refills": 500,
    "energydrinkused": 300,
    "daysbeendonator": 400,
    "statenhancersused": 10,
    "lsdtaken": 20,
    "exttaken": 40,
    "attackswon": 900,
    "networth": 4_740_570_924_889,
    # ... every other personal stat you have
}
 
estimate = estimator.estimate(personal_stats)
```
 
`personal_stats` is a mapping of personal stat names to numbers. The names are keys provided by the Torn API that the model was trained on which you can fetch with `estimator.input_fields`. Missing stats and `None` values count as 0, and unknown keys are ignored, so you can pass the whole record you have.
 
### Use an earlier stat score
If you have a stat score from earlier, such as from a stat spy, it can be provided alongside the personal stats from when the stat score was from to generate a better estimate:
```python
from statstimate import PreviousStats
 
previous = PreviousStats(
    stat_score=85_000,
    personal_stats=old_personal_stats,
    days_ago=52,
)
 
estimate = estimator.estimate(personal_stats, previous)
```
 
> [!NOTE]
> Only pass a real stat score as `PreviousStats`, never an earlier estimate. Feeding estimates back in makes their errors add up over time.
 
### Estimate many players
 
`estimate_many` runs the whole batch of users through the models at once and is much faster than calling `estimate` in a loop. `previous_stats` is optional but requires one entry per player when provided (with `None` where there's no earlier stat score):
```python
estimates = estimator.estimate_many(
    [stats_a, stats_b, stats_c],
    [None, PreviousStats(120_000, old_stats_b, 1), PreviousStats(40_000, old_stats_c, 400)],
)
```
 
### The result
Each estimate is an `EstimatedStats`:
 
| Field            | Type    | Description                                                     |
|------------------|---------|-------------------------------------------------------------------|
| `stat_score`     | `float` | The estimated stat score.                                       |
| `log_stat_score` | `float` | Natural log of the estimate; this is what the models predict.   |
| `method`         | `str`   | `"snapshot"`, `"keep previous"` or `"change model"`.            |
| `model_version`  | `str`   | The version of the model files used, such as `"2026-11-13"`.    |

### Options
```python
Estimator(
    model_path=None,  # folder with the model files; see "Model files" below
    device=None,      # XGBoost device, e.g. "cpu" or "cuda"; None keeps XGBoost's default (CPU)
    threads=None,     # threads per prediction; None uses all cores
)
```
- **`threads`**: on a web server with one worker per core, use `threads=1` so workers don't compete for cores.
- **`device`**: a GPU only pays off for large batches, and needs the full `xgboost` package (`statstimate[gpu]`).

### Accuracy (model 2026-11-13)

| Method | Typical miss | Within ±25% |
|----------------------------------------------|---|---|
| Estimation (personal stats only) | 26% | 67% |
| Change model (earlier label > 3 days old) | 9% | 93% |
| Show previous (earlier label ≤ 3 days old) | 4% | 96% |

## Contributing
To install library with development dependencies to your virtual environment, you can run `pip install -e ".[cpu]" --group dev`. To only install the dev dependencies, you can run `pip install --group dev` to install the `dev` dependency group.

Install the library in editable mode with the `dev` dependency group:
```sh
pip install -e ".[cpu]" --group dev
```
 
Alternatively, the library can beinstalled with Nix. `nix develop` opens a shell with Python, XGBoost, the dev tools and `uv`, with `src/` on `PYTHONPATH`.
 
`nix flake check` can build the Nix package and loads the bundled models with nixpkgs' XGBoost, to make sure that version can read them.
 
## License
Copyright 2026 tiksan

This project is licensed under the GPLv3 license; see [LICENSE.md](LICENSE.md) for more details.
