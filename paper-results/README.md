# Paper results bundle

Result JSONs (per-method, per-dataset metric scores) and the published
baselines, snapshotted for plotting from a laptop without needing the
fit/eval pipeline.

## Layout
- `baselines/score_uns.yaml` — published baseline scores (the pool the
  leaderboard ranks against).
- `sweeps/v3_baseline_seeded/` — affine, ne=5, wd=1e-4. The reference
  "v3 untuned baseline" condo result on all 6 datasets.
- `sweeps/v3_locscale_ne5_wd1e-5_all/` — location-scale at the auto
  default (ne=5, wd=1e-5, wd_on_bias=True).
- `sweeps/v3_dplr_r16_ne50_wd1e-5_all/` — diagonal-plus-low-rank at the
  auto default (rank=16, ne=50, wd=1e-5).

Each sweep dir has one JSON per dataset
(`dkd, gtex_v9, mouse_pancreas_atlas, immune_cell_atlas, hypomap,
tabula_sapiens`). Each JSON is a list of `{"metric": name, "score": value,
"dt": seconds}` records, one per evaluated metric.

## Computing the leaderboard
The active leaderboard script is `batch-integration/leaderboard.py` in this
same repo. By default it reads `work/fullbench/baselines/score_uns.yaml`;
to use this bundle's copy, set the path explicitly or symlink:
```
ln -s ../paper-results/baselines work/fullbench/baselines  # one-time
```
Then run as usual:
```
python batch-integration/leaderboard.py \
  --results-dir paper-results/sweeps/v3_dplr_r16_ne50_wd1e-5_all \
  --condo-label "DPLR r=16 ne=50" \
  --score composite
```

## Producing the leaderboard pool
The `leaderboard_extras/scvi_tabula_sapiens.json` and
`leaderboard_extras/scanvi_tabula_sapiens.json` files (in
`batch-integration/leaderboard_extras/`) fill gaps in the published
baselines for tabula_sapiens; pass them via `--inject method:dataset:json`.
