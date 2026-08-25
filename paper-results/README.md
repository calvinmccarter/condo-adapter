# Paper results bundle

Result JSONs (per-method, per-dataset metric scores) and the published
baselines, snapshotted for plotting from a laptop without needing the
fit/eval pipeline.

## Layout

### Published baselines
- `baselines/score_uns.yaml` — published baseline scores (the pool the
  leaderboard ranks against).

### Main condo results (all 6 datasets)
- `sweeps/v3_baseline_seeded/` — affine, ne=5, wd=1e-4. The "v3 untuned
  baseline" condo result. Preceded `torch.manual_seed`, so scores are
  NOT bit-reproducible across machines.
- `sweeps/v3_locscale_ne5_wd1e-5_all/` — location-scale at the auto
  default (ne=5, wd=1e-5).
- `sweeps/v3_dplr_r16_ne50_wd1e-5_all/` — diagonal-plus-low-rank at the
  auto default (rank=16, ne=50, wd=1e-5).

### Official location-scale result (wd=1e-4, matches affine)
- `sweeps/abl_locscale_celltype_silhouette_wd1e-4/` — **the official
  location-scale baseline for every downstream comparison.** Identical to
  the affine official baseline `abl_celltype_silhouette` in every
  hyperparameter except the transform (mmd, features, ne=5, lr=1e-3,
  mmd_size=40, bs=8, patience=3, celltype_silhouette ranking, **wd=1e-4**),
  so affine and location-scale are now reported under the SAME
  hyperparameters. All 6 datasets, one machine, torch-seeded. Replaces the
  wd=1e-5 loc-scale as the headline result; the wd=1e-5
  `abl_locscale_celltype_silhouette/` is retained only as the anchor cell
  of the merge-order ablation below.

### Ranking-strategy ablation
Same config as v3_baseline_seeded / v3_locscale_ne5_wd1e-5 (mmd, features,
ne=5), varying only the `--ranking-strategy` flag that drives BOTH the
seed batch and the compatible-neighbour merge order in the agglomerative
integrator. All ablation runs are torch-seeded (reproducible on-machine).

**Affine variant** (wd=1e-4):
- `sweeps/abl_celltype_silhouette/` — per-batch cell-type silhouette on
  X_pca, highest first. **This is the official condo affine baseline for
  every downstream comparison** (replaces v3_baseline_seeded; same
  config, but bit-reproducible).
- `sweeps/abl_celltype_sil_low/` — same score, lowest first (mirror).
- `sweeps/abl_random_rs42/` — fixed per-batch random priority.
- `sweeps/abl_biggest/` — batch size in cells, biggest first.
- `sweeps/abl_smallest/` — batch size in cells, smallest first (mirror).
- `sweeps/abl_batch_sil_low/` — per-batch batch silhouette on X_pca,
  lowest first (merge already-mixed batches earliest).
- `sweeps/abl_batch_sil_high/` — batch silhouette, highest first.

**Location-scale variant** (wd=1e-5): the seven `sweeps/abl_locscale_*/`
directories mirror the affine ones one-to-one with the same seven
strategy suffixes. These stay at wd=1e-5 as a self-consistent merge-order
ablation (vary only the strategy). NB: the headline official loc-scale
result is the wd=1e-4 rerun above (`abl_locscale_celltype_silhouette_wd1e-4/`);
the wd=1e-5 `abl_locscale_celltype_silhouette/` is only the anchor of this
ablation, not the reported loc-scale result.

Each sweep dir has one JSON per dataset (`dkd, gtex_v9,
mouse_pancreas_atlas, immune_cell_atlas, hypomap, tabula_sapiens`). Each
JSON is a list of `{"metric": name, "score": value, "dt": seconds}`
records, one per evaluated metric (13 metrics per dataset).

## Notes on cross-machine drift and kbet coverage

The ablation sweep was split across two GPU boxes. All seven strategies
for a single dataset are always on the same machine, so the within-
dataset (strategy vs strategy) comparisons are internally consistent.
However, comparing `abl_celltype_silhouette` scores across datasets
(some datasets on one box, some on the other) inherits a 3rd-decimal
drift on non-deterministic torch/CUDA ops. Documented, expected, called
out during the ablation.

kbet is not part of the 7-metric leaderboard composite (excluded
because the published baselines lack it on hypomap and
mouse_pancreas_atlas). The kbet field in each ablation JSON is either
numeric or `{"skipped": true}`:

- **Affine ablation** (`abl_*/`): kbet numeric on dkd, gtex_v9,
  immune_cell_atlas, tabula_sapiens (28/42 cells). Skipped on hypomap
  and mouse_pancreas_atlas (14/42) — the runner set `--skip-kbet` on
  those two datasets because kbet on the 385k-cell hypomap reliably
  triggered an amazon-efs proxy stall requiring an instance reboot.
- **Loc-scale ablation** (`abl_locscale_*/`): kbet skipped on all
  42/42 cells. Same reason (three consecutive reboots during
  hypomap-kbet, once with h5ads on NFS, once on retry, once with the
  h5ads copied to local ext4 — the trigger appears to be cumulative
  uptime under sustained heavy compute rather than kbet's NFS I/O
  specifically). Not worth chasing given kbet is out of the composite.

## Computing the leaderboard

The active leaderboard script is `batch-integration/leaderboard.py` in
this same repo. By default it reads `work/fullbench/baselines/score_uns.yaml`;
to use this bundle's copy, set the path explicitly or symlink:
```
ln -s ../paper-results/baselines work/fullbench/baselines  # one-time
```
Then run as usual:
```
python batch-integration/leaderboard.py \
  --results-dir paper-results/sweeps/abl_celltype_silhouette \
  --condo-label "condo (affine)" \
  --score composite
```

For the ranking-strategy ablation table, compare each `abl_*/` dir
against `abl_celltype_silhouette/` (affine) or each `abl_locscale_*/`
against `abl_locscale_celltype_silhouette/` (loc-scale) — same
transform, same config, same machine per dataset.

## Producing the leaderboard pool
The `leaderboard_extras/scvi_tabula_sapiens.json` and
`leaderboard_extras/scanvi_tabula_sapiens.json` files (in
`batch-integration/leaderboard_extras/`) fill gaps in the published
baselines for tabula_sapiens; pass them via `--inject method:dataset:json`.
