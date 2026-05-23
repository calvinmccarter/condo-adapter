# batch-integration

Local tooling for evaluating ConDo on the openproblems
[task_batch_integration](https://openproblems.bio/benchmarks/batch_integration)
benchmark. The viash component itself lives in the
[task_batch_integration fork](https://github.com/calvinmccarter/task_batch_integration)
under `src/methods/condo/` (branch `condo-method`). This directory holds
the *local* harness used to fit + score variants without going through
viash/nextflow/docker — useful for iterating on the method itself.

## Layout

```
batch-integration/
├── run_local.py        # Fit-only entry point (calls condo_runner.run_condo).
├── eval_variant.py     # Aligned-to-scIB metric harness (asw, pcr, clustering).
├── summarize.py        # Aggregate JSON results into a leaderboard table.
├── scripts/
│   ├── run_one.sh      # Durable fit+eval for a single variant (nohup/setsid).
│   └── run_sweep.sh    # Loop over (divergence × transform × rep × hvg_only).
└── README.md
```

`run_local.py` imports `condo_runner.py` from the fork — by default it
looks for it at `../../task_batch_integration_forked/src/methods/condo/`
relative to this repo. Override with `--method-dir` or
`$CONDO_METHOD_DIR`.

## Quickstart

```bash
# 1. Create a venv and install deps from public PyPI.
python -m venv .venv
.venv/bin/pip install --index-url https://pypi.org/simple/ \
    "anndata>=0.10" scanpy h5py scipy numpy pandas scikit-learn \
    torch "miceforest<6.0.0" pytorch-minimize condo scib

# 2. Point at the test resources from openproblems (publicly hosted on S3).
export CONDO_BENCH_PY="$(pwd)/.venv/bin/python"
export CONDO_BENCH_DATASET=resources_test/dataset.h5ad
export CONDO_BENCH_SOLUTION=resources_test/solution.h5ad

# 3. Run a single variant.
batch-integration/scripts/run_one.sh kld location-scale features 0

# 4. Or sweep all four named variants in parallel.
CONDO_BENCH_PARALLEL=1 batch-integration/scripts/run_sweep.sh

# 5. Read off the leaderboard.
"$CONDO_BENCH_PY" batch-integration/summarize.py --results-dir work/results
```

## Variants

`run_local.py` exposes four orthogonal axes that combine with the
runner's parameterization in the fork:

| flag                    | values                          | effect |
|-------------------------|----------------------------------|--------|
| `--divergence`          | `kld` \| `mmd`                  | ConDo objective. |
| `--transform-type`      | `location-scale` \| `affine`    | Per-feature shift vs full d×d map. |
| `--rep`                 | `features` \| `pca`             | Fit on normalized expression (feature method) or on obsm['X_pca'] (embedding method). |
| `--hvg-only`            | (boolean)                       | Restrict feature-space fit to var['hvg']; non-HVGs pass through. |

The four "official" named variants used in the openproblems config:

| name                          | divergence | transform-type   |
|-------------------------------|------------|------------------|
| `condo_kld_location_scale`    | `kld`      | `location-scale` |
| `condo_kld_affine`            | `kld`      | `affine`         |
| `condo_mmd_location_scale`    | `mmd`      | `location-scale` |
| `condo_mmd_affine`            | `mmd`      | `affine`         |

## Notes

- The `affine` variants scale O(d²) in feature space — use `--hvg-only`
  on real-sized datasets, or `--rep pca` to operate in 50-D embedding
  space.
- Run artifacts (logs, h5ad outputs, score JSON files) live under
  `work/` and are gitignored — they're regenerated cheaply from these
  scripts on any machine with the deps installed.
