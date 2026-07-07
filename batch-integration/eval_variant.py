"""Run benchmark metrics on a single integrated.h5ad output.

Aligned with the official openproblems task_batch_integration metric scripts
in src/metrics/*. We replicate:

1. ``data_processors/transform``: for feature methods, PCA on
   ``corrected_counts`` -> obsm['X_emb']; kNN -> obsp['connectivities'/
   'distances'] + uns['neighbors'].
2. Per-metric label/uns grafting from the solution h5ad.
3. The exact scib calls used by the official scripts:
   * asw_label:      silhouette(label_key='cell_type', embed='X_emb')
   * asw_batch:      silhouette_batch(batch_key='batch', label_key='cell_type', embed='X_emb')
   * graph_connectivity: graph_connectivity(label_key='cell_type')
   * pcr:            pcr_comparison(solution[:, batch_hvg], integrated, embed='X_emb', covariate='batch')
   * clustering_overlap: cluster_optimal_resolution + ari/nmi + ari_batch/nmi_batch
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import anndata as ad
import numpy as np
import scanpy as sc


_HERE = Path(__file__).resolve().parent

# Resolutions for leiden, matching openproblems clustering_overlap /
# isolated_label_f1 defaults.
_RESOLUTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Leiden backend: igraph flavor, as openproblems' precompute_clustering_run
# uses on CPU (`flavor='igraph', n_iterations=2`). leidenalg (scanpy's old
# default) is orders of magnitude slower at scale and gives a slightly
# different partition, so igraph is both faster AND more consistent with the
# published baselines.
_leiden_igraph = functools.partial(
    sc.tl.leiden, flavor="igraph", n_iterations=2, directed=False
)
_KBET_VENV_PY = Path(
    os.environ.get(
        "CONDO_KBET_PYTHON",
        _HERE.parent.parent / ".venv-kbet" / "bin" / "python",
    )
)
_KBET_SCRIPT = _HERE / "scripts" / "compute_kbet.py"
# kbet subprocess wall-clock budget. Default 1h; raise via CONDO_KBET_TIMEOUT
# (seconds) on big-memory machines where kbet on 300k+ cells needs longer than
# the default before it would otherwise be recorded as a timeout error.
_KBET_TIMEOUT = int(os.environ.get("CONDO_KBET_TIMEOUT", 60 * 60))


def _ensure_processed(integrated: ad.AnnData, dataset: ad.AnnData) -> ad.AnnData:
    """Mirror data_processors/transform/script.py behavior."""
    if not integrated.obs.index.equals(dataset.obs.index):
        if integrated.obs.index.sort_values().equals(dataset.obs.index.sort_values()):
            integrated = integrated[dataset.obs.index].copy()
        else:
            raise AssertionError("Cell index mismatch between integrated and dataset")

    if "corrected_counts" in integrated.layers and "X_emb" not in integrated.obsm:
        print(">> PCA on corrected_counts", flush=True)
        integrated.obsm["X_emb"] = sc.pp.pca(
            integrated.layers["corrected_counts"],
            n_comps=50,
            use_highly_variable=False,
            svd_solver="arpack",
            return_info=False,
        )

    if "X_emb" in integrated.obsm and "neighbors" not in integrated.uns:
        print(">> kNN on X_emb", flush=True)
        sc.pp.neighbors(integrated, use_rep="X_emb")

    return integrated


def _precompute_leiden(adata: ad.AnnData) -> None:
    """Cluster once per resolution with igraph and store as ``leiden_{res}``
    columns. scib's cluster_optimal_resolution / isolated_labels_f1 reuse any
    existing ``{cluster_key}_{res}`` column (they only cluster if it's
    missing), so this is computed once and shared by both nmi/ari and
    isolated_label_f1 -- mirroring openproblems' precompute_clustering step
    and avoiding the ~14 leidenalg runs the old code did at eval time."""
    t0 = time.time()
    for res in _RESOLUTIONS:
        _leiden_igraph(adata, resolution=res, key_added=f"leiden_{res}")
    print(
        f">> precomputed leiden (igraph) at {len(_RESOLUTIONS)} resolutions "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )


def _safe(metric_name: str, fn: Callable[[], float]) -> dict[str, Any]:
    t0 = time.time()
    try:
        score = float(fn())
        return {"metric": metric_name, "score": score, "dt": time.time() - t0}
    except Exception as exc:
        return {
            "metric": metric_name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
            "dt": time.time() - t0,
        }


def evaluate(
    integrated_path: str,
    dataset_path: str,
    solution_path: str,
    skip_kbet: bool = False,
) -> list[dict]:
    print(">> Read integrated", flush=True)
    integrated = ad.read_h5ad(integrated_path)
    print(">> Read dataset", flush=True)
    dataset = ad.read_h5ad(dataset_path)
    print(">> Read solution", flush=True)
    solution = ad.read_h5ad(solution_path)

    integrated = _ensure_processed(integrated, dataset)

    # Graft labels & uns from the solution (the official per-metric scripts
    # do `adata.obs = solution.obs; adata.uns |= solution.uns`).
    integrated.obs = solution.obs.loc[integrated.obs.index].copy()
    integrated.uns = {**integrated.uns, **solution.uns}

    results: list[dict] = []

    # ------------------------------------------------------------------ bio
    from scib.metrics import silhouette as _scib_silhouette
    from scib.metrics import silhouette_batch as _scib_silhouette_batch
    from scib.metrics import graph_connectivity as _scib_graph_conn

    results.append(
        _safe(
            "asw_label",
            lambda: _scib_silhouette(integrated, label_key="cell_type", embed="X_emb"),
        )
    )
    results.append(
        _safe(
            "asw_batch",
            lambda: _scib_silhouette_batch(
                integrated,
                batch_key="batch",
                label_key="cell_type",
                embed="X_emb",
            ),
        )
    )
    results.append(
        _safe(
            "graph_connectivity",
            lambda: _scib_graph_conn(integrated, label_key="cell_type"),
        )
    )

    # ------------------------------------------------------------------ pcr
    # Official recipe: pre = solution restricted to batch_hvg columns,
    # post = integrated with X_emb in obsm. embed='X_emb' tells scib to use
    # the integrated obsm rather than recomputing PCA from .X.
    try:
        from scib.metrics import pcr_comparison

        batch_hvg = solution.var["batch_hvg"].astype(bool).values
        adata_pre = ad.AnnData(
            X=solution.layers["normalized"][:, batch_hvg],
            obs=solution.obs.copy(),
            var=solution.var.loc[batch_hvg].copy(),
        )
        results.append(
            _safe(
                "pcr",
                lambda: pcr_comparison(
                    adata_pre,
                    integrated,
                    embed="X_emb",
                    covariate="batch",
                    n_comps=50,
                ),
            )
        )
    except Exception as exc:
        results.append(
            {
                "metric": "pcr",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3),
            }
        )

    # Precompute the leiden clustering once (igraph) and share it between
    # nmi/ari and isolated_label_f1, matching openproblems' precompute step.
    _precompute_leiden(integrated)

    # ----------------------------------------------- clustering_overlap (ari + nmi)
    try:
        from scib.metrics.clustering import cluster_optimal_resolution
        from scib.metrics import nmi, ari

        adata_clust = integrated.copy()
        cluster_optimal_resolution(
            adata=adata_clust,
            label_key="cell_type",
            cluster_key="leiden",
            cluster_function=_leiden_igraph,
            resolutions=_RESOLUTIONS,
        )

        results.append(
            _safe(
                "ari",
                lambda: ari(adata_clust, cluster_key="leiden", label_key="cell_type"),
            )
        )
        results.append(
            _safe(
                "nmi",
                lambda: nmi(adata_clust, cluster_key="leiden", label_key="cell_type"),
            )
        )
    except Exception as exc:
        results.append(
            {
                "metric": "clustering_overlap",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3),
            }
        )

    # ---------------------------------------------------------- isolated labels
    # Matches src/metrics/isolated_label_{f1,asw}/script.py, but with one
    # tweak: scib >= 1.1.7 added a guard in get_isolated_labels that returns
    # NaN when every cell type appears in every batch (the metric
    # degenerates into mean asw_label / mean f1 per cell type in that case).
    # Older scib (which the published baseline ran on) silently computed
    # the degenerate quantity, so its yaml has real numbers. We bypass the
    # guard in that case to keep our composite directly comparable.
    try:
        from scib.metrics import isolated_labels_asw, isolated_labels_f1

        tmp = integrated.obs[["cell_type", "batch"]].drop_duplicates()
        batch_per_lab = tmp.groupby("cell_type", observed=True).agg({"batch": "count"})
        default_threshold = int(batch_per_lab.min().tolist()[0])
        n_batches = integrated.obs["batch"].nunique()
        iso_thr_override = (
            n_batches + 1 if default_threshold == n_batches else None
        )

        results.append(
            _safe(
                "isolated_label_asw",
                lambda: isolated_labels_asw(
                    integrated,
                    label_key="cell_type",
                    batch_key="batch",
                    embed="X_emb",
                    iso_threshold=iso_thr_override,
                    verbose=False,
                ),
            )
        )
        # isolated_labels_f1 re-optimises resolution against its own F1
        # metric, but reuses the precomputed leiden_{res} columns carried in
        # on the copy of `integrated` (no new leiden runs).
        adata_clust2 = integrated.copy()
        results.append(
            _safe(
                "isolated_label_f1",
                lambda: isolated_labels_f1(
                    adata_clust2,
                    label_key="cell_type",
                    batch_key="batch",
                    cluster_key="leiden",
                    resolutions=_RESOLUTIONS,
                    embed=None,
                    iso_threshold=iso_thr_override,
                    verbose=False,
                ),
            )
        )
    except Exception as exc:
        results.append(
            {
                "metric": "isolated_labels",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3),
            }
        )

    # ---------------------------------------------------------- hvg_overlap
    # Mirrors src/metrics/hvg_overlap/script.py.
    try:
        from scib.metrics import hvg_overlap
        from scib.utils import split_batches

        adata_solution = ad.AnnData(
            X=solution.layers["normalized"],
            obs=solution.obs.copy(),
            var=solution.var.copy(),
            uns=dict(solution.uns),
        )
        # The integrated AnnData needs .X populated for the per-batch HVG
        # computation; the official script reads X='layers/corrected_counts'
        # directly, so do the equivalent here.
        if "corrected_counts" in integrated.layers:
            corrected_X = integrated.layers["corrected_counts"]
        elif integrated.X is not None:
            corrected_X = integrated.X
        else:
            raise RuntimeError(
                "hvg_overlap needs either corrected_counts layer or .X on integrated"
            )
        adata_integrated_hvg = ad.AnnData(
            X=corrected_X,
            obs=integrated.obs.copy(),
            var=integrated.var.copy(),
        )
        adata_integrated_hvg.obs["batch"] = solution.obs.loc[
            integrated.obs.index, "batch"
        ].values

        adata_list = split_batches(
            adata_solution, "batch", hvg=adata_integrated_hvg.var_names
        )
        skip = []
        for ab in adata_list:
            sc.pp.filter_genes(ab, min_cells=1)
            n_hvg_tmp = np.minimum(500, int(0.5 * ab.n_vars))
            if n_hvg_tmp < 500:
                # .iloc[0] avoids the pandas positional-vs-label warning
                # the official script still triggers; semantically the
                # same value.
                skip.append(ab.obs["batch"].iloc[0])
        if skip:
            adata_solution = adata_solution[
                ~adata_solution.obs["batch"].isin(skip)
            ].copy()
            adata_integrated_hvg = adata_integrated_hvg[
                ~adata_integrated_hvg.obs["batch"].isin(skip)
            ].copy()

        results.append(
            _safe(
                "hvg_overlap",
                lambda: hvg_overlap(
                    adata_solution[
                        :, adata_solution.var_names.isin(adata_integrated_hvg.var_names)
                    ],
                    adata_integrated_hvg,
                    batch_key="batch",
                ),
            )
        )
    except Exception as exc:
        results.append(
            {
                "metric": "hvg_overlap",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3),
            }
        )

    # ---------------------------------------------------------- kbet
    # Runs in the sibling .venv-kbet (numpy<2, scipy<=1.13, rpy2 3.5.x +
    # theislab/kBET R package). Skipped if --skip-kbet is set or the venv
    # isn't present. kbet is the slowest metric (it times out on the large
    # datasets, where the published baselines also lack it) and is not part
    # of the all/feature composites.
    try:
        if skip_kbet:
            results.append({"metric": "kbet", "skipped": True})
        elif not _KBET_VENV_PY.exists():
            results.append(
                {"metric": "kbet", "error": f"venv not found at {_KBET_VENV_PY}"}
            )
        else:
            t0 = time.time()
            env = dict(os.environ, R_HOME="/usr/lib/R")
            proc = subprocess.run(
                [
                    str(_KBET_VENV_PY),
                    str(_KBET_SCRIPT),
                    "--integrated",
                    integrated_path,
                    "--solution",
                    solution_path,
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=_KBET_TIMEOUT,
            )
            if proc.returncode == 0:
                # The subprocess prints lines of progress then a single
                # JSON object at the end; parse the last non-empty line.
                last = next(
                    ln for ln in reversed(proc.stdout.splitlines()) if ln.strip()
                )
                score = float(json.loads(last)["score"])
                results.append(
                    {"metric": "kbet", "score": score, "dt": time.time() - t0}
                )
            else:
                results.append(
                    {
                        "metric": "kbet",
                        "error": f"subprocess exit {proc.returncode}",
                        "stderr": proc.stderr[-2000:],
                        "dt": time.time() - t0,
                    }
                )
    except Exception as exc:
        results.append(
            {
                "metric": "kbet",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3),
            }
        )

    # ---------------------------------------------------------- iLISI / cLISI
    # Matches src/metrics/lisi/script.py. The lisi_graph_py call returns raw
    # LISI scores; the normalization below converts to [0,1] with higher =
    # better, as the official script does.
    try:
        from scib.metrics.lisi import lisi_graph_py

        n_batches = integrated.obs["batch"].nunique()
        n_labels = integrated.obs["cell_type"].nunique()

        def _ilisi():
            scores = lisi_graph_py(
                adata=integrated,
                obs_key="batch",
                n_neighbors=90,
                perplexity=None,
                subsample=None,
                n_cores=1,
                verbose=False,
            )
            med = np.nanmedian(scores)
            return (med - 1) / (n_batches - 1)

        def _clisi():
            scores = lisi_graph_py(
                adata=integrated,
                obs_key="cell_type",
                n_neighbors=90,
                perplexity=None,
                subsample=None,
                n_cores=1,
                verbose=False,
            )
            med = np.nanmedian(scores)
            return (n_labels - med) / (n_labels - 1)

        results.append(_safe("ilisi", _ilisi))
        results.append(_safe("clisi", _clisi))
    except Exception as exc:
        results.append(
            {
                "metric": "lisi",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3),
            }
        )

    # -------------------------------------------- cell_cycle_conservation
    # Mirrors src/metrics/cell_cycle_conservation/script.py: pre = normalized
    # solution with gene-symbol var_names, post = integrated (embed='X_emb'),
    # organism from solution.uns. scib needs cell-cycle marker genes present
    # in var; _safe records the error if they are absent (openproblems only
    # enables ccc on some datasets), so this is harmless to attempt always.
    def _ccc():
        from scib.metrics import cell_cycle

        adata_pre = ad.AnnData(
            X=solution.layers["normalized"],
            obs=solution.obs.copy(),
            var=solution.var.copy(),
        )
        adata_pre.var_names = solution.var["feature_name"].astype(str).values
        return cell_cycle(
            adata_pre,
            integrated,
            batch_key="batch",
            embed="X_emb",
            organism=solution.uns["dataset_organism"],
        )

    results.append(_safe("cell_cycle_conservation", _ccc))

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrated", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--output", required=True, help="path to JSON output")
    parser.add_argument(
        "--skip-kbet", action="store_true",
        help="skip the kbet metric (slow R subprocess; times out on large "
             "datasets and not used in the all/feature composites).",
    )
    args = parser.parse_args()

    results = evaluate(
        args.integrated, args.dataset, args.solution, skip_kbet=args.skip_kbet
    )

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
