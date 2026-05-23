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
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import anndata as ad
import numpy as np
import scanpy as sc


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


def evaluate(integrated_path: str, dataset_path: str, solution_path: str) -> list[dict]:
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

    # ----------------------------------------------- clustering_overlap (4)
    try:
        from scib.metrics.clustering import cluster_optimal_resolution
        from scib.metrics import nmi, ari

        adata_clust = integrated.copy()
        cluster_optimal_resolution(
            adata=adata_clust,
            label_key="cell_type",
            cluster_key="leiden",
            cluster_function=sc.tl.leiden,
            resolutions=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
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
        results.append(
            _safe(
                "ari_batch",
                lambda: 1
                - ari(adata_clust, cluster_key="leiden", label_key="batch"),
            )
        )
        results.append(
            _safe(
                "nmi_batch",
                lambda: 1
                - nmi(adata_clust, cluster_key="leiden", label_key="batch"),
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

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrated", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--output", required=True, help="path to JSON output")
    args = parser.parse_args()

    results = evaluate(args.integrated, args.dataset, args.solution)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
