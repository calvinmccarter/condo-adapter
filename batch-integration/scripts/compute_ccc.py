"""Compute scib's cell_cycle_conservation metric in isolation.

Mirrors openproblems task_batch_integration metrics/cell_cycle_conservation:
  - adata_pre  = solution, X=normalized, var_names set to gene symbols
                 (var['feature_name']), batch in obs
  - adata_post = integrated, with embedding in obsm['X_emb']
  - score = cell_cycle(pre, post, batch_key='batch', embed='X_emb',
                       organism=solution.uns['dataset_organism'])

For feature-method outputs (corrected_counts, no X_emb) we derive X_emb by
PCA on corrected_counts exactly as eval_variant.py's _ensure_processed does,
so the embedding matches the rest of the eval.

Prints a single JSON object to stdout: {"score": <float>}.
"""
import argparse
import json
import warnings

import anndata as ad
import scanpy as sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrated", required=True)
    ap.add_argument("--solution", required=True)
    args = ap.parse_args()

    integrated = ad.read_h5ad(args.integrated)
    solution = ad.read_h5ad(args.solution)

    # Derive X_emb for feature output (matches eval_variant._ensure_processed).
    if "corrected_counts" in integrated.layers and "X_emb" not in integrated.obsm:
        integrated.obsm["X_emb"] = sc.pp.pca(
            integrated.layers["corrected_counts"],
            n_comps=50,
            use_highly_variable=False,
            svd_solver="arpack",
            return_info=False,
        )
    if "X_emb" not in integrated.obsm:
        raise ValueError(
            "integrated has neither obsm['X_emb'] nor a corrected_counts layer"
        )

    # adata_pre: normalized solution, gene-symbol var_names, batch in obs.
    adata_pre = ad.AnnData(
        X=solution.layers["normalized"],
        obs=solution.obs.copy(),
        var=solution.var.copy(),
    )
    adata_pre.var_names = solution.var["feature_name"].astype(str).values

    # adata_post: align to solution order, carry batch + X_emb.
    integrated = integrated[adata_pre.obs.index].copy()
    integrated.obs["batch"] = adata_pre.obs["batch"]

    organism = solution.uns["dataset_organism"]

    from scib.metrics import cell_cycle

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = cell_cycle(
            adata_pre,
            integrated,
            batch_key="batch",
            embed="X_emb",
            organism=organism,
        )

    print(json.dumps({"score": float(score)}))


if __name__ == "__main__":
    main()
