"""Compute scib's kBET metric in isolation.

Run inside ``.venv-kbet`` (numpy<2, scipy<=1.13, rpy2, anndata2ri,
plus the R kBET package). The main eval (eval_variant.py) is on a
different env (numpy 2 + modern scanpy) and calls this script via
subprocess.

Usage:
    .venv-kbet/bin/python compute_kbet.py \\
        --integrated /path/to/integrated.h5ad \\
        --solution   /path/to/solution.h5ad

Prints a single JSON object to stdout: {"score": <float>}.
"""
import argparse
import json
import sys
import warnings

import anndata as ad
import numpy as np
import scanpy as sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrated", required=True)
    ap.add_argument("--solution", required=True)
    args = ap.parse_args()

    integrated = ad.read_h5ad(args.integrated)
    solution = ad.read_h5ad(args.solution)

    # Match eval_variant.py: derive X_emb if it's a feature method, then
    # graft obs/uns from solution exactly like the official script.
    if "corrected_counts" in integrated.layers and "X_emb" not in integrated.obsm:
        integrated.obsm["X_emb"] = sc.pp.pca(
            integrated.layers["corrected_counts"],
            n_comps=50,
            svd_solver="arpack",
            return_info=False,
        )
    if "X_emb" in integrated.obsm and "neighbors" not in integrated.uns:
        sc.pp.neighbors(integrated, use_rep="X_emb")

    integrated.obs = solution.obs.loc[integrated.obs.index].copy()
    integrated.uns = {**integrated.uns, **solution.uns}

    from scib.metrics import kBET

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = kBET(
            integrated,
            batch_key="batch",
            label_key="cell_type",
            type_="embed",
            embed="X_emb",
            scaled=True,
            verbose=False,
        )

    print(json.dumps({"score": float(score)}))


if __name__ == "__main__":
    main()
