"""Train scvi + scanvi on tabula_sapiens (or any dataset) and save the
integrated h5ads in the same format other methods produce.

Mirrors the openproblems task_batch_integration scvi and scanvi components:
  - Read X='layers/counts', subset to top --n-hvg by hvg_score.
  - SCVI.setup_anndata(batch_key='batch'); train scvi (max_epochs).
  - SCANVI.from_scvi_model(labels_key='cell_type',
      unlabeled_category='UnknownUnknown'); train scanvi.
  - Save each embedding into obsm['X_emb'] of an output h5ad.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import anndata as ad
import numpy as np


def _read_dataset(path: str) -> ad.AnnData:
    a = ad.read_h5ad(path)
    a.X = a.layers["counts"]
    return a


def _subset_hvg(a: ad.AnnData, n_hvg: int) -> ad.AnnData:
    idx = a.var["hvg_score"].to_numpy().argsort()[::-1][:n_hvg]
    return a[:, idx].copy()


def _save(a: ad.AnnData, X_emb: np.ndarray, method_id: str, out: str) -> None:
    output = ad.AnnData(
        obs=a.obs[[]], var=a.var[[]],
        obsm={"X_emb": X_emb},
        uns={
            "dataset_id": a.uns["dataset_id"],
            "normalization_id": a.uns["normalization_id"],
            "method_id": method_id,
        },
    )
    output.write_h5ad(out, compression="gzip")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-scvi", required=True)
    ap.add_argument("--out-scanvi", required=True)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--max-epochs-scvi", type=int, default=None,
                    help="None => scvi auto-heuristic (recommended)")
    ap.add_argument("--max-epochs-scanvi", type=int, default=None,
                    help="None => scvi auto-heuristic (recommended)")
    args = ap.parse_args()

    print(">> Read input", flush=True)
    adata = _read_dataset(args.input)
    print(f">> Subset to top {args.n_hvg} HVGs by hvg_score", flush=True)
    adata = _subset_hvg(adata, args.n_hvg)
    print(f"   shape after HVG subset: {adata.shape}", flush=True)

    from scvi.model import SCVI, SCANVI

    print(">> SCVI.setup_anndata + train", flush=True)
    SCVI.setup_anndata(adata, batch_key="batch")
    vae = SCVI(adata)
    vae.train(max_epochs=args.max_epochs_scvi, train_size=1.0)
    print(">> Save scvi embedding", flush=True)
    _save(adata, vae.get_latent_representation(), "scvi_local", args.out_scvi)

    print(">> SCANVI.from_scvi_model + train", flush=True)
    scanvae = SCANVI.from_scvi_model(
        scvi_model=vae,
        labels_key="cell_type",
        unlabeled_category="UnknownUnknown",
    )
    scanvae.train(max_epochs=args.max_epochs_scanvi, train_size=1.0)
    print(">> Save scanvi embedding", flush=True)
    _save(adata, scanvae.get_latent_representation(), "scanvi_local",
          args.out_scanvi)

    print(">> Done", flush=True)


if __name__ == "__main__":
    main()
