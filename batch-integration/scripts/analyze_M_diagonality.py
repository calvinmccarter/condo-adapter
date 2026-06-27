"""Fit condo agglomerative on a dataset, capture each per-merge affine
matrix (M, b), and report how close each M is to a diagonal.

Metrics per merge (X = source batch's raw input rows, M = applied affine):
  - off_F   = ||M - diag(M)||_F / ||M||_F                 (matrix Frobenius)
  - off_2   = ||M - diag(M)||_2 / ||M||_2                 (matrix spectral)
  - rho_emp = ||X M^T - X diag(M)||_F / ||X M^T||_F       (input-aware F)
  - gap     = ||X M^T - X d*||_F / ||X M^T||_F            (best-diagonal residual,
              where d_j* = sum_i (XM^T)_{i,j} X_{i,j} / sum_i X_{i,j}^2)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="benchmark dataset.h5ad")
    ap.add_argument("--method-dir", required=True,
                    help="task_batch_integration_forked/src/methods/condo")
    ap.add_argument("--utils-dir", required=True,
                    help="task_batch_integration_forked/src/utils")
    ap.add_argument("--output", required=True,
                    help=".npz to store M/b matrices and metadata")
    ap.add_argument("--n-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, args.method_dir)
    from condo_runner import (
        _read_input, _pick_target_by_pre_asw,
        _build_adapter, _to_dense, _select_feature_columns,
    )

    par = {
        "input": args.input,
        "rep": "features",
        "hvg_only": False,
        "divergence": "mmd",
        "transform_type": "affine",
        "n_epochs": args.n_epochs,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "wd_on_bias": False,
        "learning_rate": 1e-3,
        "mmd_size": 40,
        "batch_size": 8,
        "bootstrap_fraction": 1.0,
        "random_state": 42,
        "device": args.device,
    }
    meta = {"resources_dir": args.utils_dir}

    print(">> Reading dataset", flush=True)
    adata = _read_input(par, meta)
    batches = np.asarray(adata.obs["batch"].astype(str).values, dtype="U")
    cell_types = np.asarray(adata.obs["cell_type"].astype(str).values, dtype="U")
    _, per_batch_asw = _pick_target_by_pre_asw(adata, batches, cell_types)
    Y = _to_dense(adata.X).astype(np.float64)
    print(f"   Y shape: {Y.shape}, batches: {len(set(batches.tolist()))}", flush=True)

    captured = []

    def factory():
        a = _build_adapter(par)
        captured.append(a)
        return a

    from condo.batch_integration import agglomerative_integrate

    t0 = time.time()
    result = agglomerative_integrate(
        Y=Y, batches=batches, confounders=cell_types,
        batch_score=per_batch_asw, adapter_factory=factory, verbose=True,
    )
    print(f">> Fit complete in {time.time() - t0:.0f}s; "
          f"{len(captured)} merges captured", flush=True)

    d = Y.shape[1]
    rows = []
    save = {"seed": result.initial_target,
            "merge_order": np.array(result.merge_order, dtype="U"),
            "n_features": d}
    print(f"\n  per-merge: {'src':24s}  "
          f"{'off_F':>7s} {'off_2':>7s} {'rho_emp':>7s} {'gap':>7s}")
    for i, (src, a) in enumerate(zip(result.merge_order, captured)):
        M = a.M_
        b = a.b_
        save[f"M_{i:03d}"] = M
        save[f"b_{i:03d}"] = b
        save[f"src_{i:03d}"] = np.array(src, dtype="U")

        off = M.copy()
        np.fill_diagonal(off, 0.0)
        nM_F   = np.linalg.norm(M)
        noff_F = np.linalg.norm(off)
        off_F  = noff_F / nM_F if nM_F > 0 else 0.0
        off_2  = np.linalg.norm(off, 2) / np.linalg.norm(M, 2)

        X = Y[batches == src]                       # source samples (raw input)
        AX  = X @ M.T
        DAX = X * np.diag(M)
        rho_emp = np.linalg.norm(AX - DAX) / np.linalg.norm(AX)
        num = np.einsum("ij,ij->j", AX, X)
        den = np.einsum("ij,ij->j", X, X)
        d_star = np.where(den > 0, num / den, 0.0)
        gap = np.linalg.norm(AX - X * d_star) / np.linalg.norm(AX)

        rows.append((src, off_F, off_2, rho_emp, gap))
        print(
            f"  [{i:>2}] {src[:24]:24s}  "
            f"{off_F:7.4f} {off_2:7.4f} {rho_emp:7.4f} {gap:7.4f}"
        )

    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    cols = ["off_F", "off_2", "rho_emp", "gap"]
    print()
    print("  aggregated across merges:")
    print(f"  {'metric':>15s}  {'mean':>9s} {'min':>9s} {'max':>9s} {'std':>9s}")
    for j, c in enumerate(cols):
        v = arr[:, j]
        print(f"  {c:>15s}  {v.mean():9.4f} {v.min():9.4f} {v.max():9.4f} {v.std():9.4f}")

    save["rows_names"] = np.array(["src"] + cols, dtype="U")
    save["rows_values"] = np.array([(r[0],) + tuple(r[1:]) for r in rows],
                                    dtype=object)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **save)
    print(f"\n>> Saved {Path(args.output)}  ({len(captured)} merges, d={d})")


if __name__ == "__main__":
    main()
