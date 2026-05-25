"""Sweep over target-batch choice on a single dataset.

For each batch in the dataset, fit condo MMD-affine with that batch as
target, then compute the embedding-geometry bio metrics that respond to
inter-cell-type spacing: asw_label, isolated_label_asw, clisi. Also
compute asw_batch, graph_connectivity, ilisi as sanity checks that
batch-mixing isn't changing wildly.

Skips: nmi/ari/leiden (slow + topology not geometry), kbet (slow R
subprocess), hvg_overlap, pcr. We just want a quick "does target choice
affect inter-type spacing" answer.

Output: JSON file with per-target results.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import anndata as ad
import numpy as np
import scanpy as sc

REPO = Path(__file__).resolve().parents[2]


def fit_one(
    target_batch: str,
    *,
    dataset: str,
    sweep_dir: Path,
    method_dir: Path,
    utils_dir: Path,
    python: str,
    gpu_id: int | None,
) -> dict:
    tag = f"target_{target_batch}".replace("/", "_")
    out_h5 = sweep_dir / "outputs" / f"{tag}.h5ad"
    fit_log = sweep_dir / "logs" / f"{tag}.fit.log"
    if out_h5.exists() and out_h5.stat().st_size > 0:
        return {"target": target_batch, "status": "fit_skipped"}

    cmd = [
        python,
        str(REPO / "batch-integration" / "run_local.py"),
        "--divergence", "mmd",
        "--transform-type", "affine",
        "--rep", "features",
        "--device", "cuda",
        "--method-dir", str(method_dir),
        "--utils-dir", str(utils_dir),
        "--n-epochs", "5",
        "--learning-rate", "1e-3",
        "--mmd-size", "40",
        "--batch-size", "8",
        "--weight-decay", "1e-4",
        "--random-state", "42",
        "--optimizer", "adamw",
        "--target-batch", target_batch,
        "--input", dataset,
        "--output", str(out_h5),
    ]
    env = dict(os.environ)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with open(fit_log, "wb") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if rc != 0:
        return {"target": target_batch, "status": "fit_failed", "dt": time.time() - t0}
    return {"target": target_batch, "status": "fit_ok", "dt": time.time() - t0}


def quick_bio_eval(
    integrated_path: str, solution_path: str
) -> dict:
    """Compute fast embedding-geometry bio + batch metrics."""
    integrated = ad.read_h5ad(integrated_path)
    solution = ad.read_h5ad(solution_path)

    integrated.obsm["X_emb"] = sc.pp.pca(
        integrated.layers["corrected_counts"],
        n_comps=50,
        svd_solver="arpack",
        return_info=False,
    )
    sc.pp.neighbors(integrated, use_rep="X_emb")
    integrated.obs = solution.obs.loc[integrated.obs.index].copy()
    integrated.uns = {**integrated.uns, **solution.uns}

    from scib.metrics import (
        silhouette as _silhouette,
        silhouette_batch as _silhouette_batch,
        graph_connectivity as _graph_conn,
        isolated_labels_asw,
    )
    from scib.metrics.lisi import lisi_graph_py

    out: dict = {}
    out["asw_label"] = float(
        _silhouette(integrated, label_key="cell_type", embed="X_emb")
    )
    out["asw_batch"] = float(
        _silhouette_batch(
            integrated, batch_key="batch", label_key="cell_type", embed="X_emb"
        )
    )
    out["graph_connectivity"] = float(
        _graph_conn(integrated, label_key="cell_type")
    )

    # iso_label_asw with the iso_threshold workaround
    tmp = integrated.obs[["cell_type", "batch"]].drop_duplicates()
    bpl = tmp.groupby("cell_type", observed=True).agg({"batch": "count"})
    default_thr = int(bpl.min().tolist()[0])
    n_batches = integrated.obs["batch"].nunique()
    iso_thr = n_batches + 1 if default_thr == n_batches else None
    out["isolated_label_asw"] = float(
        isolated_labels_asw(
            integrated,
            label_key="cell_type",
            batch_key="batch",
            embed="X_emb",
            iso_threshold=iso_thr,
            verbose=False,
        )
    )

    # ilisi / clisi
    nb = n_batches
    nl = integrated.obs["cell_type"].nunique()
    scores = lisi_graph_py(
        adata=integrated,
        obs_key="batch",
        n_neighbors=90,
        perplexity=None,
        subsample=None,
        n_cores=1,
        verbose=False,
    )
    out["ilisi"] = float((np.nanmedian(scores) - 1) / (nb - 1))
    scores = lisi_graph_py(
        adata=integrated,
        obs_key="cell_type",
        n_neighbors=90,
        perplexity=None,
        subsample=None,
        n_cores=1,
        verbose=False,
    )
    out["clisi"] = float((nl - np.nanmedian(scores)) / (nl - 1))
    return out


def eval_one(
    target_batch: str, *, sweep_dir: Path, solution: str
) -> dict:
    tag = f"target_{target_batch}".replace("/", "_")
    out_h5 = sweep_dir / "outputs" / f"{tag}.h5ad"
    res_json = sweep_dir / "results" / f"{tag}.json"
    if res_json.exists() and res_json.stat().st_size > 0:
        return {
            "target": target_batch,
            "status": "eval_skipped",
            "scores": json.loads(res_json.read_text()),
        }
    if not out_h5.exists():
        return {"target": target_batch, "status": "no_fit"}
    t0 = time.time()
    scores = quick_bio_eval(str(out_h5), solution)
    res_json.write_text(json.dumps(scores, indent=2))
    return {
        "target": target_batch,
        "status": "eval_ok",
        "dt": time.time() - t0,
        "scores": scores,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument(
        "--method-dir",
        default=str(REPO.parent / "task_batch_integration_forked" / "src" / "methods" / "condo"),
    )
    ap.add_argument(
        "--utils-dir",
        default=str(REPO.parent / "task_batch_integration_forked" / "src" / "utils"),
    )
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--n-gpus", type=int, default=2)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    (sweep_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (sweep_dir / "results").mkdir(parents=True, exist_ok=True)
    (sweep_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Enumerate batches
    print("Reading batches...", flush=True)
    ds = ad.read_h5ad(args.dataset)
    batches = sorted(ds.obs["batch"].astype(str).unique().tolist())
    counts = ds.obs["batch"].value_counts().to_dict()
    print(f"Found {len(batches)} batches:", flush=True)
    for b in batches:
        print(f"  {b}  n={counts[b]}", flush=True)

    common = dict(
        dataset=args.dataset,
        sweep_dir=sweep_dir,
        method_dir=Path(args.method_dir),
        utils_dir=Path(args.utils_dir),
        python=args.python,
    )

    def gpu_for(i): return i % args.n_gpus if args.n_gpus > 0 else None

    # Stage 1: fits (GPU)
    print("\n=== Stage 1: fits ===", flush=True)
    with ProcessPoolExecutor(max_workers=args.parallel) as ex:
        futs = {
            ex.submit(fit_one, b, **common, gpu_id=gpu_for(i)): b
            for i, b in enumerate(batches)
        }
        for fut in as_completed(futs):
            r = fut.result()
            print(
                f"[{r['status']}] target={r['target']} dt={r.get('dt', 0):.0f}s",
                flush=True,
            )

    # Stage 2: evals (CPU)
    print("\n=== Stage 2: evals ===", flush=True)
    all_results = {}
    for b in batches:
        r = eval_one(b, sweep_dir=sweep_dir, solution=args.solution)
        print(
            f"[{r['status']}] target={b} dt={r.get('dt', 0):.0f}s",
            flush=True,
        )
        if "scores" in r:
            all_results[b] = {**r["scores"], "target_n": counts[b]}

    summary_path = sweep_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {summary_path}", flush=True)

    # Print sorted table
    if all_results:
        print()
        print(
            f"{'target':<20} {'n':>6}  {'asw_label':>9} {'iso_asw':>8} "
            f"{'clisi':>6}  {'asw_batch':>9} {'graph_co':>9} {'ilisi':>6}"
        )
        print("-" * 100)
        rows = sorted(all_results.items(), key=lambda kv: -kv[1]["asw_label"])
        for tgt, d in rows:
            print(
                f"{tgt:<20} {d['target_n']:>6}  "
                f"{d['asw_label']:>9.4f} {d['isolated_label_asw']:>8.4f} "
                f"{d['clisi']:>6.4f}  {d['asw_batch']:>9.4f} "
                f"{d['graph_connectivity']:>9.4f} {d['ilisi']:>6.4f}"
            )


if __name__ == "__main__":
    main()
