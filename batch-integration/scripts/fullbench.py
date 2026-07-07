"""Run condo MMD-affine on the openproblems task_batch_integration datasets.

Iterates a list of dataset names, runs the fit + eval on each, parallel
across GPUs via CUDA_VISIBLE_DEVICES. Resumable: skips datasets whose
result JSON already exists.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FORK = REPO.parent / "task_batch_integration_forked"
BENCH = REPO.parent / "fullbench"

DATASETS = [
    "dkd",
    "gtex_v9",
    "hypomap",
    "immune_cell_atlas",
    "mouse_pancreas_atlas",
    "tabula_sapiens",
]

# Sort largest-first so the slow jobs start early and overlap with the small
# ones at the end (better wall-time packing on N workers).
DATASETS_BY_SIZE_DESC = [
    "tabula_sapiens",     # 483k cells, 29 batches
    "mouse_pancreas_atlas",  # 302k cells, 56 batches (most source fits!)
    "hypomap",            # 385k cells, 24 batches
    "immune_cell_atlas",  # 330k cells, 12 batches
    "gtex_v9",            # 209k cells, 16 batches
    "dkd",                # 39k cells, 11 batches
]


def fit_and_eval(
    ds: str,
    *,
    bench_dir: Path,
    sweep_dir: Path,
    method_dir: Path,
    utils_dir: Path,
    python: str,
    config: dict,
    gpu_id: int | None,
) -> dict:
    out_h5 = sweep_dir / "outputs" / f"{ds}.h5ad"
    res_json = sweep_dir / "results" / f"{ds}.json"
    fit_log = sweep_dir / "logs" / f"{ds}.fit.log"
    eval_log = sweep_dir / "logs" / f"{ds}.eval.log"
    if res_json.exists() and res_json.stat().st_size > 0:
        return {"dataset": ds, "status": "skipped"}

    dataset = bench_dir / "datasets" / ds / "dataset.h5ad"
    solution = bench_dir / "datasets" / ds / "solution.h5ad"
    for p in (dataset, solution):
        if not p.exists():
            return {"dataset": ds, "status": "missing", "path": str(p)}

    fit_cmd = [
        python,
        str(REPO / "batch-integration" / "run_local.py"),
        "--divergence", config.get("divergence", "mmd"),
        "--transform-type", config.get("transform_type", "affine"),
        "--rep", config.get("rep", "features"),
        "--device", "cuda",
        "--method-dir", str(method_dir),
        "--utils-dir", str(utils_dir),
        "--n-epochs", str(config.get("n_epochs", "auto")),
        "--learning-rate", str(config.get("learning_rate", 1e-3)),
        "--mmd-size", str(config.get("mmd_size", 40)),
        "--batch-size", str(config.get("batch_size", 8)),
        "--weight-decay", str(config.get("weight_decay", "auto")),
        "--patience", str(config.get("patience", 3)),
        "--dplr-rank", str(config.get("dplr_rank", 16)),
        "--random-state", str(config.get("random_state", 42)),
        "--ranking-strategy", str(config.get("ranking_strategy", "celltype_silhouette")),
        "--input", str(dataset),
        "--output", str(out_h5),
    ]

    env = dict(os.environ)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if out_h5.exists() and out_h5.stat().st_size > 0:
        # Reuse a prior fit; only the eval is missing.
        fit_dt = 0.0
    else:
        t0 = time.time()
        with open(fit_log, "wb") as f:
            rc = subprocess.call(
                fit_cmd, stdout=f, stderr=subprocess.STDOUT, env=env
            )
        if rc != 0:
            return {
                "dataset": ds, "status": "fit_failed",
                "fit_log": str(fit_log), "dt": time.time() - t0,
            }
        fit_dt = time.time() - t0

    eval_cmd = [
        python,
        str(REPO / "batch-integration" / "eval_variant.py"),
        "--integrated", str(out_h5),
        "--dataset", str(dataset),
        "--solution", str(solution),
        "--output", str(res_json),
    ]
    if config.get("skip_kbet"):
        eval_cmd.append("--skip-kbet")
    t1 = time.time()
    with open(eval_log, "wb") as f:
        rc = subprocess.call(eval_cmd, stdout=f, stderr=subprocess.STDOUT)
    eval_dt = time.time() - t1
    if rc != 0:
        return {
            "dataset": ds, "status": "eval_failed",
            "eval_log": str(eval_log),
            "fit_dt": fit_dt, "eval_dt": eval_dt,
        }
    return {"dataset": ds, "status": "ok", "fit_dt": fit_dt, "eval_dt": eval_dt}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", default=str(BENCH))
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument(
        "--method-dir", default=str(DEFAULT_FORK / "src" / "methods" / "condo")
    )
    parser.add_argument(
        "--utils-dir", default=str(DEFAULT_FORK / "src" / "utils")
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument(
        "--n-gpus", type=int, default=0,
        help="Rotate CUDA_VISIBLE_DEVICES across this many GPUs.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=DATASETS_BY_SIZE_DESC,
        help="Datasets to run (default: all, sorted largest-first)",
    )
    # MMD-affine config; defaults match the AdamW pull_to_identity winner.
    parser.add_argument("--divergence", default="mmd")
    parser.add_argument("--transform-type", dest="transform_type", default="affine",
                        choices=["location-scale", "affine", "diagonal-plus-low-rank"])
    parser.add_argument("--dplr-rank", dest="dplr_rank", type=int, default=16,
                        help="rank for diagonal-plus-low-rank transform")
    parser.add_argument("--rep", default="features")
    parser.add_argument("--n-epochs", default="auto",
                        help="integer or 'auto' (transform-specific default)")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mmd-size", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", default="auto",
                        help="float or 'auto' (transform-specific default)")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--ranking-strategy", dest="ranking_strategy",
        choices=[
            "celltype_silhouette", "random", "biggest",
            "batch_silhouette_low", "batch_silhouette_high",
        ],
        default="celltype_silhouette",
        help="Agglomerative seed + neighbor ranking criterion (ablation axis).",
    )
    parser.add_argument("--skip-kbet", dest="skip_kbet", action="store_true",
                        help="skip kbet in eval (slow; not in composites)")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    (sweep_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (sweep_dir / "results").mkdir(parents=True, exist_ok=True)
    (sweep_dir / "logs").mkdir(parents=True, exist_ok=True)

    config = {
        "divergence": args.divergence,
        "transform_type": args.transform_type,
        "rep": args.rep,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "mmd_size": args.mmd_size,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "dplr_rank": args.dplr_rank,
        "random_state": args.random_state,
        "ranking_strategy": args.ranking_strategy,
        "skip_kbet": args.skip_kbet,
    }
    (sweep_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(
        f"Run: {len(args.datasets)} datasets, parallel={args.parallel}, "
        f"config={config}",
        flush=True,
    )

    common = dict(
        bench_dir=Path(args.bench_dir),
        sweep_dir=sweep_dir,
        method_dir=Path(args.method_dir),
        utils_dir=Path(args.utils_dir),
        python=args.python,
        config=config,
    )

    def gpu_for(ix: int) -> int | None:
        if args.n_gpus <= 0:
            return None
        return ix % args.n_gpus

    summary_path = sweep_dir / "summary.jsonl"
    summary_path.touch()
    with summary_path.open("a") as fsum:
        if args.parallel <= 1:
            for ix, ds in enumerate(args.datasets):
                res = fit_and_eval(ds, **common, gpu_id=gpu_for(ix))
                fsum.write(json.dumps(res) + "\n"); fsum.flush()
                print(f"[{res['status']}] {ds}  fit_dt={res.get('fit_dt', 0):.0f}s  eval_dt={res.get('eval_dt', 0):.0f}s", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.parallel) as ex:
                futures = {
                    ex.submit(fit_and_eval, ds, **common, gpu_id=gpu_for(ix)): ds
                    for ix, ds in enumerate(args.datasets)
                }
                for fut in as_completed(futures):
                    res = fut.result()
                    fsum.write(json.dumps(res) + "\n"); fsum.flush()
                    print(
                        f"[{res['status']}] {res['dataset']}  "
                        f"fit_dt={res.get('fit_dt', 0):.0f}s  "
                        f"eval_dt={res.get('eval_dt', 0):.0f}s",
                        flush=True,
                    )

    print("Full benchmark run complete", flush=True)


if __name__ == "__main__":
    main()
