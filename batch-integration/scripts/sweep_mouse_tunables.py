"""Sweep MMD-affine training tunables on mouse_pancreas_atlas (or any
single dataset). Fits + evals the cross product of n_epochs x patience x
weight_decay x wd_on_bias; eval is igraph + skip-kbet.

Resumable: skips configs whose result JSON already exists. Per-config
fit/eval logs live under {sweep_dir}/{logs|outputs|results}/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO.parent / "fullbench"

GRID = {
    "n_epochs": [15, 30],
    "patience": [3, 10],
    "weight_decay": [1e-4, 1e-3, 1e-2],
    "wd_on_bias": [False, True],
}


def tag(cfg: dict) -> str:
    return (
        f"ne{cfg['n_epochs']}_pat{cfg['patience']}"
        f"_wd{cfg['weight_decay']:.0e}"
        f"_wdb{int(cfg['wd_on_bias'])}"
    )


def expand(grid: dict) -> list[dict]:
    keys = list(grid)
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*(grid[k] for k in keys))
    ]


def fit_and_eval(
    cfg: dict, *, dataset_name: str, sweep_dir: Path, python: str,
    gpu_id: int | None,
) -> dict:
    t = tag(cfg)
    out_h5 = sweep_dir / "outputs" / f"{t}.h5ad"
    res_json = sweep_dir / "results" / f"{t}.json"
    fit_log = sweep_dir / "logs" / f"{t}.fit.log"
    eval_log = sweep_dir / "logs" / f"{t}.eval.log"
    if res_json.exists() and res_json.stat().st_size > 0:
        return {"tag": t, "status": "skipped", "cfg": cfg}

    dataset = BENCH / "datasets" / dataset_name / "dataset.h5ad"
    solution = BENCH / "datasets" / dataset_name / "solution.h5ad"
    env = dict(os.environ)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if not (out_h5.exists() and out_h5.stat().st_size > 0):
        fit_cmd = [
            python, str(REPO / "batch-integration" / "run_local.py"),
            "--divergence", "mmd", "--transform-type", "affine",
            "--rep", "features", "--device", "cuda",
            "--n-epochs", str(cfg["n_epochs"]),
            "--learning-rate", "1e-3",
            "--mmd-size", "40",
            "--batch-size", "8",
            "--weight-decay", str(cfg["weight_decay"]),
            "--patience", str(cfg["patience"]),
            *(["--wd-on-bias"] if cfg["wd_on_bias"] else []),
            "--random-state", "42",
            "--input", str(dataset),
            "--output", str(out_h5),
        ]
        t0 = time.time()
        with open(fit_log, "wb") as f:
            rc = subprocess.call(
                fit_cmd, stdout=f, stderr=subprocess.STDOUT, env=env
            )
        if rc != 0:
            return {"tag": t, "status": "fit_failed", "cfg": cfg,
                    "fit_log": str(fit_log), "dt": time.time() - t0}
        fit_dt = time.time() - t0
    else:
        fit_dt = 0.0

    eval_cmd = [
        python, str(REPO / "batch-integration" / "eval_variant.py"),
        "--integrated", str(out_h5),
        "--dataset", str(dataset),
        "--solution", str(solution),
        "--output", str(res_json),
        "--skip-kbet",
    ]
    t1 = time.time()
    with open(eval_log, "wb") as f:
        rc = subprocess.call(eval_cmd, stdout=f, stderr=subprocess.STDOUT)
    eval_dt = time.time() - t1
    if rc != 0:
        return {"tag": t, "status": "eval_failed", "cfg": cfg,
                "eval_log": str(eval_log),
                "fit_dt": fit_dt, "eval_dt": eval_dt}
    return {"tag": t, "status": "ok", "cfg": cfg,
            "fit_dt": fit_dt, "eval_dt": eval_dt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="work/sweep_mouse_tunables")
    ap.add_argument("--dataset", default="mouse_pancreas_atlas")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument(
        "--n-gpus", type=int, default=2,
        help="Rotate CUDA_VISIBLE_DEVICES across this many GPUs.",
    )
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    for sub in ("outputs", "results", "logs"):
        (sweep_dir / sub).mkdir(parents=True, exist_ok=True)

    configs = expand(GRID)
    (sweep_dir / "grid.json").write_text(
        json.dumps({"grid": GRID, "configs": configs}, indent=2)
    )

    print(
        f"Sweep on {args.dataset}: {len(configs)} configs, "
        f"parallel={args.parallel}, n_gpus={args.n_gpus}",
        flush=True,
    )

    summary_path = sweep_dir / "summary.jsonl"
    summary_path.touch()
    with summary_path.open("a") as fsum:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = {
                ex.submit(
                    fit_and_eval, cfg,
                    dataset_name=args.dataset, sweep_dir=sweep_dir,
                    python=args.python,
                    gpu_id=(ix % args.n_gpus) if args.n_gpus > 0 else None,
                ): cfg
                for ix, cfg in enumerate(configs)
            }
            for fut in as_completed(futures):
                res = fut.result()
                fsum.write(json.dumps(res) + "\n")
                fsum.flush()
                print(
                    f"[{res['status']}] {res['tag']}  "
                    f"fit_dt={res.get('fit_dt', 0):.0f}s  "
                    f"eval_dt={res.get('eval_dt', 0):.0f}s",
                    flush=True,
                )
    print("sweep complete", flush=True)


if __name__ == "__main__":
    main()
