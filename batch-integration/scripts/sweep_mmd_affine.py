"""Pruned hyperparameter sweep for ConDo MMD-affine on task_batch_integration.

Grid (3 seeds = 54 total fits):
    n_epochs       ∈ {5, 30, 100}
    learning_rate  ∈ {1e-3, 3e-3, 1e-2}
    mmd_size       ∈ {20, 40}
    batch_size     ∈ {8}
    weight_decay   ∈ {1e-4}
    random_state   ∈ {42, 7, 1729}

Each config is fit + evaluated by spawning ``run_local.py`` and
``eval_variant.py`` as subprocesses. Output goes to
``<sweep_dir>/{outputs,results,logs}/`` with one h5ad + one JSON per
config. Configs whose result JSON already exists are skipped, so the
sweep is resumable.

Usage::

    python batch-integration/scripts/sweep_mmd_affine.py \\
        --dataset  /path/to/dataset.h5ad \\
        --solution /path/to/solution.h5ad \\
        --sweep-dir /path/to/work/sweep_mmd_affine \\
        --parallel 4
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

REPO = Path(__file__).resolve().parents[2]  # condo-adapter
DEFAULT_FORK = REPO.parent / "task_batch_integration_forked"

# --- AdamW grids ----------------------------------------------------------
# Original 54-config grid (n_epochs × lr × mmd_size × seed).
GRID_V1 = {
    "n_epochs": [5, 30, 100],
    "learning_rate": [1e-3, 3e-3, 1e-2],
    "mmd_size": [20, 40],
    "batch_size": [8],
    "weight_decay": [1e-4],
    "random_state": [42, 7, 1729],
    "optimizer": ["adamw"],
}
# Low-lr follow-up at n_epochs=5 with extended mmd_size.
GRID_V2 = {
    "n_epochs": [5],
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "mmd_size": [20, 40, 80],
    "batch_size": [8],
    "weight_decay": [1e-4],
    "random_state": [42, 7, 1729],
    "optimizer": ["adamw"],
}

# --- Muon grids -----------------------------------------------------------
# Same shape as the AdamW grid, lr ranges shifted ~10x upward to match Muon's
# normalized-update step size. Recommended Muon defaults around lr=2e-2.
GRID_MUON_V1 = {
    "n_epochs": [5, 30, 100],
    "learning_rate": [1e-2, 3e-2, 1e-1],
    "mmd_size": [20, 40],
    "batch_size": [8],
    "weight_decay": [1e-4],
    "random_state": [42, 7, 1729],
    "optimizer": ["muon"],
}
GRID_MUON_V2 = {
    "n_epochs": [5],
    "learning_rate": [1e-3, 3e-3, 1e-2],
    "mmd_size": [20, 40, 80],
    "batch_size": [8],
    "weight_decay": [1e-4],
    "random_state": [42, 7, 1729],
    "optimizer": ["muon"],
}

GRIDS = {
    "adamw": [GRID_V1, GRID_V2],
    "muon": [GRID_MUON_V1, GRID_MUON_V2],
}


def _expand(grid: dict) -> list[dict]:
    keys = list(grid)
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*(grid[k] for k in keys))
    ]


def iter_configs(which: str = "adamw") -> list[dict]:
    if which not in GRIDS:
        raise ValueError(f"unknown grid {which!r}; pick one of {list(GRIDS)}")
    seen: set[str] = set()
    out: list[dict] = []
    for grid in GRIDS[which]:
        for cfg in _expand(grid):
            t = tag_for(cfg)
            if t in seen:
                continue
            seen.add(t)
            out.append(cfg)
    return out


def tag_for(cfg: dict) -> str:
    base = (
        "ne{n_epochs}_lr{learning_rate:.0e}"
        "_ms{mmd_size}_bs{batch_size}"
        "_wd{weight_decay:.0e}_seed{random_state}"
    ).format(**cfg)
    # Include optimizer in the tag only when it's not the default 'adamw',
    # so legacy adamw tags (and the existing pull_to_zero / pull_to_identity
    # archives) keep their original names and remain resumable.
    opt = cfg.get("optimizer", "adamw")
    if opt != "adamw":
        base = base + f"_{opt}"
    return base


def fit_one(
    cfg: dict,
    *,
    dataset: str,
    solution: str,
    sweep_dir: Path,
    method_dir: Path,
    utils_dir: Path,
    python: str,
    device: str = "cpu",
    gpu_id: int | None = None,
) -> dict:
    tag = tag_for(cfg)
    out_h5 = sweep_dir / "outputs" / f"{tag}.h5ad"
    res_json = sweep_dir / "results" / f"{tag}.json"
    fit_log = sweep_dir / "logs" / f"{tag}.fit.log"
    eval_log = sweep_dir / "logs" / f"{tag}.eval.log"

    if res_json.exists() and res_json.stat().st_size > 0:
        return {"tag": tag, "status": "skipped", "cfg": cfg}

    # Fit via run_local.py (does the same as the viash component shim).
    fit_cmd = [
        python,
        str(REPO / "batch-integration" / "run_local.py"),
        "--divergence",
        "mmd",
        "--transform-type",
        "affine",
        "--rep",
        "features",
        "--method-dir",
        str(method_dir),
        "--utils-dir",
        str(utils_dir),
        "--n-epochs",
        str(cfg["n_epochs"]),
        "--learning-rate",
        str(cfg["learning_rate"]),
        "--mmd-size",
        str(cfg["mmd_size"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--weight-decay",
        str(cfg["weight_decay"]),
        "--random-state",
        str(cfg["random_state"]),
        "--device",
        device,
        "--optimizer",
        str(cfg.get("optimizer", "adamw")),
        "--input",
        dataset,
        "--output",
        str(out_h5),
    ]

    # Optionally pin to a specific GPU for parallel workers.
    env = dict(os.environ)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    t0 = time.time()
    with open(fit_log, "wb") as f:
        rc = subprocess.call(fit_cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if rc != 0:
        return {
            "tag": tag,
            "status": "fit_failed",
            "cfg": cfg,
            "fit_log": str(fit_log),
            "dt": time.time() - t0,
        }

    # Eval
    eval_cmd = [
        python,
        str(REPO / "batch-integration" / "eval_variant.py"),
        "--integrated",
        str(out_h5),
        "--dataset",
        dataset,
        "--solution",
        solution,
        "--output",
        str(res_json),
    ]
    with open(eval_log, "wb") as f:
        rc = subprocess.call(eval_cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if rc != 0:
        return {
            "tag": tag,
            "status": "eval_failed",
            "cfg": cfg,
            "eval_log": str(eval_log),
            "dt": dt,
        }

    return {"tag": tag, "status": "ok", "cfg": cfg, "dt": dt}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument(
        "--method-dir",
        default=str(DEFAULT_FORK / "src" / "methods" / "condo"),
        help="Path to fork's src/methods/condo (contains condo_runner.py)",
    )
    parser.add_argument(
        "--utils-dir",
        default=str(DEFAULT_FORK / "src" / "utils"),
        help="Path to fork's src/utils (contains read_anndata_partial.py)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for subprocesses",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device passed to the fit subprocess ('cpu', 'cuda', 'cuda:0', ...)",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=0,
        help="If >0, rotate CUDA_VISIBLE_DEVICES across this many GPUs (round-robin by submission order). Implies --device cuda.",
    )
    parser.add_argument(
        "--grid",
        choices=list(GRIDS),
        default="adamw",
        help="Which grid set to expand (adamw or muon).",
    )
    args = parser.parse_args()
    if args.n_gpus > 0 and not args.device.startswith("cuda"):
        args.device = "cuda"

    sweep_dir = Path(args.sweep_dir)
    (sweep_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (sweep_dir / "results").mkdir(parents=True, exist_ok=True)
    (sweep_dir / "logs").mkdir(parents=True, exist_ok=True)

    configs = iter_configs(args.grid)
    print(
        f"Sweep: {len(configs)} configs, parallel={args.parallel}, grid={args.grid}",
        flush=True,
    )
    (sweep_dir / "grid.json").write_text(json.dumps(configs, indent=2))

    common = dict(
        dataset=args.dataset,
        solution=args.solution,
        sweep_dir=sweep_dir,
        method_dir=Path(args.method_dir),
        utils_dir=Path(args.utils_dir),
        python=args.python,
        device=args.device,
    )

    def gpu_for(ix: int) -> int | None:
        if args.n_gpus <= 0:
            return None
        return ix % args.n_gpus

    summary_path = sweep_dir / "sweep_summary.jsonl"
    summary_path.touch()
    with summary_path.open("a") as fsum:
        if args.parallel <= 1:
            for ix, cfg in enumerate(configs):
                res = fit_one(cfg, **common, gpu_id=gpu_for(ix))
                fsum.write(json.dumps(res) + "\n")
                fsum.flush()
                print(
                    f"[{res['status']}] {res['tag']} dt={res.get('dt', 0):.1f}s",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=args.parallel) as ex:
                futures = {
                    ex.submit(fit_one, cfg, **common, gpu_id=gpu_for(ix)): cfg
                    for ix, cfg in enumerate(configs)
                }
                for fut in as_completed(futures):
                    res = fut.result()
                    fsum.write(json.dumps(res) + "\n")
                    fsum.flush()
                    print(
                        f"[{res['status']}] {res['tag']} dt={res.get('dt', 0):.1f}s",
                        flush=True,
                    )

    print("Sweep complete", flush=True)


if __name__ == "__main__":
    main()
