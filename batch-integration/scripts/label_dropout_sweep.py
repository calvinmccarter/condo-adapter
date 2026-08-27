"""Label-availability robustness sweep for ConDo.

Randomly hides a fraction f of cell-type labels from the ConDo fit and measures
how integration quality degrades. The merge ordering is fixed to the best
LABEL-FREE strategy (batch_silhouette_low) so the only thing that varies with f
is the conditioning, not the seed/merge order. Eval uses the true labels and the
composite-7 metric set only (fast).

Per (transform, dataset): f=0 baseline + f in {0.25,0.5,0.75} x {bucket,excluded}
x seeds + f=1.0 (bucket only, deterministic). Seeds: 3 on small datasets, 1 on
the atlases (per-type stats are stable at scale). Resumable, GPU-parallel.

Example:
    setsid nohup .venv-condo-bench/bin/python \
        batch-integration/scripts/label_dropout_sweep.py \
        --sweep-dir work/label_dropout --n-gpus 2 --parallel 2 > LOG 2>&1 &
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

# smallest-first
DATASETS = [
    "dkd", "gtex_v9", "mouse_pancreas_atlas",
    "immune_cell_atlas", "hypomap", "tabula_sapiens",
]
SMALL = {"dkd", "gtex_v9"}  # 3 seeds; atlases get 1

# fixed official-ish config for this experiment
FIXED = dict(
    divergence="mmd", rep="features", n_epochs="5", learning_rate="1e-3",
    mmd_size="40", batch_size="8", weight_decay="1e-4", patience="3",
    dplr_rank="16", random_state="42", ranking_strategy="batch_silhouette_low",
)


def build_runs(transforms, datasets, fractions, handlings, seeds_small, seeds_atlas):
    runs = []
    for t in transforms:
        for ds in datasets:
            S = seeds_small if ds in SMALL else seeds_atlas
            runs.append((t, ds, 0.0, "full", 0))  # baseline
            for f in fractions:
                if f == 0.0:
                    continue
                for h in handlings:
                    if h == "excluded" and f >= 1.0:
                        continue  # undefined (no labels to condition on)
                    seed_list = [0] if f >= 1.0 else list(range(S))
                    for s in seed_list:
                        runs.append((t, ds, f, h, s))
    return runs


def _key(t, ds, f, h, s):
    return f"{t.replace('-', '_')}__{ds}__f{f}__{h}__s{s}"


def run_one(spec, *, sweep_dir, bench_dir, method_dir, utils_dir, python, gpu_id):
    t, ds, f, h, s = spec
    key = _key(t, ds, f, h, s)
    out_h5 = sweep_dir / "outputs" / f"{key}.h5ad"
    res_json = sweep_dir / "results" / f"{key}.json"
    fit_log = sweep_dir / "logs" / f"{key}.fit.log"
    eval_log = sweep_dir / "logs" / f"{key}.eval.log"
    meta = dict(transform=t, dataset=ds, frac=f, handling=h, seed=s, key=key)
    if res_json.exists() and res_json.stat().st_size > 0:
        return {**meta, "status": "skipped"}

    dataset = bench_dir / "datasets" / ds / "dataset.h5ad"
    solution = bench_dir / "datasets" / ds / "solution.h5ad"
    for p in (dataset, solution):
        if not p.exists():
            return {**meta, "status": "missing", "path": str(p)}

    env = dict(os.environ)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    fit_dt = 0.0
    if not (out_h5.exists() and out_h5.stat().st_size > 0):
        fit_cmd = [
            python, str(REPO / "batch-integration" / "run_local.py"),
            "--divergence", FIXED["divergence"], "--transform-type", t,
            "--rep", FIXED["rep"], "--device", "cuda",
            "--method-dir", str(method_dir), "--utils-dir", str(utils_dir),
            "--n-epochs", FIXED["n_epochs"], "--learning-rate", FIXED["learning_rate"],
            "--mmd-size", FIXED["mmd_size"], "--batch-size", FIXED["batch_size"],
            "--weight-decay", FIXED["weight_decay"], "--patience", FIXED["patience"],
            "--dplr-rank", FIXED["dplr_rank"], "--random-state", FIXED["random_state"],
            "--ranking-strategy", FIXED["ranking_strategy"],
            "--label-dropout", str(f),
            "--dropout-handling", ("bucket" if h == "full" else h),
            "--dropout-seed", str(s),
            "--input", str(dataset), "--output", str(out_h5),
        ]
        t0 = time.time()
        with open(fit_log, "wb") as fh:
            rc = subprocess.call(fit_cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        fit_dt = time.time() - t0
        if rc != 0:
            return {**meta, "status": "fit_failed", "fit_log": str(fit_log), "fit_dt": fit_dt}

    eval_cmd = [
        python, str(REPO / "batch-integration" / "eval_variant.py"),
        "--integrated", str(out_h5), "--dataset", str(dataset),
        "--solution", str(solution), "--output", str(res_json),
        "--skip-kbet", "--metrics", "composite",
    ]
    t1 = time.time()
    with open(eval_log, "wb") as fh:
        rc = subprocess.call(eval_cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    eval_dt = time.time() - t1
    if rc != 0:
        return {**meta, "status": "eval_failed", "eval_log": str(eval_log),
                "fit_dt": fit_dt, "eval_dt": eval_dt}
    return {**meta, "status": "ok", "fit_dt": fit_dt, "eval_dt": eval_dt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--bench-dir", default=str(BENCH))
    ap.add_argument("--method-dir", default=str(DEFAULT_FORK / "src" / "methods" / "condo"))
    ap.add_argument("--utils-dir", default=str(DEFAULT_FORK / "src" / "utils"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--n-gpus", type=int, default=0)
    ap.add_argument("--transforms", nargs="*", default=["affine", "location-scale"])
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--fractions", nargs="*", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--handlings", nargs="*", default=["bucket", "excluded"])
    ap.add_argument("--seeds-small", type=int, default=3)
    ap.add_argument("--seeds-atlas", type=int, default=1)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    for sub in ("outputs", "results", "logs"):
        (sweep_dir / sub).mkdir(parents=True, exist_ok=True)
    (sweep_dir / "config.json").write_text(json.dumps({**FIXED, "fractions": args.fractions,
        "handlings": args.handlings, "transforms": args.transforms,
        "seeds_small": args.seeds_small, "seeds_atlas": args.seeds_atlas}, indent=2))

    order = {ds: i for i, ds in enumerate(DATASETS)}
    runs = build_runs(args.transforms, args.datasets, args.fractions, args.handlings,
                      args.seeds_small, args.seeds_atlas)
    runs.sort(key=lambda r: (order.get(r[1], 99), r[2]))  # smallest dataset first, then frac
    print(f"Label-dropout sweep: {len(runs)} runs, parallel={args.parallel}", flush=True)

    common = dict(sweep_dir=sweep_dir, bench_dir=Path(args.bench_dir),
                  method_dir=Path(args.method_dir),
                  utils_dir=Path(args.utils_dir), python=args.python)

    def gpu_for(ix):
        return None if args.n_gpus <= 0 else ix % args.n_gpus

    summary = sweep_dir / "summary.jsonl"
    summary.touch()
    with summary.open("a") as fsum:
        if args.parallel <= 1:
            for ix, spec in enumerate(runs):
                r = run_one(spec, **common, gpu_id=gpu_for(ix))
                fsum.write(json.dumps(r) + "\n"); fsum.flush()
                print(f"[{r['status']}] {r['key']}  fit={r.get('fit_dt',0):.0f}s eval={r.get('eval_dt',0):.0f}s", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.parallel) as ex:
                futs = {ex.submit(run_one, spec, **common, gpu_id=gpu_for(ix)): spec
                        for ix, spec in enumerate(runs)}
                for fut in as_completed(futs):
                    r = fut.result()
                    fsum.write(json.dumps(r) + "\n"); fsum.flush()
                    print(f"[{r['status']}] {r['key']}  fit={r.get('fit_dt',0):.0f}s eval={r.get('eval_dt',0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
