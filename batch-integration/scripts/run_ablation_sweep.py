"""Driver for the agglomerative-ranking ablation sweep.

Runs (dataset x ranking_strategy) fit+eval pipelines through fullbench.py,
sequentially (concurrency=1 -- concurrency>1 both slows fits ~2.4x via GPU
contention and risks OOM on the big datasets). Resumable: a (dataset, strategy)
whose result JSON already exists is skipped; a fit whose output h5ad exists is
reused (only its eval re-runs). Retries once, with a CUDA health probe, if a
job produces no result.

All variants hold the v3_baseline_seeded config fixed (mmd/affine/features,
ne=5, wd=1e-4, lr=1e-3, mmd_size=40, batch_size=8, patience=3) and vary only
--ranking-strategy. The 'celltype_silhouette' strategy reproduces
v3_baseline_seeded; the others are the ablations.

Typical uses
------------
This 124 GB box (fits <=70 GB): dkd, gtex, hypomap, mouse_pancreas.
    python run_ablation_sweep.py --datasets dkd gtex_v9 hypomap mouse_pancreas_atlas
Big-memory GPU box (immune/tabula fits need 100-170 GB) WITH kbet:
    python run_ablation_sweep.py --datasets immune_cell_atlas tabula_sapiens

Memory note: peak fit RAM = 2*min(src,tgt)*mmd_size*d*8 bytes (driven by the
largest batch); peak eval RAM is ~60-110 GB on 300k+ cells and kbet loads a
2nd copy -- budget accordingly (or --skip-kbet on a memory-tight box).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]           # condo-adapter/
FULLBENCH = REPO / "batch-integration" / "scripts" / "fullbench.py"

ALL_DATASETS = ["dkd", "gtex_v9", "immune_cell_atlas", "hypomap",
                "mouse_pancreas_atlas", "tabula_sapiens"]
ALL_STRATEGIES = ["celltype_silhouette", "celltype_silhouette_low",
                  "random", "biggest", "smallest",
                  "batch_silhouette_low", "batch_silhouette_high"]


def label_for(strategy: str) -> str:
    return {
        "celltype_silhouette": "abl_celltype_silhouette",
        "celltype_silhouette_low": "abl_celltype_sil_low",
        "random": "abl_random_rs42",
        "biggest": "abl_biggest",
        "smallest": "abl_smallest",
        "batch_silhouette_low": "abl_batch_sil_low",
        "batch_silhouette_high": "abl_batch_sil_high",
    }[strategy]


def gpu_ok(python: str) -> bool:
    try:
        r = subprocess.run(
            [python, "-c", "import torch;a=torch.randn(64,64,device='cuda');"
                           "assert float((a@a).sum())==float((a@a).sum())"],
            capture_output=True, timeout=120, env=dict(os.environ))
        return r.returncode == 0
    except Exception:
        return False


def run_job(ds, strategy, args, fixed) -> dict:
    sweep_dir = Path(args.work_dir) / label_for(strategy)
    res = sweep_dir / "results" / f"{ds}.json"
    if res.exists() and res.stat().st_size > 0:
        return {"ds": ds, "strategy": strategy, "status": "skipped", "dt": 0}
    cmd = [
        args.python, str(FULLBENCH),
        "--bench-dir", args.bench_dir,
        "--sweep-dir", str(sweep_dir),
        "--method-dir", args.method_dir,
        "--utils-dir", args.utils_dir,
        "--datasets", ds,
        "--ranking-strategy", strategy,
        "--random-state", str(args.random_state),
        "--n-gpus", "1",
        *fixed,
    ]
    if args.skip_kbet:
        cmd.append("--skip-kbet")
    t0 = time.time()
    for attempt in (1, 2):
        if not gpu_ok(args.python):
            print(f"   !! GPU unhealthy before {strategy}/{ds}; wait 60s", flush=True)
            time.sleep(60)
        (sweep_dir / "logs").mkdir(parents=True, exist_ok=True)
        with open(sweep_dir / "logs" / f"{ds}.driver.log", "ab") as f:
            subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ))
        if res.exists() and res.stat().st_size > 0:
            return {"ds": ds, "strategy": strategy,
                    "status": "ok" if attempt == 1 else "ok(retry)",
                    "dt": time.time() - t0}
        print(f"   !! {strategy}/{ds} attempt {attempt} produced no result", flush=True)
        time.sleep(30)
    return {"ds": ds, "strategy": strategy, "status": "FAILED", "dt": time.time() - t0}


def main() -> None:
    default_fork = REPO.parent / "task_batch_integration"
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    ap.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES, choices=ALL_STRATEGIES)
    ap.add_argument("--skip-kbet", action="store_true",
                    help="skip kbet (use on memory-tight boxes; kbet double-loads "
                         "the data and OOMs big-dataset evals under ~124 GB)")
    ap.add_argument("--work-dir", default=str(REPO.parent / "work"))
    ap.add_argument("--bench-dir", default=str(REPO.parent / "fullbench"))
    ap.add_argument("--method-dir", default=str(default_fork / "src" / "methods" / "condo"))
    ap.add_argument("--utils-dir", default=str(default_fork / "src" / "utils"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--random-state", type=int, default=42)
    # v3_baseline_seeded config (do not change for the ablation)
    ap.add_argument("--transform-type", default="affine",
                    choices=["affine", "location-scale"])
    ap.add_argument("--mmd-size", default="40")
    ap.add_argument("--n-epochs", default="5")
    ap.add_argument("--weight-decay", default="1e-4",
                    help="v3_baseline_seeded uses 1e-4 for affine; "
                         "location-scale conventionally uses 1e-5.")
    args = ap.parse_args()

    fixed = [
        "--divergence", "mmd", "--transform-type", args.transform_type,
        "--rep", "features",
        "--n-epochs", str(args.n_epochs), "--weight-decay", str(args.weight_decay),
        "--learning-rate", "1e-3", "--mmd-size", str(args.mmd_size),
        "--batch-size", "8", "--patience", "3",
    ]
    jobs = [(ds, st) for ds in args.datasets for st in args.strategies]
    print(f">> {len(jobs)} jobs, SEQUENTIAL, skip_kbet={args.skip_kbet}, "
          f"work_dir={args.work_dir}", flush=True)
    print(f">> CONDO_KBET_PYTHON={os.environ.get('CONDO_KBET_PYTHON', '(unset!)')}",
          flush=True)
    failures = []
    for i, (ds, st) in enumerate(jobs, 1):
        r = run_job(ds, st, args, fixed)
        if r["status"] == "FAILED":
            failures.append((st, ds))
        print(f"[{i}/{len(jobs)}] [{r['status']:>9s}] {label_for(st):26s} "
              f"{ds:22s} dt={r.get('dt', 0):.0f}s", flush=True)
    if failures:
        print(f">> {len(failures)} FAILURES: {failures}", flush=True)
    print(">> SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
